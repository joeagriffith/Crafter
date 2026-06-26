#!/usr/bin/env python3
"""Live Weights & Biases streamer for the running DreamerV3 size-ladder sweep.

Run with the MAIN project venv (it has ``wandb`` + ``crafter_rl``):
    .venv/bin/python experiments/20260625-dreamerv3/wandb_streamer.py

Behaviour (mirrors ttlcd_sidecar.py, but the sink is W&B instead of the LCD):
  * Every ~5s, determine the ACTIVE trial from the last ``[START]`` line of
    ``sweep_summary.txt`` (model size, seed, out dir).
  * Maintain ONE W&B run per (sweep_id, size, seed) via
    ``crafter_rl.wandb.init_run`` with a DETERMINISTIC id
    (``crafter_rl.wandb.run_id``) and ``resume="allow"``. Because the post-hoc
    ``report_results.py`` derives the SAME id, the live stream and the final
    summary land on ONE run -- no duplicates.
  * Tail that trial's ``metrics.jsonl`` incrementally (by byte offset) and
    ``wandb.log`` each new record's numeric metrics at its ``step``.
  * On trial change, ``wandb.finish()`` the old run and open the new one.
  * Never crash: every file read / parse / W&B call is guarded.
  * Exit cleanly when the orchestrator is gone AND no trial is active.

Dependency-light: stdlib + wandb + crafter_rl.
"""

import json
import os
import sys
import time
import traceback

# Make crafter_rl importable regardless of cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import wandb
    from crafter_rl.wandb import init_run, run_id
except Exception as exc:  # pragma: no cover - deps should be installed
    print(f"[wandb-stream] FATAL: cannot import wandb/crafter_rl: {exc}", flush=True)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "sweep_summary.txt")
PROJECT = "crafter"
EXP_TYPE = "baseline"
ALGO = "dreamerv3"
POLL_S = 5.0
PINNED_SHA = "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f"


def log(msg):
    print(f"[wandb-stream {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_active_trial():
    """Return (size, seed, out_dir) from the last [START] line, or None."""
    try:
        last = None
        with open(SUMMARY, "r") as fh:
            for line in fh:
                if line.startswith("[START]"):
                    last = line
    except FileNotFoundError:
        return None
    except Exception as exc:
        log(f"warn: reading summary failed: {exc}")
        return None
    if not last:
        return None
    size = seed = out = None
    for tok in last.split():
        if tok.startswith("model="):
            size = tok[len("model="):]
        elif tok.startswith("seed="):
            seed = tok[len("seed="):]
        elif tok.startswith("out="):
            out = tok[len("out="):]
    if not (size and seed and out):
        return None
    return size, seed, out


def sweep_id_from_summary():
    """Best-effort sweep_id from the [HEADER] block; falls back to dv3MMDD."""
    try:
        with open(SUMMARY, "r") as fh:
            for line in fh:
                if line.startswith("[HEADER] sweep_id="):
                    return line.split("sweep_id=", 1)[1].strip()
    except Exception:
        pass
    import datetime as dt
    return f"dv3{dt.date.today():%m%d}"


def orchestrator_running():
    """True if the sweep orchestrator process is still alive."""
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if "run_sweep_dreamerv3.sh" in cmd or "run_dreamerv3.sh" in cmd:
                return True
    except Exception as exc:
        log(f"warn: orchestrator check failed: {exc}")
        return True  # be conservative; don't exit on a transient error
    return False


def clean_key(k):
    """Map a raw metrics key to a clean W&B key.

    ``episode/eval_<x>`` -> ``eval/<x>`` (matches report_results.py's namespace
    so the streamed curve and the final summary share keys); everything else
    (``train/*``, ``episode/score`` ...) is passed through unchanged.
    """
    if k.startswith("episode/eval_"):
        return "eval/" + k[len("episode/eval_"):]
    return k


def clean_metrics(rec):
    """Numeric metrics from a record as clean {wandb_key: float}, sans 'step'."""
    out = {}
    for k, v in rec.items():
        if k == "step":
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[clean_key(k)] = float(v)
    return out


class Streamer:
    """Tails one trial's metrics.jsonl into one W&B run."""

    def __init__(self, size, seed, out_dir, sweep_id):
        self.size = size
        self.seed = seed
        self.out_dir = out_dir
        self.sweep_id = sweep_id
        self.algo_label = f"{ALGO}-{size}"
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.offset = 0
        self.last_step = -1
        self.run = None
        config = {
            "algo": ALGO, "seed": int(seed) if str(seed).isdigit() else seed,
            "sweep_id": sweep_id, "exp_type": EXP_TYPE,
            "n_envs": 1, "impl": "NM512/r2dreamer", "impl_sha": PINNED_SHA,
            "model_size": size, "rep_loss": "dreamer", "source": "live-stream",
        }
        rid = run_id(sweep_id, self.algo_label, seed)
        try:
            self.run = init_run(
                algo=self.algo_label, seed=seed, sweep_id=sweep_id,
                exp_type=EXP_TYPE, project=PROJECT, config=config,
                out_dir=out_dir, id=rid, resume="allow",
                sync_tensorboard=False,  # we log explicit env-step; keep that axis
                name=f"{sweep_id}-{self.algo_label}-s{seed}",
                notes=f"DreamerV3 ({size}, r2dreamer rep_loss=dreamer) Crafter "
                      f"baseline -- LIVE stream")
            log(f"opened run id={rid} ({self.size} s{self.seed})")
        except Exception as exc:
            log(f"warn: init_run failed for {size} s{seed}: {exc}")

    def key(self):
        return (self.size, self.seed, self.out_dir)

    def poll(self):
        """Read any new metrics lines and forward them to W&B."""
        if self.run is None:
            return
        try:
            size = os.path.getsize(self.metrics_path)
        except FileNotFoundError:
            return
        except Exception as exc:
            log(f"warn: stat metrics failed: {exc}")
            return
        if size < self.offset:
            # File truncated/rotated; restart from the top.
            self.offset = 0
        if size == self.offset:
            return
        try:
            with open(self.metrics_path, "r") as fh:
                fh.seek(self.offset)
                data = fh.read()
                self.offset = fh.tell()
        except Exception as exc:
            log(f"warn: reading metrics failed: {exc}")
            return
        # Keep only complete lines; rewind offset over any partial trailing line.
        if data and not data.endswith("\n"):
            nl = data.rfind("\n")
            if nl == -1:
                self.offset -= len(data.encode("utf-8", "replace"))
                return
            partial = data[nl + 1:]
            self.offset -= len(partial.encode("utf-8", "replace"))
            data = data[: nl + 1]
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            step = rec.get("step")
            if not isinstance(step, (int, float)):
                continue
            step = int(step)
            # W&B requires non-decreasing steps; r2dreamer's log is ordered, so
            # drop the rare out-of-order record rather than let W&B reject it.
            if step < self.last_step:
                continue
            payload = clean_metrics(rec)
            if not payload:
                self.last_step = max(self.last_step, step)
                continue
            try:
                self.run.log(payload, step=step)
                self.last_step = step
            except Exception as exc:
                log(f"warn: wandb.log failed: {exc}")

    def finish(self):
        if self.run is None:
            return
        try:
            self.run.finish()
            log(f"finished run {self.size} s{self.seed} (last step={self.last_step})")
        except Exception as exc:
            log(f"warn: run.finish failed: {exc}")
        self.run = None


def main():
    log("starting DreamerV3 W&B live streamer")
    streamer = None
    try:
        while True:
            active = read_active_trial()
            if active is not None:
                if streamer is None or streamer.key() != active:
                    if streamer is not None:
                        streamer.finish()
                    streamer = Streamer(*active, sweep_id=sweep_id_from_summary())
                try:
                    streamer.poll()
                except Exception as exc:
                    log(f"warn: poll error: {exc}\n{traceback.format_exc()}")
            else:
                if not orchestrator_running():
                    log("no active trial and orchestrator gone; exiting")
                    break
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        if streamer is not None:
            streamer.finish()
    log("streamer stopped")


if __name__ == "__main__":
    main()
