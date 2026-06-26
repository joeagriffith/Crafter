#!/usr/bin/env python3
"""ttlcd sidecar: stream the running DreamerV3 sweep's progress to the
physical Thermaltake LCD via the ttlcd_panel SDK.

Run with the MAIN project venv:
    .venv/bin/python experiments/20260625-dreamerv3/ttlcd_sidecar.py

Behaviour:
  * Every ~3s, determine the ACTIVE trial from the last `[START]` line of
    `sweep_summary.txt` (model size, seed, out dir).
  * Maintain a Panel(project=f"dv3-{size}", ...) for that trial, tailing its
    metrics.jsonl incrementally and forwarding cleaned metrics.
  * On trial change, finish the old Panel and start a fresh one.
  * Never crash: every file read / parse / SDK call is guarded.
  * Exit cleanly when the orchestrator is gone AND no trial is active.

Dependency-light: stdlib + ttlcd_panel only.
"""

import json
import os
import sys
import time
import traceback

try:
    from ttlcd_panel import Panel
except Exception as exc:  # pragma: no cover - SDK should be installed
    print(f"[sidecar] FATAL: cannot import ttlcd_panel: {exc}", flush=True)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "sweep_summary.txt")
ORCHESTRATOR = os.path.join(HERE, "run_sweep_dreamerv3.sh")
OWNER = "claude-code"
POLL_S = 3.0
STEPS_PER_EPOCH = 1010000

# Candidate loss keys in DreamerV3 metrics, in preference order.
LOSS_KEYS = (
    "train/opt/loss",
    "train/loss/value",
    "train/loss/policy",
    "train/loss/rep",
    "train/loss/dyn",
)


def log(msg):
    print(f"[sidecar {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_active_trial():
    """Return (size, seed, out_dir) from the last [START] line, or None."""
    try:
        with open(SUMMARY, "r") as fh:
            last = None
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
            if "run_sweep_dreamerv3.sh" in cmd:
                return True
    except Exception as exc:
        log(f"warn: orchestrator check failed: {exc}")
        return True  # be conservative; don't exit on a transient error
    return False


def clean_metrics(rec):
    """Map a raw metrics record to clean panel keys. Returns dict (may be empty)."""
    out = {}
    v = rec.get("episode/eval_score")
    if isinstance(v, (int, float)):
        out["eval_score"] = float(v)
    fps = rec.get("fps/fps", rec.get("fps"))
    if isinstance(fps, (int, float)):
        out["fps"] = float(fps)
    for k in LOSS_KEYS:
        v = rec.get(k)
        if isinstance(v, (int, float)):
            out["loss"] = float(v)
            break
    return out


class Tracker:
    """Tails one trial's metrics.jsonl into one Panel."""

    def __init__(self, size, seed, out_dir):
        self.size = size
        self.seed = seed
        self.out_dir = out_dir
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.offset = 0
        self.last_step = -1
        self.panel = None
        try:
            self.panel = Panel(
                project=f"dv3-{size}",
                epochs=1,
                steps_per_epoch=STEPS_PER_EPOCH,
                owner=OWNER,
            )
            try:
                self.panel.message(f"DreamerV3 {size} s{seed} started")
            except Exception:
                pass
            log(f"started panel project=dv3-{size} seed={seed}")
        except Exception as exc:
            log(f"warn: could not start panel for {size} s{seed}: {exc}")

    def key(self):
        return (self.size, self.seed, self.out_dir)

    def poll(self):
        """Read any new metrics lines and forward them to the panel."""
        if self.panel is None:
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
            self.last_step = -1
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
            metrics = clean_metrics(rec)
            if not metrics:
                self.last_step = max(self.last_step, step)
                continue
            try:
                self.panel.log(metrics, epoch=0, batch=step)
                self.last_step = step
            except Exception as exc:
                log(f"warn: panel.log failed: {exc}")

    def finish(self):
        if self.panel is None:
            return
        try:
            self.panel.finish()
            log(f"finished panel dv3-{self.size} s{self.seed} (last step={self.last_step})")
        except Exception as exc:
            log(f"warn: panel.finish failed: {exc}")
        self.panel = None


def main():
    log("starting DreamerV3 ttlcd sidecar")
    tracker = None
    try:
        while True:
            active = read_active_trial()
            if active is not None:
                if tracker is None or tracker.key() != active:
                    if tracker is not None:
                        tracker.finish()
                    tracker = Tracker(*active)
                try:
                    tracker.poll()
                except Exception as exc:
                    log(f"warn: poll error: {exc}\n{traceback.format_exc()}")
            else:
                # No active trial. If the orchestrator is gone, exit cleanly.
                if not orchestrator_running():
                    log("no active trial and orchestrator gone; exiting")
                    break
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        if tracker is not None:
            tracker.finish()
    log("sidecar stopped")


if __name__ == "__main__":
    main()
