"""crafter_rl: shared, promoted utilities for the Crafter RL research repo.

Importable from any cwd once installed editable (``uv pip install -e .``):

    from crafter_rl.env import CrafterGym
    from crafter_rl.wandb import init_run
    from crafter_rl.utils import get_git_sha, set_seed, write_metrics, append_ledger

Submodules:
    env      - the CrafterGym Gymnasium adapter.
    wandb    - the one reusable W&B run-naming/grouping convention.
    utils    - git/version/seed helpers + metrics.json / ledger.jsonl writers.
    kernels  - lazy loader stub for custom CUDA ops (see csrc/).
"""

__version__ = "0.1.0"
