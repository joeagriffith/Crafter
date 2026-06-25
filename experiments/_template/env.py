"""Experiment-local env handle. Re-exports the shared Crafter adapter.

Import from here within the experiment so swapping in a wrapped/modified env
later only touches this one file.
"""
from crafter_rl.env import CrafterGym  # noqa: F401
