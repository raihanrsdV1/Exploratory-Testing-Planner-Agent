"""Passive UI graph. No browser actions, model calls, or planner policy."""
from .recorder import Recorder, create_recorder

__all__ = ["Recorder", "create_recorder"]
