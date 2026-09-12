"""Fail-open hooks consuming the player's existing observations and action results."""
import logging
from pathlib import Path

from .state import NAVIGATION_KEYS, action_descriptor, observation_signature, route, state
from .store import GraphStore

log = logging.getLogger(__name__)


class Recorder:
    def __init__(self, store, navigation_keys=NAVIGATION_KEYS):
        self.store = store
        self.navigation_keys = navigation_keys
        self.pending = None
        self.disabled = False

    def _safe(self, operation):
        if self.disabled:
            return
        try:
            operation()
        except Exception:
            self.disabled = True
            # Do not log exception payloads: they may contain observed user data.
            log.warning("Passive exploration recording disabled after a storage/normalization error; test continues.")

    def observe(self, snapshot):
        def record():
            node_id, payload = state(snapshot, self.navigation_keys)
            self.store.node(node_id, payload)
            if self.pending:
                source, action, before = self.pending
                outcome = ("state_changed" if source != node_id else
                           "observation_changed" if before != observation_signature(snapshot) else "no_visible_change")
                self.store.edge(source, node_id, action, outcome)
                self.pending = None
        self._safe(record)

    def attempted(self, action, snapshot, error_category=None):
        def record():
            source, _ = state(snapshot, self.navigation_keys)
            descriptor = action_descriptor(action, snapshot, self.navigation_keys)
            if error_category:
                self.pending = None
                # A failed dispatch does not prove the page stayed unchanged.
                self.store.edge(source, None, descriptor, "dispatch_failed", error_category)
            else:
                self.pending = (source, descriptor, observation_signature(snapshot))
        self._safe(record)

    def close(self):
        def flush():
            if self.pending:
                source, action, _ = self.pending
                self.store.edge(source, None, action, "destination_unobserved")
                self.pending = None
        self._safe(flush)
        try:
            self.store.close()
        except Exception:
            pass


def create_recorder(cfg):
    if not getattr(cfg, "WEB_EXPLORATION_ENABLED", False):
        return None
    try:
        keys = getattr(cfg, "WEB_EXPLORATION_QUERY_KEYS", NAVIGATION_KEYS)
        # Namespace includes project, origin/base route, and normalization policy.
        namespace = str(cfg.PROJECT) + "|" + route(cfg.WEB_BASE_URL, keys) + "|" + ",".join(keys)
        path = getattr(cfg, "WEB_EXPLORATION_DB", Path(__file__).resolve().parents[2] / "data/exploration/web.sqlite3")
        return Recorder(GraphStore(path, namespace), keys)
    except Exception:
        log.warning("Passive exploration store unavailable; test continues without recording.")
        return None
