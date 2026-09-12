"""SQLite persistence boundary; separate from test verdicts and Neo4j availability."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .state import digest


class GraphStore:
    def __init__(self, path, namespace, read_only=False):
        self.namespace = namespace
        if read_only:
            self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2)
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=0.2)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS ui_states (
                namespace TEXT, id TEXT, payload TEXT NOT NULL,
                observations INTEGER NOT NULL DEFAULT 1, first_seen TEXT, last_seen TEXT,
                PRIMARY KEY(namespace, id));
            CREATE TABLE IF NOT EXISTS ui_transitions (
                namespace TEXT, id TEXT, source TEXT NOT NULL, destination TEXT,
                action TEXT NOT NULL, outcome TEXT NOT NULL, error_category TEXT,
                observations INTEGER NOT NULL DEFAULT 1, first_seen TEXT, last_seen TEXT,
                PRIMARY KEY(namespace, id));
            PRAGMA user_version=1;
        """)

    def node(self, node_id, payload):
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute("""INSERT INTO ui_states VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(namespace,id) DO UPDATE SET
                observations=observations+1, last_seen=excluded.last_seen""",
                (self.namespace, node_id, json.dumps(payload), now, now))

    def edge(self, source, destination, action, outcome, error_category=""):
        key = digest([source, destination, action, outcome, error_category])
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute("""INSERT INTO ui_transitions VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(namespace,id) DO UPDATE SET
                observations=observations+1, last_seen=excluded.last_seen""",
                (self.namespace, key, source, destination, json.dumps(action),
                 outcome, error_category, now, now))

    def export(self):
        nodes = [dict(id=r[0], **json.loads(r[1]), observations=r[2], first_seen=r[3], last_seen=r[4])
                 for r in self.db.execute("SELECT id,payload,observations,first_seen,last_seen FROM ui_states WHERE namespace=? ORDER BY id", (self.namespace,))]
        edges = [dict(id=r[0], source=r[1], destination=r[2], action=json.loads(r[3]),
                      outcome=r[4], error_category=r[5], observations=r[6], first_seen=r[7], last_seen=r[8])
                 for r in self.db.execute("SELECT id,source,destination,action,outcome,error_category,observations,first_seen,last_seen FROM ui_transitions WHERE namespace=? ORDER BY id", (self.namespace,))]
        return {"schema_version": 1, "namespace": self.namespace, "nodes": nodes, "edges": edges}

    def close(self):
        self.db.close()
