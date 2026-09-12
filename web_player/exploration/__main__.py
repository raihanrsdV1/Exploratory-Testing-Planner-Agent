"""Read-only graph inspection: python -m web_player.exploration [--project NAME]."""
import argparse
import json
import sqlite3
from pathlib import Path

from .store import GraphStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(Path(__file__).resolve().parents[2] / "data/exploration/web.sqlite3"))
    parser.add_argument("--project", help="Filter namespaces by project name")
    parser.add_argument("--json", action="store_true", help="Print graph JSON instead of counts")
    args = parser.parse_args()
    path = Path(args.db).resolve()
    if not path.is_file():
        parser.error("Graph database does not exist yet. Run a web test with recording enabled first.")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        namespaces = [row[0] for row in db.execute("SELECT DISTINCT namespace FROM ui_states ORDER BY namespace")
                      if not args.project or row[0].split("|", 1)[0] == args.project]
        graphs = []
        for namespace in namespaces:
            reader = GraphStore(path, namespace, read_only=True)
            try:
                graph = reader.export()
            finally:
                reader.close()
            graphs.append(graph)
            if not args.json:
                print(f"{namespace}: {len(graph['nodes'])} states, {len(graph['edges'])} transitions")
        if args.json:
            print(json.dumps(graphs, ensure_ascii=True, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
