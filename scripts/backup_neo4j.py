"""
Backup Neo4j Knowledge Graph to a structured JSON file.
Usage: python scripts/backup_neo4j.py [output_path]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase


def backup_neo4j(output_path: str | None = None) -> str:
    os.makedirs("data/backups", exist_ok=True)
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/backups/neo4j_backup_{timestamp}.json"

    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as s:
        print("Exporting nodes...")
        nodes_res = s.run(
            "MATCH (n) RETURN elementId(n) as element_id, labels(n) as labels, properties(n) as properties"
        )
        nodes = []
        for r in nodes_res:
            props = dict(r["properties"])
            for k, v in props.items():
                if hasattr(v, "iso_format"):
                    props[k] = v.iso_format()
                elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    props[k] = str(v)
            nodes.append({
                "element_id": r["element_id"],
                "labels": r["labels"],
                "properties": props,
            })

        print(f"Exported {len(nodes)} nodes. Exporting relationships...")
        rels_res = s.run(
            "MATCH (a)-[r]->(b) "
            "RETURN elementId(r) as element_id, type(r) as type, properties(r) as properties, "
            "elementId(a) as start_elem_id, elementId(b) as end_elem_id, "
            "labels(a) as start_labels, labels(b) as end_labels"
        )
        relationships = []
        for r in rels_res:
            props = dict(r["properties"])
            for k, v in props.items():
                if hasattr(v, "iso_format"):
                    props[k] = v.iso_format()
                elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    props[k] = str(v)
            relationships.append({
                "element_id": r["element_id"],
                "type": r["type"],
                "properties": props,
                "start_element_id": r["start_elem_id"],
                "end_element_id": r["end_elem_id"],
                "start_labels": r["start_labels"],
                "end_labels": r["end_labels"],
            })

    backup_data = {
        "timestamp": datetime.now().isoformat(),
        "total_nodes": len(nodes),
        "total_relationships": len(relationships),
        "nodes": nodes,
        "relationships": relationships,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Backup complete: {output_path}")
    print(f"  Nodes: {len(nodes)}, Relationships: {len(relationships)}")
    return output_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    backup_neo4j(out)
