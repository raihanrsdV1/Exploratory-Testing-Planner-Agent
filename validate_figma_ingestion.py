"""
Task 5: Validate Figma ingestion correctness.
Uses the RAG API (HTTP) instead of direct Neo4j connection.
"""
import json, os, sys, requests

RAG_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:9010")
PROJECT = "contacts-app"
JSON_PATH = os.path.join(os.path.dirname(__file__), "GENERATED_JSON.json")


# ── 1.  Parse source JSON ──

def extract_screens(data):
    """Extract top-level frames as 'screens' from the Figma JSON."""
    screens = {}
    pages = data.get("document", {}).get("children", [])
    for page in pages:
        for frame in page.get("children", []):
            if frame.get("type") == "FRAME":
                name = frame.get("name", "")
                texts = set()
                buttons = set()
                inputs = set()
                _extract_all(frame, texts, buttons, inputs)
                screens[name] = {
                    "texts": texts,
                    "buttons": buttons,
                    "inputs": inputs,
                    "total": len(texts) + len(buttons) + len(inputs),
                    "figma_id": frame.get("id"),
                }
    return screens


def _extract_all(node, texts, buttons, inputs, depth=0):
    if depth > 20:
        return
    name = node.get("name", "")
    ntype = node.get("type", "")
    chars = node.get("characters", "")

    if ntype == "TEXT" and chars.strip():
        n = name.lower()
        if any(kw in n for kw in ["button", "save", "cancel", "delete", "add", "create", "edit", "back", "send", "fab"]):
            buttons.add(chars.strip())
        elif any(kw in n for kw in ["input", "field", "text field", "search"]):
            inputs.add(chars.strip())
        else:
            texts.add(chars.strip())
    elif ntype == "FRAME":
        n = name.lower()
        if any(kw in n for kw in ["button", "fab", "btn", "link -"]):
            buttons.add(name)
        elif any(kw in n for kw in ["input", "text field", "search"]):
            inputs.add(name)

    for child in node.get("children", []):
        _extract_all(child, texts, buttons, inputs, depth + 1)


# ── 2.  Query RAG API ──

def get_rag_figma(project):
    try:
        r = requests.get(f"{RAG_URL}/figma/overview", params={"project": project, "top_labels": 20}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠ RAG API error: {e}")
        return None


def get_rag_stats(project):
    try:
        r = requests.get(f"{RAG_URL}/graph/stats", params={"project": project}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠ RAG API error: {e}")
        return None


# ── 3.  Compare ──

def compare(source_screens, rag_data, stats):
    report = {}

    # Parse RAG screens
    rag_screens = {}
    if rag_data and "screens" in rag_data:
        for s in rag_data["screens"]:
            rag_screens[s["screen_name"]] = {
                "purpose": s.get("purpose", "—"),
                "buttons": set(s.get("buttons", [])),
                "inputs": set(s.get("inputs", [])),
                "nav": set(s.get("nav", [])),
                "interactive_count": s.get("interactive_count", 0),
            }

    src_names = set(source_screens.keys())
    rag_names = set(rag_screens.keys())

    report["summary"] = {
        "source_screens": len(src_names),
        "neo4j_screens": stats.get("figma_screen_count", 0) if stats else len(rag_names),
        "neo4j_elements": stats.get("figma_element_count", 0) if stats else "?",
        "matching_screens": len(src_names & rag_names),
        "missing_from_neo4j": sorted(src_names - rag_names),
        "extra_in_neo4j": sorted(rag_names - src_names),
    }

    report["screen_details"] = []
    for name in sorted(src_names | rag_names):
        src = source_screens.get(name, None)
        rag = rag_screens.get(name, None)

        detail = {
            "screen": name,
            "in_source": name in src_names,
            "in_neo4j": name in rag_names,
        }

        if src:
            detail["source_buttons"] = sorted(src["buttons"])[:8]
            detail["source_inputs"] = sorted(src["inputs"])[:8]
            detail["source_text_count"] = len(src["texts"])

        if rag:
            detail["neo4j_purpose"] = rag["purpose"]
            detail["neo4j_buttons"] = sorted(rag["buttons"])[:8]
            detail["neo4j_inputs"] = sorted(rag["inputs"])[:8]
            detail["neo4j_nav"] = sorted(rag["nav"])[:5]
            detail["neo4j_interactive_count"] = rag["interactive_count"]

        # Cross-check elements
        if src and rag:
            src_btn_lower = {b.lower() for b in src["buttons"]}
            rag_btn_lower = {b.lower() for b in rag["buttons"]}
            detail["matched_buttons"] = sorted(src_btn_lower & rag_btn_lower)
            detail["missing_buttons"] = sorted(src_btn_lower - rag_btn_lower)[:5]

            src_inp_lower = {i.lower() for i in src["inputs"]}
            rag_inp_lower = {i.lower() for i in rag["inputs"]}
            detail["matched_inputs"] = sorted(src_inp_lower & rag_inp_lower)
            detail["missing_inputs"] = sorted(src_inp_lower - rag_inp_lower)[:5]

        report["screen_details"].append(detail)

    return report


# ── 4.  Pretty print ──

def print_report(report):
    s = report["summary"]
    print("=" * 70)
    print("  FIGMA INGESTION VALIDATION REPORT")
    print("=" * 70)

    print(f"\n{'Source screens (JSON):':<30} {s['source_screens']}")
    print(f"{'Neo4j screens:':<30} {s['neo4j_screens']}")
    print(f"{'Neo4j elements:':<30} {s['neo4j_elements']}")
    print(f"{'Matching names:':<30} {s['matching_screens']}")

    if s["missing_from_neo4j"]:
        print(f"\n⚠  MISSING SCREENS (in source but NOT in Neo4j):")
        for name in s["missing_from_neo4j"]:
            print(f"   ✗ {name}")
    else:
        print(f"\n✅ All source screens are present in Neo4j!")

    if s["extra_in_neo4j"]:
        print(f"\n📌 EXTRA SCREENS (in Neo4j but NOT in source):")
        for name in s["extra_in_neo4j"]:
            print(f"   + {name}")

    print(f"\n{'─' * 70}")
    print(f"  SCREEN-BY-SCREEN COMPARISON")
    print(f"{'─' * 70}")

    for d in report["screen_details"]:
        status = "✓" if d["in_source"] and d["in_neo4j"] else ("✗ MISSING" if not d["in_neo4j"] else "+ EXTRA")
        print(f"\n  [{status}] {d['screen']}")

        if d.get("neo4j_purpose"):
            print(f"       Purpose: {d['neo4j_purpose']}")

        if d.get("source_buttons"):
            print(f"       Source buttons: {d['source_buttons']}")
        if d.get("neo4j_buttons"):
            print(f"       Neo4j  buttons: {d['neo4j_buttons']}")
        if d.get("missing_buttons"):
            print(f"       ⚠ Missing buttons: {d['missing_buttons']}")

        if d.get("source_inputs"):
            print(f"       Source inputs: {d['source_inputs']}")
        if d.get("neo4j_inputs"):
            print(f"       Neo4j  inputs: {d['neo4j_inputs']}")
        if d.get("missing_inputs"):
            print(f"       ⚠ Missing inputs: {d['missing_inputs']}")

        if d.get("neo4j_nav"):
            print(f"       Neo4j  nav: {d['neo4j_nav']}")

        if d.get("neo4j_interactive_count") is not None:
            print(f"       Interactive elements: {d.get('neo4j_interactive_count', '?')}")

    print(f"\n{'=' * 70}")
    missing = len(s["missing_from_neo4j"])
    if missing == 0:
        print(f"  VERDICT: ✅ All {s['source_screens']} screens correctly ingested")
    else:
        print(f"  VERDICT: ⚠ {missing} screen(s) missing from ingestion")
    print(f"{'=' * 70}")


# ── Main ──

def main():
    print("Loading source JSON...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    source_screens = extract_screens(data)
    print(f"  Found {len(source_screens)} screens in GENERATED_JSON.json:")
    for name in sorted(source_screens.keys()):
        src = source_screens[name]
        print(f"    • {name} ({len(src['buttons'])} btns, {len(src['inputs'])} inputs, {len(src['texts'])} texts)")

    print("\nQuerying RAG API...")
    rag_data = get_rag_figma(PROJECT)
    stats = get_rag_stats(PROJECT)

    if not rag_data:
        print("⚠ Cannot connect to RAG API. Make sure it's running on port 9010.")
        sys.exit(1)

    report = compare(source_screens, rag_data, stats)
    print_report(report)

    # Save JSON report
    # Convert sets to lists for JSON serialization
    def serialize(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    report_path = os.path.join(os.path.dirname(__file__), "figma_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=serialize)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
