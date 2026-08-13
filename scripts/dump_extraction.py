"""
Show exactly what the LLM returns when it ingests a document.

Ingestion is the step every downstream number depends on — if the extracted
requirements are wrong, the tests, the coverage, and the "bugs found" figure are
all wrong underneath. This prints the model's own output at each pass so a human
can judge it directly, instead of inferring quality from what landed in Neo4j.

Usage:
    ./venv/bin/python scripts/dump_extraction.py                    # default SRS
    ./venv/bin/python scripts/dump_extraction.py --path other.txt
    ./venv/bin/python scripts/dump_extraction.py --raw              # unparsed model text
    ./venv/bin/python scripts/dump_extraction.py --json out.json    # save structured result
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import SRS_PATH  # noqa: E402
from ingestion import document_loader, extractor  # noqa: E402
from planner import model_client  # noqa: E402


def model_call(prompt: str, max_new_tokens: int, enable_thinking: bool) -> dict:
    return model_client.call_model(prompt, max_new_tokens, enable_thinking)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=SRS_PATH)
    ap.add_argument("--raw", action="store_true", help="also print the model's unparsed reply")
    ap.add_argument("--json", default="", help="write the structured extraction here")
    ap.add_argument("--single", action="store_true", help="use the single-call path instead of multipass")
    args = ap.parse_args()

    text = document_loader.load_text(args.path) if hasattr(document_loader, "load_text") \
        else open(args.path, encoding="utf-8", errors="ignore").read()
    print(f"source          : {args.path}")
    print(f"source chars    : {len(text)}\n")

    if args.raw:
        print("=" * 78)
        print("RAW MODEL REPLY (single extraction call, before any parsing)")
        print("=" * 78)
        reply = model_call(extractor.EXTRACTION_PROMPT + text.strip(), 12000, False)
        print((reply.get("answer") or "")[:6000])
        print("\n")

    print("=" * 78)
    print(f"STRUCTURED EXTRACTION ({'single-call' if args.single else 'multipass'})")
    print("=" * 78)
    data, source = extractor.extract(
        text, model_call=model_call, require_model=False,
        multipass=not args.single, max_new_tokens=12000,
    )
    reqs = data.get("requirements", []) or []
    rules = data.get("validation_rules", []) or []
    ents = data.get("entities", []) or []

    print(f"extraction path : {source}")
    print(f"document_title  : {data.get('document_title','')}")
    print(f"requirements    : {len(reqs)}")
    print(f"validation rules: {len(rules)}")
    print(f"entities        : {len(ents)}")
    oq = data.get("open_questions") or []
    if oq:
        print(f"open questions  : {len(oq)}")

    print("\n--- REQUIREMENTS ---")
    seen_text: dict[str, list[str]] = {}
    for r in reqs:
        rid = str(r.get("id", "?"))
        txt = str(r.get("text", ""))
        print(f"[{rid}]  ({r.get('type','?')}, {r.get('priority','?')}, feature={r.get('feature','?')})")
        print(f"    {txt[:190]}")
        if r.get("acceptance"):
            print(f"    acceptance: {'; '.join(str(a) for a in r['acceptance'])[:160]}")
        seen_text.setdefault(txt.strip()[:80], []).append(rid)

    print("\n--- VALIDATION RULES ---")
    for v in rules[:30]:
        print(f"[{v.get('requirement_id','?')}] {str(v.get('rule',''))[:150]}"
              + (f"  (conf={v.get('confidence')})" if v.get("confidence") is not None else ""))

    if ents:
        print("\n--- ENTITIES ---")
        print("   " + ", ".join(str(e.get("name", e)) for e in ents[:40]))

    if oq:
        print("\n--- OPEN QUESTIONS (ambiguities the model flagged) ---")
        for q in oq:
            print(f"   - {str(q)[:180]}")

    dupes = {t: ids for t, ids in seen_text.items() if len(ids) > 1}
    if dupes:
        print("\n--- ⚠ DUPLICATE REQUIREMENTS (same text, multiple ids) ---")
        for t, ids in dupes.items():
            print(f"   {ids}\n      -> {t}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"\nstructured extraction -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
