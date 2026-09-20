# Rendered diagrams

PNG renders of every Mermaid block in [`../DIAGRAMS.md`](../DIAGRAMS.md), at 2x
device scale so they stay sharp in slides and print.

**These files are generated — do not edit them.** Edit the Mermaid source in
`docs/DIAGRAMS.md`, then regenerate:

```bash
./venv/bin/python scripts/render_diagrams.py
```

The renderer uses headless Google Chrome and a cached copy of `mermaid.min.js`
(in `.mermaid-cache/`, fetched once). No Node toolchain is required.

| # | Diagram | File |
|---|---|---|
| 1 | System architecture | [`01-system-architecture.png`](01-system-architecture.png) |
| 2 | One test round, end to end | [`02-one-test-round-end-to-end.png`](02-one-test-round-end-to-end.png) |
| 3 | Knowledge graph schema | [`03-knowledge-graph-schema.png`](03-knowledge-graph-schema.png) |
| 4 | Planner — tool-calling mode | [`04-planner-tool-calling-mode.png`](04-planner-tool-calling-mode.png) |
| 5 | Planner — pipeline mode (default) | [`05-planner-pipeline-mode-default.png`](05-planner-pipeline-mode-default.png) |
| 6 | Investigator — trajectory to knowledge | [`06-investigator-trajectory-to-knowledge.png`](06-investigator-trajectory-to-knowledge.png) |
| 7 | Finding taxonomy and who consumes it | [`07-finding-taxonomy-and-who-consumes-it.png`](07-finding-taxonomy-and-who-consumes-it.png) |
| 8 | Finding lifecycle | [`08-finding-lifecycle.png`](08-finding-lifecycle.png) |
| 9 | Failure attribution | [`09-failure-attribution.png`](09-failure-attribution.png) |
| 10 | Campaign lifecycle | [`10-campaign-lifecycle.png`](10-campaign-lifecycle.png) |
| 11 | Ingestion — document to queryable requirements | [`11-ingestion-document-to-queryable-requirements.png`](11-ingestion-document-to-queryable-requirements.png) |
| 12 | Runtime topology | [`12-runtime-topology.png`](12-runtime-topology.png) |
| 13 | Why tools replaced retrieval sources | [`13-why-tools-replaced-retrieval-sources.png`](13-why-tools-replaced-retrieval-sources.png) |
| 14 | The nine tools as an investigation | [`14-the-nine-tools-as-an-investigation.png`](14-the-nine-tools-as-an-investigation.png) |
| 15 | Three rules that hold for every tool | [`15-three-rules-that-hold-for-every-tool.png`](15-three-rules-that-hold-for-every-tool.png) |
| 16 | The executor, in detail | [`16-the-executor-in-detail.png`](16-the-executor-in-detail.png) |
| 17 | State identity — is this a screen we have seen? | [`17-state-identity-is-this-a-screen-we-have-seen.png`](17-state-identity-is-this-a-screen-we-have-seen.png) |

## Using these elsewhere

The PNGs have a white background and no transparency, so they drop straight into
slides, a document or the LaTeX report. For the report, prefer re-drawing in TikZ
(as Chapter 4 does) — vector output scales better in print — or include the PNG
with `\includegraphics[width=\textwidth]{...}` if time is short.

Colours are consistent across all twelve: amber = inputs, green = knowledge,
blue = services, purple = the three agents, red = language-model calls,
grey = device and app under test.
