"""Generate docs/USER_GUIDE.pdf — a complete step-by-step operator guide.

Pure-Python (fpdf2), ASCII-only content so the PDF core fonts render cleanly.
"""
from fpdf import FPDF

# ── Colors ───────────────────────────────────────────────────────────────────
INK = (28, 32, 36)
ACCENT = (79, 70, 229)
MUTED = (110, 118, 129)
CODE_BG = (244, 245, 248)
NOTE_BG = (238, 242, 255)
NOTE_BAR = (79, 70, 229)
WARN_BG = (253, 243, 227)
WARN_BAR = (217, 119, 6)
RULE = (223, 227, 232)


class Guide(FPDF):
    def multi_cell(self, w, h=None, text="", *args, **kwargs):
        # fpdf2 leaves the cursor at the right margin by default, which makes the
        # next full-width multi_cell zero-width. Always return to the left margin.
        kwargs.setdefault("new_x", "LMARGIN")
        kwargs.setdefault("new_y", "NEXT")
        return super().multi_cell(w, h, text, *args, **kwargs)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_y(10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, "Exploratory Testing Planner Agent - Operator Guide",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(self.l_margin, 16, self.w - self.r_margin, 16)
        self.set_y(20)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


pdf = Guide(format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(18, 20, 18)
EPW = pdf.w - pdf.l_margin - pdf.r_margin


def h1(txt):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 8, txt)
    pdf.set_draw_color(*ACCENT)
    pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.l_margin + 30, pdf.get_y() + 1)
    pdf.ln(4)


def h2(txt):
    if pdf.get_y() > pdf.h - 45:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 7, txt)
    pdf.ln(1)


def h3(txt):
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 6, txt)
    pdf.ln(0.5)


def para(txt):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.4, txt)
    pdf.ln(1.5)


def bullets(items):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    for it in items:
        bold = None
        if isinstance(it, tuple):
            bold, it = it
        x = pdf.get_x()
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.cell(5, 5.2, "-")
        if bold:
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.cell(pdf.get_string_width(bold + " "), 5.2, bold + " ")
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 5.2, it, new_x="LMARGIN")
        pdf.set_x(x)
    pdf.ln(1.5)


def code(txt):
    pdf.ln(0.5)
    pdf.set_font("Courier", "", 9)
    lines = txt.split("\n")
    pad = 2.5
    line_h = 4.6
    height = line_h * len(lines) + pad * 2
    if pdf.get_y() + height > pdf.h - 20:
        pdf.add_page()
    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.set_fill_color(*CODE_BG)
    pdf.rect(x0, y0, EPW, height, style="F")
    pdf.set_draw_color(*ACCENT)
    pdf.rect(x0, y0, 1.1, height, style="F")
    pdf.set_xy(x0 + 4, y0 + pad)
    pdf.set_text_color(30, 40, 55)
    for ln in lines:
        pdf.set_x(x0 + 4)
        pdf.cell(0, line_h, ln, new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(x0, y0 + height + 2)
    pdf.ln(1)


def callout(txt, kind="note"):
    bg = NOTE_BG if kind == "note" else WARN_BG
    bar = NOTE_BAR if kind == "note" else WARN_BAR
    label = "NOTE" if kind == "note" else "IMPORTANT"
    pdf.set_font("Helvetica", "", 10)
    # measure
    pdf.set_xy(pdf.l_margin, pdf.get_y())
    tmp_y = pdf.get_y()
    # estimate lines using split_only
    lines = pdf.multi_cell(EPW - 10, 5, txt, dry_run=True, output="LINES")
    height = 5 * len(lines) + 9
    if tmp_y + height > pdf.h - 20:
        pdf.add_page()
        tmp_y = pdf.get_y()
    x0 = pdf.l_margin
    pdf.set_fill_color(*bg)
    pdf.rect(x0, tmp_y, EPW, height, style="F")
    pdf.set_fill_color(*bar)
    pdf.rect(x0, tmp_y, 1.4, height, style="F")
    pdf.set_xy(x0 + 5, tmp_y + 2.5)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*bar)
    pdf.cell(0, 4, label, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(x0 + 5)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*INK)
    pdf.multi_cell(EPW - 10, 5, txt)
    pdf.set_xy(x0, tmp_y + height + 2)
    pdf.ln(1)


def kvtable(rows, col0=58):
    pdf.set_font("Helvetica", "", 9.5)
    for k, v in rows:
        if pdf.get_y() > pdf.h - 24:
            pdf.add_page()
        x0, y0 = pdf.get_x(), pdf.get_y()
        # key cell
        kl = pdf.multi_cell(col0, 5, k, dry_run=True, output="LINES")
        vl = pdf.multi_cell(EPW - col0, 5, v, dry_run=True, output="LINES")
        rh = 5 * max(len(kl), len(vl)) + 2
        pdf.set_draw_color(*RULE)
        pdf.rect(x0, y0, col0, rh)
        pdf.rect(x0 + col0, y0, EPW - col0, rh)
        pdf.set_xy(x0 + 1.5, y0 + 1)
        pdf.set_font("Courier", "", 8.8)
        pdf.set_text_color(*ACCENT)
        pdf.multi_cell(col0 - 3, 5, k)
        pdf.set_xy(x0 + col0 + 1.5, y0 + 1)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*INK)
        pdf.multi_cell(EPW - col0 - 3, 5, v)
        pdf.set_xy(x0, y0 + rh)
    pdf.ln(2)


def spacer(n=2):
    pdf.ln(n)


# ═══════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.ln(30)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(*ACCENT)
pdf.multi_cell(0, 12, "Exploratory Testing\nPlanner Agent")
pdf.ln(2)
pdf.set_font("Helvetica", "B", 14)
pdf.set_text_color(*INK)
pdf.multi_cell(0, 8, "Operator Guide - Start, Run & Verify")
pdf.ln(4)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(*MUTED)
pdf.multi_cell(0, 6,
    "A step-by-step guide to starting the stack, running the full testing suite on the "
    "emulator, verifying every feature, and understanding what each capability means.")
pdf.ln(10)
pdf.set_draw_color(*RULE)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(6)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(*INK)
pdf.multi_cell(0, 6,
    "Scope: ETA-REQ-301 to 308  (Work Packages WP1 - WP9)\n"
    "Autonomous, self-learning GraphRAG test engineer over Neo4j + a live Android app.\n"
    "Assumes the emulator, Neo4j, and the .env file are already set up.")
pdf.ln(16)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(*ACCENT)
pdf.multi_cell(0, 6, "Contents")
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(*INK)
toc = [
    "1.  System overview & architecture",
    "2.  Prerequisites (already set up)",
    "3.  Part 1 - Starting the stack",
    "4.  Part 2 - Running the full testing suite on the emulator",
    "5.  Part 3 - Running the verification suite (101 checks)",
    "6.  Part 4 - Manual feature checks (what each does + how to verify)",
    "7.  Part 5 - Dashboard walkthrough",
    "8.  Part 6 - Feature glossary (WP1-WP9)",
    "9.  Troubleshooting",
    "10. Appendix - endpoint reference",
]
for t in toc:
    pdf.multi_cell(0, 5.6, t)

# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("1. System Overview & Architecture")
para("The agent turns whatever knowledge exists about an app - a requirements spec (SRS), a "
     "UI design export (Figma), a defect history, and above all the LIVE running app itself - into "
     "high-value exploratory tests. Every execution feeds results back into a Neo4j knowledge graph, "
     "so the agent gets measurably smarter each cycle. No single source is required: with nothing but "
     "an installed app, it explores the device and builds its own map.")

h2("The moving parts")
kvtable([
    ("Neo4j", "The knowledge graph (bolt://localhost:7687). Stores SRS, UI model, defects, "
              "navigation memory, execution logs, strategies, risk and anomalies. Already set up."),
    ("RAG API  :9010", "rag_api.main - the graph service. Ingestion, hybrid retrieval, and all "
                        "learning endpoints. Talks to Neo4j."),
    ("Gateway  :9100", "gateway.main - the thin public router + the agent loop (LangGraph). Serves "
                        "the operator dashboard and proxies to the RAG API."),
    ("Emulator + mobilerun", "The Android device under test. The executor drives it, observes each "
                             "screen, and executes generated test steps. Already set up."),
    ("Dashboard", "http://localhost:9100/dashboard - a single live view of everything the agent "
                  "knows and is doing."),
])

h2("Data flow (one cycle)")
code(
    "SOURCES            RAG API (:9010)        AGENT LOOP (:9100)         EMULATOR\n"
    "SRS  ---------\\                                                      \n"
    "Figma --------  ingest ->  Neo4j graph  <->  retrieve + generate  -> execute test\n"
    "Defects ------/               ^                     |                    |\n"
    "Live UI <----------- observe  |                     v                    |\n"
    "                              +---------- log execution, verdict, path <-+\n"
    "                                 (defects, navtree, strategies, risk,\n"
    "                                  coverage, anomalies -> next cycle)")
para("Because the loop writes back what it learns (navigation paths, which strategies find bugs, "
     "regression risk, emerging anomalies), the next test is generated with more context than the last.")

# ═══════════════════════════════════════════════════════════════════════════
# 2. PREREQUISITES
# ═══════════════════════════════════════════════════════════════════════════
h1("2. Prerequisites (already set up)")
para("This guide assumes the following are already configured on your machine. You only need to "
     "confirm they are running.")
bullets([
    ("Neo4j:", "a local instance is started and reachable at bolt://localhost:7687 with the "
               "credentials in .env (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)."),
    ("Emulator:", "an Android emulator (or device) is running and visible to adb, with the target "
                  "app installed (TARGET_APP_PACKAGE in .env)."),
    ("Python venv:", "the project virtual environment .venv_py_3_13 exists with dependencies "
                     "installed."),
    (".env:", "present in the project root with model backend, embeddings, project name, and paths "
              "configured."),
])
callout("All commands below are run from the project root: "
        "d:\\Projects\\Exploratory-Testing-Planner-Agent . The Python interpreter is referenced as "
        ".venv_py_3_13\\Scripts\\python.exe (Windows). On Git Bash use forward slashes.")

# ═══════════════════════════════════════════════════════════════════════════
# 3. PART 1 - STARTING THE STACK
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("3. Part 1 - Starting the Stack")
para("Start the two services in two separate terminals. Keep them running while you work.")

h3("Step 1 - Confirm Neo4j is up")
para("Quick connectivity check (should print NEO4J OK):")
code('.venv_py_3_13\\Scripts\\python.exe -c "from neo4j import GraphDatabase; import os; '
     "from dotenv import load_dotenv; load_dotenv(); "
     "d=GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), "
     "os.getenv('NEO4J_PASSWORD'))); d.verify_connectivity(); print('NEO4J OK')\"")

h3("Step 2 - Start the RAG API (terminal 1)")
code(".venv_py_3_13\\Scripts\\python.exe -m uvicorn rag_api.main:app --port 9010")
para("Wait for startup, then confirm health (should return status ok):")
code("curl http://localhost:9010/health")

h3("Step 3 - Start the Gateway (terminal 2)")
code(".venv_py_3_13\\Scripts\\python.exe -m uvicorn gateway.main:app --port 9100")
para("Confirm it is up:")
code("curl http://localhost:9100/health")

h3("Step 4 - Open the dashboard")
para("Open a browser to the live operator dashboard. Pass the project name that matches PROJECT in "
     "your .env (default contacts-app):")
code("http://localhost:9100/dashboard?project=contacts-app")
para("The dashboard auto-refreshes every 4 seconds. It will be mostly empty until you ingest "
     "knowledge and/or run a testing loop.")

h3("Step 5 (optional) - Ingest knowledge sources")
para("This loads the SRS, Figma UI, and sample defects for the project into the graph. It is "
     "OPTIONAL - the agent also works with no documents at all (it explores the live app). Paths "
     "come from SRS_PATH / FIGMA_PATH in .env.")
code(".venv_py_3_13\\Scripts\\python.exe scripts/ingest_all.py")
callout("Ingesting the SRS uses the configured model backend to extract requirements and business "
        "rules, so it consumes model tokens (unless you use a local/ngrok backend). Ingesting Figma "
        "and defects does not.")

# ═══════════════════════════════════════════════════════════════════════════
# 4. PART 2 - TESTING SUITE ON EMULATOR
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("4. Part 2 - Running the Full Testing Suite on the Emulator")
para("There are three runner scripts in clients/. Each drives the same agent loop; they differ in "
     "whether a real device is involved.")
kvtable([
    ("executor_runner.py", "THE MAIN ONE. Generates a test, EXECUTES it on the emulator via "
                           "mobilerun, captures the real trajectory (screens + steps), self-heals on "
                           "failure, and logs everything back. This is the full testing suite."),
    ("crawl_runner.py", "Autonomous explorer. Maps the app into the Live App Model BEFORE testing - "
                        "useful for a zero-doc app so the agent has a map to reason over."),
    ("simulator_runner.py", "No device. Runs the planner/learning loop but SIMULATES pass/fail "
                            "verdicts. Use it to exercise generation + learning without an emulator."),
])

h2("Recommended run order")
h3("Step 1 (optional) - Crawl the app to seed the Live App Model")
para("Drives the emulator to explore the app and record its screens/transitions. Controlled by "
     "CRAWL_ROUNDS and CRAWL_MAX_STEPS in .env.")
code(".venv_py_3_13\\Scripts\\python.exe clients/crawl_runner.py")

h3("Step 2 (optional) - Ingest documents")
para("If you have an SRS / Figma / defects for the app, run the ingest from Part 1, Step 5. Skip for "
     "a pure zero-doc exploration.")

h3("Step 3 - Run the executor loop on the emulator")
para("This is the main testing suite. It runs EXECUTOR_ROUNDS rounds; each round the agent generates "
     "one test and executes it on the device against TARGET_APP_PACKAGE.")
code(".venv_py_3_13\\Scripts\\python.exe clients/executor_runner.py")
para("Watch it live in two places while it runs:")
bullets([
    ("Dashboard:", "the Live Activity strip shows the current test and a verdict stream; the App "
                   "Model graph grows as new screens are discovered; risk, coverage and anomalies "
                   "update each round."),
    ("Console / logs:", "mobilerun's on-device 'thinking' and actions are teed to logs/mobilerun.log "
                        "and streamed into the dashboard log view."),
])

h2("Key configuration knobs (.env)")
kvtable([
    ("PROJECT", "The project name (graph namespace). All runners and the dashboard must agree on it."),
    ("TARGET_APP_PACKAGE", "The Android package under test, e.g. com.android.contacts."),
    ("EXECUTOR_ROUNDS", "How many generate-and-execute rounds the executor runs (default 2)."),
    ("EXECUTOR_MAX_STEPS", "Max on-device steps mobilerun may take per test."),
    ("SELF_HEAL", "1 = classify failures and attempt an adaptive recovery + retry (WP7); 0 = off."),
    ("SIM_ROUNDS / SIM_FAIL_EVERY", "For simulator_runner: number of rounds and how often to force a "
                                    "simulated failure."),
    ("MODEL_BACKEND", "gemini | openrouter | ngrok - which LLM generates the tests."),
    ("EMBEDDING_BACKEND", "fastembed (local, free) recommended - powers semantic retrieval & dedup."),
])

h2("How many tests does a run produce?")
para("One test per round. executor_runner runs EXECUTOR_ROUNDS tests on the device; simulator_runner "
     "runs SIM_ROUNDS simulated tests. Increase the rounds to build up more history (the learning "
     "signals get richer with more executions).")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PART 3 - VERIFICATION SUITE
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("5. Part 3 - Running the Verification Suite (101 checks)")
para("Before (or instead of) a live emulator run, you can prove that every feature works end to end "
     "with a single LLM-free script. It seeds throwaway projects with synthetic data and asserts that "
     "each endpoint returns correct graph-derived results. No emulator and no model tokens required - "
     "only the RAG API (:9010) and Neo4j.")
code(".venv_py_3_13\\Scripts\\python.exe scripts/verify_enhancements.py")
para("Expected final line:")
code("RESULT: 101 passed, 0 failed")
para("The output is grouped by feature (REQ-301 Defect History, REQ-302 Navigation Tree, ... WP8 "
     "anomaly detection, WP9 live observability). A PASS line for each acceptance criterion is your "
     "proof that the corresponding capability is wired correctly.")
callout("This is the fastest way to confirm a healthy install. If all 101 pass, the backend, graph "
        "schema, learning loop, and dashboard data contract are all correct.")

# ═══════════════════════════════════════════════════════════════════════════
# 6. PART 4 - MANUAL FEATURE CHECKS
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("6. Part 4 - Manual Feature Checks")
para("Each capability below lists: what it MEANS, how to CHECK it by hand (an endpoint on the RAG API "
     ":9010), and what a healthy RESULT looks like. Replace PROJECT with your project name. These use "
     "GET unless noted; use a browser, curl, or the Swagger UI at http://localhost:9010/docs .")

def feature(title, meaning, how, result, req=""):
    h3(title + (f"   [{req}]" if req else ""))
    pdf.set_font("Helvetica", "B", 9.5); pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5, "MEANS"); pdf.set_font("Helvetica", "", 10.5); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.2, meaning); pdf.ln(0.5)
    if how:
        code(how)
    pdf.set_font("Helvetica", "B", 9.5); pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5, "HEALTHY RESULT"); pdf.set_font("Helvetica", "", 10.5); pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.2, result); pdf.ln(2.5)

feature("Live App Model (self-built UI map)",
    "The agent builds its own map of the app by observing each running screen. Distinct screens "
    "become UIState nodes; navigations between them become transitions. Re-seeing a screen does not "
    "duplicate it (deduped by a structural signature); reaching a new screen adds a node. Stale "
    "screens fade after app updates (knowledge decay).",
    "curl \"http://localhost:9010/appmodel/graph?project=PROJECT\"",
    "A JSON object with nodes (each screen with label, visit_count, has_shot) and edges (action-"
    "labelled transitions). state_count grows as new screens are found; revisits only bump "
    "visit_count. On the dashboard this renders as the App Model graph you can watch grow.",
    "WP1")

feature("Defect Intelligence",
    "Historical defects are ingested as first-class knowledge. The agent computes which feature areas "
    "are most defect-prone and biases test generation toward them - broken areas tend to break again.",
    "curl \"http://localhost:9010/defects/summary?project=PROJECT\"\n"
    "curl \"http://localhost:9010/defects/prone-areas?project=PROJECT\"",
    "total_defects, unresolved_defects, a severity distribution, and a ranked prone_areas list. "
    "Generated tests should skew toward the highest-density areas versus a no-defect baseline.",
    "WP2 / REQ-301")

feature("Execution-trace capture",
    "Every test run is persisted with its verdict, timing, step counts, error type, and the exact "
    "ordered path of screens it walked. This is the raw material for all experiential learning.",
    "curl \"http://localhost:9010/execution/logs?project=PROJECT&limit=10\"",
    "One ExecutionLog per run with verdict, duration_ms, device_steps, error_type, and path_labels. "
    "After a real executor loop there is one per round.",
    "WP3 / REQ-303.1")

feature("Navigation memory (NavTree)",
    "The agent remembers the shortest successful route to each screen and reuses it instead of "
    "re-exploring; routes that repeatedly fail are flagged 'avoid'.",
    "curl \"http://localhost:9010/navtree/retrieve-path?project=PROJECT&screen=SCREEN_LABEL\"\n"
    "curl \"http://localhost:9010/navtree/failed-paths?project=PROJECT\"",
    "retrieve-path returns an ordered step list (the proven shortest path). failed-paths lists "
    "dead-ends with a success/visit ratio below the avoid threshold. Re-running a test reuses the "
    "stored path, so it takes fewer exploratory steps.",
    "WP4 / REQ-302")

feature("Experiential learning + business logic",
    "Recurring failures are mined into ErrorPatterns with suggested mitigations; test STRATEGIES that "
    "find defects gain effectiveness score; a COVERAGE heatmap tracks what is tested; SESSIONS give "
    "continuity; and SRS extraction is multi-pass with confidence, provenance, versioning and drift "
    "detection (re-ingesting a changed SRS flags the affected areas for re-test). Stale knowledge "
    "decays (90-day half-life).",
    "curl \"http://localhost:9010/execution/error-patterns?project=PROJECT\"\n"
    "curl \"http://localhost:9010/strategy/memory?project=PROJECT\"\n"
    "curl \"http://localhost:9010/coverage/heatmap?project=PROJECT\"\n"
    "curl \"http://localhost:9010/srs/drift?project=PROJECT\"",
    "error-patterns lists recurring failure signatures + mitigations; strategy/memory lists "
    "strategies with a decay-weighted effectiveness score; the heatmap reports covered vs total "
    "areas/requirements; srs/drift reports the SRS version and any areas needing re-test.",
    "WP5 / REQ-303 + BLI")

feature("Multi-dimensional knowledge",
    "Knowledge can be partitioned by profile / platform / application so retrieval only returns "
    "in-dimension context, and tests from one environment can be SUGGESTED for another (cross-"
    "dimensional transfer with a confidence score).",
    "curl \"http://localhost:9010/dimensions/list?project=PROJECT\"\n"
    "curl \"http://localhost:9010/dimensions/transfer-suggestions?project=PROJECT&platform=android\"",
    "dimensions/list shows the registered profile/platform/application values. Retrieval with a "
    "dimension filter excludes out-of-dimension content (no leakage). transfer-suggestions proposes "
    "same-app tests for an untested environment with a transfer_confidence.",
    "WP6 / REQ-304")

feature("Self-healing + regression risk",
    "On failure the executor classifies the cause (navigation, element-not-found, assertion, timeout, "
    "crash, permission) and attempts a category-appropriate recovery + retry, logging the outcome. "
    "Separately, a regression-risk score per area (from defect density + failure ratio + recency + "
    "navigation instability) biases generation toward fragile areas.",
    "curl \"http://localhost:9010/risk/scores?project=PROJECT\"",
    "risk_scores ranks areas by regression_risk_score in [0,1] with the contributing factors exposed "
    "(defect_density, fail_ratio, defect_recency, nav_instability). In a live run, a recoverable "
    "failure shows a recovery_action ending in RECOVERED on its ExecutionLog.",
    "WP7 / REQ-305, 306")

feature("Quality metrics + semantic dedup + anomalies",
    "Per-test effectiveness (defect-discovery rate, execution stability, coverage contribution) is "
    "tracked. Duplicate detection uses embedding-cosine similarity, not just word overlap, so "
    "reworded duplicates are caught. An anomaly engine scans execution logs for emerging issues "
    "(failure-rate spikes, execution-time regressions, new error types, navigation instability) and "
    "surfaces them so the agent generates targeted investigation tests.",
    "curl \"http://localhost:9010/tests/effectiveness?project=PROJECT\"\n"
    "curl -X POST \"http://localhost:9010/anomalies/detect\" -H \"Content-Type: application/json\" "
    "-d '{\"project\":\"PROJECT\"}'\n"
    "curl \"http://localhost:9010/anomalies?project=PROJECT\"",
    "effectiveness returns per-test metrics; a reworded-but-equivalent test title scores a high "
    "similarity on /tests/dedup-check while an unrelated one scores low; anomalies/detect returns "
    "current alerts (type, area, severity, description) which then bias the next generated test.",
    "WP8 / REQ-307, 308")

feature("Live observability",
    "A single dashboard shows what the agent is doing now (current test + live verdict stream), the "
    "growing App Model graph, and 'getting smarter' trend charts (cumulative bugs, states discovered, "
    "steps-per-run, pass-rate).",
    "curl \"http://localhost:9010/session/live?project=PROJECT\"\n"
    "curl \"http://localhost:9010/metrics/trends?project=PROJECT\"",
    "session/live reports status (executing/idle), the most-recent test, and a recent verdict stream; "
    "metrics/trends returns per-run series. Both drive the dashboard's top strip and trend charts.",
    "WP9")

# ═══════════════════════════════════════════════════════════════════════════
# 7. PART 5 - DASHBOARD WALKTHROUGH
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("7. Part 5 - Dashboard Walkthrough")
para("Open http://localhost:9100/dashboard?project=PROJECT . One poll drives every panel. From top "
     "to bottom:")
bullets([
    ("Header pills:", "project, model backend, a live status pill (executing / idle), the active "
                      "session, and its dimension tags."),
    ("KPI tiles:", "total tests, pass rate, bugs found, coverage, top regression risk, defects in the "
                   "knowledge base, self-heals, anomalies, requirements."),
    ("Live Activity strip:", "the test running right now (or the last one) with its verdict, plus a "
                             "colored verdict stream of recent rounds."),
    ("Execution:", "the test-case table and an Execution Timeline (verdict, duration, steps, "
                   "classified error type, and self-healing outcome per run)."),
    ("Risk & Defects:", "regression-risk heat meters, defect intelligence (severity + prone areas), "
                        "and live bugs found."),
    ("Learning:", "strategy memory (effectiveness bars), error patterns (+ mitigations), and "
                  "navigation memory (nodes, dead-ends, paths to avoid)."),
    ("Quality & Anomalies:", "anomaly alerts by severity, and per-test effectiveness metrics."),
    ("Getting Smarter - Trends:", "sparkline charts proving improvement over runs."),
    ("Knowledge:", "SRS stats, business-logic health (versions, drift, low-confidence rules), and "
                   "dimensions."),
    ("Live App Model:", "the force-directed screen graph (hover a node for its screenshot) plus a "
                        "screenshot card grid. Watch it grow as the executor explores."),
])
callout("If a panel is empty, that source simply has no data yet for this project - run an ingest or "
        "an executor loop. Panels degrade independently; one empty source never blanks the board.")

# ═══════════════════════════════════════════════════════════════════════════
# 8. PART 6 - GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("8. Part 6 - Feature Glossary (WP1 - WP9)")
kvtable([
    ("WP1 Live App Model", "Self-built, screenshot-grounded UI state graph; the always-available "
                           "knowledge source for zero-doc apps."),
    ("WP2 / REQ-301 Defects", "Defect history as a knowledge source; biases tests toward defect-prone "
                              "areas."),
    ("WP3 / REQ-303.1 Traces", "Persist the executor's real trajectory + rich verdict per run."),
    ("WP4 / REQ-302 NavTree", "Remember shortest successful paths; reuse them; avoid failed ones."),
    ("WP5 / REQ-303 Learning", "Error patterns, strategy memory, coverage heatmap, sessions, knowledge "
                               "decay, and evolving business-logic extraction (multi-pass, confidence, "
                               "provenance, versioning, drift)."),
    ("WP6 / REQ-304 Dimensions", "Partition + filter knowledge by profile/platform/application; "
                                 "cross-dimensional transfer."),
    ("WP7 / REQ-305,306", "Self-healing recovery on failure; regression-risk-weighted prioritization."),
    ("WP8 / REQ-307,308", "Test effectiveness metrics, embedding-based semantic dedup, and anomaly "
                          "detection from execution patterns."),
    ("WP9 Observability", "The operator dashboard: live status, verdict stream, App Model graph, and "
                          "gets-smarter trend charts."),
])

# ═══════════════════════════════════════════════════════════════════════════
# 9. TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════
h1("9. Troubleshooting")
kvtable([
    ("RAG API won't start / ServiceUnavailable", "Neo4j is not running or the URI/credentials in .env "
                                                 "are wrong. Start Neo4j and re-run the Step-1 check."),
    ("Dashboard banner: cannot reach gateway", "The gateway (:9100) is not running, or you opened the "
                                               "dashboard on the wrong port. Start gateway.main."),
    ("Panels are empty", "No data for this project yet. Confirm the ?project= in the URL matches "
                         "PROJECT in .env, then ingest or run a loop."),
    ("UnicodeEncodeError on Windows console", "The runners already force UTF-8; if you wrapped them, "
                                              "set PYTHONUTF8=1."),
    ("Executor: no device / adb", "Ensure the emulator is running and 'adb devices' lists it before "
                                  "starting executor_runner.py."),
    ("Semantic dedup shows enabled=false", "EMBEDDING_BACKEND is off/unavailable. Set it to fastembed "
                                           "(local, no key required)."),
])

# ═══════════════════════════════════════════════════════════════════════════
# 10. APPENDIX
# ═══════════════════════════════════════════════════════════════════════════
pdf.add_page()
h1("10. Appendix - Endpoint Reference")
para("All on the RAG API (:9010) unless noted. The Gateway (:9100) mirrors the agent-facing ones and "
     "serves the dashboard. Full interactive docs: http://localhost:9010/docs .")
kvtable([
    ("GET /health", "Service + Neo4j status."),
    ("POST /ingest/srs | /ingest/figma | /ingest/defects", "Load knowledge sources."),
    ("POST /retrieve", "Hybrid (vector+keyword+graph) context retrieval, dimension-filterable."),
    ("POST /agent/next-testcase   (gateway)", "Generate the next exploratory test."),
    ("POST /agent/log-verdict-and-next  (gateway)", "Log a verdict and get the next test."),
    ("POST /liveui/observe  |  GET /appmodel/graph", "Record an observed screen | read the UI map."),
    ("GET /defects/summary | /defects/prone-areas", "Defect intelligence."),
    ("GET /navtree/retrieve-path | /navtree/failed-paths", "Navigation memory."),
    ("GET /execution/logs | /execution/error-patterns", "Execution history + mined patterns."),
    ("GET /strategy/memory | /coverage/heatmap", "Strategy scores | coverage snapshot."),
    ("POST /session/start | /session/end ; GET /session/context | /session/live", "Sessions + live status."),
    ("GET /srs/drift | /business-logic/rules", "SRS versioning/drift | extracted rules."),
    ("GET /dimensions/list | /dimensions/transfer-suggestions", "Dimensions + transfer."),
    ("GET /risk/scores", "Regression risk per area."),
    ("GET /tests/effectiveness ; POST /tests/dedup-check", "Effectiveness metrics | semantic dedup."),
    ("POST /anomalies/detect ; GET /anomalies", "Detect + read anomaly alerts."),
    ("GET /metrics/trends", "Gets-smarter trend series."),
    ("GET /dashboard  |  /dashboard/data   (gateway)", "The dashboard page | its aggregated payload."),
])

out = "d:/Projects/Exploratory-Testing-Planner-Agent/docs/USER_GUIDE.pdf"
pdf.output(out)
print("WROTE", out, "pages:", pdf.page_no())
