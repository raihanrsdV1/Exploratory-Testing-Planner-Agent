# System Diagrams

Every diagram below describes the system as it is actually built, not as it was
planned. Rendered PNGs live in [`diagrams/`](diagrams/) — regenerate them with
`scripts/render_diagrams.py` after editing any Mermaid block here.

Colour convention, used consistently across all diagrams:

| Colour | Means |
|---|---|
| 🟧 amber | inputs — things a human supplies |
| 🟩 green | the knowledge graph and what is stored in it |
| 🟦 blue | long-running services (HTTP) |
| 🟪 purple | the three LLM agents |
| 🟥 red | language-model calls (billed, slow, non-deterministic) |
| ⬜ grey | the device and the app under test |

---

## 1. System architecture

The three agents never call each other. They communicate only by reading and
writing the knowledge graph, which is why any of them can be changed or replaced
without touching the others.

```mermaid
flowchart TB
    subgraph IN["INPUTS"]
        SRS["SRS document<br/><i>requirements, in prose</i>"]
        FIG["Figma export<br/><i>optional</i>"]
        DEF["Defect history<br/><i>optional</i>"]
    end

    subgraph KN["KNOWLEDGE"]
        NEO[("Neo4j :7687<br/>requirements · app map<br/>findings · execution history")]
        VEC[["Vector index<br/>fastembed BAAI/bge-small-en-v1.5"]]
    end

    subgraph SV["SERVICES"]
        RAG["RAG API :9010<br/><i>rag_api/</i><br/>retrieval · coverage · findings"]
        GW["Agent Gateway :9100<br/><i>gateway/ + planner/</i>"]
    end

    subgraph AG["AGENTS"]
        PL["PLANNER<br/><i>what to test next</i>"]
        IV["INVESTIGATOR<br/><i>what the run proved</i>"]
        EX["EXECUTOR<br/><i>clients/executor_runner.py</i><br/>mobilerun 0.6.8"]
    end

    subgraph DV["DEVICE"]
        PORT["On-device portal<br/>accessibility service + IME"]
        APP["App under test"]
    end

    OR{{"OpenRouter<br/>qwen3.7-flash · glm-5.3-flash"}}

    SRS --> RAG
    FIG --> RAG
    DEF --> RAG
    RAG <--> NEO
    RAG <--> VEC
    GW <-->|HTTP| RAG
    GW --> PL
    GW --> IV
    PL -->|"test case"| EX
    EX -->|"trajectory"| IV
    EX -->|"observed UI states"| RAG
    IV -->|"verdict + findings"| RAG
    EX <-->|"ADB"| PORT
    PORT <--> APP
    PL -.->|"chat/completions"| OR
    IV -.-> OR
    EX -.-> OR

    classDef input fill:#FFE8CC,stroke:#E8892B,stroke-width:2px,color:#7A4A10
    classDef know fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef svc fill:#D0EBFF,stroke:#1971C2,stroke-width:2px,color:#0B4A87
    classDef agent fill:#E5DBFF,stroke:#6741D9,stroke-width:2px,color:#3B2185
    classDef dev fill:#E9ECEF,stroke:#868E96,stroke-width:2px,color:#343A40
    classDef model fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    class SRS,FIG,DEF input
    class NEO,VEC know
    class RAG,GW svc
    class PL,IV,EX agent
    class PORT,APP dev
    class OR model
```

---

## 2. One test round, end to end

This cycle is the unit of work. A 50-round campaign is this diagram fifty times,
with the graph larger on each pass. Nothing else carries between rounds — every
LLM call is stateless.

```mermaid
sequenceDiagram
    autonumber
    participant P as Planner
    participant G as Knowledge graph
    participant X as Executor
    participant D as Device
    participant I as Investigator

    P->>G: what is known? (coverage, findings,<br/>open questions, untested requirements)
    G-->>P: retrieved context
    P->>P: build prompt → LLM → propose test case
    Note over P: validation gate: real screen?<br/>real requirement id? not a duplicate?
    P-->>P: rejected → fixable reason → retry (max 3)
    P->>G: store TestCase, verdict = "planned"
    P->>X: objective + screen_hint + addresses
    loop up to 50 steps
        X->>D: observe screen, decide, act
        D-->>X: accessibility tree + screenshot
        X->>G: observed UIState + transition
    end
    X->>I: trajectory
    I->>G: findings already known for these screens
    G-->>I: prior findings (so repeats reinforce, not duplicate)
    I->>I: LLM: did this prove the objective?
    I->>G: verdict + new findings + resolved questions
    Note over G: the next round starts from this larger graph
```

---

## 3. Knowledge graph schema

Only the decision-bearing nodes are shown. A node type earns a place here when
some module queries it to make a decision.

```mermaid
erDiagram
    PROJECT ||--o{ REQUIREMENT : "HAS_REQUIREMENT"
    PROJECT ||--o{ TESTCASE : "HAS_TEST"
    PROJECT ||--o{ FINDING : "HAS_FINDING"
    PROJECT ||--o{ FEATUREAREA : "HAS_FEATURE"
    TESTCASE ||--o{ REQUIREMENT : "COVERS"
    TESTCASE ||--o{ FEATUREAREA : "COVERS_FEATURE"
    TESTCASE ||--o{ TESTRUN : "HAS_RUN"
    TESTCASE ||--o{ EXECUTIONLOG : "produced"
    EXECUTIONLOG ||--o{ UISTATE : "visited"
    UISTATE ||--o{ UIELEMENT : "HAS_CONTROL"
    FINDING }o--|| UISTATE : "observed on"
    FINDING }o--o{ FINDING : "GENERALISED_BY"
    SRS ||--o{ CHUNK : "chunked into"
    CHUNK ||--o{ REQUIREMENT : "extracted"

    REQUIREMENT {
        string ref_id "FR-VAL-03"
        string text
        string priority
        int covered_count
    }
    TESTCASE {
        string external_id "TC-014"
        string title
        string area
        string last_verdict "pass|failed|planned"
    }
    EXECUTIONLOG {
        string verdict
        string error_type "STEP_LIMIT_EXCEEDED|ASSERTION_FAILURE|..."
        int device_steps
        int duration_ms
    }
    FINDING {
        string kind "8 kinds"
        string status "open|resolved|inconclusive"
        string claim
        int times_seen
        int attempts
    }
    UISTATE {
        string signature "structural, for dedupe"
        string label
    }
```

---

## 4. Planner — tool-calling mode

`PLANNER_MODE=tools`. The model investigates the graph through nine read-only
tools, then proposes. The gate is what makes it honest: a proposal naming a
screen the app has never shown is rejected with a reason it can act on.

```mermaid
flowchart TD
    S([POST /agent/next-testcase]) --> SEED["Seed message<br/>objective · session constraints<br/>last 3 runs · open questions<br/>exploration directive · step budget"]
    SEED --> LLM{"Planner model<br/>+ 9 tools"}
    LLM -->|tool call| T["run tool, append result"]
    T --> LLM
    LLM -->|propose_test_case| V{"Validation gate"}
    V -->|"screen does not exist"| LLM
    V -->|"requirement id is invented"| LLM
    V -->|"duplicate of earlier test"| LLM
    V -->|"out of scope"| LLM
    V -->|accepted| A["record attempt on the open question<br/>auto-log verdict=planned + COVERS edges"]
    A --> E([return test case])

    TOOLS["search_requirements · list_untested_requirements<br/>get_screen · list_screens · get_nav_path<br/>findings_summary · list_findings<br/>list_open_questions · get_coverage"]
    TOOLS -.-> T

    classDef gate fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    classDef agent fill:#E5DBFF,stroke:#6741D9,stroke-width:2px,color:#3B2185
    classDef tool fill:#D0EBFF,stroke:#1971C2,stroke-width:2px,color:#0B4A87
    classDef ok fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    class V gate
    class LLM agent
    class TOOLS,T tool
    class A,E ok
```

---

## 5. Planner — pipeline mode (default)

`PLANNER_MODE=pipeline`. A LangGraph state machine that decides what to retrieve
before spending the one expensive generation call. **Note:** the proposal
validation gate in diagram 4 belongs to tools mode only — in pipeline mode it
does not run.

```mermaid
flowchart TD
    START([POST /agent/next-testcase]) --> BC["bootstrap_context<br/><i>no LLM call</i>"]
    BC --> PS["planner_step<br/>1 SMALL LLM call<br/><i>what do I still need?</i>"]
    PS --> ROUTE{should_continue?}
    ROUTE -->|"retrieve more"| ER["execute_retrieval<br/><i>no LLM call</i><br/>1–3 sources per round"]
    ER --> PS
    ROUTE -->|"ready · round > 6 ·<br/>context > 9000 chars ·<br/>nothing new"| GT["generate_testcase<br/>1 BIG LLM call<br/>+ screenshot if resolved"]
    GT --> DC{"duplicate_check<br/>Jaccard + cosine"}
    DC -->|"too similar"| RETRY["1 more LLM call:<br/>alternate screens + blocked titles"]
    RETRY --> LOG
    DC -->|distinct| LOG["auto-log to Neo4j<br/>verdict = planned"]
    LOG --> END([return to executor])

    classDef cheap fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef costly fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    classDef decide fill:#FFF3BF,stroke:#F08C00,stroke-width:2px,color:#7A4A10
    class BC,ER,LOG cheap
    class PS,GT,RETRY costly
    class ROUTE,DC decide
```

---

## 6. Investigator — trajectory to knowledge

The investigator is deliberately not the planner. The component deciding what was
proved is not the one that hoped to prove it.

```mermaid
flowchart LR
    T["trajectory.json<br/>up to 50 steps"] --> INV["POST /execution/evaluate<br/>QA Evaluator"]
    K["GET /findings?screens=…<br/><i>what is already known</i>"] --> INV
    M["mission block<br/><i>which open question<br/>this run addresses</i>"] --> INV
    INV --> REC["POST /findings/record<br/>dedupe + reinforce"]
    REC --> G[("Neo4j :Finding")]
    G -->|"group = oracle"| P["planner prompt<br/><i>what previous runs established</i>"]
    G -->|"group = agent"| A["agent-difficulty steering"]
    G -->|"screens = …"| INV
    INV --> EL["ExecutionLog.trajectory_summary<br/><i>one line, for humans</i>"]
    INV --> VD{"verdict<br/>pass / failed"}

    classDef store fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef agent fill:#E5DBFF,stroke:#6741D9,stroke-width:2px,color:#3B2185
    classDef io fill:#D0EBFF,stroke:#1971C2,stroke-width:2px,color:#0B4A87
    classDef decide fill:#FFF3BF,stroke:#F08C00,stroke-width:2px,color:#7A4A10
    class G store
    class INV agent
    class T,K,M,REC,P,A,EL io
    class VD decide
```

---

## 7. Finding taxonomy and who consumes it

Eight kinds, routed to four consumer groups. The split that matters most:
`AGENT_DIFFICULTY` describes *our tester*, not the app, and must never be counted
in a findings headline about application quality.

```mermaid
flowchart LR
    subgraph KINDS["8 FINDING KINDS"]
        SD["SUSPECTED_DEFECT"]
        SV["SPEC_VIOLATION"]
        SG["SPEC_GAP"]
        UB["UNEXPECTED_BEHAVIOUR"]
        CB["CONFIRMED_BEHAVIOUR"]
        CD["CONTROL_DISCOVERED"]
        UV["UNVERIFIED"]
        AD["AGENT_DIFFICULTY"]
    end

    SD --> DEFECT["defect<br/><i>needs a human</i>"]
    SV --> DEFECT
    SG --> ORACLE["oracle<br/><i>feeds the planner prompt</i>"]
    UB --> ORACLE
    CB --> ORACLE
    CD --> UI["ui<br/><i>enriches the app map</i>"]
    UV --> ORACLE
    AD --> AGENTG["agent<br/><i>our own limits —<br/>NOT app evidence</i>"]

    classDef defect fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    classDef oracle fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef ui fill:#D0EBFF,stroke:#1971C2,stroke-width:2px,color:#0B4A87
    classDef agent fill:#E9ECEF,stroke:#868E96,stroke-width:2px,color:#343A40
    classDef kind fill:#FFF9DB,stroke:#F08C00,stroke-width:1.5px,color:#7A4A10
    class SD,SV,SG,UB,CB,CD,UV,AD kind
    class DEFECT defect
    class ORACLE oracle
    class UI ui
    class AGENTG agent
```

---

## 8. Finding lifecycle

A finding that poses a question stays open until a later run settles it. After
three inconclusive attempts it is closed as `inconclusive` rather than retried
forever — an unbounded retry loop is not a result.

```mermaid
stateDiagram-v2
    [*] --> OPEN: recorded, and it poses a question
    [*] --> RESOLVED: recorded, and it states a fact<br/>(CONFIRMED_BEHAVIOUR)
    [*] --> NOSTATUS: AGENT_DIFFICULTY<br/>(not a question at all)

    OPEN --> OPEN: seen again → times_seen++
    OPEN --> ATTEMPTED: a test is written to address it
    ATTEMPTED --> RESOLVED: the run settled it
    ATTEMPTED --> OPEN: inconclusive, attempts < 3
    ATTEMPTED --> INCONCLUSIVE: attempts reached 3

    RESOLVED --> [*]
    INCONCLUSIVE --> [*]
    NOSTATUS --> [*]

    note right of OPEN
        Open questions are what the
        planner picks its next mission from
    end note
```

---

## 9. Failure attribution

The single most important distinction in the results. A testing agent that cannot
finish its own test produces a failure that looks identical to a real defect
unless the two are separated at the point of measurement.

```mermaid
flowchart TD
    V{"verdict"} -->|pass| OK["the app did what<br/>the test expected"]
    V -->|failed| ET{"error_type"}

    ET --> AF["ASSERTION_FAILURE"]
    ET --> SL["STEP_LIMIT_EXCEEDED"]
    ET --> NF["NAVIGATION_FAILURE"]
    ET --> CR["CRASH"]
    ET --> PN["PRECONDITION_NOT_MET"]

    AF --> APP["APP EVIDENCE<br/><i>the agent reached its checkpoint<br/>and the app misbehaved.<br/>This is the real finding count.</i>"]
    CR --> APP
    SL --> OURS["OUR CEILING<br/><i>the executor ran out of budget<br/>or could not navigate.<br/>Says nothing about the app.</i>"]
    NF --> OURS
    PN --> ENV["ENVIRONMENT<br/><i>the precondition was unavailable</i>"]

    classDef ok fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef bad fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    classDef ours fill:#E9ECEF,stroke:#868E96,stroke-width:2px,color:#343A40
    classDef env fill:#FFE8CC,stroke:#E8892B,stroke-width:2px,color:#7A4A10
    classDef decide fill:#FFF3BF,stroke:#F08C00,stroke-width:2px,color:#7A4A10
    class OK ok
    class APP bad
    class OURS ours
    class ENV env
    class V,ET decide
```

---

## 10. Campaign lifecycle

What a run does to stored state. `CLEAN_SLATE` resets execution history but
**findings survive it** — they are the accumulated knowledge, kept deliberately.
This asymmetry is why campaign reports must scope findings by date.

```mermaid
flowchart LR
    A(["start campaign"]) --> B{"RESUME?"}
    B -->|"RESUME=1"| C["keep everything<br/><i>ids continue</i>"]
    B -->|"CLEAN_SLATE=1"| D["snapshot to CampaignSummary,<br/>then delete tests + execution logs"]
    D --> E{"CLEAN_SLATE_APPMODEL?"}
    E -->|"=1"| F["also delete app map<br/>+ findings → fully blind"]
    E -->|"=0 (default)"| G["keep app map + findings<br/><i>a warm start</i>"]
    C --> H["run N rounds"]
    F --> H
    G --> H
    H --> I["write batch CSV<br/><i>the only record that<br/>survives the next reset</i>"]
    I --> J(["end"])

    classDef keep fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef wipe fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    classDef decide fill:#FFF3BF,stroke:#F08C00,stroke-width:2px,color:#7A4A10
    class C,G,I keep
    class D,F wipe
    class B,E decide
```

---

## 11. Ingestion — document to queryable requirements

Runs once per document version, not once per test.

```mermaid
flowchart LR
    DOC["SRS document<br/><i>FR-XXX-NN, one per line</i>"] --> CH["chunk"]
    CH --> EMB["embed<br/><i>fastembed</i>"]
    CH --> EX["extract requirements<br/><i>multipass LLM</i>"]
    EX --> REQ[(":Requirement<br/>ref_id · text · priority")]
    EX --> VR[(":ValidationRule")]
    EX --> ENT[(":Entity")]
    EMB --> VEC[("vector index")]
    REQ --> RET["retrieval at plan time<br/><i>semantic + keyword</i>"]
    VEC --> RET

    classDef input fill:#FFE8CC,stroke:#E8892B,stroke-width:2px,color:#7A4A10
    classDef know fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef proc fill:#D0EBFF,stroke:#1971C2,stroke-width:2px,color:#0B4A87
    class DOC input
    class REQ,VR,ENT,VEC know
    class CH,EMB,EX,RET proc
```

---

## 12. Runtime topology

What actually runs, and where. Everything except OpenRouter is local.

```mermaid
flowchart TB
    subgraph MAC["Developer machine (macOS)"]
        NEO[("Neo4j<br/>:7687")]
        RAG["uvicorn rag_api.main:app<br/>:9010"]
        GW["uvicorn gateway.main:app<br/>:9100"]
        CLI["clients/executor_runner.py<br/><i>the campaign driver</i>"]
    end

    subgraph EMU["Android emulator / device"]
        PORTAL["com.mobilerun.portal<br/>accessibility service + IME"]
        AUT["app under test"]
    end

    CLOUD{{"OpenRouter API<br/><i>the only external dependency</i>"}}

    RAG <--> NEO
    GW <--> RAG
    CLI <--> GW
    CLI <-->|"adb"| PORTAL
    PORTAL <--> AUT
    GW -.-> CLOUD
    CLI -.-> CLOUD

    classDef svc fill:#D0EBFF,stroke:#1971C2,stroke-width:2px,color:#0B4A87
    classDef know fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef dev fill:#E9ECEF,stroke:#868E96,stroke-width:2px,color:#343A40
    classDef model fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    class RAG,GW,CLI svc
    class NEO know
    class PORTAL,AUT dev
    class CLOUD model
```

---

## 13. Why tools replaced retrieval sources

The planner's original design had *sources*: each returned a block of text that
was stashed in a bucket and concatenated into one large prompt at the very end.
The model never read any of it while deciding what to fetch next — it chose each
round from one-line summaries. It was retrieving blind.

A tool returns its result **into the conversation**. The model reads the actual
content before deciding what to ask for next, so the second question can depend
on the answer to the first. That is the whole difference.

```mermaid
flowchart TB
    subgraph OLD["BEFORE — sources (blind retrieval)"]
        direction TB
        O1["planner picks a source<br/><i>from a one-line note</i>"] --> O2["source returns a text block"]
        O2 --> O3[("bucket")]
        O3 --> O1
        O3 --> O4["concatenate everything<br/>into one large prompt"]
        O4 --> O5["ONE generation call<br/><i>first time the model sees<br/>any of the content</i>"]
    end

    subgraph NEW["AFTER — tools (informed investigation)"]
        direction TB
        N1["model asks a question"] --> N2["tool returns bounded data<br/><i>into the conversation</i>"]
        N2 --> N3["model reads it<br/><i>and now knows something</i>"]
        N3 -->|"the next question<br/>depends on this answer"| N1
        N3 --> N4["propose test case<br/><i>grounded in what was read</i>"]
    end

    %% Disconnected subgraphs are ordered arbitrarily; an invisible link (~~~)
    %% pins BEFORE to the left of AFTER so it reads in the right order.
    O5 ~~~ N1

    classDef bad fill:#FFE3E3,stroke:#E03131,stroke-width:2px,color:#8B1A1A
    classDef good fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef neutral fill:#E9ECEF,stroke:#868E96,stroke-width:2px,color:#343A40
    class O1,O4,O5 bad
    class O2,O3 neutral
    class N1,N3,N4 good
    class N2 neutral
```

---

## 14. The nine tools as an investigation

The tools are not a menu of data feeds. They answer four different questions, and
a competent round moves through them in roughly that order: what *should* exist,
what the app *actually* has, what we *already know*, and therefore where the
*gap* is. The model chooses the order and stops when it has enough — the
reject-and-retry loop around the proposal is in diagram 4.

```mermaid
flowchart LR
    Q1["Q1 · What SHOULD exist?<br/><i>the specification</i>"]
    Q2["Q2 · What does the app ACTUALLY have?<br/><i>the observed world</i>"]
    Q3["Q3 · What do we ALREADY know?<br/><i>prior evidence</i>"]
    Q4["Q4 · Where is the GAP?<br/><i>the decision</i>"]

    Q1 --> T1["search_requirements<br/><i>semantic + keyword, ≤3000 chars</i>"]
    Q1 --> T2["list_untested_requirements<br/><i>the coverage frontier</i>"]
    Q2 --> T3["list_screens<br/><i>everything seen so far</i>"]
    Q2 --> T4["get_screen<br/><i>real controls — grounds screen_hint</i>"]
    Q2 --> T5["get_nav_path<br/><i>how to reach it</i>"]
    Q3 --> T6["findings_summary<br/><i>the whole graph, bounded</i>"]
    Q3 --> T7["list_findings<br/><i>by group or screen</i>"]
    Q3 --> T8["list_open_questions<br/><i>raised but never settled</i>"]
    Q4 --> T9["get_coverage<br/><i>by area and requirement</i>"]

    T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8 & T9 --> P["propose_test_case<br/><i>objective · screen_hint ·<br/>requirement ids · addresses</i>"]

    classDef q fill:#FFF3BF,stroke:#F08C00,stroke-width:2px,color:#7A4A10
    classDef spec fill:#FFE8CC,stroke:#E8892B,stroke-width:2px,color:#7A4A10
    classDef world fill:#E9ECEF,stroke:#868E96,stroke-width:2px,color:#343A40
    classDef known fill:#D3F9D8,stroke:#2F9E44,stroke-width:2px,color:#1B5E27
    classDef gap fill:#D0EBFF,stroke:#1971C2,stroke-width:2px,color:#0B4A87
    classDef act fill:#E5DBFF,stroke:#6741D9,stroke-width:2px,color:#3B2185
    class Q1,Q2,Q3,Q4 q
    class T1,T2 spec
    class T3,T4,T5 world
    class T6,T7,T8 known
    class T9 gap
    class P act
```

The ordering is a tendency, not a rule the code enforces. What the code does
enforce is that a proposal naming a screen `get_screen` never returned is
rejected — so Q2 is the one question a round cannot skip and still pass the gate.

---

## 15. Three rules that hold for every tool

These are what keep a tool-calling planner from degenerating into an unbounded, hijackable context. They are enforced in `planner/tools.py`, not left to the model's discretion.

```mermaid
flowchart TB
    R1["EVERY RESULT IS BOUNDED<br/>requirements 3000 · findings 2500<br/>screen 1200 · generic 2000 chars"]
    R1 --> R1W["There is no global prompt budget.<br/>Each tool is responsible for not<br/>flooding the context on its own."]

    R2["RESULTS ARE DATA, NEVER INSTRUCTIONS"]
    R2 --> R2W["Tool output is app content — screen labels,<br/>requirement text, findings written by a model.<br/>It must never redirect the planner."]

    R3["A TOOL WITH NO DATA IS NOT OFFERED"]
    R3 --> R3W["If its backing source is disabled or<br/>un-ingested the model never sees the tool,<br/>so it cannot be misled by an empty result."]

    classDef rule fill:#E5DBFF,stroke:#6741D9,stroke-width:2px,color:#3B2185
    classDef why fill:#F8F9FA,stroke:#ADB5BD,stroke-width:1.5px,color:#343A40
    class R1,R2,R3 rule
    class R1W,R2W,R3W why
```

> **Applies to `PLANNER_MODE=tools` only.** The default is `pipeline`
> (diagram 5), where none of this runs — including the validation gate.
