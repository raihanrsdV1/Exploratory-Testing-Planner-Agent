# QA Test-Case Agent with Notebook Model + Local Neo4j RAG

This project runs a **Qwen model in a notebook** (GPU side) and keeps **RAG + orchestration on your local machine**.

It is designed for iterative QA testing:
1. Generate next test case
2. Run in simulator
3. Log verdict (`pass` / `failed`)
4. Generate the next informed test case using:
   - SRS context
   - already executed tests
   - failed-test history

---

## Architecture

- **Notebook (planner.ipynb)**
  - Loads Qwen model
  - Exposes model-only API:
    - `GET /health`
    - `POST /generate`
  - Exposed publicly via ngrok

- **Local RAG API (local_rag_api.py)**
  - Connects to Neo4j
  - Stores SRS chunks
  - Stores executed tests + verdict runs
  - Retrieves context from SRS + test history

- **Local Gateway (local_agent_gateway.py)**
  - Calls RAG API + Notebook model API
  - Generates next test case
  - Logs verdict and returns next test case
  - Adds duplicate-avoidance and failed-history prioritization

- **Simulator loop client (simulator_runner.py)**
  - Demonstrates continuous loop automatically

---

## Files

- `planner.ipynb` — model server notebook
- `local_rag_api.py` — Neo4j-backed RAG storage/retrieval
- `local_agent_gateway.py` — orchestration service
- `simulator_runner.py` — demo runner for iterative test generation
- `test_loop_client.py` — interactive loop client
- `requirements-device.txt` — local Python dependencies
- `neo4j_setup.md` — Neo4j setup reference
- `device_setup.md` — quick setup reference

---

## Prerequisites

- Python 3.10+
- Neo4j 5.x (Desktop or Docker)
- Notebook environment with GPU (for `planner.ipynb`)
- ngrok account/token (for notebook model endpoint)

---

## 1) Local install

```bash
pip install -r requirements-device.txt
```

---

## 2) Start Neo4j

Use Neo4j Desktop or Docker.

Default local connection used by RAG service:
- `NEO4J_URI=neo4j://127.0.0.1:7687`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=<your_password>`

---

## 3) Run Notebook model API (planner.ipynb)

In `planner.ipynb`, run cells in order to:
1. install notebook deps
2. load model
3. start FastAPI model server (`/generate`)
4. open ngrok tunnel to port `8000`

You should get a public URL like:

```text
https://xxxx.ngrok-free.app
```

Check it:

```bash
curl https://xxxx.ngrok-free.app/health
```

---

## 4) Start local RAG API

```bash
uvicorn local_rag_api:app --host 0.0.0.0 --port 9000 --reload
```

Health check:

```bash
curl http://127.0.0.1:9000/health
```

---

## 5) Start local gateway (IMPORTANT: set current model URL)

Set the **current** ngrok URL from notebook before starting gateway:

```bash
export MODEL_API_URL="https://xxxx.ngrok-free.app"
uvicorn local_agent_gateway:app --host 0.0.0.0 --port 9100 --reload
```

Health check:

```bash
curl http://127.0.0.1:9100/health
```

If ngrok URL changes, restart gateway with updated `MODEL_API_URL`.

---

## 6) Ingest SRS

```bash
curl -X POST http://127.0.0.1:9100/srs/ingest \
  -H 'Content-Type: application/json' \
  -d '{"project":"contacts-app","source_path":"./SRS1.txt"}'
```

Expected output includes `chunks_written`.

---

## 7) Generate next test case

```bash
curl -X POST http://127.0.0.1:9100/agent/next-testcase \
  -H 'Content-Type: application/json' \
  -d '{
    "project":"contacts-app",
    "app_name":"contacts app",
    "objective":"generate next high-value non-duplicate test case",
    "top_k":8,
    "max_new_tokens":420,
    "enable_thinking":false
  }'
```

Response fields:
- `next_testcase_json` (raw model text)
- `next_testcase` (parsed object with `test_case_id`, `title`, `steps`, etc.)
- `recent_tests_count`
- `failed_tests_count`

---

## 8) Log verdict and get next in one call

```bash
curl -X POST http://127.0.0.1:9100/agent/log-verdict-and-next \
  -H 'Content-Type: application/json' \
  -d '{
    "project":"contacts-app",
    "app_name":"contacts app",
    "test_case_id":"TC-001",
    "title":"Verify ...",
    "verdict":"failed",
    "notes":"simulator failed at step 3",
    "area":"Contact Creation",
    "top_k":8,
    "max_new_tokens":420,
    "enable_thinking":false
  }'
```

---

## 9) Run automatic simulator loop

```bash
python simulator_runner.py
```

This demonstrates multiple rounds and prints recently saved tests from Neo4j.

---

## Neo4j data model (high-level)

Nodes:
- `Project`
- `SRS`
- `Chunk`
- `TestCase`
- `TestRun`

Relationships:
- `(:Project)-[:HAS_SRS]->(:SRS)-[:HAS_CHUNK]->(:Chunk)`
- `(:Project)-[:HAS_TEST]->(:TestCase)-[:HAS_RUN]->(:TestRun)`

---

## Useful checks

Recent tests via API:

```bash
curl "http://127.0.0.1:9000/tests/recent?project=contacts-app&limit=10"
```

Recent tests via Cypher:

```cypher
MATCH (t:TestCase)-[:HAS_RUN]->(r:TestRun)
RETURN t.id, t.title, r.verdict, r.created_at
ORDER BY r.created_at DESC
LIMIT 30;
```

---

## Common issues

### 1) Gateway returns model backend unavailable (404/503)
- `MODEL_API_URL` is stale
- Notebook model API not running
- ngrok tunnel changed

Fix: update `MODEL_API_URL`, restart gateway.

### 2) `curl http://127.0.0.1:9100/health` fails
- Gateway not running/crashed

Fix: restart `uvicorn local_agent_gateway:app ...` and check logs.

### 3) Neo4j auth errors
- Wrong `NEO4J_PASSWORD`

Fix: set correct env vars before starting RAG API.

### 4) Repetitive test cases
- Increase `top_k`
- Increase `max_new_tokens` (e.g., 420–700)
- Keep verdict notes descriptive to improve context quality

---

## Security note

- Do **not** commit ngrok auth tokens or credentials to git.
- Prefer env vars for:
  - `NEO4J_PASSWORD`
  - `MODEL_API_URL`
  - `RAG_API_KEY`
  - `GATEWAY_API_KEY`

---

## Quick start (minimal)

```bash
# terminal 1
uvicorn local_rag_api:app --host 0.0.0.0 --port 9000 --reload

# terminal 2 (after notebook gives fresh ngrok URL)
export MODEL_API_URL="https://xxxx.ngrok-free.app"
uvicorn local_agent_gateway:app --host 0.0.0.0 --port 9100 --reload

# terminal 3
python simulator_runner.py
```
