# Device-side setup (RAG + agent orchestrator)

Run these on your own machine. Keep the notebook as model-only server.

## 1) Install dependencies

- pip install -r requirements-device.txt

## 2) Start Neo4j-backed RAG service

- File: local_rag_api.py
- Start:
  - uvicorn local_rag_api:app --host 0.0.0.0 --port 9000

Main endpoints:
- POST /ingest/srs
- POST /retrieve
- POST /tests/log
- GET /tests/recent

## 3) Start notebook model API and expose it

- In planner notebook, run model + API cells and expose port 8000 (ngrok)
- Set env var on your device:
  - MODEL_API_URL=https://<notebook-ngrok-url>

## 4) Start local agent gateway (orchestrator)

- File: local_agent_gateway.py
- Start:
  - uvicorn local_agent_gateway:app --host 0.0.0.0 --port 9100

Main endpoints:
- POST /srs/ingest
- POST /agent/next-testcase
- POST /agent/log-verdict-and-next

## 5) Suggested flow for continuous testing

1. Ingest SRS once:
   - POST http://127.0.0.1:9100/srs/ingest
   - body: {"project":"contacts-app","source_path":"./SRS1.txt"}
2. Ask next test:
   - POST http://127.0.0.1:9100/agent/next-testcase
3. Execute in simulator and produce verdict (pass/failed).
4. Log verdict + get next test in one call:
   - POST http://127.0.0.1:9100/agent/log-verdict-and-next
5. Repeat step 3-4.

## Environment variables

- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
- MODEL_API_URL
- RAG_API_KEY (optional)
- GATEWAY_API_KEY (optional)
