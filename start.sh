#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Start all QA Agent services and verify they are healthy
#
# Usage:
#   ./start.sh              # start everything + ingest SRS & Figma
#   ./start.sh --no-ingest  # start services only, skip ingest
#   ./start.sh --stop       # kill all managed services
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAG_PORT=9010
GATEWAY_PORT=9100

RAG_LOG="$DIR/logs/rag_api.log"
GATEWAY_LOG="$DIR/logs/gateway.log"

PID_FILE="$DIR/logs/services.pid"

PROJECT="contacts-app"
SRS_PATH="./SRS1.txt"
FIGMA_PATH="./GENERATED_JSON.json"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }
info() { echo -e "${CYAN}[..] ${NC} $*"; }
warn() { echo -e "${YELLOW}[!!] ${NC} $*"; }

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    while IFS= read -r pid; do
      kill "$pid" 2>/dev/null && echo "  killed PID $pid" || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    ok "All managed services stopped."
  else
    warn "No PID file found. Nothing to stop."
  fi
  exit 0
fi

NO_INGEST=false
[[ "${1:-}" == "--no-ingest" ]] && NO_INGEST=true

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$DIR/logs"
> "$PID_FILE"
cd "$DIR"

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  QA Agent System — Startup                    ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

# ── Helper: wait for HTTP health endpoint ─────────────────────────────────────
wait_for_health() {
  local name="$1" url="$2" retries=20 delay=1
  info "Waiting for $name at $url ..."
  for i in $(seq 1 $retries); do
    if curl -sf "$url" > /dev/null 2>&1; then
      ok "$name is up"
      return 0
    fi
    sleep "$delay"
  done
  err "$name did NOT start within $((retries * delay))s"
  err "  Check log: ${url/health/} → see logs/ directory"
  return 1
}

# ── Check for port conflicts ───────────────────────────────────────────────────
check_port() {
  local port="$1" name="$2"
  if lsof -ti :"$port" > /dev/null 2>&1; then
    local pids owner
    pids=$(lsof -ti :"$port" | tr '\n' ' ')
    owner=$(lsof -ti :"$port" | xargs ps -o comm= -p 2>/dev/null | head -1 || echo "unknown")
    warn "Port $port ($name) already in use by PID(s) $pids ($owner)"
    warn "  Will attempt to use existing process — if health check fails, run: kill $pids"
    return 1
  fi
  return 0
}

ERRORS=0

# ── 1. RAG API ────────────────────────────────────────────────────────────────
echo -e "${CYAN}[1/2] RAG API (Neo4j + SRS/Figma store)${NC}"
if check_port $RAG_PORT "RAG API"; then
  uvicorn local_rag_api:app --host 0.0.0.0 --port $RAG_PORT \
    > "$RAG_LOG" 2>&1 &
  RAG_PID=$!
  echo "$RAG_PID" >> "$PID_FILE"
  info "Started RAG API (PID $RAG_PID) → $RAG_LOG"
fi
wait_for_health "RAG API" "http://127.0.0.1:$RAG_PORT/health" || ERRORS=$((ERRORS+1))
echo ""

# ── 2. Agent Gateway ──────────────────────────────────────────────────────────
echo -e "${CYAN}[2/2] Agent Gateway${NC}"
if [[ -z "${MODEL_API_URL:-}" ]]; then
  err "MODEL_API_URL is not set. Point it to your Kaggle/ngrok planner /generate API."
  err "Example: export MODEL_API_URL=https://xxxx.ngrok-free.app"
  exit 1
fi
if check_port $GATEWAY_PORT "Gateway"; then
  RAG_API_URL="http://127.0.0.1:$RAG_PORT" \
  MODEL_API_URL="$MODEL_API_URL" \
  uvicorn local_agent_gateway:app --host 0.0.0.0 --port $GATEWAY_PORT \
    > "$GATEWAY_LOG" 2>&1 &
  GATEWAY_PID=$!
  echo "$GATEWAY_PID" >> "$PID_FILE"
  info "Started Gateway (PID $GATEWAY_PID) → $GATEWAY_LOG"
fi
wait_for_health "Gateway" "http://127.0.0.1:$GATEWAY_PORT/health" || ERRORS=$((ERRORS+1))
echo ""

# ── Abort if any service failed ───────────────────────────────────────────────
if [[ $ERRORS -gt 0 ]]; then
  echo -e "${RED}═══════════════════════════════════════════════${NC}"
  err "$ERRORS service(s) failed to start. Check logs in $DIR/logs/"
  echo -e "${RED}═══════════════════════════════════════════════${NC}"
  exit 1
fi

# ── 4. Ingest SRS + Figma ─────────────────────────────────────────────────────
if [[ "$NO_INGEST" == false ]]; then
  echo -e "${CYAN}[4/4] Ingesting SRS + Figma into Neo4j${NC}"

  if [[ -f "$SRS_PATH" && -f "$FIGMA_PATH" ]]; then
    info "Running ingest_all.py (reset + SRS + Figma + stats)"
    INGEST_OUTPUT=$(GATEWAY_URL="http://127.0.0.1:$GATEWAY_PORT" \
      RAG_URL="http://127.0.0.1:$RAG_PORT" \
      PROJECT="$PROJECT" \
      SRS_PATH="$SRS_PATH" \
      FIGMA_PATH="$FIGMA_PATH" \
      python ingest_all.py 2>&1) || {
      err "ingest_all.py failed"
      echo "$INGEST_OUTPUT"
      ERRORS=$((ERRORS+1))
    }
    if [[ $ERRORS -eq 0 ]]; then
      ok "Graph rebuilt successfully"
      echo "$INGEST_OUTPUT"
    fi
  else
    warn "SRS or Figma file missing (SRS=$SRS_PATH, FIGMA=$FIGMA_PATH) — skipping ingest"
  fi
  echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
if [[ $ERRORS -eq 0 ]]; then
  ok "All systems running."
else
  err "$ERRORS error(s) during ingest — services are up but data may be incomplete."
fi
echo ""
echo "  RAG API    → http://127.0.0.1:$RAG_PORT"
echo "  Model      → $MODEL_API_URL"
echo "  Gateway    → http://127.0.0.1:$GATEWAY_PORT"
echo ""
echo "  Next steps:"
echo "    python test_loop_client.py    # interactive QA loop"
echo "    python simulator_runner.py    # automated simulation"
echo "    ./start.sh --stop             # stop all services"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""

exit $ERRORS
