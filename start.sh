#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh — Bring up the whole QA Agent stack, idempotently.
#
#   Neo4j (local Desktop DBMS)  →  Android emulator  →  RAG API  →  Gateway
#
# A plain start brings up infrastructure only — it does NOT touch your data
# (no re-ingest, no executor). Opt in with flags.
#
# Usage:
#   ./start.sh                 # neo4j + emulator + rag_api + gateway (no data changes)
#   ./start.sh --ingest        # ALSO reset + ingest SRS/Figma  (DESTRUCTIVE: wipes tests/app-model)
#   ./start.sh --with-executor # ALSO start the executor test loop
#   ./start.sh --build         # ALSO (re)build the React dashboard first
#   ./start.sh --no-neo4j      # skip starting Neo4j (managed elsewhere)
#   ./start.sh --no-emulator   # skip the emulator (e.g. using a physical device)
#   ./start.sh --stop          # delegate to ./stop.sh
#
# Machine-specific paths (Neo4j DBMS dir, emulator AVD) can be overridden in .env.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
ENV_FILE="$DIR/.env"

# ── Delegate stop ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--stop" ]]; then exec "$DIR/stop.sh"; fi

# ── Load .env (for config + overrides) ──────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi

# ── Config (override any of these in .env) ──────────────────────────────────────
RAG_PORT=9010
GATEWAY_PORT=9100
PID_FILE="$DIR/logs/services.pid"
PROJECT="${PROJECT:-contacts-app}"
SRS_PATH="${SRS_PATH:-./data/inputs/Sample-Contacts-App-SRS.txt}"
FIGMA_PATH="${FIGMA_PATH:-./data/inputs/GENERATED_JSON.json}"

# Local Neo4j Desktop-managed instance (the "test" DBMS).
NEO4J_DBMS_DIR="${NEO4J_DBMS_DIR:-$HOME/Library/Application Support/neo4j-desktop/Application/Data/dbmss/dbms-15751e0d-b9e8-437e-8199-e0fb4c954865}"
NEO4J_JAVA_HOME="${NEO4J_JAVA_HOME:-$(ls -d "$HOME/Library/Application Support/neo4j-desktop/Application/Cache/runtime/"zulu*jre* 2>/dev/null | head -1)}"

# Android emulator.
ANDROID_EMULATOR="${ANDROID_EMULATOR:-$HOME/Library/Android/sdk/emulator/emulator}"
EMULATOR_AVD="${EMULATOR_AVD:-MyEmulator}"
PORTAL_A11Y="com.mobilerun.portal/com.mobilerun.portal.service.MobilerunAccessibilityService"

# ── Flags ────────────────────────────────────────────────────────────────────
DO_INGEST=false; DO_EXECUTOR=false; DO_BUILD=false; DO_NEO4J=true; DO_EMULATOR=true
for arg in "$@"; do
  case "$arg" in
    --ingest)        DO_INGEST=true ;;
    --with-executor) DO_EXECUTOR=true ;;
    --build)         DO_BUILD=true ;;
    --no-neo4j)      DO_NEO4J=false ;;
    --no-emulator)   DO_EMULATOR=false ;;
  esac
done

# ── Colours + helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }
info() { echo -e "${CYAN}[..]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }

mkdir -p "$DIR/logs"
: > "$PID_FILE"

# ── Resolve Python ──────────────────────────────────────────────────────────────
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if   [[ -x "$DIR/venv/bin/python" ]]; then PYTHON_BIN="$DIR/venv/bin/python"
  elif command -v python  >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"
  else err "No Python found. Create venv or set PYTHON_BIN in .env"; exit 1; fi
fi

wait_for_health() {
  local name="$1" url="$2" retries="${3:-60}"
  info "Waiting for $name ($url) ..."
  for _ in $(seq 1 "$retries"); do
    curl -sf "$url" >/dev/null 2>&1 && { ok "$name is up"; return 0; }
    sleep 1
  done
  err "$name did not start in ${retries}s — check logs/"; return 1
}

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  QA Agent System — Startup                     ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

ERRORS=0

# ── 1. Neo4j ─────────────────────────────────────────────────────────────────────
echo -e "${CYAN}[1/4] Neo4j${NC}"
if [[ "$DO_NEO4J" == true ]]; then
  if curl -sf http://localhost:7474 >/dev/null 2>&1; then
    ok "Neo4j already running (http 7474 / bolt 7687)"
  elif [[ -x "$NEO4J_DBMS_DIR/bin/neo4j" ]]; then
    info "Starting Neo4j DBMS at $NEO4J_DBMS_DIR"
    JAVA_HOME="$NEO4J_JAVA_HOME" "$NEO4J_DBMS_DIR/bin/neo4j" start >/dev/null 2>&1 || true
    wait_for_health "Neo4j" "http://localhost:7474" 60 || ERRORS=$((ERRORS+1))
  else
    warn "Neo4j binary not found at \$NEO4J_DBMS_DIR — start Neo4j Desktop manually,"
    warn "  or set NEO4J_DBMS_DIR in .env. (Skipping.)"
  fi
else
  info "Skipping Neo4j (--no-neo4j)"
fi
echo ""

# ── 2. Android emulator ──────────────────────────────────────────────────────────
echo -e "${CYAN}[2/4] Android emulator${NC}"
if [[ "$DO_EMULATOR" == true ]]; then
  if ! command -v adb >/dev/null 2>&1; then
    warn "adb not found — skipping emulator (brew install android-platform-tools)"
  elif [[ "$(adb devices 2>/dev/null | grep -c 'device$')" -gt 0 ]]; then
    ok "An Android device/emulator is already connected"
  elif [[ -x "$ANDROID_EMULATOR" ]]; then
    info "Launching emulator '$EMULATOR_AVD' (boot can take a minute) ..."
    nohup "$ANDROID_EMULATOR" -avd "$EMULATOR_AVD" > "$DIR/logs/emulator.log" 2>&1 &
    for _ in $(seq 1 90); do
      [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]] && break
      sleep 2
    done
    if [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; then
      ok "Emulator booted"
    else
      warn "Emulator did not report boot within ~3min — check logs/emulator.log"
    fi
  else
    warn "emulator binary not found at \$ANDROID_EMULATOR — set ANDROID_EMULATOR/EMULATOR_AVD in .env"
  fi
  # Enable the mobilerun Portal accessibility service (needed for device control).
  if [[ "$(adb devices 2>/dev/null | grep -c 'device$')" -gt 0 ]]; then
    adb shell settings put secure enabled_accessibility_services "$PORTAL_A11Y" >/dev/null 2>&1 || true
    adb shell settings put secure accessibility_enabled 1 >/dev/null 2>&1 || true
    info "mobilerun accessibility service enabled"
  fi
else
  info "Skipping emulator (--no-emulator)"
fi
echo ""

# ── 3. (optional) build dashboard ────────────────────────────────────────────────
if [[ "$DO_BUILD" == true ]]; then
  echo -e "${CYAN}[--] Building React dashboard${NC}"
  if command -v npm >/dev/null 2>&1; then
    ( cd "$DIR/dashboard-react" && npm install --silent && npm run build ) && ok "Dashboard built" \
      || warn "Dashboard build failed — gateway will serve the fallback dashboard"
  else
    warn "npm not found — skipping dashboard build"
  fi
  echo ""
fi

# ── 4. Services (RAG API + Gateway) ──────────────────────────────────────────────
if [[ "${MODEL_BACKEND:-openrouter}" == "ngrok" && -z "${MODEL_API_URL:-}" ]]; then
  err "MODEL_BACKEND=ngrok but MODEL_API_URL is not set (add it to .env)."; exit 1
fi

echo -e "${CYAN}[3/4] RAG API (:$RAG_PORT)${NC}"
if lsof -ti :"$RAG_PORT" >/dev/null 2>&1; then
  warn "Port $RAG_PORT already in use — reusing existing process"
else
  "$PYTHON_BIN" -m uvicorn rag_api.main:app --host 0.0.0.0 --port "$RAG_PORT" > "$DIR/logs/rag_api.log" 2>&1 &
  echo $! >> "$PID_FILE"; info "Started RAG API (PID $!)"
fi
wait_for_health "RAG API" "http://127.0.0.1:$RAG_PORT/health" || ERRORS=$((ERRORS+1))
echo ""

echo -e "${CYAN}[4/4] Agent Gateway (:$GATEWAY_PORT)${NC}"
if lsof -ti :"$GATEWAY_PORT" >/dev/null 2>&1; then
  warn "Port $GATEWAY_PORT already in use — reusing existing process"
else
  RAG_API_URL="http://127.0.0.1:$RAG_PORT" \
  "$PYTHON_BIN" -m uvicorn gateway.main:app --host 0.0.0.0 --port "$GATEWAY_PORT" > "$DIR/logs/gateway.log" 2>&1 &
  echo $! >> "$PID_FILE"; info "Started Gateway (PID $!)"
fi
wait_for_health "Gateway" "http://127.0.0.1:$GATEWAY_PORT/health" || ERRORS=$((ERRORS+1))
echo ""

if [[ $ERRORS -gt 0 ]]; then
  err "$ERRORS component(s) failed to start — check logs/"; exit 1
fi

# ── (optional) ingest ─────────────────────────────────────────────────────────────
if [[ "$DO_INGEST" == true ]]; then
  echo -e "${CYAN}[+] Ingesting SRS + Figma (reset → SRS → Figma)${NC}"
  warn "This RESETS the project graph (wipes tests + app model)."
  if [[ -f "$SRS_PATH" && -f "$FIGMA_PATH" ]]; then
    GATEWAY_URL="http://127.0.0.1:$GATEWAY_PORT" RAG_URL="http://127.0.0.1:$RAG_PORT" \
    PROJECT="$PROJECT" SRS_PATH="$SRS_PATH" FIGMA_PATH="$FIGMA_PATH" \
      "$PYTHON_BIN" scripts/ingest_all.py && ok "Graph ingested" || err "Ingest failed"
  else
    warn "SRS/Figma file missing — skipping ingest"
  fi
  echo ""
fi

# ── (optional) executor ────────────────────────────────────────────────────────────
if [[ "$DO_EXECUTOR" == true ]]; then
  echo -e "${CYAN}[+] Droidrun Executor${NC}"
  if [[ "$(adb devices 2>/dev/null | grep -c 'device$')" -gt 0 ]]; then
    "$PYTHON_BIN" clients/executor_runner.py > "$DIR/logs/simulation_result.txt" 2>&1 &
    echo $! >> "$PID_FILE"
    ok "Executor started (PID $!) → tail -f logs/simulation_result.txt"
  else
    warn "No ADB device — cannot start executor"
  fi
  echo ""
fi

# ── Summary ─────────────────────────────────────────────────────────────────────
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
ok "Stack is up."
echo ""
echo "  Neo4j      → http://localhost:7474   (bolt :7687)"
echo "  RAG API    → http://127.0.0.1:$RAG_PORT"
echo "  Gateway    → http://127.0.0.1:$GATEWAY_PORT"
echo "  Dashboard  → http://127.0.0.1:$GATEWAY_PORT/dashboard?project=$PROJECT"
echo ""
echo "  Run a test loop:  $PYTHON_BIN clients/executor_runner.py"
echo "  Stop everything:  ./stop.sh"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
