#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# stop.sh — Tear down the QA Agent stack (services + emulator + Neo4j).
#
# Usage:
#   ./stop.sh                  # stop services + emulator + Neo4j
#   ./stop.sh --services-only  # stop only RAG API / Gateway / Executor
#   ./stop.sh --keep-emulator  # leave the Android emulator running
#   ./stop.sh --keep-neo4j     # leave Neo4j running
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
[[ -f "$DIR/.env" ]] && { set -a; source "$DIR/.env"; set +a; }

PID_FILE="$DIR/logs/services.pid"
NEO4J_DBMS_DIR="${NEO4J_DBMS_DIR:-$HOME/Library/Application Support/neo4j-desktop/Application/Data/dbmss/dbms-15751e0d-b9e8-437e-8199-e0fb4c954865}"
NEO4J_JAVA_HOME="${NEO4J_JAVA_HOME:-$(ls -d "$HOME/Library/Application Support/neo4j-desktop/Application/Cache/runtime/"zulu*jre* 2>/dev/null | head -1)}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
info() { echo -e "${CYAN}[..]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }

STOP_EMULATOR=true; STOP_NEO4J=true
for arg in "$@"; do
  case "$arg" in
    --services-only) STOP_EMULATOR=false; STOP_NEO4J=false ;;
    --keep-emulator) STOP_EMULATOR=false ;;
    --keep-neo4j)    STOP_NEO4J=false ;;
  esac
done

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  QA Agent System — Shutdown                    ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo ""

# ── 1. Services (RAG API, Gateway, Executor) ──────────────────────────────────────
info "Stopping services ..."
if [[ -f "$PID_FILE" ]]; then
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null && echo "  killed PID $pid"
  done < "$PID_FILE"
  rm -f "$PID_FILE"
fi
# Fallback: kill by process signature in case the PID file was stale.
pkill -f "uvicorn rag_api.main:app"  2>/dev/null && echo "  killed rag_api"  || true
pkill -f "uvicorn gateway.main:app"  2>/dev/null && echo "  killed gateway"  || true
pkill -f "clients/executor_runner.py" 2>/dev/null && echo "  killed executor" || true
ok "Services stopped"
echo ""

# ── 2. Android emulator ──────────────────────────────────────────────────────────
if [[ "$STOP_EMULATOR" == true ]]; then
  info "Stopping Android emulator ..."
  if command -v adb >/dev/null 2>&1; then
    # Kill every connected emulator instance.
    for serial in $(adb devices 2>/dev/null | grep 'device$' | grep '^emulator-' | cut -f1); do
      adb -s "$serial" emu kill 2>/dev/null && echo "  killed $serial" || true
    done
    ok "Emulator stopped"
  else
    warn "adb not found — cannot stop emulator"
  fi
  echo ""
else
  info "Leaving emulator running (--keep-emulator / --services-only)"
  echo ""
fi

# ── 3. Neo4j ─────────────────────────────────────────────────────────────────────
if [[ "$STOP_NEO4J" == true ]]; then
  info "Stopping Neo4j ..."
  if [[ -x "$NEO4J_DBMS_DIR/bin/neo4j" ]]; then
    JAVA_HOME="$NEO4J_JAVA_HOME" "$NEO4J_DBMS_DIR/bin/neo4j" stop >/dev/null 2>&1 \
      && ok "Neo4j stopped" || warn "Neo4j stop returned non-zero (may already be down)"
  else
    warn "Neo4j binary not found at \$NEO4J_DBMS_DIR — stop it via Neo4j Desktop"
  fi
  echo ""
else
  info "Leaving Neo4j running (--keep-neo4j / --services-only)"
  echo ""
fi

echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
ok "Shutdown complete."
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
