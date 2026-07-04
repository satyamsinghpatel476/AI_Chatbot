#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${HOME}/ai_robotics_assistant"
VENV_DIR="${PROJECT_ROOT}/env"
APP_DIR="${PROJECT_ROOT}/voice_assistant_website"
APP_FILE="${APP_DIR}/app.py"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

SERVER_PID=""
BROWSER_OPENER_PID=""

log() {
    printf '[Voice Assistant] %s\n' "$*"
}

pause_on_error() {
    local message="$1"
    printf '\n[Voice Assistant] ERROR: %s\n' "$message" >&2
    if [[ -t 0 ]]; then
        read -r -p "Press Enter to close this terminal..." _ || true
    fi
    exit 1
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM HUP

    if [[ -n "${BROWSER_OPENER_PID}" ]] && kill -0 "${BROWSER_OPENER_PID}" 2>/dev/null; then
        kill "${BROWSER_OPENER_PID}" 2>/dev/null || true
    fi

    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping server on port ${PORT}..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM HUP

port_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
        return
    fi

    if command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "${PORT}" 2>/dev/null || true
        return
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null \
            | awk -v port=":${PORT}" '$4 ~ port "$" {print $0}' \
            | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
            | sort -u \
            || true
    fi
}

kill_existing_port_processes() {
    local pids
    pids="$(port_pids | tr '\n' ' ' | xargs 2>/dev/null || true)"
    if [[ -z "${pids}" ]]; then
        return
    fi

    log "Port ${PORT} is already in use. Stopping old process(es): ${pids}"
    kill ${pids} 2>/dev/null || true
    sleep 1

    local pid
    for pid in ${pids}; do
        if kill -0 "${pid}" 2>/dev/null; then
            log "Old process ${pid} did not stop cleanly; forcing it to stop."
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done
}

open_browser_later() {
    sleep 2
    log "Opening ${URL}"
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${URL}" >/dev/null 2>&1 || true
    elif command -v sensible-browser >/dev/null 2>&1; then
        sensible-browser "${URL}" >/dev/null 2>&1 || true
    else
        log "No browser opener found. Open this URL manually: ${URL}"
    fi
}

[[ -d "${PROJECT_ROOT}" ]] || pause_on_error "Project root not found: ${PROJECT_ROOT}"
[[ -d "${VENV_DIR}" ]] || pause_on_error "Virtual environment not found: ${VENV_DIR}"
[[ -f "${VENV_DIR}/bin/activate" ]] || pause_on_error "Venv activate script missing: ${VENV_DIR}/bin/activate"
[[ -f "${APP_FILE}" ]] || pause_on_error "Website app not found: ${APP_FILE}"

log "Project root: ${PROJECT_ROOT}"
log "Activating virtual environment: ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

kill_existing_port_processes

cd "${APP_DIR}"
log "Starting website server at ${URL}"
open_browser_later &
BROWSER_OPENER_PID=$!

python "${APP_FILE}" &
SERVER_PID=$!
wait "${SERVER_PID}"
