#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$HOME/ai_robotics_assistant"
APP_DIR="$PROJECT_ROOT/voice_assistant_website"
PORT=8000
PID_FILE="$APP_DIR/server.pid"
LAUNCHER_LOG="$APP_DIR/launcher.log"

log() {
    local message="$1"
    mkdir -p "$APP_DIR" 2>/dev/null || true
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$message" >> "$LAUNCHER_LOG"
    printf '%s\n' "$message"
}

port_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
        return
    fi

    if command -v fuser >/dev/null 2>&1; then
        fuser -n tcp "$PORT" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
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

pid_is_running() {
    local pid="${1:-}"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

pid_cmdline() {
    local pid="${1:-}"
    [[ "$pid" =~ ^[0-9]+$ ]] || return
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
}

pid_belongs_to_app() {
    local pid="${1:-}"
    local cmdline
    cmdline="$(pid_cmdline "$pid")"
    [[ "$cmdline" == *"$APP_DIR/app.py"* ]]
}

child_pids() {
    local pid="${1:-}"
    [[ "$pid" =~ ^[0-9]+$ ]] || return
    if command -v pgrep >/dev/null 2>&1; then
        pgrep -P "$pid" 2>/dev/null || true
    fi
}

kill_pid_tree() {
    local pid="${1:-}"
    local child
    [[ "$pid" =~ ^[0-9]+$ ]] || return

    while IFS= read -r child; do
        [[ "$child" =~ ^[0-9]+$ ]] && kill_pid_tree "$child"
    done < <(child_pids "$pid")

    if ! pid_is_running "$pid"; then
        return
    fi

    log "Stopping backend process $pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    if pid_is_running "$pid"; then
        log "Force stopping backend process $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
}

stop_pid_file_backend() {
    local pid=""
    if [[ -f "$PID_FILE" ]]; then
        pid="$(sed -n '1p' "$PID_FILE" 2>/dev/null | tr -cd '0-9')"
    fi

    if pid_is_running "$pid" && pid_belongs_to_app "$pid"; then
        kill_pid_tree "$pid"
    elif pid_is_running "$pid"; then
        log "PID file points to non-app process $pid; leaving it untouched."
    fi
}

stop_app_port_backend() {
    local pid
    while IFS= read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if pid_belongs_to_app "$pid"; then
            log "Stopping app-owned process on port ${PORT}: $pid"
            kill_pid_tree "$pid"
        else
            log "Port ${PORT} is used by another process; leaving PID $pid untouched."
        fi
    done < <(port_pids)
}

main() {
    stop_pid_file_backend
    stop_app_port_backend
    rm -f "$PID_FILE"
    log "Local Multi-Domain Assistant backend stopped cleanly."
}

main "$@"
