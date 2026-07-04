#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$HOME/ai_robotics_assistant"
APP_DIR="$PROJECT_ROOT/voice_assistant_website"
VENV="$PROJECT_ROOT/env"
PORT="8000"
URL="http://127.0.0.1:${PORT}"
PID_FILE="$APP_DIR/server.pid"
SERVER_LOG="$APP_DIR/server.log"
LAUNCHER_LOG="$APP_DIR/launcher.log"
BROWSER_PROFILE_DIR="$APP_DIR/.browser_profile"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LAUNCHER_LOG"
}

notify_user() {
    local message="$1"
    log "$message"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "Local Multi-Domain Assistant" "$message" >/dev/null 2>&1 || true
    fi
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

pid_owns_port() {
    local wanted_pid="${1:-}"
    local pid
    [[ "$wanted_pid" =~ ^[0-9]+$ ]] || return 1
    while IFS= read -r pid; do
        [[ "$pid" == "$wanted_pid" ]] && return 0
    done < <(port_pids)
    return 1
}

kill_port_processes() {
    local pids=()
    local pid
    while IFS= read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid")
    done < <(port_pids)

    if [[ "${#pids[@]}" -eq 0 ]]; then
        return
    fi

    log "Stopping old process(es) on port ${PORT}: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
    sleep 1

    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log "Force stopping old process ${pid}"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

read_existing_pid() {
    if [[ -f "$PID_FILE" ]]; then
        sed -n '1p' "$PID_FILE" 2>/dev/null | tr -cd '0-9'
    fi
}

ensure_paths() {
    [[ -d "$PROJECT_ROOT" ]] || {
        notify_user "Project root not found: $PROJECT_ROOT"
        exit 1
    }
    [[ -f "$VENV/bin/activate" ]] || {
        notify_user "Virtual environment not found: $VENV"
        exit 1
    }
    [[ -f "$APP_DIR/app.py" ]] || {
        notify_user "Website app not found: $APP_DIR/app.py"
        exit 1
    }
}

start_backend_if_needed() {
    local existing_pid
    existing_pid="$(read_existing_pid || true)"

    if pid_is_running "$existing_pid" && pid_owns_port "$existing_pid"; then
        log "Server already running with PID $existing_pid; not starting a duplicate."
        return
    fi

    rm -f "$PID_FILE"
    kill_port_processes

    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    cd "$APP_DIR"

    log "Starting backend on $URL"
    nohup python "$APP_DIR/app.py" > "$SERVER_LOG" 2>&1 &
    echo "$!" > "$PID_FILE"
}

detect_browser() {
    local candidate
    for candidate in google-chrome chromium-browser chromium microsoft-edge; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

open_browser_and_wait() {
    local browser
    browser="$(detect_browser || true)"

    if [[ -n "$browser" ]]; then
        mkdir -p "$BROWSER_PROFILE_DIR"
        log "Opening app window with $browser"
        "$browser" \
            --user-data-dir="$BROWSER_PROFILE_DIR" \
            --no-first-run \
            --disable-default-apps \
            --app="$URL" \
            >/dev/null 2>&1 &
        local browser_pid=$!
        wait "$browser_pid" 2>/dev/null || true
        log "Browser app window closed; stopping backend."
        "$APP_DIR/stop_voice_assistant.sh"
        return
    fi

    if command -v xdg-open >/dev/null 2>&1; then
        log "No Chrome/Chromium/Edge app browser found; opening with xdg-open."
        xdg-open "$URL" >/dev/null 2>&1 || true
        notify_user "Opened in your default browser. Backend will keep running until you run stop_voice_assistant.sh."
    else
        notify_user "Open $URL manually. Backend will keep running until you run stop_voice_assistant.sh."
    fi
}

main() {
    ensure_paths
    start_backend_if_needed
    sleep 2
    open_browser_and_wait
}

main "$@"
