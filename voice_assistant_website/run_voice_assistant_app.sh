#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$HOME/ai_robotics_assistant"
APP_DIR="$PROJECT_ROOT/voice_assistant_website"
VENV="$PROJECT_ROOT/env"
PORT=8000
URL="http://127.0.0.1:${PORT}"
PID_FILE="$APP_DIR/server.pid"
SERVER_LOG="$APP_DIR/server.log"
LAUNCHER_LOG="$APP_DIR/launcher.log"
CHROME_PROFILE="$APP_DIR/chrome_profile"

log() {
    mkdir -p "$APP_DIR" 2>/dev/null || true
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LAUNCHER_LOG"
}

notify_user() {
    local message="$1"
    log "$message"
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "Local Multi-Domain Assistant" "$message" >/dev/null 2>&1 || true
    fi
}

detect_chrome() {
    local candidate
    for candidate in google-chrome google-chrome-stable; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
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

kill_app_port_processes() {
    local pid
    while IFS= read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        if pid_belongs_to_app "$pid"; then
            log "Stopping stale app process on port ${PORT}: ${pid}"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        else
            notify_user "Port ${PORT} is already used by another app. Backend was not started."
            exit 1
        fi
    done < <(port_pids)
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
    [[ -x "$APP_DIR/stop_voice_assistant.sh" ]] || {
        notify_user "Stop script is not executable: $APP_DIR/stop_voice_assistant.sh"
        exit 1
    }
}

wait_for_server() {
    local attempt
    for attempt in $(seq 1 20); do
        if command -v curl >/dev/null 2>&1; then
            if curl -fsS "$URL/api/health" >/dev/null 2>&1 || curl -fsS "$URL" >/dev/null 2>&1; then
                log "Backend responded after ${attempt} second(s)."
                return 0
            fi
        else
            if python - "$URL/api/health" >/dev/null 2>&1 <<'PY'
import sys
from urllib.request import urlopen
urlopen(sys.argv[1], timeout=1).read(1)
PY
            then
                log "Backend responded after ${attempt} second(s)."
                return 0
            fi
        fi
        sleep 1
    done

    notify_user "Backend did not respond at $URL within 20 seconds. See $SERVER_LOG."
    "$APP_DIR/stop_voice_assistant.sh" || true
    exit 1
}

start_backend() {
    "$APP_DIR/stop_voice_assistant.sh" || true
    rm -f "$PID_FILE"
    kill_app_port_processes

    cd "$PROJECT_ROOT"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"

    log "Starting backend at $URL"
    nohup python "$APP_DIR/app.py" > "$SERVER_LOG" 2>&1 &
    echo "$!" > "$PID_FILE"
}

open_chrome_app() {
    local chrome="$1"
    mkdir -p "$CHROME_PROFILE"
    log "Opening Chrome app window with $chrome"
    "$chrome" \
        --app="$URL" \
        --user-data-dir="$CHROME_PROFILE" \
        --no-first-run \
        --disable-default-apps \
        --new-window \
        >/dev/null 2>&1 &
    local browser_pid=$!
    wait "$browser_pid" 2>/dev/null || true
    log "Chrome app window closed; stopping backend."
    "$APP_DIR/stop_voice_assistant.sh" || true
}

main() {
    local chrome
    chrome="$(detect_chrome || true)"
    if [[ -z "$chrome" ]]; then
        notify_user "Google Chrome not found. Please install Chrome."
        exit 1
    fi

    ensure_paths
    start_backend
    wait_for_server
    open_chrome_app "$chrome"
}

main "$@"
