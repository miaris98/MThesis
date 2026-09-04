#!/usr/bin/env bash
# ==============================================================================
# Standalone Cloudflare HTTPS tunnel launcher for an MLflow UI server.
#
# Extracted from run_multi_carla_training.sh so any trainer (train_wor.py
# included) can get a public MLflow link without going through the online-RL
# launcher. Reuses an existing tunnel (tmux session or cached URL) when one is
# already up, otherwise launches cloudflared and retries through 429 rate
# limits, writing the URL to /tmp/mlflow_tunnel.log and /tmp/mlflow_cf_url -
# the same files ExperimentLogger already reads to print the link.
#
# Usage: bash start_mlflow_tunnel.sh [MLFLOW_PORT]   (default 10100)
# ==============================================================================
set -uo pipefail

MLFLOW_PORT="${1:-10100}"
PYTHON_BIN=$(command -v python3.8 2>/dev/null || command -v python 2>/dev/null || echo "python")

echo "--> Waiting for MLflow UI server on port ${MLFLOW_PORT}..."
for i in $(seq 1 30); do
    if curl -s -I "http://127.0.0.1:${MLFLOW_PORT}" 2>/dev/null | grep -q -E "HTTP/|200|302|mlflow"; then
        break
    fi
    sleep 1
done
echo "✓ MLflow UI server active on port ${MLFLOW_PORT}"

is_valid_cloudflared() {
    local bin="$1"
    [ -n "$bin" ] && [ -x "$bin" ] && "$bin" --version &>/dev/null
}

extract_cf_url() {
    local logfile="$1"
    [ -f "$logfile" ] || return 1
    local url=""
    url=$("$PYTHON_BIN" -c "
import re
try:
    with open('$logfile', 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    m = re.search(r'https://[-a-zA-Z0-9]+\.trycloudflare\.com', txt)
    if m:
        print(m.group(0))
except Exception:
    pass
" 2>/dev/null || true)
    if [ -z "$url" ]; then
        url=$(grep -oE 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' "$logfile" 2>/dev/null | head -1 || true)
    fi
    [ -n "$url" ] && echo "$url" && return 0
    return 1
}

CLOUDFLARED_BIN=""
for candidate in "$(command -v cloudflared 2>/dev/null)" "/usr/local/bin/cloudflared" "/usr/bin/cloudflared"; do
    if is_valid_cloudflared "$candidate"; then
        CLOUDFLARED_BIN="$candidate"
        break
    fi
done

if [ -z "$CLOUDFLARED_BIN" ]; then
    echo "--> Installing cloudflared binary for public HTTPS dashboard access..."
    rm -f /usr/local/bin/cloudflared /tmp/cloudflared*
    if curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o /usr/local/bin/cloudflared 2>/dev/null || \
       wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -O /usr/local/bin/cloudflared 2>/dev/null; then
        chmod +x /usr/local/bin/cloudflared 2>/dev/null || true
    fi
    if is_valid_cloudflared "/usr/local/bin/cloudflared"; then
        CLOUDFLARED_BIN="/usr/local/bin/cloudflared"
    else
        if wget -q "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb" -O /tmp/cloudflared.deb 2>/dev/null; then
            dpkg -i /tmp/cloudflared.deb >/dev/null 2>&1 || true
            rm -f /tmp/cloudflared.deb
        fi
        for candidate in "$(command -v cloudflared 2>/dev/null)" "/usr/local/bin/cloudflared"; do
            if is_valid_cloudflared "$candidate"; then
                CLOUDFLARED_BIN="$candidate"
                break
            fi
        done
    fi
fi

CLOUDFLARE_URL=""

if tmux has-session -t mlflow_tunnel 2>/dev/null && [ -f /tmp/mlflow_tunnel.log ]; then
    CLOUDFLARE_URL=$(extract_cf_url /tmp/mlflow_tunnel.log) || true
    if [ -n "$CLOUDFLARE_URL" ]; then
        echo "✓ Reusing active Cloudflare HTTPS tunnel from tmux session 'mlflow_tunnel': $CLOUDFLARE_URL"
    fi
fi

if [ -z "$CLOUDFLARE_URL" ] && [ -f /tmp/mlflow_cf_url ]; then
    CACHED_URL=$(cat /tmp/mlflow_cf_url 2>/dev/null)
    if [ -n "$CACHED_URL" ] && curl -s -k --max-time 3 -I "$CACHED_URL" >/dev/null 2>&1; then
        CLOUDFLARE_URL="$CACHED_URL"
        echo "✓ Reusing cached Cloudflare tunnel: $CLOUDFLARE_URL"
    fi
fi

if [ -z "$CLOUDFLARE_URL" ] && [ -n "$CLOUDFLARED_BIN" ]; then
    tmux kill-session -t mlflow_tunnel 2>/dev/null || true
    pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
    sleep 1

    MAX_RETRIES=3
    RETRY_DELAYS=(5 15 30)

    for attempt_num in $(seq 1 $MAX_RETRIES); do
        rm -f /tmp/mlflow_tunnel.log
        echo "--> 🌐 Launching Cloudflare HTTPS tunnel in protected tmux 'mlflow_tunnel' (port ${MLFLOW_PORT}) [attempt ${attempt_num}/${MAX_RETRIES}]..."
        tmux new-session -d -s mlflow_tunnel \
            "$CLOUDFLARED_BIN tunnel --url http://127.0.0.1:${MLFLOW_PORT} --no-autoupdate > /tmp/mlflow_tunnel.log 2>&1"

        for i in $(seq 1 15); do
            if [ -f /tmp/mlflow_tunnel.log ]; then
                CLOUDFLARE_URL=$(extract_cf_url /tmp/mlflow_tunnel.log) || true
                [ -n "$CLOUDFLARE_URL" ] && break 2
            fi
            if ! tmux has-session -t mlflow_tunnel 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if [ -f /tmp/mlflow_tunnel.log ] && grep -q "429\|Too Many Requests\|rate" /tmp/mlflow_tunnel.log 2>/dev/null; then
            DELAY=${RETRY_DELAYS[$((attempt_num - 1))]}
            echo "--> ⚠️  Cloudflare rate-limited (429). Waiting ${DELAY}s before retry..."
            tmux kill-session -t mlflow_tunnel 2>/dev/null || true
            pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
            sleep "$DELAY"
        elif [ -f /tmp/mlflow_tunnel.log ] && ! tmux has-session -t mlflow_tunnel 2>/dev/null; then
            echo "--> [Note] cloudflared exited unexpectedly:"
            head -n 3 /tmp/mlflow_tunnel.log 2>/dev/null | sed 's/^/    /' || true
            tmux kill-session -t mlflow_tunnel 2>/dev/null || true
            pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
            if [ "$attempt_num" -lt "$MAX_RETRIES" ]; then
                DELAY=${RETRY_DELAYS[$((attempt_num - 1))]}
                echo "--> Retrying in ${DELAY}s..."
                sleep "$DELAY"
            fi
        else
            tmux kill-session -t mlflow_tunnel 2>/dev/null || true
            pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
            if [ "$attempt_num" -lt "$MAX_RETRIES" ]; then
                echo "--> Timed out. Retrying..."
                sleep 2
            fi
        fi
    done
fi

if [ -n "$CLOUDFLARE_URL" ]; then
    echo "$CLOUDFLARE_URL" > /tmp/mlflow_cf_url
    echo "--> Verifying Cloudflare public DNS propagation..."
    for i in $(seq 1 10); do
        if curl -s -k -m 3 -I "$CLOUDFLARE_URL" 2>/dev/null | grep -q -E "HTTP/|200|302|301|404|403|502|503"; then
            break
        fi
        sleep 1
    done
fi

echo "=============================================================="
echo "   📊 MLFLOW DASHBOARD ONLINE (PORT ${MLFLOW_PORT})           "
if [ -n "$CLOUDFLARE_URL" ]; then
    echo -e "   👉 \033[1;32mlink to mlflow :     $CLOUDFLARE_URL\033[0m"
else
    echo "   👉 Vast.ai Tunnel:    Open Port ${MLFLOW_PORT} in Vast.ai Tunnels UI"
fi
echo "=============================================================="
