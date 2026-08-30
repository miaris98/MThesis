# Cloudflare Tunnel & MLflow Remote Access Troubleshooting Guide

This guide provides a comprehensive troubleshooting reference for managing, diagnosing, and repairing **Cloudflare Quick Tunnels** (`trycloudflare.com`) and remote **MLflow UI access** on Vast.ai and remote GPU training servers.

---

## 1. How Cloudflare Quick Tunnels Work

When you launch `run_multi_carla_training.sh` or `start_dashboard.sh`:
1. The **MLflow UI server** runs locally on port `10100` inside a persistent `tmux` session named `mlflow_server`.
2. A **Cloudflare Quick Tunnel** (`cloudflared tunnel --url http://127.0.0.1:10100`) runs inside a persistent `tmux` session named `mlflow_tunnel`.
3. Cloudflare registers an ephemeral, randomly generated public HTTPS URL (e.g., `https://xxxx-yyyy-zzzz.trycloudflare.com`) and saves it to `/tmp/mlflow_cf_url` and `/tmp/mlflow_tunnel.log`.

---

## 2. Common Errors & Exact Solutions

### Issue A: Error 1033 (Cloudflare Tunnel Error / Host Not Found)

#### 🔴 Symptom
```text
Error 1033 — Cloudflare Tunnel error
The host (xxxx-yyyy.trycloudflare.com) is configured as a Cloudflare Tunnel, and Cloudflare is currently unable to resolve it.
```

#### 🔍 Root Cause
1. **The tunnel process died:** The `mlflow_tunnel` tmux session crashed or was terminated.
2. **The URL is outdated:** The training script restarted and generated a **new random subdomain**, but you are opening the **old link**.
3. **The script was paused/stopped:** You pressed `Ctrl + Z` or `Ctrl + C` on the supervisor script, which broke the tunnel connection.

#### ✅ Fix
Check if the local MLflow server is still alive, restart the tunnel, and grab the fresh URL:

```bash
# 1. Verify MLflow is responding locally
curl -I http://127.0.0.1:10100

# 2. Restart the Cloudflare tunnel in background tmux
pkill -9 -f "cloudflared tunnel" 2>/dev/null || true
tmux kill-session -t mlflow_tunnel 2>/dev/null || true

tmux new-session -d -s mlflow_tunnel \
    "cloudflared tunnel --url http://127.0.0.1:10100 --no-autoupdate > /tmp/mlflow_tunnel.log 2>&1"

# 3. Wait 4 seconds and print new URL
sleep 4
NEW_URL=$(grep -oE 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' /tmp/mlflow_tunnel.log | tail -n 1)
echo "$NEW_URL" > /tmp/mlflow_cf_url
echo -e "\n👉 NEW LIVE MLFLOW URL: $NEW_URL\n"
```

---

### Issue B: DNS Error (`DNS_PROBE_POSSIBLE` / `NXDOMAIN`)

#### 🔴 Symptom
```text
This site can’t be reached
xxxx-yyyy.trycloudflare.com’s DNS address could not be found.
DNS_PROBE_POSSIBLE  or  DNS_PROBE_FINISHED_NXDOMAIN
```

#### 🔍 Root Cause
Cloudflare Quick Tunnels generate a brand-new domain name that takes **10 to 30 seconds to propagate** across global DNS nameservers (Google DNS `8.8.8.8`, Cloudflare `1.1.1.1`, and your local ISP). If you click the link the exact second it appears in the console, your browser gets a temporary cache of the non-existent record.

#### ✅ Fix
1. **Wait 15–20 seconds.**
2. **Perform a Hard Refresh** in your browser to bypass cached DNS lookup failures:
   * **Windows / Linux:** `Ctrl + Shift + R` or `Ctrl + F5`
   * **Mac:** `Cmd + Shift + R`
3. **Verify edge connectivity from the server:**
   ```bash
   # Test if Cloudflare edge has registered the tunnel
   curl -s -I "$(cat /tmp/mlflow_cf_url 2>/dev/null)" | head -n 5
   ```
   If it returns `HTTP/2 200` or `HTTP/1.1 200`, the domain is live and accessible.

---

### Issue C: HTTP 429 (Too Many Requests / Rate Limited)

#### 🔴 Symptom
Inside `/tmp/mlflow_tunnel.log`:
```text
ERR Request failed error="HTTP 429 Too Many Requests"
```

#### 🔍 Root Cause
Creating too many new Cloudflare Quick Tunnels in rapid succession causes Cloudflare to temporarily throttle tunnel creation for your IP address for 1–2 minutes.

#### ✅ Fix
Reuse the existing active tunnel rather than killing and recreating it:
```bash
# Check if an existing tunnel is already running
if tmux has-session -t mlflow_tunnel 2>/dev/null; then
    cat /tmp/mlflow_cf_url 2>/dev/null || grep -oE 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' /tmp/mlflow_tunnel.log | tail -n 1
else
    echo "Wait 30 seconds before attempting a new tunnel..."
    sleep 30
    tmux new-session -d -s mlflow_tunnel "cloudflared tunnel --url http://127.0.0.1:10100 > /tmp/mlflow_tunnel.log 2>&1"
fi
```

---

### Issue D: 502 Bad Gateway / Connection Refused

#### 🔴 Symptom
Cloudflare page loads, but displays:
```text
502 Bad Gateway
Host Error / Origin server connection failed
```

#### 🔍 Root Cause
Cloudflare is online, but the local MLflow server on port `10100` stopped, crashed, or was killed.

#### ✅ Fix
Restart the MLflow UI server:
```bash
tmux kill-session -t mlflow_server 2>/dev/null || true
fuser -k 10100/tcp 2>/dev/null || true
sleep 1

# Launch MLflow UI in background tmux
tmux new-session -d -s mlflow_server \
    "python -m mlflow ui --host 0.0.0.0 --port 10100 --backend-store-uri /workspace/MThesis/mlruns > /workspace/mlflow_server.log 2>&1"

# Verify local health
sleep 3
curl -I http://127.0.0.1:10100
```

---

## 3. All-in-One Quick Diagnostic Script

Run this single snippet anytime to inspect the entire MLflow + Cloudflare stack:

```bash
echo "=== 1. Active Tmux Sessions ==="
tmux ls 2>/dev/null || echo "No tmux sessions active!"

echo -e "\n=== 2. Local MLflow Server Check (Port 10100) ==="
curl -s -I http://127.0.0.1:10100 | head -n 3 || echo "MLflow is NOT responding on port 10100!"

echo -e "\n=== 3. Cloudflare Tunnel Log Status ==="
cat /tmp/mlflow_tunnel.log 2>/dev/null | grep -E "trycloudflare|Registered tunnel connection|ERR|WRN" | tail -n 5 || echo "No tunnel log found"

echo -e "\n=== 4. Active Public URL ==="
ACTIVE_URL=$(cat /tmp/mlflow_cf_url 2>/dev/null || grep -oE 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' /tmp/mlflow_tunnel.log 2>/dev/null | tail -n 1)
echo "URL: ${ACTIVE_URL:-None found}"

if [ -n "$ACTIVE_URL" ]; then
    echo -e "\n=== 5. Public URL Response Check ==="
    curl -s -I "$ACTIVE_URL" | head -n 3
fi
```

---

## 4. Alternative Direct Access (No Cloudflare Needed)

If Cloudflare is ever blocked, down, or rate-limited, you have **two zero-dependency fallback options**:

### Option A: SSH Local Port Forwarding (Recommended & 100% Reliable)
From your **local machine's terminal / PowerShell**, connect to Vast.ai with the `-L 10100:127.0.0.1:10100` flag:

```powershell
ssh -p <VAST_SSH_PORT> root@<VAST_SSH_IP> -L 10100:127.0.0.1:10100
```

Then open your local browser to:
👉 **`http://localhost:10100`**

* **Benefits:** Zero DNS latency, encrypted connection, completely immune to Cloudflare rate-limits.

---

### Option B: Vast.ai Direct Port Mapping
1. On the **Vast.ai Console**, open your instance details and check the **Port Forwarding / Mapped Ports** table.
2. Find the external port mapped to internal container port **`10100`** (or add a mapping for `10100`).
3. Open `http://<VAST_PUBLIC_IP>:<EXTERNAL_MAPPED_PORT>` in your browser.
