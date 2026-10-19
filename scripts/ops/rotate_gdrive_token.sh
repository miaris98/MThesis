#!/usr/bin/env bash
# Rotate the Google Drive refresh token: new token -> ship to every box -> revoke the old one(s). RUN BY THE USER (or the scheduled task);
# Claude must not read .gdrive_rclone.conf. Prints no secrets. Log: .gdrive_rotation.log (gitignored).
#
# Order matters (no box is ever left without a valid token):
#   1. remember the old refresh token in .gdrive_pending_revoke
#   2. `rclone config reconnect` -> browser sign-in (ONE click; Google needs a person for consent, that step cannot be automated)
#   3. check the new token works (rclone lsd)
#   4. ship to every running box (push_gdrive_token_to_boxes.sh must report 0 failures)
#   5. only then revoke every token in .gdrive_pending_revoke, and check the new token still works
# If any step fails the old token stays valid and nothing is revoked; rerun later (an unreachable box is not a destroyed box).
#
# usage: scripts/ops/rotate_gdrive_token.sh [--no-ship] [--no-revoke] [--wait-min N]      (--wait-min: browser sign-in timeout, default 15)
cd "$(dirname "$0")/../.." || exit 1
LOG=.gdrive_rotation.log
SHIP=1; REVOKE=1; WAIT=15
while [ "$#" -gt 0 ]; do case "$1" in --no-ship) SHIP=0;; --no-revoke) REVOKE=0;; --wait-min) WAIT=$2; shift;; esac; shift; done
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

PY=""; for c in python3 python py; do "$c" -c 'import json' >/dev/null 2>&1 && { PY=$c; break; }; done
[ -n "$PY" ] || { log "no working python"; exit 1; }
RC=$(command -v rclone || ls /c/Users/miari/AppData/Local/Microsoft/WinGet/Packages/Rclone.Rclone_*/rclone-*/rclone.exe 2>/dev/null | head -1)
[ -n "$RC" ] || { log "rclone not found"; exit 1; }
CONF=.gdrive_rclone.conf
[ -s "$CONF" ] || { log "missing $CONF"; exit 1; }

refresh_token() { "$PY" -c "
import json, re, sys
m = re.search(r'^token\s*=\s*(\{.*\})\s*$', open(sys.argv[1], encoding='utf-8').read(), re.M)
print(json.loads(m.group(1)).get('refresh_token', '') if m else '')" "$1" | tr -d '\r'; }
works() { "$RC" --config "$CONF" lsd gdrive:MThesis_backups >/dev/null 2>&1; }

OLD=$(refresh_token "$CONF")
[ -n "$OLD" ] || { log "no refresh token in $CONF"; exit 1; }
works || { log "current token does not work before rotating (already revoked?) - reconnecting anyway"; }
cp "$CONF" .gdrive_rclone.conf.prev; chmod 600 .gdrive_rclone.conf.prev 2>/dev/null
touch .gdrive_pending_revoke; grep -qxF "$OLD" .gdrive_pending_revoke || echo "$OLD" >> .gdrive_pending_revoke

log "step 2: browser sign-in (waiting up to ${WAIT} min)"
if ! timeout "$((WAIT*60))" "$RC" config reconnect gdrive: --config "$CONF" --auto-confirm >/dev/null 2>&1; then
  log "reconnect failed or timed out - restoring the old config, nothing revoked"; cp .gdrive_rclone.conf.prev "$CONF"; exit 1
fi
NEW=$(refresh_token "$CONF")
if [ -z "$NEW" ] || [ "$NEW" = "$OLD" ]; then log "no NEW refresh token was issued (Google kept the old one) - nothing to revoke; keeping the config"; REVOKE=0; fi
works || { log "new token does not work - restoring the old config"; cp .gdrive_rclone.conf.prev "$CONF"; exit 1; }
log "step 3 ok: new token works"

if [ "$SHIP" = 1 ]; then
  if ! bash scripts/ops/push_gdrive_token_to_boxes.sh >>"$LOG" 2>&1; then
    log "step 4 FAILED: not every box took the new token - old token(s) NOT revoked, see $LOG; rerun when the boxes answer"; exit 1
  fi
  log "step 4 ok: every running box has the new token"
fi

if [ "$REVOKE" = 1 ] && [ "$SHIP" = 1 ]; then
  : > .gdrive_pending_revoke.left
  while read -r T; do
    [ -n "$T" ] && [ "$T" != "$NEW" ] || continue
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -d "token=$T" https://oauth2.googleapis.com/revoke)
    # 200 = revoked; 400 = already invalid/revoked (nothing left to do)
    case "$CODE" in 200|400) ;; *) echo "$T" >> .gdrive_pending_revoke.left;; esac
  done < .gdrive_pending_revoke
  mv .gdrive_pending_revoke.left .gdrive_pending_revoke
  if works; then log "step 5 ok: old token(s) revoked, new token still works (pending revokes left: $(grep -c . .gdrive_pending_revoke))"
  else log "step 5 WARNING: the new token stopped working after the revoke - Google revoked the whole grant. Run rotate_gdrive_token.sh --no-revoke to sign in again and re-ship, then see docs/ops/google_drive_sync.md"; exit 1; fi
fi
log "rotation done"
