#!/bin/bash
# One-time root setup on cerebro: GPU wired limit 14 GiB (at every boot), the gateway as a
# LaunchDaemon running as cucostudio (starts at boot, restarts on exit), and log rotation.
#   sudo ~/srv/brain-src/deploy/cerebro/install.sh            install / refresh
#   sudo ~/srv/brain-src/deploy/cerebro/install.sh --remove   undo all of it
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
U=cucostudio; B=/Users/$U/srv
GPU=/Library/LaunchDaemons/local.brain.gpu-memory-limit.plist
GW=/Library/LaunchDaemons/local.brain.gateway.plist
ROT=/etc/newsyslog.d/local.brain.conf
for p in "$GW" "$GPU"; do launchctl bootout system "$p" 2>/dev/null || true; done
if [ "${1:-}" = "--remove" ]; then
  rm -f "$GW" "$GPU" "$ROT"; sysctl iogpu.wired_limit_mb=0; echo removed; exit 0
fi
pkill -u $U -f "$B/brain-src/.venv/bin/brain serve" 2>/dev/null && sleep 3 || true   # a hand-started gateway
mkdir -p $B/brain/logs $B/brain/cache && chown -R $U:staff $B/brain

cat > "$GPU" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>local.brain.gpu-memory-limit</string>
  <key>ProgramArguments</key><array><string>/usr/sbin/sysctl</string><string>iogpu.wired_limit_mb=14336</string></array>
  <key>RunAtLoad</key><true/>
</dict></plist>
PL
cat > "$GW" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>local.brain.gateway</string>
  <key>ProgramArguments</key><array><string>$B/brain-src/deploy/cerebro/run.sh</string></array>
  <key>UserName</key><string>$U</string>
  <key>EnvironmentVariables</key><dict>
    <key>HOME</key><string>/Users/$U</string>
    <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$B/brain/logs/gateway.log</string>
  <key>StandardErrorPath</key><string>$B/brain/logs/gateway.log</string>
</dict></plist>
PL
chown root:wheel "$GPU" "$GW"; chmod 644 "$GPU" "$GW"
echo "$B/brain/logs/gateway.log $U:staff 644 5 20480 * NJ" > "$ROT"   # keep 5 × 20 MB, bzip2
launchctl bootstrap system "$GPU"
sleep 1; sysctl iogpu.wired_limit_mb
launchctl bootstrap system "$GW"
echo "gateway: launchctl print system/local.brain.gateway | grep state; log: $B/brain/logs/gateway.log"
