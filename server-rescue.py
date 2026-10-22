# -*- coding: utf-8 -*-
"""Rescue the saturated server and restart the big build in the background.

The deploy's remote cargo build outlived its ssh channel (a local read timeout
kills nobody), kept chewing the small machine, and locked everyone out. This:
kills the orphans, guarantees swap headroom, makes sure the web service is up
(reusing the binary already on disk), then relaunch the build detached with
-j1 + nice so the machine stays usable while it runs.
"""
import os
import subprocess
import sys
import tarfile

from cloud_ssh import connect, read_credentials, run

ROOT = r"I:\nguruvilu"
REMOTE = "/home/ubuntu/Nguruvilu"
EXCLUDE_PREFIX = ("dist-next/", ".playwright-mcp/", "multi-test/", "relay/", "shots/", "dist/")

# 1. current git tree (now including the pack-list refresh fix)
files = subprocess.run(
    ["git", "-C", ROOT, "-c", "core.quotepath=off", "ls-files"],
    capture_output=True, text=True, encoding="utf-8"
).stdout.splitlines()
files = [f for f in files if f and not f.startswith(EXCLUDE_PREFIX)]
tmp = os.path.join(os.environ.get("TEMP", "."), "ngu-update.tar.gz")
with tarfile.open(tmp, "w:gz") as tf:
    for f in files:
        tf.add(os.path.join(ROOT, f), arcname=f)
print(f"tarball: {len(files)} files, {os.path.getsize(tmp)} bytes", flush=True)

try:
    ssh = connect(attempts=8, backoff=20)
except SystemExit:
    sys.exit("RESCUE FAILED: ssh unreachable (machine still saturated)")

pw = read_credentials()[2]
sftp = ssh.open_sftp()
sftp.put(tmp, "/tmp/ngu-update.tar.gz")
sftp.close()
print("uploaded", flush=True)

prep = f"""
pkill -9 -f 'cargo buil[d]' || true
pkill -9 -f '[r]ustc' || true
sleep 2
free -m | sed -n 1,3p
if [ -z "$(swapon --show --noheadings)" ]; then
  if [ ! -f /swap2g ]; then
    echo '{pw}' | sudo -S -p '' fallocate -l 2G /swap2g || echo FALLOC_FAIL
    echo '{pw}' | sudo -S -p '' chmod 600 /swap2g
    echo '{pw}' | sudo -S -p '' mkswap /swap2g >/dev/null && echo SWAP_MADE
  fi
  echo '{pw}' | sudo -S -p '' swapon /swap2g && echo SWAP_ON
fi
swapon --show --noheadings || echo NO_SWAP
echo ---SERVICE---
pgrep -af 'release/ngu[-]desktop' || \\
  (cd /home/ubuntu; timeout 12 bash -c 'nohup {REMOTE}/desktop/target/release/ngu-desktop --serve 0.0.0.0:8088 --token nguru-demo-2026 --ui paged >> /home/ubuntu/ngu-serve.log 2>&1 < /dev/null & disown; sleep 2; pgrep -af "release/ngu[-]desktop"')
echo ---EXTRACT---
tar xzf /tmp/ngu-update.tar.gz -C {REMOTE} && echo EXTRACT_OK
"""
out = run(ssh, prep, timeout=180)
print(out, flush=True)

# Separate exec: the launch line contains the very words pkill hunts; one
# script holding both lets the killer read its own cmdline and commit suicide.
launch = """
timeout 15 bash -c 'cd /home/ubuntu/Nguruvilu && . $HOME/.cargo/env && rm -f /home/ubuntu/ngu-build.log && nohup nice -n 19 cargo build --release --manifest-path desktop/Cargo.toml -j 1 >> /home/ubuntu/ngu-build.log 2>&1 < /dev/null & disown'
sleep 2
pgrep -x cargo || echo BUILD_NOT_STARTED
"""
out2 = run(ssh, launch, timeout=60)
print(out2)
ssh.close()
print("RESCUE_DONE" if "EXTRACT_OK" in out and out2.strip().splitlines()[-1].isdigit() else "RESCUE_INCOMPLETE")
