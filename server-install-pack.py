# -*- coding: utf-8 -*-
"""Install the long-horizon pack on the server through the real web pipeline:
upload the archive, pick_install by path, load_pack to activate it."""
import json

from cloud_ssh import connect, run

PACK_LOCAL = r"I:\nguruvilu\packs\long-horizon\long-horizon-1.0.0.dshpack"
TOKEN = "nguru-demo-2026"

ssh = connect()

sftp = ssh.open_sftp()
sftp.put(PACK_LOCAL, "/tmp/long-horizon-1.0.0.dshpack")
sftp.close()
print("archive uploaded to /tmp")

# 1) the page's own upload door: POST /upload
out = run(
    ssh,
    "curl -s -X POST 'http://127.0.0.1:8088/upload?name=long-horizon-1.0.0.dshpack' "
    f"-H 'Authorization: Bearer {TOKEN}' --data-binary @/tmp/long-horizon-1.0.0.dshpack",
    timeout=30,
)
print("upload ->", out[:200])
try:
    path = json.loads(out)["path"]
except Exception:
    raise SystemExit("upload did not return a path")

# 2) the same verify-then-install the Install button triggers
payload = json.dumps({"cmd": "pick_install", "path": path}).replace('"', '\\"')
out = run(
    ssh,
    f"curl -s -X POST 'http://127.0.0.1:8088/cmd' -H 'Authorization: Bearer {TOKEN}' "
    f"-H 'Content-Type: application/json' -d '{payload}'",
    timeout=60,
)
print("pick_install ->", out[:200])

# 3) activate it (load_pack): plugin fibers + tools + the mission bar
cmd = json.dumps({"cmd": "load_pack", "name": "long-horizon", "version": "1.0.0"}).replace('"', '\\"')
out = run(
    ssh,
    f"curl -s -X POST 'http://127.0.0.1:8088/cmd' -H 'Authorization: Bearer {TOKEN}' "
    f"-H 'Content-Type: application/json' -d '{cmd}'",
    timeout=60,
)
print("load_pack ->", out[:200])

import time

time.sleep(2)
print("== server log tail ==")
print(run(ssh, "grep -n -E 'long-horizon|pack\\] (installed|loaded|loaded long)' /home/ubuntu/ngu-serve.log | tail -12"))
