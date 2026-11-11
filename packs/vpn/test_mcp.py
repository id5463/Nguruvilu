"""Drive the vpn MCP server and print each response.

Binary pipes on purpose: on Windows a text-mode pipe decodes with the console
codepage (cp936 here), and the node names carry emoji, which killed the parent
with `OSError: Invalid argument` before a single tool call completed.
"""
import json
import os
import subprocess
import sys
import threading
import time

env = dict(os.environ)
env.setdefault("MIHOMO_CONTROLLER", "http://127.0.0.1:19090")
env["PYTHONIOENCODING"] = "utf-8"

here = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(here, "scripts", "vpn_mcp.py")

p = subprocess.Popen(
    [sys.executable, script],
    env=env,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)


def send(obj):
    p.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    p.stdin.flush()


def reader():
    for raw in p.stdout:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("{"):
            continue
        msg = json.loads(line)
        if not msg.get("id"):
            continue
        result = msg.get("result", {})
        text = result["content"][0]["text"] if "content" in result else json.dumps(result, ensure_ascii=False)
        print("id=%s %s" % (msg["id"], text[:600].replace("\n", " ")), flush=True)


def errors():
    for raw in p.stderr:
        line = raw.decode("utf-8", "replace").strip()
        if line:
            print("[stderr]", line[:200], flush=True)


threading.Thread(target=reader, daemon=True).start()
threading.Thread(target=errors, daemon=True).start()

send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                 "clientInfo": {"name": "probe", "version": "1"}}})
time.sleep(1)
send({"jsonrpc": "2.0", "method": "notifications/initialized"})

calls = [
    ("vpn_status", {}),
    ("vpn_groups", {}),
    ("vpn_nodes", {}),
    ("vpn_nodes", {"probe": True}),
]
for index, (name, args) in enumerate(calls, start=2):
    send({"jsonrpc": "2.0", "id": index, "method": "tools/call",
          "params": {"name": name, "arguments": args}})
    time.sleep(6 if args.get("probe") else 2)

time.sleep(35)
p.terminate()
