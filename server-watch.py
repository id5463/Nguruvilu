# -*- coding: utf-8 -*-
"""Watch the detached server build; when it lands, restart the web service on
the new binary and install the long-horizon pack. Each poll is one quick
connection — no held-open session to die on."""
import re
import subprocess
import sys
import time

from cloud_ssh import connect, run

BUILD_LOG = "/home/ubuntu/ngu-build.log"
REMOTE = "/home/ubuntu/Nguruvilu"
DEADLINE = time.time() + 75 * 60

while time.time() < DEADLINE:
    try:
        ssh = connect(attempts=1, backoff=1)
    except SystemExit:
        print("[watch] unreachable, retrying soon", flush=True)
        time.sleep(45)
        continue
    try:
        tail = run(
            ssh,
            f"tail -3 {BUILD_LOG} 2>/dev/null; echo ===DONE; "
            "pgrep -x cargo >/dev/null && echo CARGO_RUNNING || echo CARGO_GONE",
            timeout=25,
        )
        print(f"[watch] {tail.strip()[-200:]}", flush=True)
        gone = "CARGO_GONE" in tail
        success = "Finished" in tail or "BUILD_EXIT:0" in tail

        if gone and success:
            print("[watch] build landed; restarting service on the new binary", flush=True)
            restart = run(
                ssh,
                "old=$(pgrep -f 'release/ngu[-]desktop' | head -1); "
                "if [ -n \"$old\" ]; then kill $old; sleep 1; kill -9 $old 2>/dev/null; fi; "
                f"cd /home/ubuntu; timeout 12 bash -c 'nohup {REMOTE}/desktop/target/release/ngu-desktop "
                "--serve 0.0.0.0:8088 --token nguru-demo-2026 --ui paged "
                ">> /home/ubuntu/ngu-serve.log 2>&1 < /dev/null & disown; sleep 3; "
                "pgrep -af 'release/ngu[-]desktop'; curl -s -o /dev/null -w \"health:%{http_code}\" "
                "http://127.0.0.1:8088/ ; echo'",
                timeout=40,
            )
            print(restart, flush=True)
            if "health:200" in restart:
                print("[watch] service healthy; installing long-horizon", flush=True)
                done = subprocess.run(
                    [sys.executable, r"I:\nguruvilu\server-install-pack.py"],
                    capture_output=True, text=True, cwd=r"I:\nguruvilu", timeout=240,
                )
                print(done.stdout[-1500:], flush=True)
                if done.returncode != 0:
                    print("install stderr:", done.stderr[-400:], flush=True)
                print("WATCH_DONE", flush=True)
                sys.exit(0)
            print("[watch] restart did not produce health:200", flush=True)
        elif gone:
            print("[watch] BUILD FAILED", flush=True)
            print(run(ssh, f"tail -40 {BUILD_LOG}", timeout=25), flush=True)
            sys.exit(1)
    finally:
        ssh.close()
    time.sleep(45)

print("[watch] deadline reached without a verdict", flush=True)
sys.exit(1)
