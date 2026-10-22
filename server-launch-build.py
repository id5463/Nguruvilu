# -*- coding: utf-8 -*-
"""Start the -j1 nice build (or report it already runs); one quick exec."""
from cloud_ssh import connect, run

ssh = connect(attempts=3, backoff=10)
state = run(
    ssh,
    "pgrep -x cargo || echo NO_CARGO; ls -la /home/ubuntu/ngu-build.log 2>/dev/null || echo NO_LOG",
    timeout=30,
)
print("state:", state.strip())

if "NO_CARGO" in state:
    launch = (
        "timeout 15 bash -c 'cd /home/ubuntu/Nguruvilu && . $HOME/.cargo/env && "
        "nohup nice -n 19 cargo build --release --manifest-path desktop/Cargo.toml -j 1 "
        ">> /home/ubuntu/ngu-build.log 2>&1 < /dev/null & disown'; "
        "sleep 3; pgrep -x cargo || echo START_FAIL"
    )
    out = run(ssh, launch, timeout=45)
    print("launch:", out.strip())
ssh.close()
