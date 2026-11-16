---
name: restart-guide
description: Restart the process the agent runs in, after changing something only a fresh start reads. Use when a new binary, a changed assembly, or an edited settings file has landed and the running process still has the old one.
---

# Restarting yourself

`restart` asks the running process to exit so its supervisor starts it again.
It exists because a turn can change something only startup reads — a pack's
assembly, `settings.json`, the binary on disk — and the recovery path used to
be "ask the user to restart it", which is exactly the case where nobody is
there.

## Calling it

```
restart(reason="pack `judge` installed; needs a fresh kernel")
```

`reason` is required. It is written to the log **before** the exit, so a restart
always has a cause on disk afterwards; an unexplained restart loop is the worst
outcome this tool can produce.

## It refuses when it cannot work

The tool checks for a supervisor first. Under `systemd` with `Restart=always`,
`launchd` with `KeepAlive`, or Docker with a restart policy, exiting *is* the
restart. Started by hand in a terminal, exiting is just death — the tool says so
and does nothing rather than leaving a box with no service.

If it refuses, the honest next step is to tell the user which command restarts
it, not to try again.

## What it does not do

It does not wait for the new process. The reply to the tool call has to reach
you and the page before the process goes away — the restart is delayed a few
hundred milliseconds for exactly that — so the call returns first and the new
process is a separate matter. After a successful restart, your conversation is
still on disk; reconnecting resumes it.
