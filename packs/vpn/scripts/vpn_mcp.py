#!/usr/bin/env python3
"""VPN / proxy node manager for a local mihomo (Clash Meta) instance.

Talks to mihomo's own REST API rather than editing its YAML: the running process
is the truth about which node is selected and how fast each one actually is, and
a file edit cannot answer either question. Nothing here writes configuration —
selecting a node is a runtime decision, and mihomo keeps it in memory.

Every tool returns JSON as text, which is what an MCP client reads.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

# Where mihomo listens. Overridable so the same script works on a machine whose
# controller sits elsewhere; the defaults match a stock `external-controller`.
CONTROLLER = os.environ.get("MIHOMO_CONTROLLER", "http://127.0.0.1:9090")
SECRET = os.environ.get("MIHOMO_SECRET", "")

# The URL a latency probe asks for. A 204 endpoint is the cheapest thing that
# proves the tunnel carries traffic end to end.
PROBE_URL = os.environ.get("MIHOMO_PROBE_URL", "http://www.gstatic.com/generate_204")

# Proxy types that are real endpoints, as opposed to groups that hold them.
GROUP_TYPES = {"Selector", "URLTest", "Fallback", "LoadBalance", "Relay"}

# mihomo's built-in outbounds and adapters. They answer to /proxies like any
# other entry, and listing them as "nodes" would tell a caller that REJECT is a
# place its traffic can go — it is the opposite.
BUILTIN = {
    "DIRECT", "REJECT", "REJECT-DROP", "PASS", "COMPATIBLE", "GLOBAL", "PASS-RULE",
}
BUILTIN_TYPES = {"Direct", "Reject", "RejectDrop", "Pass", "Compatible", "PassRule"}


def _request(path, method="GET"):
    url = CONTROLLER.rstrip("/") + path
    req = urllib.request.Request(url, method=method)
    if SECRET:
        req.add_header("Authorization", "Bearer " + SECRET)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError("mihomo %s %s returned %d: %s" % (method, path, e.code, detail))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "cannot reach mihomo at %s (%s). Is it running, and is "
            "`external-controller` set in its config?" % (CONTROLLER, e.reason)
        )


def _proxies():
    return _request("/proxies").get("proxies", {})


def _nodes(proxies):
    """Real outbound nodes: not a group, and not one of mihomo's adapters."""
    return {
        name: info
        for name, info in proxies.items()
        if info.get("type") not in GROUP_TYPES
        and info.get("type") not in BUILTIN_TYPES
        and name not in BUILTIN
    }


def _groups(proxies):
    return {name: info for name, info in proxies.items() if info.get("type") in GROUP_TYPES}


def _quote(name):
    return urllib.parse.quote(name, safe="")


# --------------------------------------------------------------------------- tools


def vpn_status(_args):
    """Which node is in force, which groups exist, and whether mihomo answers."""
    proxies = _proxies()
    groups = _groups(proxies)
    nodes = _nodes(proxies)
    out = {
        "controller": CONTROLLER,
        "reachable": True,
        "nodes": len(nodes),
        "groups": {},
    }
    for name, info in sorted(groups.items()):
        out["groups"][name] = {
            "type": info.get("type"),
            "now": info.get("now"),
            "members": len(info.get("all", [])),
        }
    # The node a user means by "what am I on" is the one the main group selected.
    main = groups.get("PROXY") or groups.get("GLOBAL")
    if main:
        out["active"] = main.get("now")
        out["active_group"] = "PROXY" if "PROXY" in groups else "GLOBAL"
    return out


def vpn_nodes(args):
    """List every node, optionally with a live latency for each.

    `probe=true` costs one request per node against the real endpoint, so it is
    opt-in; the cheap listing is what a caller wants most of the time.
    """
    proxies = _proxies()
    nodes = _nodes(proxies)
    group = args.get("group") or ("PROXY" if "PROXY" in proxies else "GLOBAL")
    current = proxies.get(group, {}).get("now")
    rows = []
    for name, info in sorted(nodes.items()):
        row = {
            "name": name,
            "type": info.get("type"),
            "current": name == current,
        }
        if info.get("udp"):
            row["udp"] = True
        rows.append(row)

    if args.get("probe"):
        timeout = int(args.get("timeout_ms") or 5000)
        for row in rows:
            try:
                detail = _request(
                    "/proxies/%s/delay?timeout=%d&url=%s"
                    % (_quote(row["name"]), timeout, urllib.parse.quote(PROBE_URL, safe=""))
                )
                row["delay_ms"] = detail.get("delay")
            except Exception as error:
                row["delay_ms"] = None
                row["error"] = str(error)[:120]
        rows.sort(key=lambda r: (r.get("delay_ms") is None, r.get("delay_ms") or 0))

    return {"group": group, "current": current, "count": len(rows), "nodes": rows}


def vpn_switch(args):
    """Point a group at a node. This is the "connect me to X" action."""
    name = (args.get("node") or "").strip()
    if not name:
        raise RuntimeError("missing required argument: node (see vpn_nodes)")
    group = args.get("group") or ("PROXY" if "PROXY" in _proxies() else "GLOBAL")
    proxies = _proxies()
    if group not in proxies:
        raise RuntimeError("no group named %r; groups are %s" % (group, sorted(_groups(proxies))))
    if proxies[group].get("type") != "Selector":
        # A url-test group picks by latency itself; a manual choice needs a
        # Selector. Saying so beats a silent no-op that looks like a failure.
        raise RuntimeError(
            "group %r is a %s group: it chooses by latency on its own, so it cannot be "
            "pointed at a node. Switch that group's type to `select` in the config, or "
            "pick a Selector group."
            % (group, proxies[group].get("type"))
        )
    known = proxies[group].get("all", [])
    if name not in known:
        raise RuntimeError(
            "node %r is not a member of %r. Members: %s" % (name, group, known[:20])
        )
    # PUT carries the choice in the body, so it cannot go through `_request`,
    # which sends no body.
    url = CONTROLLER.rstrip("/") + "/proxies/" + _quote(group)
    req = urllib.request.Request(
        url, data=json.dumps({"name": name}).encode(), method="PUT"
    )
    req.add_header("Content-Type", "application/json")
    if SECRET:
        req.add_header("Authorization", "Bearer " + SECRET)
    with urllib.request.urlopen(req, timeout=20) as r:
        r.read()
    now = _proxies().get(group, {}).get("now")
    return {"group": group, "requested": name, "now": now, "ok": now == name}


def vpn_test(args):
    """Latency of every node in a group, fastest first.

    The one action a person actually repeats: "which node is alive right now".
    """
    return vpn_nodes(dict(args, probe=True))


def vpn_groups(_args):
    """The groups and their members, for a caller that wants to choose a target."""
    proxies = _proxies()
    out = []
    for name, info in sorted(_groups(proxies).items()):
        out.append(
            {
                "name": name,
                "type": info.get("type"),
                "now": info.get("now"),
                "members": info.get("all", []),
            }
        )
    return {"groups": out}


TOOLS = [
    {
        "name": "vpn_status",
        "description": "Show which proxy node is currently in force, plus the groups "
                       "and their selections. The first call to make.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": vpn_status,
    },
    {
        "name": "vpn_nodes",
        "description": "List proxy nodes. probe=true measures each one's latency "
                       "(costs one request per node); probe=false is a cheap listing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "probe": {"type": "boolean", "description": "measure latency per node"},
                "group": {"type": "string", "description": "group whose current node to mark"},
                "timeout_ms": {"type": "integer", "description": "per-node probe timeout"},
            },
            "additionalProperties": False,
        },
        "fn": vpn_nodes,
    },
    {
        "name": "vpn_test",
        "description": "Measure every node's latency and sort fastest first. Use this to "
                       "answer 'which node works right now'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group": {"type": "string"},
                "timeout_ms": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        "fn": vpn_test,
    },
    {
        "name": "vpn_switch",
        "description": "Point a Selector group at a named node. This is the action that "
                       "changes what traffic uses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "exact node name from vpn_nodes"},
                "group": {"type": "string", "description": "defaults to PROXY"},
            },
            "required": ["node"],
            "additionalProperties": False,
        },
        "fn": vpn_switch,
    },
    {
        "name": "vpn_groups",
        "description": "List proxy groups with their type, current selection and members.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "fn": vpn_groups,
    },
]

BY_NAME = {t["name"]: t for t in TOOLS}


def handle(msg):
    method = msg.get("method")
    if method == "initialize":
        return {
            "protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "vpn-nodes", "version": "1.0.0"},
        }
    if method == "tools/list":
        return {
            "tools": [
                {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]}
                for t in TOOLS
            ]
        }
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        tool = BY_NAME.get(name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": "unknown tool: %s" % name}],
                "isError": True,
            }
        try:
            result = tool["fn"](params.get("arguments") or {})
            return {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=1)}],
                "isError": False,
            }
        except Exception as error:
            return {"content": [{"type": "text", "text": str(error)}], "isError": True}
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        result = handle(msg)
        if result is None or "id" not in msg:
            continue
        out = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
