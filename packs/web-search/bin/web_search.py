# -*- coding: utf-8 -*-
"""Web search MCP server (stdio JSON-RPC 2.0, zero third-party deps).

Search is not one API. The three that publish for agents disagree about the HTTP
method, the auth header, where the results live in the response, and what the
snippet field is called:

  |        | Tavily         | Brave                          | Exa          |
  | method | POST /search   | GET /res/v1/web/search?q=      | POST /search |
  | auth   | api_key body   | X-Subscription-Token header    | Bearer       |
  | at     | results[]      | web.results[]                  | results[]    |
  | snip   | content        | description                    | text         |

So "point the endpoint somewhere else" does not work. The differences are data,
so each provider is one row in a table rather than a class.

Proxy handling is the reason this is a pack and not a kernel tool. The kernel
version read only NGU_PROXY and fell back to no_proxy(), while the model route
read settings.proxy — on a machine where the provider answers directly but the
search API does not, every search returned 403 and nothing said why. Here the
order is: HTTPS_PROXY / NGU_PROXY from the environment, then settings.proxy.

Keys come from the environment: TAVILY_API_KEY, BRAVE_API_KEY, EXA_API_KEY. The
provider is chosen in that order unless SEARCH_PROVIDER names one.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HOME = os.path.expanduser("~")
SETTINGS = os.path.join(HOME, ".nguruvilu", "settings.json")

# name -> (method, url, auth style, results path, snippet key, title key, url key)
DIALECTS = {
    "tavily": ("POST", "https://api.tavily.com/search", "body",
               ("results",), "content", "title", "url"),
    "brave": ("GET", "https://api.search.brave.com/res/v1/web/search", "header",
              ("web", "results"), "description", "title", "url"),
    "exa": ("POST", "https://api.exa.ai/search", "bearer",
            ("results",), "text", "title", "url"),
}

KEY_ENV = {
    "tavily": "TAVILY_API_KEY",
    "brave": "BRAVE_API_KEY",
    "exa": "EXA_API_KEY",
}


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def proxy_url():
    """Environment first, then the panel.

    NGU_PROXY is this program's own variable; HTTPS_PROXY is the ambient one a
    shell or a launcher would set. Both lose to nothing: if the user configured a
    proxy in settings.json and exported nothing, that proxy is used, because the
    alternative is a request that cannot succeed.
    """
    for name in ("NGU_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    configured = (read_json(SETTINGS, {}) or {}).get("proxy") or ""
    return configured.strip()


def pick_provider():
    named = (os.environ.get("SEARCH_PROVIDER") or "").strip().lower()
    if named in DIALECTS and os.environ.get(KEY_ENV[named]):
        return named
    for name in ("tavily", "brave", "exa"):
        if os.environ.get(KEY_ENV[name]):
            return name
    return None


def dig(payload, path):
    node = payload
    for step in path:
        if isinstance(node, dict):
            node = node.get(step)
        else:
            return []
    return node if isinstance(node, list) else []


def search(query, max_results=5, provider=None):
    name = provider or pick_provider()
    if not name:
        return {"ok": False,
                "error": "no provider configured: set TAVILY_API_KEY, BRAVE_API_KEY or EXA_API_KEY"}
    method, url, auth, path, snip, title_key, url_key = DIALECTS[name]
    key = os.environ.get(KEY_ENV[name], "")
    if not key:
        return {"ok": False, "error": "%s is not set" % KEY_ENV[name]}

    headers = {"Accept": "application/json",
               "User-Agent": "nguruvilu-web-search/1.0"}
    body = None
    if name == "tavily":
        body = json.dumps({"query": query, "max_results": max_results,
                           "api_key": key}).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif name == "brave":
        url = url + "?" + urllib.parse.urlencode({"q": query, "count": max_results})
        headers["X-Subscription-Token"] = key
    elif name == "exa":
        body = json.dumps({"query": query, "numResults": max_results}).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Authorization"] = "Bearer " + key

    # An explicit ProxyHandler beats urllib's ambient environment handling,
    # which would otherwise pick up the very variables we just resolved.
    if proxy_url():
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url(), "https": proxy_url()}))
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with opener.open(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        return {"ok": False, "provider": name, "status": exc.code,
                "error": "HTTP %s: %s" % (exc.code, detail)}
    except Exception as exc:
        return {"ok": False, "provider": name,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "proxy": proxy_url() or None}

    hits = []
    for row in dig(payload, path)[:max_results]:
        if not isinstance(row, dict):
            continue
        hits.append({"title": row.get(title_key) or row.get("title") or "",
                     "url": row.get(url_key) or row.get("url") or "",
                     "snippet": row.get(snip) or row.get("content") or row.get("text") or ""})
    return {"ok": True, "provider": name, "query": query,
            "count": len(hits), "results": hits}


TOOLS = {
    "search_web": (search, "Search the web and return titles, URLs and snippets. "
                            "Choose the provider by setting TAVILY_API_KEY, BRAVE_API_KEY "
                            "or EXA_API_KEY (Tavily wins if several are set), or name one "
                            "with SEARCH_PROVIDER.",
                   {"query": "What to search for.",
                    "max_results": "How many results to return (default 5).",
                    "provider": "Optional: tavily, brave or exa."}),
}


def schema_for(params):
    return {"type": "object",
            "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
            "additionalProperties": True}


def handle(msg):
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "web-search", "version": "1.0.0"}}}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": n, "description": f[1], "inputSchema": schema_for(f[2])}
            for n, f in TOOLS.items()]}}
    if method == "tools/call":
        p = msg.get("params") or {}
        name = p.get("name")
        args = p.get("arguments") or {}
        if name not in TOOLS:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": "unknown tool %s" % name}}
        try:
            out = TOOLS[name][0](**args)
        except TypeError as exc:
            out = {"ok": False, "error": "bad arguments: %s" % exc}
        except Exception as exc:
            out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=1)}],
            "isError": not out.get("ok", True)}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": "Method not found: %s" % method}}


def main():
    # MCP is UTF-8 on the wire. On Windows the default stdout encoding is the
    # ANSI code page, so a result carrying Chinese arrives as bytes no client can
    # decode and the whole call is lost — the failure names a codec, not a search.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
