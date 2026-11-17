---
name: web-search
description: Search the web with one tool over Tavily, Brave or Exa. Use when a question needs current information you do not have, or when the user asks you to look something up.
---

# Web search

One tool: `search_web`. It reaches Tavily, Brave and Exa behind a single
interface, because the three disagree about almost everything else.

## Calling it

```
mcp__web_search__search_web(query="...", max_results=5)
```

Arguments:

- `query` — what to search for. Write the query you would type into a search
  box, not a sentence describing what you want.
- `max_results` — how many hits. Default 5, and more is rarely better: five good
  snippets beat twenty you will not read.
- `provider` — optional. Leave it out unless you have a reason; the configured
  provider is chosen for you.

The result is JSON:

```json
{"ok": true, "provider": "tavily", "count": 2,
 "results": [{"title": "...", "url": "...", "snippet": "..."}]}
```

When it fails, `ok` is false and `error` says why. The failure worth knowing is
`HTTP 403` — that is almost always a missing or revoked key, not a bad query.

## Which provider runs

The first of these that has a key set wins, unless `SEARCH_PROVIDER` names one:

| Order | Provider | Key |
|---|---|---|
| 1 | Tavily | `TAVILY_API_KEY` |
| 2 | Brave | `BRAVE_API_KEY` |
| 3 | Exa | `EXA_API_KEY` |

With no key at all the tool says so instead of failing silently.

## Proxies

The search API is often unreachable without one, on machines where the model
provider itself is fine. The order is:

1. `NGU_PROXY`
2. `HTTPS_PROXY` / `https_proxy`
3. `proxy` in `~/.nguruvilu/settings.json`

That last one is why this is a pack: the kernel version read only the
environment, so a panel setting that worked for the model did nothing here.

## Reading results

Cite the URLs you actually used. A snippet is a claim by whichever page the
provider indexed — if the answer matters, open the page rather than trusting the
snippet, which is often truncated mid-sentence.

Do not paste the raw JSON back at the user. They asked a question; answer it.
