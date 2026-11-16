---
name: mcp-proxy-guide
description: How to call MCP tools through the single `call` entry point. Use when a task needs a browser, GitHub, or any other MCP server, or when a tool named `call` is the only MCP tool available.
license: MIT
---

# 通过 `call` 使用 MCP

工具表里只有一个 `call`，不是缺了工具——**所有 MCP 服务器都在它后面**。

## 先看目录，再调用

**不要猜工具名。** 猜错是最常见的失败。

```json
{"action": "list"}
```

返回每个服务器和它的工具：

```json
{
  "github": [
    {"name": "search_repositories", "description": "..."},
    {"name": "get_file_contents", "description": "..."}
  ],
  "chrome-devtools": [
    {"name": "navigate_page", "description": "..."}
  ]
}
```

只想看一个服务器：`{"action": "list", "server": "github"}`

## 调用

```json
{
  "action": "call",
  "server": "github",
  "tool": "search_repositories",
  "args": {"query": "nguruvilu"}
}
```

`args` **原样转发**给目标工具，字段名和类型跟该工具自己的 schema 完全一致。

## 猜错工具名会怎样

代理会列出真实的工具然后停下：

```
github 没有 get_me 这个工具。它有：create_branch, create_or_update_file, ...
```

看到这个就照着的清单重试，不要重复同一个错误名字。

## 什么时候该先 list

- 第一次用某个服务器
- 任务是新的，不确定哪个工具合适
- 上次调用报了“没有这个工具”

## 什么时候不用 list

- 已经知道确切工具名（比如上一轮刚用过）
- 多花一次往返不值得

## 服务器起不来

`list` 的结果里有 `__failed`：

```json
{"__failed": ["chrome-devtools: 启动失败 spawn chrome-devtools-mcp ENOENT"]}
```

这说明那个服务器没启动，不是工具不存在。检查它是否装在 PATH 上，或者 `NGU_MCP_TARGETS` 里的路径写对没有。

## 三模态的区别

包可以用 `NGU_MCP_MODE` 切换：

| 模式 | 工具表里是什么 | 什么时候用 |
|---|---|---|
| `proxy` | 一个 `call` | 默认。服务器多、工具多 |
| `single` | 每服务器一个 `<名字>_call` | 想按服务器分开权限或预算 |
| `full` | 每个工具原名 | 不代理，用于对照 |

`single` 模式的调用形状一样，只是不用写 `server`：

```json
{"action": "call", "tool": "search_repositories", "args": {"query": "x"}}
```
