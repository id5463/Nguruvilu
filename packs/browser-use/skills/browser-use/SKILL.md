---
name: browser-use
description: Drive a real browser through the MCP proxy — open pages, click, fill forms, take screenshots, read console and network. Use when a task needs a page that renders with JavaScript, a login flow, or visual confirmation of what a page actually shows.
license: MIT
---

# 浏览器操作

工具在 MCP 代理后面，入口只有一个 `call`。

## 第一步永远是列目录

```json
{"action": "list", "server": "browser-use"}
```

不要凭记忆猜工具名。猜错会拿到一份真实清单——照着它重试，别重复同一个错误名字。

## 单个工具的参数形状

```json
{"action": "call", "server": "browser-use", "tool": "<工具名>", "args": {...}}
```

`args` 原样转发给那个工具，字段名和类型跟它自己的 schema 一致。

## 快照驱动，不是坐标驱动

```json
{"action":"call","server":"browser-use","tool":"take_snapshot","args":{}}
```

返回的是可访问性树：每个元素有 `uid`、角色、名字、是否可点。

```json
{"action":"call","server":"browser-use","tool":"click","args":{"uid":"1_23"}}
```

`uid` 只在**当前这一版快照**里有效。页面一变，旧的 uid 指向的可能是别的东西。

## 关键步骤之后要看到证据

`click` 成功只说明点到了元素，不说明点击达到了目的。要确认效果：

```json
{"action":"call","server":"browser-use","tool":"take_screenshot","args":{}}
```

截图会作为图片附件回到对话里，可以真的看见页面。

## 读控制台和网络

调试一个页面为什么不对，比截图更有用的是这两样：

```json
{"action":"call","server":"browser-use","tool":"list_console_messages","args":{}}
{"action":"call","server":"browser-use","tool":"list_network_requests","args":{}}
```

一个 404 或者一段报错，常常比看十张截图更快定位问题。

## 边界

- 这个浏览器是**独立的**，没有主人已登录的账号；登录流程要自己走
- 提交表单、发消息、下单这类**有后果的动作**，只在主人明确要求时做
- 静态页面用 `curl` 更快，不必开浏览器
