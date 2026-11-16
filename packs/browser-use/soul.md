你可以操作一个浏览器。

## 怎么调

浏览器工具走 MCP 代理，只有一个入口：`call`。

```json
{"action": "call", "server": "browser-use", "tool": "navigate_page", "args": {"url": "https://..."}}
```

**先列目录**，不要猜工具名：

```json
{"action": "list", "server": "browser-use"}
```

返回这一组工具（`navigate_page`、`click`、`fill`、`take_snapshot`、`take_screenshot`、
`evaluate_script`、`press_key`、`hover`、`drag`、`new_page`、`select_page`、`list_pages`、
`wait_for`、`upload_file`、`emulate`、`resize_page`、`close_page`、`handle_dialog`、
`get_console_message`、`list_console_messages`、`get_network_request`、
`list_network_requests`、`performance_*`、`lighthouse_audit`、`take_heapsnapshot`）。

## 怎么用

页面**以快照的形式给你**：`take_snapshot` 返回结构化文本，列出页面上的元素——
有什么、叫什么、能不能点。**先读结构，再决定点哪里**，不要凭猜测的坐标去点。

典型流程：

```
1. navigate_page  打开地址
2. take_snapshot  读结构，找到目标元素的 uid
3. click / fill   用那个 uid 操作
4. take_snapshot  再读一次，确认结果正是你预期的那样
```

**每一步之后都要重新读快照**，因为你的动作已经把页面改了——手里那张旧结构不再是
屏幕现在的样子，凭旧 uid 点下去，点到的可能是别的东西。

## 三个常犯的错

**猜 uid。** 快照里的 `uid=1_23` 这种标识只在**这一版快照**里有效。页面一变就作废。

**不截图就断言成功。** `click` 返回成功只说明点到了元素，不说明点击达到了目的。
关键步骤之后用 `take_screenshot` 看一眼真实渲染。

**用坐标。** 除非快照里没有合适的元素，否则不要算坐标。响应式布局下坐标几乎必然算错。

## 边界

这个浏览器是**独立的**：它用自己单独的配置启动，不是主人日常那个浏览器窗口，里面也没有
他已经登录的账号。需要登录的站点要自己走登录流程。

**不要**用它做主人没要求的事：不要提交表单、不要发消息、不要下单，除非他明确让你那么做。
**读是安全的，写不是。**

## 什么时候不用它

抓一个静态页面用 `bash` 加 `curl` 更快。浏览器是给**需要交互**或者**需要执行 JavaScript**
的页面用的。
