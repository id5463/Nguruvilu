# computer-use —— 浏览器操作

给模型一个可以操作的浏览器:打开页面、读结构、点击、填表、执行 JavaScript。

## 装

```
ngu install computer-use-1.0.0.dshpack
ngu --assembly ~/.nguruvilu/packs/computer-use-1.0.0/assembly.yaml
```

**需要 Node。** 服务器是 `npx @playwright/mcp@latest`,首次运行会下载 Playwright。

## 里面是什么

| 文件 | 做什么 |
|---|---|
| `mcp.json` | 声明一个 MCP 服务器:`npx -y @playwright/mcp@latest --headless --isolated` |
| `soul.md` | 告诉模型怎么用它、边界在哪 |

**零代码。**

## 为什么用 MCP 而不是自己写插件

MCP 服务器是**别人在维护**的。Playwright 由微软维护,浏览器驱动、无障碍树提取、
各种边界情况都在里面。自己写一遍要重做的正是这些。

而 Nguruvilu 已经有 MCP 支持 —— 所以这件事只需要一个声明。

## 为什么是无障碍树而不是截图

模型拿到的是**页面的结构**(有哪些元素、叫什么、能不能点),不是像素。

这比看图更快也更准:结构是文本,直接进上下文;截图要占几百上千 token,而且模型
还得从像素里认元素。所以这个包**不需要视觉能力**。

## 边界

浏览器是 `--isolated` 的:独立的、临时的,**看不到用户自己的浏览器**,也碰不到
用户已经登录的账号。需要登录的站点要自己走登录流程。

`soul.md` 里明确写了:读是安全的,写不是 —— 不要提交表单、不要发消息、不要下单,
除非用户明确要求。

## 什么时候不该用它

抓一个静态页面用 `bash` 加 `curl` 更快。浏览器是用来处理**需要交互**或者
**需要执行 JavaScript** 的页面的。
