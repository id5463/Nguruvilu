# UI 层

**UI 层是独立的。内核只提供接口,界面通过接口操作内核。**

```
              内核
               │
        ┌──────┴──────┐
        │   接口      │   JSON 事件出,JSON 命令进
        └──────┬──────┘
               │
      ┌────────┼────────┐
      │        │        │
   内置界面   包的界面   别的什么
```

**一次只有一个界面是活动的。** 它是完整的 —— 不是一个面板,不是一个视图,是整个界面。

---

## 一、接口

接口只有两样东西:**内核往外发事件,界面往里发命令。**

### 内核 → 界面(事件)

**会话**

| 事件 | 载荷 |
|---|---|
| `ready` | — |
| `sessions` | `{ list: [{ id, title, messages, updated_at, current }] }` |
| `transcript` | `{ entries: [...] }` |
| `status` | 见下 |

**一轮对话**

| 事件 | 载荷 |
|---|---|
| `turn_start` | — |
| `step` | `{ index }` |
| `text` | `{ delta }` |
| `reasoning` | `{ delta }` |
| `tool_start` | `{ name, arguments }` |
| `tool_end` | `{ name, ok, text }` |
| `compaction_start` / `compaction` | `{ window, summarize }` / `{ replaced }` |
| `injection` | `{ activated, relocated, budget_used }` |
| `turn_end` | `{ steps, tool_calls, text, timing }` |

**配置与包**

| 事件 | 载荷 |
|---|---|
| `models` | `{ list: [{ id, owned_by }] }` |
| `packs` | `{ dir, packs: [...] }` |
| `pack_installed` / `pack_applied` / `pack_built` / `pack_verified` | 各自载荷 |
| `settings_saved` | — |
| `ui` | `{ id, title, dir }` —— 哪个界面是活动的 |

**其它**

| 事件 | 载荷 |
|---|---|
| `notice` | `{ text, bad? }` |
| `error` | `{ message }` |

`status` 的形状:

```jsonc
{
  "session_id": "...", "busy": false, "model": "...", "reasoning_effort": "minimal",
  "last_error": null, "last_prompt_tokens": 12345,
  "context": { "window": 128000, "window_source": "table",
               "threshold_percent": 75, "keep_recent": 8 },
  "max_output_tokens": "8K",
  "network": { "request_timeout_secs": 300, "pool_idle_timeout_secs": 30,
               "retry_attempts": 3, "retry_backoff_ms": 200 },
  "tools": ["bash", "edit", "pack", "read", "write"]
}
```

### 界面 → 内核(命令)

| 命令 | 参数 |
|---|---|
| `ready` | — |
| `status` | — |
| `prompt` | `{ text }` |
| `new_session` / `open_session` / `delete_session` | — / `{ id }` / `{ id }` |
| `save_settings` | 见下 |
| `fetch_models` | — |
| `list_packs` / `install_pack` / `uninstall_pack` / `apply_pack` | — / `{ path }` / `{ name, version? }` / `{ path }` |
| `set_ui` | `{ id }` —— 换一个界面 |

`save_settings`:

```jsonc
{ "base_url": "...", "api_key": "...", "keep_api_key": true, "model": "...",
  "reasoning_effort": "minimal", "proxy": "", "context_window": "128K",
  "compact_percent": 75, "compact_keep_recent": 8, "max_output_tokens": "8K" }
```

`api_key` 留空 + `keep_api_key: true` 表示"不改密钥" —— 界面拿不到已保存的密钥。

**接口就这些。** 界面怎么画、怎么摆、有没有快捷键、有没有动画、用不用框架 —— **内核一概不管**。

---

## 二、包怎么带一个界面

```jsonc
{
  "formatVersion": 1,
  "game": "nguruvilu",
  "name": "my-workbench",
  "versionId": "1.0.0",
  "license": "MIT",
  "kernelVersion": "0.1.0",
  "dependencies": { "nguruvilu": ">=0.1.0" },

  "ui": {
    "id": "my-workbench",
    "title": "我的工作台",
    "entry": "index.html",
    "source": "github:owner/my-ui@dist@v1.0.0"   // 省略则用包内自带
  }
}
```

**`source` 和 `path` 的关系和内容文件一样** —— 有 source 就先取下来,`entry` 在取来的目录里找。

**这个包不需要任何原生代码。** 桌面壳本身就是网页视图,所以一份资产三个平台通用。**这就是为什么"界面可以很大"和"界面不该分平台"不矛盾。**

**界面可以很大** —— 几百 MB 的引擎、字体、模型文件,都是下载来的。

---

## 三、界面拿到什么

界面就是**一个网页**。它的入口 HTML 由壳加载,所以相对路径天然可用:

```html
<link rel="stylesheet" href="style.css">
<script src="app.js"></script>
<img src="assets/logo.svg">
```

**没有框架要求,没有强制结构。** 一个空的 `<body>` 也可以,用 `React`、`Vue`、`Svelte` 打包成一个 js 也可以,用 `canvas` 画也可以。

它通过一个全局对象操作内核:

```js
ngu.send({ cmd: "prompt", text: "..." })   // 发命令
ngu.on("text", (e) => { /* ... */ })       // 收事件
ngu.onAny((name, payload) => {})           // 所有事件
ngu.state()                                 // 一次拿到完整状态
```

**就这些。** 没有槽位、没有视图注册、没有命令注册 —— 那些是界面自己的事。

---

## 四、一个最小的完整界面

```
clock/
├── dsh.index.json
└── index.html
```

**`dsh.index.json`**

```json
{
  "formatVersion": 1,
  "game": "nguruvilu",
  "name": "clock",
  "versionId": "1.0.0",
  "license": "MIT",
  "kernelVersion": "0.1.0",
  "dependencies": { "nguruvilu": ">=0.1.0" },
  "ui": { "id": "clock", "title": "时钟", "entry": "index.html" }
}
```

**`index.html`** —— 一个只说时间的界面,但仍然能用:

```html
<!doctype html>
<html><body style="font:14px monospace;background:#14161a;color:#d8dee9;padding:24px">
<h1 id="clock"></h1>
<p id="state"></p>
<form id="ask"><input id="q" placeholder="问点什么…" style="width:60%"><button>发送</button></form>
<pre id="out"></pre>
<script>
  const tick = () => document.getElementById("clock").textContent = new Date().toLocaleTimeString();
  tick(); setInterval(tick, 1000);

  ngu.send({ cmd: "ready" });                       // 报到,壳随后推状态

  ngu.on("status", (e) => {
    document.getElementById("state").textContent =
      e.status.model + " · " + e.status.tools.length + " 个工具";
  });

  ngu.on("text", (e) => {                            // 流式回答
    document.getElementById("out").textContent += e.delta;
  });

  document.getElementById("ask").onsubmit = (ev) => {
    ev.preventDefault();
    document.getElementById("out").textContent = "";
    ngu.send({ cmd: "prompt", text: document.getElementById("q").value });
  };
</script>
</body></html>
```

**这就是一个能用的界面了。** 它不会排版得像内置那个好看,但它**能对话、能看到状态、能看流式输出** —— 因为接口就是这些。

---

## 五、内置界面是什么

**内置界面就是上面那种东西,只是写得更全。** 它没有任何特权。

- 它也是通过 `ngu.send` / `ngu.on` 操作内核
- 它也不能读到 API key
- 它也不能做别的界面做不到的事

**检验标准:** 把内置界面的 HTML 原样拷进一个包,声明成 `ui` —— 如果功能一模一样,这层就做对了。做不到,就说明内置界面用了什么私有通道,那是缺陷。

---

## 六、多个界面怎么共存

**一次只有一个活动。** 用户在设置里选,或者启动时指定:

```bash
ngu-desktop --ui my-workbench
```

装了一个带界面的包,不会自动换过去 —— 界面是"你整天面对的东西",换它得是一个明确的动作。

`set_ui` 命令让界面能请求换掉自己(比如一个"选择界面"的界面)。

---

## 七、需要改的

| # | 改什么 | 大小 |
|---|---|---|
| 1 | 协议处理器从"返回固定页面"改成"能按路径取包里的文件" | 中 |
| 2 | `ui` 字段:解析 + 下载 + 落盘(复用 ContentRef) | 小 |
| 3 | 记住哪个界面是活动的,壳加载它而不是内置的 | 小 |
| 4 | 把现在页面的内部函数整理成公开的 `ngu` 对象,写进文档 | 小 |
| 5 | `ngu ui eject` 导出内置界面,让人有东西可改 | 小 |

**没有第 4 项之外的大改。** 上一版设计里的区域、视图、命令注册、事件总线**全部删掉** —— 那些是在解决"面板怎么组合",而面板不组合,**界面是整体的**。

---

## 八、和"面板"的区别

一个包也可能只想要一个**状态栏小件**,不想接管整个界面。

那种东西的答案不是"槽位",是:**写一个完整界面,或者不写。**

因为一旦允许"塞一块进去",内核就必须知道那块东西放哪、什么时候显示、和别的块怎么协调 —— **那就又回到我上一版的复杂度了**。

**一个包带来一个界面,界面是完整的。** 想要小件的人,可以基于内置界面导出的版本改一个自己的。

---

## 九、这一版和上一版比,少了什么

| 上一版 | 这一版 |
|---|---|
| 六个槽位 | 无 |
| 视图注册 + 切换 | 无(界面自己画切换) |
| 命令注册表 | 无 |
| 快捷键注册 | 无(界面自己的事) |
| 菜单注册 | 无 |
| 浮层注册 | 无 |
| 事件总线 | 无(界面内部的通信是界面的事) |

**全部删掉,因为它们是同一个错误的不同表现:把界面拆开,让内核去协调。**

界面不拆开。它通过接口操作内核,别的都是它自己的事。
