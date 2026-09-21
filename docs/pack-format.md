# 整合包格式 (formatVersion 1)

Nguruvilu 整合包(`.dshpack`)是**一份发行清单**,不是自带全部内容的压缩包。

一个包声明两件事:

| | 是什么 | 在包里吗 |
|---|---|---|
| **引用** | 技能、插件 —— 只有 `地址 + 哈希 + 协议` | ❌ 按需下载 |
| **内容** | 人设、模型路由、上下文策略、MCP、外观、注入规则 | ✅ 实际文件 |

装包时:**解包 → 校验清单 → 校验内核版本兼容 → 按哈希去重下载引用 → 落到包目录**。
同一个技能被两个包引用,只下载一次。

**零实质内容**:API key 只写环境变量名,绝不进包。

---

## 包的形态

```
my-pack-1.0.0.dshpack          ← zip
├── dsh.index.json             ← 清单(引用 + 内容文件索引)
├── soul.md                    ← 人设
├── models.json                ← 模型路由 / 输出 / 请求塑形
├── context.json               ← 上下文窗口 / 压缩 / 缓存策略
├── mcp.json                   ← MCP 服务器
├── look.json                  ← 主题 / UI 面板
└── injections.json            ← 注入规则
```

内容文件**全部可选**。一个只加一条注入规则的包,可以只有一个 `dsh.index.json` 和一个 `injections.json`。

---

## `dsh.index.json`

### 身份与兼容

```jsonc
{
  "formatVersion": 1,
  "game": "nguruvilu",

  "name": "themes",
  "versionId": "1.0.0",
  "license": "MIT",

  // 打包时的内核版本
  "kernelVersion": "0.1.0",
  // 可接受的内核版本范围。不匹配则拒绝加载。
  "dependencies": { "nguruvilu": ">=0.1.0 <0.2.0" }
}
```

**为什么必须有兼容声明**:内核是预览版,工具表、清单字段、ABI 都在变。一个按 0.1 打的包装到 0.3 上,失败方式会很难懂。**不匹配就拒绝,并说清楚为什么。**

### 引用:要下载的东西

```jsonc
{
  "skills": [
    {
      "id": "zhihu-search",
      "source": "github:owner/skills@zhihu-search@v1.0.0",
      "sha256": "…",              // 目录哈希
      "license": "MIT",
      "deps": { "node": ">=18" }  // 仅供参考,内核不解释
    }
  ],

  "plugins": [
    {
      "id": "subagent",
      "source": "github:owner/ngu-plugins@subagent@v1.0.0",
      "license": "MIT",
      // 按平台给不同的文件与哈希。没有 platforms 时用 sha256 单文件。
      "platforms": {
        "win-x64":   { "file": "subagent.dll",   "sha256": "…" },
        "linux-x64": { "file": "subagent.so",    "sha256": "…" },
        "mac-arm64": { "file": "subagent.dylib", "sha256": "…" }
      }
    }
  ]
}
```

**来源前缀**:

| 前缀 | 含义 |
|---|---|
| `github:owner/repo@path@ref` | GitHub 仓库里的某个路径(目录或文件),`ref` 默认 `HEAD` |
| `https://…` | 直接下载一个文件 |
| `builtin:name` | 内核已注册,不下载 |
| `dylib:相对路径` | 包目录内的本地文件,不下载 |

**平台标签**:`win-x64` `win-arm64` `linux-x64` `linux-arm64` `mac-x64` `mac-arm64`。

### 分发协议

```jsonc
{
  "components": [
    { "id": "zhihu-search", "type": "skill",  "license": "MIT" },
    { "id": "subagent",     "type": "plugin", "license": "MIT" }
  ]
}
```

每个下载来的组件**必须**标明协议。这是给用的人看的:一个包引用了 GPL 的插件,使用者有权知道。

### 内容文件

字段值是**包内路径**。字段存在但文件缺失 → 装包时报错。

```jsonc
{
  "soul":       "soul.md",
  "models":     "models.json",
  "context":    "context.json",
  "mcp":        "mcp.json",
  "look":       "look.json",
  "injections": "injections.json"
}
```

---

## 内容文件格式

### `soul.md` —— 人设 / 系统提示

纯 Markdown 正文。装载时作为 `base_prompt`。

### `models.json` —— 怎么跟模型说话

```jsonc
{
  // API key 只写环境变量名,绝不写值
  "baseUrl": "https://api.example.com/v1",
  "apiKeyEnv": "EXAMPLE_API_KEY",
  "model": "some-model",
  "reasoningEffort": "minimal",     // none|minimal|low|medium|high|xhigh|max
  "proxy": "",
  "maxOutputTokens": "8K",          // 接受 128K / 1M / 纯数字
  "extraBody": {                    // 请求塑形:合并进每个请求体
    "enable_thinking": true,        // null 表示删除该字段
    "top_k": 40
  }
}
```

**API key 只写环境变量名** —— 这是铁律。包是要分发的,写进 key 等于泄露。

### `context.json` —— 记多少东西

```jsonc
{
  "window": "128K",
  "compactPercent": 75,
  "compactKeepRecent": 8,
  "cachePolicy": "balanced"         // freshness|balanced|cache-first
}
```

### `mcp.json` —— MCP 服务器

```jsonc
{
  "servers": [
    {
      "id": "filesystem",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": { "TOKEN": "${MY_TOKEN}" },   // 引用环境变量
      "scope": "session"                    // session|global
    }
  ]
}
```

### `look.json` —— 外观

```jsonc
{
  "themes": [
    { "name": "light",
      "tokens": { "--bg": "#ffffff", "--text": "#1a1f26", "--accent": "#1a6fd4" } }
  ],
  "panels": [
    { "id": "clock",
      "slot": "status-bar",
      "title": "Clock",
      "html": "<span id='clock'></span>",
      "script": "setInterval(() => { document.getElementById('clock').textContent = new Date().toLocaleTimeString() }, 1000)",
      "order": 0 }
  ]
}
```

**主题是数据,不需要插件** —— 这条让纯外观包变成零代码、全平台通用。

### `injections.json` —— 注入规则

```jsonc
{
  "budgetPercent": 20,
  "scan": "user",                    // user|user-and-tool|all
  "entries": [
    { "id": "project-rules",
      "content": "Never edit files under vendor/.",
      "constant": true,
      "position": "prefix" },

    { "id": "rust-style",
      "content": "Prefer cargo commands.",
      "triggers": ["rust", "cargo"],
      "position": "history-tail" }
  ]
}
```

---

## 哈希与缓存

**sha256** —— 不是 sha1。sha1 的碰撞在 2026 年已经是实际可行的,而这个哈希是用来判断"下下来的东西是不是声明的那份"。

**目录哈希**:确定性算法 —— 相对路径排序,逐个拼接 `路径 + ':' + 内容 + '\n'`,再取 sha256。同一个目录在 Windows 和 Linux 上算出同一个值。

**缓存**:`~/.nguruvilu/cache/<sha256>/`,按内容寻址。命中即复用,**不联网**。

```
~/.nguruvilu/cache/
    ab12cd…/            ← zhihu-search 这个技能
    ef34gh…/            ← subagent.dll
```

## 台账

`~/.nguruvilu/installed.json` 记录装过什么:

```jsonc
{
  "packs":   [ { "name": "themes", "versionId": "1.0.0", "license": "MIT", "path": "…" } ],
  "skills":  [ { "id": "zhihu-search", "sha256": "…", "source": "…" } ],
  "plugins": [ { "id": "subagent", "sha256": "…", "source": "…" } ]
}
```

**装前先查台账**:同一个哈希已经装过,直接复用,不重复下载也不重复解压。

## 版本兼容

装包时校验:

```
包的 dependencies.nguruvilu  vs  当前内核版本
```

不匹配 → **拒绝加载**,并打印两个版本和范围。

---

## 完整例子:一个零代码外观包

`themes/` 目录:

```
dsh.index.json
look.json
```

**`dsh.index.json`**

```json
{
  "formatVersion": 1,
  "game": "nguruvilu",
  "name": "themes",
  "versionId": "1.0.0",
  "license": "MIT",
  "kernelVersion": "0.1.0",
  "dependencies": { "nguruvilu": ">=0.1.0 <0.2.0" },
  "look": "look.json"
}
```

**`look.json`**

```json
{
  "themes": [
    { "name": "light",
      "tokens": {
        "--bg": "#ffffff", "--panel": "#f4f6f9", "--line": "#d9dee6",
        "--text": "#1a1f26", "--dim": "#667080", "--accent": "#1a6fd4"
      } }
  ],
  "panels": []
}
```

**没有任何引用,没有任何下载,没有一个字节的代码。全平台通用。**

装完界面变白,卸载立刻还原。这就是"体验包的乐趣"的最低成本路径。

---

## 与压缩包(离线包)的关系

现有那个**自带全部内容**的 zip 继续有效,作为**离线/内网**形态:

```
ngu pack --offline <目录> -o out.dshpack    ← 把引用指向的东西也打进去
```

两种形态的区别:

| | 清单包 | 离线包 |
|---|---|---|
| 大小 | 几 KB | 几 MB × 平台 |
| 装包 | 需要网络(缓存命中则不用) | 完全离线 |
| 分平台 | 一份清单覆盖全部 | 每平台一份 |
| 用途 | 正常分发 | 内网 / 断网 / 归档 |
