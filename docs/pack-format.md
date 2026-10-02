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
├── look.json                  ← 主题 / UI 面板 / 界面文案
├── injections.json            ← 注入规则
└── search.json                ← 搜索 provider 与 key 的环境变量名
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
  "injections": "injections.json",
  "search":     "search.json"
}
```

**内核自带的插件用名字,不下载**:

```jsonc
{
  "plugins": [
    { "id": "delegate", "source": "builtin:delegate", "license": "MIT" }
  ]
}
```

`builtin:` 的来源在渲染装配时原样透传 —— 包决定"它要不要装",内核决定"它是什么",
所以内核自带的能力(比如子代理)可以由一个包启用,而包里不放任何二进制。

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
      "args": ["-y", "@modelcontextprotocol/server-filesystem@2026.8.31", "."],
      "env": { "TOKEN": "${MY_TOKEN}" },   // 引用环境变量
      "scope": "session"                    // session|global
    }
  ]
}
```

**`source` / `platforms` —— 装包时自动取来的服务器**:

```jsonc
{
  "servers": [
    {
      "id": "cua-driver",
      "command": "cua-driver",
      "args": ["mcp"],
      "platforms": {
        "win-x64":   { "source": "https://…/driver-windows.zip",   "sha256": "…", "command": "cua-driver.exe" },
        "linux-x64": { "source": "https://…/driver-linux.tar.gz",  "sha256": "…" },
        "mac-arm64": { "source": "https://…/driver-darwin.tar.gz", "sha256": "…", "command": "driver-darwin/cua-driver" }
      }
    }
  ]
}
```

- **`platforms` 非空时按 `fetch::platform_tag()` 选当平台条目**;取不到就报错,信息里
  列出声明过的平台和全部平台标签 —— 绝不退回另一个平台的二进制。
- **URL 是压缩包(`.zip` / `.tar.gz` / `.tgz`)就解包**进 `files/mcp/<id>/`;条目名带
  `..` 或绝对路径即拒绝,一个字节也不写出目录。其它来源照旧抓取后整目录拷入。
- **`sha256` 是压缩包文件本身的 sha256** —— 上游 release 的 `checksums.txt` 里就是这个
  值,直接照抄;缓存按它寻址,第二次安装不再联网。目录来源用目录哈希。
- **`command` 可按平台覆盖**(Windows 是 `cua-driver.exe`,有些平台的二进制包了一层
  目录)。装包时把选中的 `source` 和可执行文件的**绝对路径**写回,装载时直接执行,
  不查 PATH。
- 单平台、或所有平台同一来源时,继续用顶层 `source` / `sha256` / `command`,写法不变。

**`command` 指向本机已装的程序** —— 和 `builtin:` 一个道理:包只声明"要什么",
机器上得有。内核按 PATH 解析这个名字,找不到就在装载时报一条 `failed mcp:<id>`
并跳过,不影响其它条目。**版本在安装时定死**,不要写 `npx …@latest`:那会让每次
启动都去问一遍 npm registry,慢,而且随时可能拿到一个没测过的版本。

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
  ],
  // 界面文案:键是元素 id,或 `placeholder:<id>` 改输入框占位符。
  // 认不出的键原样放过;卸载这个包时这些字会一并还原。
  "strings": { "send": "发送", "placeholder:input": "问点什么…" }
}
```

**主题是数据,不需要插件** —— 这条让纯外观包变成零代码、全平台通用。文案同理:一个翻译包只有 `strings`,一样零代码。

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

### `search.json` —— 搜索

```jsonc
{
  "provider": "tavily",          // tavily | brave | exa
  "apiKeyEnv": "TAVILY_API_KEY", // 只写环境变量的名字,永远不写 key
  "endpoint": "https://…",       // 可选:代理 / 镜像
  "maxResults": 5                // 可选
}
```

- **钥匙从不进包**;变量没设 → `search_web` 不注册 —— 不出现,模型就不知道它存在,
  好过"出现了,每轮调用都失败"。
- **不盖过已配置的设置**:这个包随内核预装,所以它只补空缺 —— 你用
  `ngu config set --search-provider …` 配好的值不会被它改回去;只有什么都没配时,
  它才从零搭起来。

## 哈希与缓存

**sha256** —— 不是 sha1。sha1 的碰撞在 2026 年已经是实际可行的,而这个哈希是用来判断"下下来的东西是不是声明的那份"。

**目录哈希**:确定性算法 —— 相对路径排序,逐个拼接 `路径 + ':' + 内容 + '\n'`,再取 sha256。同一个目录在 Windows 和 Linux 上算出同一个值。

**单文件例外**:`mcp.json` 平台条目里的压缩包 URL,声明的是**文件本身**的 sha256(上游 `checksums.txt` 的值),详见上文 `mcp.json` 一节。

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

## 卸载、删除与按对话生效

三件事,分开:

| 动作 | 对话里还调得到吗 | 文件 | 怎么做 |
|---|---|---|---|
| **加载** | 调得到 | 不变 | 对话开始时装载当时处于 loaded 的包;对话里用 `pack` 工具 `action: "load"` 热加载 |
| **卸载** | 调不到 | **保留** | `ngu uninstall <名字>`;对话里用 `pack` 工具 `action: "unload"`,回合结束即生效 |
| **删除** | 调不到 | 删除 | `ngu uninstall <名字> --delete`(先卸载,再删文件) |

- **状态记在包目录里**:`.ngu-disabled` 标记文件。清单是包作者的声明,标记是用户对它的
  决定 —— 两者分开,所以卸载不用改包的内容,也不影响包的哈希。
- **按对话生效**:对话开始时只加载 loaded 的包;对话中途的热加载/热卸载作用于当前这个
  对话所在的进程,回合结束生效 —— 不重启、不暂停对话;之后新开的对话按当时的 loaded
  状态来,所以卸载之后新开的对话没有它。
- **卸载撤掉什么**:插件实例(`kernel.unload`,effect 逆序撤销)、MCP 工具(最后一个句柄
  落下时子进程随之停止)、技能根目录(并重建 `skill` 目录工具)、外观纤维(`pack:<name>`)。
  台账里属于该包的条目一并移除,所以重新加载是真正的重新加载,而不是"已经装过"。
- **卸载不碰人设和已经写进对话历史的文本**:那是对话本身的一部分,不是可以拔掉的插件。

## 预装:随程序带来的五个包

程序自带五个包,**嵌在可执行文件里**(不联网、不带额外文件),第一次运行时放进
packs 目录:

| 包 | 带来什么 |
|---|---|
| `chinese` | 中文人设、两条常驻注入、**界面文案** |
| `ui` | 界面的可改副本 —— 窗口服务的就是它;卸载则回到内核里的内置界面 |
| `search` | 搜索 provider 与 key 的环境变量名(没 key 就没有工具) |
| `subagent` | `delegate` 工具 —— 内核自带能力,由这个包启用 |
| `computer-use` | 桌面操作:`cua-driver` MCP —— 装包时按平台自动下载并解包驱动(首装需联网,约 27–67MB 一次,之后走缓存);没装上则每次启动报一条 `failed mcp:cua-driver`,不要就卸载 |

放置的三条规则(记录在 `~/.nguruvilu/preinstalled.json`):

1. **用户的决定优先**:卸载或删除过的包**永远不会**被重新放置;
2. **改过的包绝不覆盖**:放置时记下每个文件的 sha256,只有"和我们放进去的一模一样"
   才会被新版本替换 —— 你自己编辑过的那份,永远是你那份;
3. **幂等**:每次启动都跑一遍,稳态下只是一次目录列举和几个哈希,什么都不做。

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
