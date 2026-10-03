# Nguruvilu

> 名字来自马普切神话(智利中南部)的水生生物 **Ngürüvilu**:蛇身狐头,用带爪的尾巴制造漩涡。名字由 `nguru`(狐狸)与 `filu`(蛇)合成,意为"狐蛇"——狐狸的狡猾,蛇的灵活。

**Nguruvilu 是一个独立的 agent 内核。** 命令行工具名 `ngu`。

## 与 dshd 的关系

**Nguruvilu 是 dshd 家族的内核,但作为项目完全独立。**

两者是内核与发行版的关系:Nguruvilu 独立开发、独立发布、独立演进,不隶属于 dshd 品牌;dshd 家族(Red / Blue / White / Green / Brown / Pink / tan)是它的使用方之一。任何其他项目都可以基于 Nguruvilu 构建,不必与 dshd 产生关系。

## 架构层次

```
┌─ 内核(Rust)─────────────────────────────────┐
│  会话 / 循环 / 模型(仅 OpenAI)/ 工具表        │
│  四个基础工具:read / write / edit / bash       │
│  插件机制:隔离域 / 依赖 / 热重载 / 撤销        │
└───────────────────────────────────────────────┘
                      ↕ 扩展点
┌─ 动态加载层(Yellow)─────────────────────────┐
│  职责:动态加载、卸载、热替换                  │
│  分发格式:整合包(.dshpack)                   │
└───────────────────────────────────────────────┘
                      ↕ 加载
┌─ 可加载对象(三者平级)────────────────────────┐
│     插件        │       MCP       │    技能    │
└───────────────────────────────────────────────┘
```

**三个要点:**

1. **Yellow 是动态加载层**——它是架构中的一层,负责"把东西动态加载进来"这件事,本身不是插件。
2. **插件、MCP、技能三者平级**——都是可被 Yellow 加载的对象,彼此没有从属关系。
3. **内核不含上述任何对象**——内核只提供加载所需的扩展点。

## 一句话定位

**内核只做四件事:读写文件、运行指令、跑模型循环、管理会话。其余一切——插件、MCP、技能——由动态加载层按需加载,三者平级。**

## 设计原则

1. **内核极简且稳定**:底层工具固定为 `read` / `write` / `edit` / `bash`。内核不随能力增长而变化。
2. **能力全部可加载**:任何新能力都以插件 / MCP / 技能的形式加载,可热加载、可卸载、可隔离。
3. **能力永不因优化而受限**:缓存命中、性能、隔离都是成本项,不是能力约束。任何变更都必须能立即生效;是否牺牲缓存由用户选择。
4. **一切可撤销**:所有副作用(工具、服务、监听、定时器)都通过 effect 登记,卸载时自动清理。
5. **性能是硬要求**:瓶颈不在模型而在框架。**每轮必经路径不得经过脚本引擎。**
6. **只支持 OpenAI 格式**:模型适配层只做 OpenAI 兼容格式,砍掉多 provider 适配的全部复杂度。

## 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| **语言** | **Rust** | 单文件原生可执行(免解压、零依赖)、无 GC 停顿、真并行、启动毫秒级 |
| **动态加载层** | Yellow(重构) | 加载插件 / MCP / 技能;整合包为分发格式 |
| 插件运行时 | **QuickJS 嵌入 JS** + 子进程 | 热加载天然成立;现有 dshd JS 代码可直接搬为插件 |
| 压缩 | **纯插件** | 内核保持最小 |
| 模型格式 | 仅 OpenAI 兼容 | 砍掉多 provider 适配 |
| 会话存储 | **JSONL + trait 接口** | 追加写、崩溃安全、无原生依赖;预留后端替换 |

Rust 的选择有先例:OpenAI 的 **Codex CLI 已从 TypeScript 重写为 Rust**,理由正是性能与安全。

## 核心决策摘要

| 议题 | 决定 |
|---|---|
| 内核边界 | 4 个基础工具 + 会话 + 循环 + 模型调用 + 工具表 + 扩展点 |
| 插件机制 | 参考 Cordis 重写:`isolate` 隔离 + `epoch` 自动重载 + `effect` 撤销 |
| 服务隔离 | 同一进程内服务实例级隔离(realm) |
| 热加载 | 全部热加载,统一入口 `apply_change`,每轮快照保证一致性 |
| 热加载作用域 | 默认**会话级**(无需同意);**全局级需用户同意**,可配置自动同意 |
| 缓存策略 | 三档可配置(新鲜优先 / 平衡 / 省钱优先)+ 按类别覆盖 + 单次强制 |
| 性能 | 并行工具调度、增量上下文、流式直连、无 GC 停顿、必经路径不经过脚本 |
| 多会话 | 进程隔离 + 会话配置分层(基线 + 覆盖) |
| 多设备 | 同一设备可开多个会话(多标签),每会话独立配置 |
| 工作区管理 | 内核提供 git 自动快照:每次改动自动提交,会话可回滚 |

## 参考材料

| 来源 | 协议 | 用途 |
|---|---|---|
| [Pi](https://github.com/earendil-works/pi)(本地 `I:\pi`) | MIT | agent 循环设计(775 行 loop)、4 工具设计(443 行)、压缩思路(777 行) |
| [DSH](https://github.com/deepseek-ai/deepseek-harness)(本地 `I:\deepseek-harness`) | MIT | Cordis 隔离机制、epoch 热重载、effect 撤销、缓存策略、审批栈 |

两者均为 MIT:可自由取用、修改、商用,保留版权与许可声明即可。**注意:它们都是 TypeScript,Rust 实现只能参考设计、重写代码,不能直接复制。**

## 目录

```
src/
  lib.rs           库入口
  message.rs       中立消息格式(provider 无关,请求边界转换)
  llm.rs           OpenAI 格式客户端(流式 + 工具调用分片拼接 + 缓存统计)
  agent.rs         agent 循环(并行工具调度 + 每轮配置快照 + 时间分解)
  session.rs       会话与 JSONL 持久化(SessionStore trait,预留后端替换)
  tools/
    mod.rs         工具注册表(冲突默认 fail loud + 覆盖记录可查)
    base.rs        四个基础工具 read / write / edit / bash(ACI 原则限界输出)
  plugin.rs        插件内核(隔离域 / epoch 自动重载 / 依赖 / effect 撤销 / 权限接口)
  skills.rs        技能系统(目录扫描 / frontmatter 解析 / 目录注入 / 加载工具)
  mcp.rs           MCP 客户端(stdio,JSON-RPC,并发安全)
  pack.rs          整合包(.dshpack zip:打包 / 校验 / 安装 / 清单;Carried 与 Fetched 两种内容形态)
  preinstall.rs    预装包(五个官方包嵌进二进制;已卸载不复活、已编辑不覆盖)
  dylib.rs         动态库插件(最小 C ABI + JSON API、ABI 校验、热更新)
  assembly.rs      装配清单(阶段 / 依赖 / order / 平台过滤 / 循环检测)
  ledger.rs        安装台账(哈希去重,原子写)
  loader.rs        动态加载层(插件·MCP·技能三者平级加载 + 回滚 + 归属来源)
  hotreload.rs     热加载(apply_change / 每轮快照 / 作用域 / 同意策略表)
  context.rs       上下文注入引擎(预算 / 触发 / 深度注入 / 分组 / 递归)
  git.rs           工作区 git 自动快照(快照 / 历史 / 回滚)
  main.rs          CLI(交互 / 打印 / JSON + 管理子命令)
docs/
  SPEC.md          完整规格(所有决策的详细定义)
tests/
  kernel.rs        跨模块集成测试
  dylib.rs         动态库插件集成测试
  ui_sync.rs       界面两条一致性:母版/包副本逐字节相同、元素绑定必须声明
desktop/           桌面应用(独立 crate,产出单文件 exe)
  src/main.rs      窗口 + 事件循环 + IPC 接线 + 回环 HTTP 界面服务
  src/state.rs     应用状态与命令处理
  ui/index.html    三面板界面(母版;运行时服务 ui 包里的那份副本)
examples/
  demo-plugin/     示例动态库插件(自己定义 ABI)
```

## 快速开始

```bash
# 配置
export NGU_API_KEY=...
export NGU_BASE_URL=https://api.b.ai/v1
export NGU_MODEL=deepseek-v4.1-flash

# 构建(产出单文件原生可执行,约 8.4 MB)
cargo build --release        # → target/release/ngu

# 三种运行模式
ngu                                     # 交互模式
ngu -p "列出当前目录"                    # 打印模式
echo "总结这个文件" | ngu --json         # JSON 模式(供其他 agent 调用)
ngu --session <id> -p "继续"             # 续接会话

# 技能
ngu skills --skill-dir ./skills          # 列出技能
ngu skills --skill-dir ./skills --verbose # 连同正文

# 装配清单(整合包)
ngu assembly ./assembly.yaml             # 校验并预览加载计划
ngu apply ./assembly.yaml                # 应用:加载插件 / MCP / 技能

# 工作区快照
ngu --git-snapshot -p "重构这个模块"      # 每轮前后自动提交
ngu snapshot list                        # 查看快照
ngu snapshot restore <commit>            # 回滚

# 运行时
ngu runtime                              # 策略表与生效配置
ngu config show                          # 当前 API 配置与设置文件路径
ngu config set --base-url <url> --api-key <key> --model <id>
ngu models                               # 可用模型
```

## 桌面应用(单文件 exe)

`desktop/` 是一个**独立 crate**,产出一个 exe:双击即开窗口,**零解压、零外部资源文件**。

| | |
|---|---|
| 窗口 | tao(原生窗口,非浏览器) |
| 渲染 | wry + 系统 WebView2(Win10/11 自带) |
| 界面 | HTML/CSS/JS;母版编译进 exe,运行时服务 `ui` 包那份副本(两份逐字节一致,有测试) |
| 产物 | `ngu-desktop.exe` 约 8.4 MB |

对比 Electron portable(391 MB 自解压包,每次启动解压 2~3 分钟):**约 8.4 MB,点开即出窗口**。

窗口服务界面走**本机回环 HTTP**(启动时绑 `127.0.0.1` 随机端口,stderr 打一行
`[ui] serving http://127.0.0.1:<port>/`)—— 自定义协议(`ngu://`)不能当文档的源,
WebView2 会白屏;`curl -I` 那个端口就能诊断界面问题。所有颜色走九个主题 token,
换主题即换肤,主题覆盖到哪、元素就跟着到哪。

### 配置(endpoint / key / model)

三处配置,优先级从高到低:**命令行参数 → 环境变量 → 设置文件**。

设置文件默认在 `~/.nguruvilu/settings.json`(`$NGU_HOME` 存在时用 `$NGU_HOME/settings.json`)。

**桌面应用**:右侧面板顶部就是 **Settings**,填 endpoint、API key、model,点 Save 立即生效
(不用重启)。未配置时面板顶部会显示醒目提示,并列出缺哪些字段。API key 只显示掩码
(`sk-xxxx…xxxx`),留空表示"保持原值"。

**命令行**:

```bash
ngu config show                  # 当前生效的配置 + 设置文件路径
ngu config set --base-url https://your-host/v1 --api-key sk-... --model your-model
ngu config path                  # 只打印设置文件路径
```

**endpoint 必须带版本段**,例如 `https://your-host/v1`。

> 刻意**没有默认 endpoint**。猜一个默认值,正是让身处服务商未覆盖地区的用户撞上
> 一个自己无法解释的 403 的原因。留空会被报告为"未配置"——那是可行动的,而错误的
> endpoint 不是。
### 思考强度(reasoning_effort)

推理模型的成本主要由「想多久」决定,所以这是一个独立开关,而不是藏在提示词里。

| 位置 | 用法 |
|---|---|
| 桌面 | 右侧面板 **Thinking effort** 下拉:`provider default / none / minimal / low / medium / high / xhigh / max` |
| 命令行 | `ngu config set --reasoning-effort minimal`,或每次 `ngu --reasoning-effort high -p "..."` |
| 环境变量 | `NGU_REASONING_EFFORT` |

实测同一道推理题(`deepseek-v4.1-flash`):默认 **269** output tokens → `minimal` **230** tokens。

发送的是**扁平顶层字段** `reasoning_effort` —— 这是 Chat Completions 的规范写法。
嵌套的 `reasoning: {effort: ...}` 属于 Responses API,不是同一个接口,不要混用。

### 模型列表

`GET /v1/models` 是事实标准,所以不必手填模型名:

* **桌面**:启动时自动拉取,填进 Model 字段的候选列表(可输入过滤,也可填自己的值);另有 `fetch list` 按钮
* **命令行**:`ngu models`

实测一个网关返回了 48 个模型,并附带 `owned_by` 与 `supported_endpoint_types`。

### 兼容性说明

不同 provider 对同一件事的字段名并不一致,而**读错不会报错,只会静默给出 0 或空串**。内核按下面的方式兼容:

| 内容 | 读取的字段(按顺序取第一个有值的) |
|---|---|
| 缓存命中 | `prompt_tokens_details.cached_tokens`、`input_tokens_details.cached_tokens`、`prompt_cache_hit_tokens`、`cached_tokens`、`cache_read_input_tokens` |
| 推理内容 | `reasoning_content`、`reasoning`、`reasoning_details[].text` |
| 用量 | `prompt_tokens` / `completion_tokens`,回退 `input_tokens` / `output_tokens` |

`prompt_tokens` 在所有查过的 provider 上都**已包含**缓存命中的部分,所以不再做减法。
输出上限发送 `max_completion_tokens`(`max_tokens` 已在 OpenAI、Groq、Moonshot、DashScope 弃用)。
### 构建与运行

```bash
cargo build --manifest-path desktop/Cargo.toml --release
# → desktop/target/release/ngu-desktop.exe

ngu-desktop.exe                             # 打开窗口
ngu-desktop.exe --prompt "读一下 notes.txt"  # 打开窗口并自动发一条消息
```

### 界面

| 面板 | 内容 |
|---|---|
| 左 | 会话列表:切换 / 删除 / 新建;启动时自动续接最近会话 |
| 中 | 对话流:流式文本、工具调用卡片、工具结果 |
| 右 | 运行时(模型 / 缓存策略 / 配置版本 / 插件)、技能、工具表、最近工具输出 |
| 下 | 输入框(Enter 发送,Shift+Enter 换行)+ 状态栏(步数 / token / 缓存命中 / 耗时) |

界面与内核的分工:窗口和 webview 在 tao 事件循环线程上,agent 轮次在 tokio runtime 上,
两者只在一处交汇——`UserEvent::ToUi` 把 JSON 送进页面,webview 的 IPC handler 把命令送回来。

### 一个值得记录的坑

`const ipc = ...` 在脚本顶层声明,会与 wry 注入的 `window.ipc` 冲突,抛出**解析期**
SyntaxError。后果极具迷惑性:

- 窗口正常打开,HTML 完整渲染,`document.scripts.length === 1`
- 但**整个脚本不执行,包括它的第一行**
- `window.ngu` 永远是 `undefined`,控制台没有可见报错
- `node --check` 检测不出来,因为 node 里没有 `window.ipc` 这个预置属性

改名即可(`ui/index.html` 里叫 `toShell`,并留有注释说明原因)。
## 整合包(`.dshpack`):打包与读包

整合包是一个 **zip 归档**,装三样东西:身份清单、装配清单、以及装配引用的内容。

```
my-pack/                        ← 包目录(打包的输入)
  dsh.index.json                ← 身份:名称/版本/协议/内核版本/依赖
  assembly.yaml                 ← 装配:加载什么、什么顺序
  skills/pdf-tools/SKILL.md     ← 内容
  plugins/ngu_demo_plugin.dll
```

### 打包

```bash
ngu pack ./my-pack                              # → my-pack-<version>.dshpack
ngu pack ./my-pack --out /tmp/custom.dshpack    # 指定输出路径
```

包目录**必须**有 `dsh.index.json`。没有清单的归档无法被识别,宁可打不出来。

打包时自动排除 `.git/`、`target/`、`node_modules/`,条目按名称排序(同样内容的两次打包可逐字节比较)。

### 读包

三种方式,按需要选:

```bash
# 1. 校验:清单是否合法、有没有装配清单、协议是否声明
ngu verify my-pack-1.0.0.dshpack
ngu verify my-pack-1.0.0.dshpack --json        # 机器可读

# 2. 用系统工具直接看(它就是标准 zip,不依赖 ngu)
unzip -l my-pack-1.0.0.dshpack
#   Windows: 右键 → 打开方式 → 资源管理器;或 Expand-Archive

# 3. 安装后看它加载什么
ngu install my-pack-1.0.0.dshpack
ngu packs
ngu assembly <安装目录>/assembly.yaml           # 预览加载计划,不执行
ngu apply    <安装目录>/assembly.yaml           # 真正加载
```

**为什么是 zip**:任何平台上的任何普通工具都能打开、查看、取出单个文件。检查一个包,
或者从里面捞一个文件出来,都不该需要本内核。条目用 deflate 压缩并保留文件模式,
所以包里带的脚本解包后仍可执行。

### 安装与应用:三种方式,一条命令

```bash
# 1. 本地文件 —— 离线包,完全不联网
ngu install my-pack-1.0.0.dshpack
ngu install my-pack-1.0.0.dshpack --into /opt/packs

# 2. 从 GitHub 下载(仓库里那条传统分发路径)
ngu install github:owner/repo@packs/my-pack@v1.0.0     # 指到包目录(推荐,唯一归档)
ngu install github:owner/repo@main                     # 指到仓库根:多个归档会列出来让你挑,不猜

# 3. 直接给归档的 URL
ngu install https://example.com/dist/my-pack-1.0.0.dshpack

# 装完之后
ngu packs                                        # 列出已安装
ngu apply <pack>/assembly.yaml                   # 加载它的插件 / MCP / 技能
ngu uninstall <name>                             # 卸载:之后的对话不再加载,文件保留
ngu uninstall <name> --delete                    # 删除:卸载之后连文件一起删
```

GitHub/URL 方式会先把归档抓进**内容寻址缓存**(`~/.nguruvilu/cache/<sha256>`),同一个
归档第二次取用不再联网;`source + sha256` 都写了的时候,哈希不符直接拒绝安装。

安装目录**带版本号**,所以同一个包的两个版本共存,而不是互相覆盖。解包逐条进行,
**并拒绝逃逸目标目录的条目路径**。

**卸载 ≠ 删除。** 卸载只让它不再生效:新对话不会加载它,已经载入的对话当场把它
的东西撤掉;文件留在原处,重新启用不用再下载。删除才动文件。

**包是按对话生效的**:对话开始时加载当时处于 loaded 状态的包;对话进行中,模型可以用
`pack` 工具的 `load` / `unload` 动作自己加载或卸载 —— 回合结束即热生效,不重启、
不打断对话;卸载之后,新的对话自然没有它。

应用之后,台账 `~/.nguruvilu/installed.json` 记录每个贡献来自哪个包:

```json
{ "kind": "plugin", "id": "demo", "pack": "demo-pack-1.0.0", "scope": "session" }
```

### 包的两种形态:离线随身 vs 按需下载

清单里的每个内容引用都归为两种之一(`ContentRef`),这是包设计的根:

| 形态 | 语义 | 典型用法 | 断网时 |
|---|---|---|---|
| **`Carried`(随包携带)** | 内容就在归档里,装包不解外部的东西 | 文案、人设、默认值、界面 —— 预装五包全是纯随身包 | ✅ 完全可用 |
| **`Fetched`(装时下载)** | 归档只有几 KB 的**引用**:`source + sha256`,装包时抓进 `files/` | 驱动二进制、远程技能目录、GitHub 上的界面 | 首次需要联网;**命中缓存后完全离线** |

所以一个"下载型包"本身仍然很小(computer-use 1.2.0 的归档 **3.8 KB**,它引用的驱动
27.7 MB 在装包时按平台取、按 sha256 校验、进缓存);一个"离线包"则一行网络都不走
(预装五包即此类,首次启动直接放置)。

**支持的来源形式**(`Source::parse`):

| 前缀 | 含义 |
|---|---|
| `github:owner/repo[@path][@ref]` | 仓库(可带路径与固定 ref;**生产建议钉 ref**) |
| `https://…` / `http://…` | 单个文件(URL 侧 sha256 校验;`zip` / `.tar.gz` / `.tgz` 会解压,路径逃逸条目拒绝) |
| `dylib:…` / 相对路径 | 包内自带(离线形态) |
| `builtin:<name>` | 内核里已注册的插件代码(包只声明启用) |

> **不设关卡**:引用解析失败、哈希不符、平台缺失 —— 只有这些**明确错误**才拦;
> 其余一律如实报告。出错再警告,不是事先猜测。

### 装配清单里能写什么

```yaml
version: 1
name: demo-pack
defaults:
  scope: session        # session | global
  on_failure: skip      # abort | skip | retry
stages:
  - name: foundation
    plugins:
      - id: demo
        source: "dylib:./plugins/ngu_demo_plugin.dll"   # 相对包目录解析
        order: 10
  - name: extensions
    skills:
      - id: pdf
        source: "./skills"
    mcp:
      - id: files
        transport: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
```

顺序优先级:依赖图 → 阶段顺序 → `order` → 声明顺序;有环直接报错。

## 插件与归属

内核只提供**机制**,能力的**单位是包** —— 这是包与插件的分工,也是归属面板的由来:

| 层 | 谁负责 | 例 |
|---|---|---|
| 插件代码 | 内核内置(`builtin:`)或动态库(`dylib:`) | `delegate`、`search` 的实现编在二进制里 |
| 启用与否 | **包的清单**:`"plugins": [{"id":"…","source":"builtin:…"}]` | `packs/search` 声明 `builtin:search` |
| 配置 | 用户设置 / 包里的默认值文件 | `settings.search` ← `search.json` 只填空缺 |
| 归属显示 | 装配时记来源,状态里回读 | 面板:`delegate ← subagent` |

- **只声明、不启用,等于没有**:`ngu` 启动时把 `builtin:*` 的代码放进去,但工具表里
  不会出现,直到某个包的装配问它要。没有包 → 内核没有这个能力。
- **卸载跟包走**:卸掉 `search` 包 → 插件纤维卸下 → `search_web` 从工具表消失;你配的
  设置不删,装回来即恢复。界面上那一节(搜索三栏)也随包动态创建/移除。
- **不搞独占**:我们研究过 DSH 的 computer-use 注册表(一个会话只允许一个电脑操作
  提供方),**刻意不做** —— 多种操控电脑的方式可以自由并存,同时驱动一台桌面时由使用者
  自己协调。自由优先于互斥。
- **彻底包化(dylib)是既定路线**:能力实现编成动态库随包分发,内核只剩装载器。
  `src/dylib.rs` 的最小 ABI、`ngu_demo_plugin.dll` 样例、`examples/demo-plugin/`
  都已就位;逐个能力迁移是下一步的事。

## 联网搜索

`search_web` 由 `search` 包启用,三家方言各按自己的方式带 key:

| 方言 | 请求 | key 位置 |
|---|---|---|
| `tavily` | POST | 请求体 |
| `brave` | GET | `X-Subscription-Token` |
| `exa` | GET | `Authorization: Bearer` |

- **没配 key 就没有这个工具** —— 一个用不了的工具,模型每试一次就白花一轮;
- 桌面设置面板里三项:`Web search provider`(下拉,`off` 即关)、`Search API key`
  (留空=保留已存)、`Search endpoint`(可选,走镜像/代理时填);**保存即生效**,工具当场
  注册或注销;密钥本体永不回传面板(只回 `(set)`/`(not set)`);
- 命令行:`ngu config set --search-provider brave --search-api-key BRAVE_KEY`
  `--search-endpoint https://…`;
- 包里的 `search.json` 只在**空缺**时补默认值(`provider`/`apiKeyEnv`/`maxResults`),
  从不覆盖你已经写好的设置;`apiKeyEnv` 只写**环境变量的名字**,钥匙从不进包。

## 命令参考

`ngu --help` 是权威;下表是常用全集(19 个子命令 + 全局选项):

| 子命令 | 作用 |
|---|---|
| `sessions` / `show` / `delete` | 列出 / 打印 / 删除存储的会话 |
| `models` | 列出当前路由上的模型(`GET /v1/models`) |
| `skills` | 列出找到的技能(`--verbose` 连正文) |
| `assembly` | 校验装配清单并打印加载计划(不执行) |
| `apply` | 应用装配清单:加载插件 / MCP / 技能 |
| `pack` | 从包目录打 `.dshpack`(`--out` 指定输出,`--pin` 钉引用,`--offline` 离线) |
| **`install`** | **装包:本地文件 / `github:owner/repo[@path][@ref]` / `https://` 归档 URL**(`--into` 指定目录) |
| `packs` | 列出已安装(版本 / loaded / assembly) |
| `uninstall` | 卸载(留文件);`--delete` 连文件一起删 |
| `verify` | 只校验归档,不安装 |
| `ui` | 把内置界面导出到目录,作为自己界面包的起点 |
| `plugin` | 装一个动态库插件并报告它的贡献(开发用) |
| `config` | `show` / `set` / `path` —— endpoint、key、model、reasoning、search… |
| `runtime` | 策略表与生效配置 |
| `injections` | 注入策略表与会命中的注入条目 |
| `snapshot` | 查看 / 回滚工作区 git 快照 |
| `help` | 子命令帮助 |

全局选项(节选):`-p <prompt>` 一次性提问、`-s <id>` 续接会话、`--new-session`、
`--json`、`-q`、`--max-steps`、`--system`、`--cwd`、`--home`、`--skill-dir`、
`--reasoning-effort`、`--persona`、`--cache-policy`、`--git-snapshot`、
`--assembly <file>`;环境变量 `NGU_API_KEY` / `NGU_BASE_URL` / `NGU_MODEL` /
`NGU_REASONING_EFFORT` / `NGU_HOME`。**优先级:命令行 > 环境变量 > 设置文件。**

## 构建、测试与发布

```bash
# 内核 + CLI(单文件,约 8.4 MB)
cargo build --release --offline          # → target/release/ngu

# 桌面(独立 crate,单文件,约 8.4 MB;界面是"母版",运行时服务的是 ui 包那份)
cargo build --release --offline --manifest-path desktop/Cargo.toml

# 测试:单元 + 动态库集成 + 内核集成 + CLI + 界面一致性,共 427 项
cargo test --offline

# 发布:出 dist/ 与桌面副本,打印 SHA256
pwsh -File release.ps1
```

两条环境规则(踩过的坑,写死):

1. **cargo 一律 `--offline`**;跑构建时才需要代理就只给构建(`HTTP_PROXY/HTTPS_PROXY`
   仅在 cargo 期间设置),**跑测试、跑 `ngu`、访问 API 之前必须清掉** —— 代理会把
   模型流量也劫走,造成超时;
2. 界面有两条一致性测试:母版 `desktop/ui/index.html` 与 `packs/ui/index.html`
   **逐字节相同**(改一处必须同步 + 重打包),脚本用到的元素绑定必须有声明
   (漏声明是运行时 `ReferenceError`,`node --check` 查不出来 —— 这类错各发生过一次,
   现在由测试拦)。

## 状态

**完整架构已实现:427 个测试通过(388 单元 + 9 动态库集成 + 26 内核集成 + 2 CLI + 2 界面)。**

| 能力 | 状态 | 说明 |
|---|---|---|
| agent 循环 | ✅ | 多步、**并行工具调度**、每轮配置快照、时间分解 |
| 模型客户端 | ✅ | 仅 OpenAI 格式;流式;工具调用分片拼接;缓存命中统计 |
| 中立消息格式 | ✅ | provider 无关,只在请求边界转换 |
| 四个基础工具 | ✅ | ACI 原则限界输出;跨平台 shell 解析(POSIX 命令在 Windows 可用) |
| **宿主内置工具** | ✅ | `pack` 打包安装、`search_web`(Tavily/Brave/Exa,**由 `search` 包的插件启用**,未配 key 不出现)、`delegate` 子代理 |
| **子代理** | ✅ | `delegate`:同路由同工具表同提示词,空历史起步;步数上限 12,深度上限 2 |
| **网络调节** | ✅ | 设置 / 包 `models.json` / 插件 `network` 服务三条路径,统一进路由后才建客户端 |
| 工具注册表 | ✅ | 冲突默认 fail loud,覆盖记录可查 |
| 会话持久化 | ✅ | JSONL 追加写 + `SessionStore` trait;损坏行容错 |
| **插件机制** | ✅ | 隔离域(realm)、epoch 自动重载、effect 逆序撤销、依赖等待、失败回滚 |
| **动态库插件** | ✅ | 最小 C ABI(char* + i32)+ 全 JSON API;ABI 版本校验;库永不卸载;热更新用新文件名 |
| **权限接口** | ✅ | 收集式 + deny-wins,顺序无关 |
| **技能系统** | ✅ | 目录扫描、frontmatter、目录注入提示词、按需加载工具 |
| **MCP** | ✅ | stdio JSON-RPC 客户端,工具自动注册与命名空间隔离 |
| **装配清单** | ✅ | 阶段 / 依赖拓扑 / order / 平台过滤 / 循环检测 |
| **动态加载层** | ✅ | 插件·MCP·技能平级加载,台账去重,失败策略(abort/skip/retry) |
| **整合包** | ✅ | .dshpack(zip):打包 / 校验 / 安装;身份与装配分离;路径逃逸防护 |
| **包的加载与卸载** | ✅ | 按对话生效;卸载(留文件)与删除是两件事;对话内热加载 / 热卸载,回合结束生效,不重启 |
| **预装包** | ✅ | 五个官方包(chinese / ui / search / subagent / computer-use)嵌在可执行文件里,首次运行自动放置;卸载过不复活,改过不覆盖 |
| **桌面 / 浏览器能力包** | ✅ | 两个包各管一条线:`computer-use`(桌面,预装,**装包时按平台自动下载并校验 `cua-driver`,之后全离线**,冷装约 20s / 二跑 0.2s,实测 57 个桌面工具)与 `browser-use`(浏览器,需自装 `chrome-devtools-mcp@1.10.1`) |
| **装包源** | ✅ | 本地文件 / `github:owner/repo[@path][@ref]` / `https://` 归档 URL;下载进内容寻址缓存,多归档列出不猜 |
| **插件归属** | ✅ | 装配记录来源,状态回读成 `delegate ← subagent`;卸载跟包走,不搞独占,多种电脑操控方式自由并存 |
| **全部热加载** | ✅ | `apply_change` 统一入口、会话/全局作用域、同意策略表、每轮快照 |
| **上下文注入引擎** | ✅ | 预算百分比+上限、触发、深度注入、分组竞争、递归激活 |
| **工作区 git 快照** | ✅ | 每轮前后自动提交、历史、回滚(含删除新增文件) |
| CLI | ✅ | 交互 / 打印 / JSON 三模式 + 19 个管理子命令 |
| **桌面应用** | ✅ | `desktop/` 独立 crate:`ngu-desktop.exe` 约 8.4 MB 单文件,tao + wry + 系统 WebView2,三面板;界面母版编译进 exe,运行时服务 `ui` 包副本 |

## 端到端验证(`deepseek-v4.1-flash` @ `api.b.ai`)

| 验证项 | 结果 |
|---|---|
| 工具调用 | ✅ 模型正确调用工具,参数与结果回灌正常 |
| 多步循环 | ✅ 最多 6 步完成写→改→读→验证链路 |
| **并行调度** | ✅ 一轮请求 3 个工具并发执行,总耗时 53 ms |
| 会话续接 | ✅ 跨进程记住上下文 |
| **KV cache 命中** | ✅ `cached=768/1664`,续接会话时 `in=33 cached=768` |
| 文件落盘 | ✅ `write` + `edit` 后文件内容正确 |
| **技能加载** | ✅ 模型看到技能目录 → 调用 `skill` 工具 → 正确读出指令 |
| **子代理** | ✅ 模型调用 `delegate` → 子代理在同路由同工具表上跑完 → 只把答案带回主会话 |
| **装配清单** | ✅ 预览加载计划;应用时技能加载成功、插件失败被 `on_failure: skip` 降级并报告 |
| **git 快照** | ✅ 每轮前后提交;回滚后新增文件被删除、原有文件保留 |
| **驱动自动下载** | ✅ `computer-use` 装包时按平台取 `cua-driver`(sha256 校验进缓存):冷装 19.8s / 二跑 0.2s,握手成功 57 个 `mcp__cua-driver__*` 工具 |
| 管道 / JSON 调用 | ✅ `echo ... \| ngu --json` 返回结构化结果 |

## 下一步

* **能力彻底包化(dylib)**:delegate / search / computer-use 的实现编成动态库随包分发,
  内核只剩装载器 —— 机制(`src/dylib.rs`、`ngu_demo_plugin.dll`)已就位,逐个迁移
* 服务端 API 层(HTTP/WebSocket)—— 手机端与 Web 前端的共同前提
* 子进程插件协议(隔离 + 任意语言 + MCP 生态复用;computer use 这类重活的长期归属)
* QuickJS 插件运行时(可选优化:改 JS 文件即生效,免编译)
* 移动端多标签会话(方案 C)
* macOS / Linux 构建矩阵与发布产物(GitHub Actions)
* `browser-use` 打磨:固定版本安装引导、会话内状态保留的边界说明

## 许可

MIT。见 [LICENSE](LICENSE)。

取用参考项目([Pi](https://github.com/earendil-works/pi)、[DSH](https://github.com/deepseek-ai/deepseek-harness))同为 MIT:保留版权与许可声明即可自由使用、修改、商用 —— 但它们是 TypeScript,本项目只参考设计、不复制代码。
