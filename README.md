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
  pack.rs          整合包(.dshpack zip:打包 / 校验 / 安装 / 清单)
  dylib.rs         动态库插件(最小 C ABI + JSON API、ABI 校验、热更新)
  assembly.rs      装配清单(阶段 / 依赖 / order / 平台过滤 / 循环检测)
  ledger.rs        安装台账(哈希去重,原子写)
  loader.rs        动态加载层(插件·MCP·技能三者平级加载 + 回滚)
  hotreload.rs     热加载(apply_change / 每轮快照 / 作用域 / 同意策略表)
  context.rs       上下文注入引擎(预算 / 触发 / 深度注入 / 分组 / 递归)
  git.rs           工作区 git 自动快照(快照 / 历史 / 回滚)
  main.rs          CLI(交互 / 打印 / JSON + 管理子命令)
docs/
  SPEC.md          完整规格(所有决策的详细定义)
tests/
  kernel.rs        跨模块集成测试
  dylib.rs         动态库插件集成测试
desktop/           桌面应用(独立 crate,产出单文件 exe)
  src/main.rs      窗口 + 事件循环 + IPC 接线
  src/state.rs     应用状态与命令处理
  ui/index.html    三面板界面(编译进 exe)
examples/
  demo-plugin/     示例动态库插件(自己定义 ABI)
```

## 快速开始

```bash
# 配置
export NGU_API_KEY=...
export NGU_BASE_URL=https://api.b.ai/v1
export NGU_MODEL=deepseek-v4.1-flash

# 构建(产出单文件原生可执行,约 5 MB)
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
| 界面 | HTML/CSS/JS,用 `include_str!` 编译进 exe |
| 产物 | `ngu-desktop.exe` 4.6 MB |

对比 Electron portable(391 MB 自解压包,每次启动解压 2~3 分钟):**4.6 MB,点开即出窗口**。

### 配置(endpoint / key / model)

三处配置,优先级从高到低:**命令行参数 → 环境变量 → 设置文件**。

设置文件默认在 `~/.nguruvilu/settings.json`(`$NGU_HOME` 存在时用 `$NGU_HOME/settings.json`)。

**桌面应用**:右侧面板顶部就是 **Settings**,填 endpoint、API key、model,点 Save 立即生效
(不用重启)。未配置时面板顶部会显示醒目提示,并列出缺哪些字段。API key 只显示掩码
(`sk-xxxx...xxxx`),留空表示"保持原值"。

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

### 安装与应用

```bash
ngu install my-pack-1.0.0.dshpack               # → ~/.nguruvilu/packs/<name>-<version>/
ngu install my-pack-1.0.0.dshpack --into /opt/packs
ngu packs                                        # 列出已安装
ngu apply <pack>/assembly.yaml                   # 加载它的插件 / MCP / 技能
ngu uninstall <name>                             # 卸载:之后的对话不再加载,文件保留
ngu uninstall <name> --delete                    # 删除:卸载之后连文件一起删
```

安装目录**带版本号**,所以同一个包的两个版本共存,而不是互相覆盖。解包逐条进行,
**卸载 ≠ 删除。** 卸载只让它不再生效:新对话不会加载它,已经载入的对话当场把它
的东西撤掉;文件留在原处,重新启用不用再下载。删除才动文件。

**包是按对话生效的**:对话开始时加载当时处于 loaded 状态的包;对话进行中,模型可以用
`pack` 工具的 `load` / `unload` 动作自己加载或卸载 —— 回合结束即热生效,不重启、
不打断对话;卸载之后,新的对话自然没有它。
并**拒绝逃逸目标目录的条目路径**。

应用之后,台账 `~/.nguruvilu/installed.json` 记录每个贡献来自哪个包:

```json
{ "kind": "plugin", "id": "demo", "pack": "demo-pack-1.0.0", "scope": "session" }
```

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
## 状态

**完整架构已实现:385 个测试通过(351 单元 + 34 集成)。**

| 能力 | 状态 | 说明 |
|---|---|---|
| agent 循环 | ✅ | 多步、**并行工具调度**、每轮配置快照、时间分解 |
| 模型客户端 | ✅ | 仅 OpenAI 格式;流式;工具调用分片拼接;缓存命中统计 |
| 中立消息格式 | ✅ | provider 无关,只在请求边界转换 |
| 四个基础工具 | ✅ | ACI 原则限界输出;跨平台 shell 解析(POSIX 命令在 Windows 可用) |
| **宿主内置工具** | ✅ | `pack` 打包安装、`search_web`(Tavily/Brave/Exa,未配置不出现)、`delegate` 子代理 |
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
| **全部热加载** | ✅ | `apply_change` 统一入口、会话/全局作用域、同意策略表、每轮快照 |
| **上下文注入引擎** | ✅ | 预算百分比+上限、触发、深度注入、分组竞争、递归激活 |
| **工作区 git 快照** | ✅ | 每轮前后自动提交、历史、回滚(含删除新增文件) |
| CLI | ✅ | 交互 / 打印 / JSON 三模式 + 12 个管理子命令 |
| **桌面应用** | ✅ | desktop/:
gu-desktop.exe 4.6 MB 单文件,tao + wry,界面编译进 exe,三面板 |

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
| 管道 / JSON 调用 | ✅ `echo ... \| ngu --json` 返回结构化结果 |

## 下一步

* 服务端 API 层(HTTP/WebSocket)——手机端与 Web 前端的共同前提
* 子进程插件协议(隔离 + 任意语言 + MCP 生态复用)
* QuickJS 插件运行时(可选优化:改 JS 文件即生效,免编译)
* 子进程插件协议(computer use 等重活)
* 移动端多标签会话(方案 C)
