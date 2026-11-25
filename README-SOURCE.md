# nguruvilu 源码结构与文件说明

> 内核只做四件事：读写文件、运行指令、跑模型循环、管理会话。其余一切——插件、MCP、技能——由动态加载层按需加载，三者平级。

本文由 `src/` 的实际内容生成：**143 个 `.rs` 文件、41 个顶层模块、39780 行**。行数由 `wc -l` 逐个读出，不是估的。

---

## 一、总览

| 项 | 数值 |
|---|---|
| 顶层模块（`lib.rs` 声明） | 41 |
| 已拆成目录的模块 | 16 |
| 源文件总数 | 143 |
| 总行数 | 39780 |
| 可替换路由（`dispatch`） | 18 条 |
| 可替换 trait | 15 个 |

---

## 二、可被包替换的东西

本仓库的核心约定：**写一个包永远不需要修改内核源码。**

### 2.1 dispatch 路由

注册同名路由即覆盖。内核那份留在链尾；包可以完全覆盖，也可以先做事再调 `next(i, args)` 回退。

| 路由名 | 做什么 | 注册在 |
|---|---|---|
| `tool.execute` | 调一个工具，参数和结果都是 JSON | `kernel_routes.rs` |
| `tool.names` | 列出全部工具名 | `kernel_routes.rs` |
| `compact.plan` | 算出要压缩多少、留多少 | `kernel_routes.rs` |
| `window.estimate` | 估一段对话占多少 token | `kernel_routes.rs` |
| `state.read` | 按路径读状态树 | `kernel_routes.rs` |
| `state.write` | 按路径写状态树 | `kernel_routes.rs` |
| `state.keys` | 列一个路径下的键 | `kernel_routes.rs` |
| `state.live` | 取一个活对象（`Arc`） | `kernel_routes.rs` |
| `cli.stats` | `ngu stats` 打印什么 | `cli/stats.rs` |
| `cli.export` | `ngu export` 导出成什么形状 | `cli/export.rs` |
| `cli.config` | 设置从哪来、里面有什么 | `cli/config.rs` |
| `cli.pack.pack` | 怎么打包 | `cli/pack.rs` |
| `cli.pack.install` | 怎么安装 | `cli/pack.rs` |
| `cli.pack.uninstall` | 怎么卸载 | `cli/pack.rs` |
| `cli.pack.verify` | 怎么校验 | `cli/pack.rs` |
| `cli.pack.list` | 装了什么 | `cli/pack.rs` |
| `cli.runtime.skills` | 技能怎么列 | `cli/cli_routes.rs` |
| `cli.runtime.assembly` | 装配清单怎么读 | `cli/cli_routes.rs` |

### 2.2 trait

实现它 = 换掉那个环节。

| trait | 换掉什么 | 定义在 |
|---|---|---|
| `AgentLoop` | 一轮对话怎么跑（最大的替换点） | `agent/contract.rs` |
| `TurnConfig` | 一轮对话的配置从哪来 | `agent/config.rs` |
| `AgentObserver` | 进度怎么显示 | `agent/observer.rs` |
| `PluginHooks` | 包在回合里想做的事 | `agent/hooks.rs` |
| `Summarizer` | 摘要怎么产生（换模型、换提示词、换外部服务） | `compaction/summarize.rs` |
| `ContextPolicy` | 上下文行为：窗口多大、什么时候压 | `window/policy.rs` |
| `InjectionSource` | 往对话里注入什么 | `context/engine.rs` |
| `SessionStore` | 会话存哪（现为 JSONL） | `session/store.rs` |
| `Plugin` | 插件本身 | `plugin/contract.rs` |
| `SchemeHandler` | 取包协议（github/https/内置/dylib） | `fetch/source.rs` |
| `PackSource` | 包从哪来 | `src/source.rs` |
| `RequestShaper` | 请求体在发出前怎么整形 | `src/request.rs` |
| `NetworkPolicy` | 超时、重试、代理 | `src/network.rs` |
| `ModelAccess` | 插件怎么拿到模型权限 | `src/model.rs` |
| `Reader` | `read` 怎么打开内核不认识的格式 | `tools/readers.rs` |

---

## 三、逐个目录的文件说明

## 3.1 `agent/` — 8 文件 2187 行

一次对话回合如何运行：发历史、流式收回复、并行执行工具、重复到模型不再要工具。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/agent/builder.rs` | 339 | `Agent` 本身和二十个 `with_*`：给一轮对话加观察者、取消信号、步数上限、摘要器。 |
| `src/agent/config.rs` | 123 | 回合配置：`TurnConfig` trait，以及「配置从不改变」的静态实现。 |
| `src/agent/contract.rs` | 142 | 一轮对话跑在什么上：`AgentLoop` 契约，以及把它装进/卸出进程。最大替换点。 |
| `src/agent/hooks.rs` | 30 | 循环与包之间的钩子接缝：包里想在回合里做的事放这里。 |
| `src/agent/mod.rs` | 35 | 模块声明与重导出。 |
| `src/agent/observer.rs` | 50 | 进度回调：`AgentObserver`，换掉它改变进度怎么显示。 |
| `src/agent/run.rs` | 1227 | 一轮的实际运行：发请求、收流、执行工具、记账、决定要不要继续。 |
| `src/agent/state.rs` | 241 | 回合携带的数据：状态、结果、单步输出、耗时分解。 |

## 3.2 `assembly/` — 3 文件 668 行

装配清单：一个包声明它加载什么、按什么顺序。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/assembly/manifest.rs` | 339 | 清单类型：一个包声明加载什么。 |
| `src/assembly/mod.rs` | 25 | 模块声明与重导出。 |
| `src/assembly/plan.rs` | 304 | 把清单条目变成本次加载的有序步骤。 |

## 3.3 `cli/` — 22 文件 4118 行

`ngu` 命令行。每个子命令一个文件，其中 10 条接进了路由。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/cli/args.rs` | 472 | 命令行参数与子命令类型。不接触内核，所以改一个 flag 不会改变一轮对话做什么。 |
| `src/cli/assembly_cli.rs` | 156 | `ngu assembly`：读并展示一份装配清单，或把它应用一遍。 |
| `src/cli/cli_routes.rs` | 69 | 命令行自己那两条路由的注册：`cli.runtime.skills`、`cli.runtime.assembly`。 |
| `src/cli/config.rs` | 480 | `ngu config`：看和改存下来的设置；`require_key` 也在这里。路由 `cli.config`。 |
| `src/cli/content_cli.rs` | 332 | 把一次包加载的内容折进当前会话：设置、路由、人格、外观。 |
| `src/cli/events.rs` | 58 | `ngu events`：读审计日志，最新在前。 |
| `src/cli/export.rs` | 129 | `ngu export`：把一条会话写成别人能自己读的文件。路由 `cli.export`。 |
| `src/cli/injections.rs` | 91 | `ngu injections`：列出注入规则，或问某段文字会触发哪些。 |
| `src/cli/mcp.rs` | 232 | `ngu mcp import`：把别的工具的 MCP 声明翻成包的清单片段。 |
| `src/cli/mission.rs` | 161 | 任务条：给模型看的那五行，和怎么把任务文字压成一行。 |
| `src/cli/mod.rs` | 74 | 模块声明与重导出。 |
| `src/cli/model_route.rs` | 65 | 这一轮用哪个模型：命令行参数 > 包提供的路由 > 设置文件。 |
| `src/cli/observer.rs` | 83 | 把一次运行的过程打到 stderr，让 stdout 只有答案。 |
| `src/cli/pack.rs` | 623 | `ngu pack` 家族：打包、安装、卸载、校验、列出。五条路由 `cli.pack.*`。 |
| `src/cli/plugin.rs` | 140 | `ngu plugin`：直接加载一个插件二进制，或调它一个工具。 |
| `src/cli/routes_cli.rs` | 63 | `ngu routes`：直接调一条内核路由，参数走 JSON。「一切皆调用」的命令行出口。 |
| `src/cli/runtime_cmd.rs` | 124 | `ngu runtime`：内核装配完成之后是什么样（路由、缓存策略、工具表、人格）。 |
| `src/cli/skills.rs` | 54 | `ngu skills`：列出能看到的技能（只读 frontmatter 已扫出的索引）。 |
| `src/cli/snapshot.rs` | 59 | `ngu snapshot`：看和回退 git 快照。 |
| `src/cli/stats.rs` | 304 | `ngu stats`：一条会话花了多少，从它记录的回合里读。路由 `cli.stats`。 |
| `src/cli/tui.rs` | 295 | 交互式模式：一行一行读输入，跑一轮，把答案打出来。 |
| `src/cli/ui.rs` | 54 | `ngu ui`：把一个目录写成内置界面，或说默认界面在哪。 |

## 3.4 `compaction/` — 7 文件 792 行

对话接近上下文窗口上限时怎么压。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/compaction/anchor.rs` | 138 | 用内容哈希定位一次压缩替换掉的那一段。 |
| `src/compaction/apply.rs` | 52 | 把一次压缩应用到历史上。 |
| `src/compaction/mod.rs` | 59 | 模块声明与重导出。 |
| `src/compaction/plan.rs` | 58 | 决定压掉前面多少条、留多少条。路由 `compact.plan`。 |
| `src/compaction/prune.rs` | 254 | 第二条控制上下文的路径：把旧的工具结果和图片裁薄。 |
| `src/compaction/record.rs` | 54 | 一次压缩的记录，以及它的摘要消息长什么样。 |
| `src/compaction/summarize.rs` | 177 | 摘要怎么产生：`Summarizer` trait，包在这里替换。 |

## 3.5 `context/` — 4 文件 876 行

往对话里注入哪些片段、放哪里、什么时候停。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/context/engine.rs` | 530 | 注入引擎本身：条目、预算、递归、缓存。`InjectionSource` trait。 |
| `src/context/entry.rs` | 167 | 一条可注入片段，以及由它解析出来的注入。 |
| `src/context/mod.rs` | 31 | 模块声明与重导出。 |
| `src/context/position.rs` | 148 | 片段能放哪里，以及一个触发器能读哪些消息。 |

## 3.6 `fetch/` — 5 文件 1559 行

把 `github:` / `https:` 引用变成磁盘上经过校验的目录。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/fetch/hash.rs` | 90 | 内容寻址，以及描述一次下载的那几个函数。 |
| `src/fetch/http.rs` | 485 | HTTP(S) 下载：客户端、重定向规则、重试规则、代理绕过。 |
| `src/fetch/mod.rs` | 46 | 模块声明与重导出。 |
| `src/fetch/source.rs` | 642 | 一个包能点名什么，以及把名字变成内容的对象。`SchemeHandler` trait。 |
| `src/fetch/unpack.rs` | 296 | 解开归档，带三重上限（条目数、单条目字节、总字节）。 |

## 3.7 `llm/` — 8 文件 2265 行

和 OpenAI 兼容接口通信：流式、工具调用、重试。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/llm/assemble.rs` | 32 | 把按 index 分片的工具调用参数拼成完整调用。 |
| `src/llm/caps.rs` | 175 | 模型能力表：哪些模型能看图、窗口多大，以及记住「这个端点拒过什么」。 |
| `src/llm/client.rs` | 1172 | `LlmClient`：发请求、收流、交给循环。也含请求体构造与代理规则。 |
| `src/llm/mod.rs` | 50 | 模块声明与重导出。 |
| `src/llm/parse.rs` | 321 | 从服务端回复里取用量、错误、上下文窗口这些数字。 |
| `src/llm/retry.rs` | 174 | 哪些错误值得重试，哪些不该。 |
| `src/llm/sse.rs` | 237 | 解析 `text/event-stream`。 |
| `src/llm/wire.rs` | 104 | 把内部的消息结构转成请求体。 |

## 3.8 `loader/` — 7 文件 1938 行

把装配清单变成运行状态：插件、MCP、技能共用一次顺序遍历。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/loader/accessors.rs` | 108 | 造一个加载器，以及把它握着的东西露出来。不加载任何东西。 |
| `src/loader/entry.rs` | 561 | 按计划装配：走一步、装一个条目。三类条目共用一次遍历与策略。 |
| `src/loader/helpers.rs` | 666 | 独立工具函数：路径解析、进程启动、命名与标签。 |
| `src/loader/mod.rs` | 29 | 模块声明与重导出。 |
| `src/loader/model.rs` | 100 | 加载层的数据模型：一个条目加载成什么、失败了怎么记。 |
| `src/loader/process.rs` | 378 | 进程插件：去哪找二进制、包里带源码就自己编、活过整次加载。 |
| `src/loader/tail.rs` | 96 | 收尾与查询：包设置的形状、源码解析、隔离域、以及两种结束方式。 |

## 3.9 `mcp/` — 4 文件 1182 行

MCP 客户端（stdio 与 Streamable HTTP）。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/mcp/client.rs` | 644 | MCP 客户端行为：连接、握手、列工具、调工具、收尾。 |
| `src/mcp/mod.rs` | 33 | 模块声明与重导出。 |
| `src/mcp/model.rs` | 432 | MCP 的类型与连接注册表（`McpSpec`/`McpTool`/`McpClient` 字段/`IdSpace`）。 |
| `src/mcp/util.rs` | 73 | 把服务端返回的形状整理成内核能用的形状（工具结果、SSE 分帧、文本清洗）。 |

## 3.10 `pack/` — 7 文件 2847 行

`.ngupack` 分发格式：打包、校验、安装。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/pack/archive.rs` | 445 | 一个目录怎么变成一个 `.ngupack`，一个归档怎么解开。 |
| `src/pack/enabled.rs` | 132 | 已安装包开没开、清单在哪、怎么读写。 |
| `src/pack/format.rs` | 195 | 归档的名字、版本、限额，以及结果类型。 |
| `src/pack/install.rs` | 689 | 安装：解开、取它点名的东西、写它要的装配清单。 |
| `src/pack/manifest.rs` | 1053 | 清单本身，以及它引用的那些数据类型。 |
| `src/pack/mod.rs` | 64 | 模块声明与重导出。 |
| `src/pack/verify.rs` | 269 | 校验一个归档而不安装它。 |

## 3.11 `plugin/` — 9 文件 2613 行

插件机制：隔离域、依赖跟踪、可撤销效果。「一个包换一个模块」的执行者。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/plugin/contract.rs` | 111 | `Plugin` 契约：一个插件能被内核问到的每一件事。实现它就是写一个插件。 |
| `src/plugin/core.rs` | 1465 | 内核本体：52 个方法的 `impl Kernel`，加载/卸载/重载/协调，以及 `KernelHooks`。 |
| `src/plugin/ctx.rs` | 379 | 插件加载时看到什么、能改什么（`PluginCtx`/`Contributions`）。 |
| `src/plugin/fiber.rs` | 163 | 一个插件实例：它的状态、它的 epoch、它贡献了什么。 |
| `src/plugin/mod.rs` | 39 | 模块声明与重导出。 |
| `src/plugin/model.rs` | 46 | 内核握着的结构：`Kernel` 的 9 个字段，每个旁边一条不变量。 |
| `src/plugin/realm.rs` | 86 | 隔离域：一个服务名在哪个域里解析。 |
| `src/plugin/seam.rs` | 91 | 一条接缝：插件可以改写的一个值。 |
| `src/plugin/service.rs` | 233 | 服务表，以及插件看到它的只读视图。一个包注册服务、按名字+域查。 |

## 3.12 `session/` — 5 文件 1302 行

会话与持久化（JSONL 追加写，崩溃安全）。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/session/jsonl.rs` | 654 | 默认后端：一个会话一个只追加的 JSONL 文件。 |
| `src/session/mod.rs` | 27 | 模块声明与重导出。 |
| `src/session/model.rs` | 234 | 会话记录与它们的用量账。 |
| `src/session/replay.rs` | 255 | 回放一条存下的会话：把压缩折回去，以及几个 helper。 |
| `src/session/store.rs` | 132 | 持久化的 trait，一个后端实现它；以及服务形态。 |

## 3.13 `settings/` — 4 文件 851 行

设置从哪读、写到哪。优先级：环境变量 > 设置文件 > 默认值。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/settings/io.rs` | 244 | 读写设置文件，以及必须打码的字段。 |
| `src/settings/mod.rs` | 24 | 模块声明与重导出。 |
| `src/settings/model.rs` | 547 | 设置文件装什么，缺的键取什么默认值。 |
| `src/settings/paths.rs` | 36 | 这个程序把文件放哪。 |

## 3.14 `skills/` — 4 文件 743 行

技能：按需加载的指令包。只读 frontmatter，正文按需加载。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/skills/mod.rs` | 43 | 模块声明与重导出。 |
| `src/skills/model.rs` | 390 | 技能模型：发现到的技能、扫描结果、技能表。 |
| `src/skills/parse.rs` | 192 | 读一份 `SKILL.md`：它的 frontmatter 和它得到的 id。 |
| `src/skills/tool.rs` | 118 | `skill` 工具：模型列和加载技能的那条路。 |

## 3.15 `tools/` — 15 文件 4545 行

工具表，以及填满它的工具。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/tools/base.rs` | 1009 | 四个基础工具：`read`、`write`、`edit`、`bash`。内核唯一面向模型的功能面。 |
| `src/tools/budget.rs` | 278 | 运行时预算，以及一次工具调用所在的会话作用域。 |
| `src/tools/jobs/mod.rs` | 50 | 作业模块声明与重导出。 |
| `src/tools/jobs/model.rs` | 419 | 作业模型：一个后台作业是什么、它怎么报告自己。 |
| `src/tools/jobs/process.rs` | 502 | 作业的进程侧：一个 pid 意味着什么、怎么停下来、怎么连整棵树一起停。 |
| `src/tools/jobs/registry.rs` | 541 | 作业表：这个进程知道的所有后台作业，以及怎么取它们的输出。 |
| `src/tools/judge.rs` | 136 | `judge` 工具：循环里的决策模型，不是挡路的闸门。 |
| `src/tools/mod.rs` | 53 | 模块声明与重导出。 |
| `src/tools/output.rs` | 52 | 一个工具产出了什么。 |
| `src/tools/pack/action.rs` | 363 | `pack` 工具：一个工具，用 `action` 参数选动作。 |
| `src/tools/pack/mod.rs` | 26 | `pack` 工具模块声明与重导出。 |
| `src/tools/pack/pending.rs` | 167 | 一次加载或卸载要等的队列，直到这一回合结束才生效。 |
| `src/tools/readers.rs` | 255 | 读取器注册表：`read` 怎么学会打开内核本身不认识的格式。 |
| `src/tools/registry.rs` | 467 | 工具表：内核可变的那张表，以及名字冲突策略（默认大声失败）。 |
| `src/tools/spill.rs` | 227 | 工具的大输出落盘，让请求里只留一个指针。 |

## 3.16 `window/` — 3 文件 460 行

上下文窗口有多大，这个数字从哪来。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/window/mod.rs` | 54 | 模块声明与重导出。 |
| `src/window/policy.rs` | 158 | 控制上下文行为的接缝（`ContextPolicy`），以及内核的默认策略。 |
| `src/window/resolve.rs` | 248 | 一个模型接受多大的窗口，以及这个数字从哪来。路由 `window.estimate`。 |

## 3.17 根目录 — 28 个单文件 10834 行

内核的其余部分：底座（`dispatch`/`catalog`）、纯数据、以及不属于上面某一组的单一功能。

| 文件 | 行数 | 用途 |
|---|---|---|
| `src/assets.rs` | 57 | 把界面文件变成响应：ETag、gzip、条件请求。 |
| `src/catalog.rs` | 215 | 内核的全部可读状态：一棵可 JSON 化的树 + 一组按 `Arc` 存活的活对象。 |
| `src/content.rs` | 489 | 应用一个包携带的内容：人格、模型路由、上下文策略、外观、注入规则。 |
| `src/dispatch.rs` | 270 | 名字 → 处理链。一切路由的底座：后注册的插到链头，`next()` 回退到内核默认实现。 |
| `src/doctor.rs` | 603 | 自检：设置、会话、技能、包、账本、价格、步数，逐项报告。 |
| `src/dylib.rs` | 994 | 动态库插件：C ABI + JSON 语义，`ABI_VERSION` 在这里。 |
| `src/events.rs` | 224 | 事件总线：`session.end` 这类事件，以及审计日志的写入。 |
| `src/export.rs` | 414 | 把一条存下的会话重述成别人能自己读的形态（Markdown / JSON）。 |
| `src/git.rs` | 322 | git 快照：动手之前记下工作区状态，可回退。 |
| `src/hotreload.rs` | 884 | 热重载：变更怎么应用、什么时候让缓存失效。 |
| `src/kernel_routes.rs` | 255 | 内核默认路由注册表：`tool.execute`、`compact.plan`、`window.estimate`、`state.*`。 |
| `src/ledger.rs` | 174 | 安装账本：谁装了什么、什么哈希，让加载幂等。 |
| `src/lib.rs` | 98 | 库入口：41 个顶层 `pub mod` 的声明，以及常用类型的重导出。 |
| `src/main.rs` | 1361 | `ngu` 二进制入口：命令行编排层。解析参数、按顺序装配内核、跑一轮对话。子命令已在 `cli/`。 |
| `src/message.rs` | 226 | 消息与角色这些纯数据类型。内核里所有东西都通过它说话。 |
| `src/mission.rs` | 100 | 任务的记录类型：目标、完成判据、当前状态。 |
| `src/model.rs` | 275 | `ModelAccess` trait：插件怎么拿到模型权限。 |
| `src/network.rs` | 319 | 网络策略：超时、重试、代理。 |
| `src/preinstall.rs` | 918 | 把随程序发布的内置包放到位。 |
| `src/pricing.rs` | 343 | 把记录下来的 token 数变成钱，费率由用户给。未定价的如实报「未定价」而不猜。 |
| `src/protocol.rs` | 1110 | 进程插件协议：插件跑成子进程时，请求和应答怎么走管道。 |
| `src/request.rs` | 134 | `RequestShaper` trait：请求体在发出前怎么整形。 |
| `src/routes.rs` | 142 | 命名路由：`provider/model` 这种名字怎么解成一个实际端点。 |
| `src/size.rs` | 100 | 文件体积的可读写法。 |
| `src/source.rs` | 135 | `PackSource` trait：包从哪来。 |
| `src/state.rs` | 151 | 发布出去的内核状态：设置、路由、人格，包可读可改。 |
| `src/theme.rs` | 198 | 主题：颜色与排版，按包提供的层叠加。 |
| `src/ui.rs` | 323 | 界面槽位：插件可以把面板放在哪。 |

---

## 四、一次对话走过的路径

```
ngu "读一下 src/pack/mod.rs"
  src/main.rs            解析参数
  cli/mission.rs         读任务条（若工作区有 mission）
  settings/              读 API key、base URL、模型（环境 > 设置文件 > 默认）
  preinstall.rs          把内置包放到位
  loader/                按 assembly.yaml 加载插件、MCP、技能
  skills/                扫描 SKILL.md 的 frontmatter
  window/                解析这个模型的上下文窗口
  agent/run.rs           跑一轮：
    context/               注入片段（人格、任务条、技能目录）
    llm/                   发请求、收流、拼工具调用
    tools/base.rs          模型请求 read → 执行
    compaction/            接近窗口上限时压缩
  session/jsonl.rs       把这一轮追加进 JSONL
```

---

## 五、拆分原则

按「**能被一个包整体替换**」切，不是按行数：

1. **一个文件一个功能**，用一句话能说清，不需要「以及」。
2. **模块内部实现私有**，对外只暴露 trait 或路由名。
3. **有些文件刻意不拆**：`plugin/core.rs` 是一个 `impl Kernel`（52 个方法共享 9 个私有字段），拆开会让「内核现在有什么」散在八个文件里；`agent/run.rs` 是 `run()` 一个函数，拆它得改函数体，不是搬家。

**没有测试。** 这是任务要求，不是疏漏。

---

## 六、风险提示

本仓库**没有经过编译验证**。143 个文件都通过 `rustfmt` 语法解析，但那只证明语法合法；
类型推导、`Send` 边界、可见性是否正确，需要 `cargo check` 才知道。
