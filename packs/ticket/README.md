# ticket —— 长程模式包（主代理 = 这条对话）

把"这条对话"变成主代理：它不动手干活，只做三件事——**把目标变成票**、**给票配资源**、
**跟干活的子代理对话并验收**。零内核改动、零编译，随时可装卸。

## 这个包是怎么设计的

一条对话是工作集，票是磁盘上的状态。四条来自实测的判断撑起全部结构：

| 观察到的事实 | 因此的设计 |
|---|---|
| 一条 76 小时的会话攒了 9,459 条消息、每步平均 39 万 input token | **一条对话只管自己开出的票**；跨对话的事实靠盘上的票目录，不靠聊天记忆 |
| 卸载包之后模型还在重复包里的话；那些话进了它自己的发言历史 | **时限与停止一律走机制**（`grant` / `job_kill` / `report.json`），话只用来讲任务内容 |
| 长任务在 15 分钟的整步里消失，没人知道它卡在哪 | worker 是**内核后台作业**（`background:true` + `notify:true`），结束会叫醒主代理 |
| "他说做完了"和"真做完了"之间没有东西 | 完成判定 = worker 自己写的 `report.json`；审查 = 主代理**问它三件事**，不是打分表 |

**没有权限层**：不审查措辞、不判断结论"够不够硬"、不因可能重复而拒绝。规则由工具签名表达
——想改时间只有 `ticket_grant`，想停只有内核的 `job_kill`，想交付只有 worker 写 `report.json`。

## 内容

```
ds.index.json      清单
soul.md             主代理的角色（几行，不复述规则）
mcp.json            把 files/ticket-mcp.mjs 作为 stdio MCP 服务器带来
files/ticket-mcp.mjs  票的状态机与参数生成（代码，Node，无依赖）
look.json           票看板面板（above-composer，从工具结果里的标记取数）
skills/ticket-protocol  开票 / 派活 / 收回 / grant / 回话 / 关闭
skills/ticket-review    审查就是对话：三问 + 三种出路
```

八个工具（前缀 `mcp__ticket__`）：

| 工具 | 作用 |
|---|---|
| `ticket_open` | 目标原文冻结成票 + 生成 `brief.md`；同目标已有活票就**照开并附一行提示** |
| `ticket_board` | 从盘上扫出所有票：状态、已跑多久、job、最后声明 |
| `ticket_start_payload` | 返回派活该用的 `bash` 参数（`background`/`notify` 都带好） |
| `ticket_start` | 记账：job id + worker 的 session id |
| `ticket_deliver` | 读 worker 的 `report.json`，状态由它决定（写了就 delivered；缺字段或 blocked 就 stuck） |
| `ticket_grant` | 追加预算并重写 `brief.md`，所以对**下一个** worker 生效 |
| `ticket_say_payload` | 生成对 worker 说话用的参数（复用它的 session）——验收的唯一手段 |
| `ticket_stop` | 关闭 + 记原因；进程请用 `job_kill` |

## 为什么 worker 不由这个包起

MCP 子进程自己 spawn 的进程不是内核作业：结束不会叫醒会话、`job_kill` 管不到、日志不进
`job_output`。所以这个包只**算参数**，起进程的是内核。这条分工是"结束能被叫醒"的前提。

## 已知摩擦（诚实记录）

- 每个工具都要传 `workspace`（会话工作区的绝对路径）。因为 MCP 子进程的 cwd 是**包目录**，
  不是会话工作区；主代理第一步用 `bash pwd` 拿到它即可。根治办法是宿主把会话工作区也注入
  MCP 子进程环境——本包刻意不靠这个。
- worker 的可执行路径由宿主经 `NGU_EXE` 传给 MCP 子进程（内核 `loader.rs` 负责），所以
  `ngu` 不在 PATH 上也能起 worker。实测过：没有这个变量时，主代理会花六七步在文件系统里
  找 `ngu.exe`，然后超时。要指定别的构建就给 `ticket_start_payload` 传 `exe`。
- 预算改动对正在运行的 worker 无效（内核在进程启动时读设置），`ticket_grant` 会重写成
  `brief.md` 让下一轮看得到。

## 本地门禁

```sh
node scripts/ticket-mcp-test.mjs   # 状态机：8 个工具全跑到，含 id 唯一性与非法收回
node scripts/ticket-e2e.mjs        # 真实一次：开票 → 派活 → 叫醒 → 回话自查（会花 token）
```

内核侧配套的两处修复由 Rust 测试覆盖：`loader` 的包内脚本 cwd 判定（4 项）、
`content` 读真包 `look.json` 的面板落地（1 项）、`agent` 的开头 system 合成一条（含 45 项 agent 测试）。
