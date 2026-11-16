---
name: ticket-protocol
description: >-
  Main-agent operating protocol: freeze the goal into a ticket (tickets/<id>/ticket.json), write the
  task brief into brief.md, and let the worker be an ngu child process started by the kernel job
  system, woken by notify when it ends; completion is the report.json the worker writes itself.
  Two ways to dispatch: ticket_start_payload starts a fresh worker that knows nothing, and
  ticket_resume_payload continues the same session that remembers what it already tried — which one
  is better is your call, not the server's. Dispatch, resume, reply, grant more budget and close are
  argument generators on ticket_* tools; the process and the wake-up belong to the kernel job system.
  One conversation owns only the tickets it opened; state is derived from facts on disk, never from
  memory. Use when delegating work that outlives this turn.
---

# ticket-protocol —— 主代理怎么干活

一条对话是工作集，票是磁盘上的状态。你忘了不要紧，盘上没忘。

## 一、目录

```
tickets/<id>/ticket.json   票本体（goal 原文、前情、预算、worker、result、state、rounds）
tickets/<id>/brief.md      交给 worker 的任务书（唯一输入）
tickets/<id>/report.json   worker 自己写的交付声明（这就是完成判定）
tickets/<id>/events.jsonl  一行一条事实，事后唯一能查的东西
```

状态由盘上事实推出来，不用你声明也不用你维护：

```
queued      没派过活
running     有 job 在跑
delivered   report 合法且 state=done
partial     report 合法且 state=partial（做了一半，剩下的写清楚了）
stuck       作业结束了但没有合法 report，或 worker 回了 blocked
closed      你或主人关掉
```

## 二、动作

| 想做 | 怎么做 |
|---|---|
| 记录一个目标 | `ticket_open(workspace, goal=主人原话, done_when?, boundary?, tried?, files?, never_touch?)` |
| 现在什么情况 | `ticket_board(workspace)` —— 交付前先读它，别靠回忆 |
| **新起一个 worker** | `ticket_start_payload(workspace, id)` → 返回的 bash 参数**原样交给 bash 工具** → `ticket_start(workspace, id, job, session)` |
| **续跑同一个 worker** | `ticket_resume_payload(workspace, id, text?)` → 同样交给 bash。它复用 session，记得上一轮读过什么、试过什么 |
| 收结果 | 被作业结束叫醒后 `ticket_deliver(workspace, id)` |
| 追加预算 | `ticket_grant(workspace, id, tool_timeout_secs?, max_steps?, wall_secs?, profile?, tried?, files?)`；重写 brief，对**下一个** worker 生效 |
| 跟 worker 说话 | `ticket_say_payload(workspace, id, text)` —— 追问、要求补证据。**这是你唯一的审查手段** |
| 关掉 | `ticket_stop(workspace, id, reason)`；进程用内核对那个 job 的 `job_kill` |

## 三、新起还是续跑

这是每一次派活都要做的判断，服务器不替你做。

**新起（`ticket_start_payload`）**

- 全新上下文，只读 brief.md
- 干净：主代理之前的试错不会传给它
- 盲：它不知道上一轮发生了什么，会重复踩过的坑
- 适合：第一次派活；任务独立；上一轮的上下文已经乱了

**续跑（`ticket_resume_payload`）**

- 复用 session，它记得读过哪些文件、试过什么、为什么失败
- 省：不用重新探索
- 脏：上一轮的误解会一直带着，越滚越偏
- 适合：做了一半接着做；worker 说 blocked 而你想换个问法；只是要它补个证据

**几个具体场景：**

```
worker 交了 partial，剩下一半很清楚   → 续跑，让它接着干
worker 交了 partial，但它理解错了方向 → 新起，brief 里把方向写明白
worker 说 blocked 缺 API key          → 先补 key 再续跑（它记得自己卡在哪）
worker 说 blocked 但它只是不会绕      → 续跑 + 换个问法
worker 交了 done 但证据是 assumed     → say_payload 要求它补 verified
连续两轮空转                          → 新起，八成是 brief 写得不对
```

**为什么不做自动续跑**：Codex 的 `/goal` 会在目标 Active 时自动起下一轮，因为它是一个长驻线程。你的 worker 是独立进程，`ticket_resume_payload` 就是你的续跑——而且它多了一道**你的判断**。那道判断不该省掉：Codex 自己的源码里也要靠「连续三轮无活动就停」来兜底。

## 四、填好任务书的前情

`ticket_open` 有三个可选参数，它们决定 worker 会不会重复你的劳动：

```
tried        你已经试过什么、为什么没成
files        相关文件的当前状态，带行号最好
never_touch  具体到文件和行，这次别动什么
```

不填也照开票，但 `brief.md` 里会明说「主代理没有记录」——worker 有权知道自己可能会重复你的坑。

**填得具体一点：**

```
tried: ["读了 src/main.rs 的 6 处引用（949/950/1140/1141/3789/3796）",
        "试过直接删 Settings.search 字段 → 3 处编译错"]
files: ["src/tools/mod.rs:27  pub mod search; 要删",
        "src/pack.rs:109      (\"search\",\"search.json\") 要删"]
```

**这比多给 3000 步预算有用。** worker 省下的不是时间，是它本来会犯的错。

`ticket_grant` 也能补 `tried`/`files`，而且是**追加**不是覆盖——第二次 grant 只想补一条时，第一次的不会被清掉。

## 五、读交付

`ticket_deliver` 会按证据等级给提示：

```
delivered  ← report.state = done
  产物：commit:8d0e38c, file:src/tools/search.rs:已删除
  verified：cargo check (exit 0) ; cargo test --lib (exit 0)
  证据：⚠ 有 1 条是推测，需要你自己确认

partial    ← report.state = partial
  还剩：pack.rs 那行还没删
  接着干用 ticket_resume_payload

stuck      ← report.state = blocked，或 report 缺失/坏
  worker 交回一个决定：给我 TAVILY_API_KEY
```

三级证据的含义：

| 级别 | 含义 | 你要做什么 |
|---|---|---|
| `verified` | 它跑过命令，有退出码 | 可以采信；重要的话自己复跑一次 |
| `observed` | 它看见了，但没跑 | 看一眼那个东西还在不在 |
| `assumed` | 它推断的 | **必须自己确认**，这是最会骗人的一级 |
| 旧格式 `evidence` | 一段自述 | 当 assumed 处理 |

## 六、几条不要做

- 不要把同一目标开第二条票，除非你真的要再跑一遍（`ticket_open` 会把已有的活票附在回答里，判由你下）。
- 不要在 `text` 里代替 worker 写答案，也不要在里面讲时限和规则：时限走 `ticket_grant`，停止走 `job_kill`/`ticket_stop`。
- 不要自己下场干活。你觉得「这我两行就改完了」的时候，正确动作是给 worker 说一句话，或者改这张票的任务书。
- 不要连续轮询 `ticket_board` 等 worker。`notify:true` 意味着作业结束会叫你；等待期间去做别的事。
- 不要忽略「连续 N 轮空转」的提示。那多半不是 worker 笨，是任务书没写清。
