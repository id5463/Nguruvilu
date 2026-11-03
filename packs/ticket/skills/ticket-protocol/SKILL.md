---
name: ticket-protocol
description: >
  主代理操作协议：目标冻成票（tickets/<id>/ticket.json），任务书写进 brief.md，
  worker 是内核后台作业里起的 ngu 子进程（结束用 notify 叫醒你），完成判定是
  worker 自己写的 report.json。派活、回话、追加预算、关闭都由 ticket_* 工具出参数，
  进程与叫醒归内核作业。一条对话只管自己开出的票；状态从盘上事实推，不靠记忆。
---

# ticket-protocol —— 主代理怎么干活

一条对话是工作集，票是磁盘上的状态。你忘了不要紧，盘上没忘。

## 一、目录

```
tickets/<id>/ticket.json   票本体（goal 原文、预算、worker、result、state）
tickets/<id>/brief.md      交给 worker 的任务书（唯一输入）
tickets/<id>/report.json   worker 自己写的交付声明（这就是完成判定）
tickets/<id>/events.jsonl  一行一条事实，事后唯一能查的东西
```

状态只有五个，由盘上事实推出来，不用你去声明也不用你维护：
`queued`（没派活）→ `running`（有 job）→ `delivered`（report 合法）/ `stuck`（作业结束但没有合法 report，或 worker 回了 blocked）→ `closed`（你或主人关掉）。

## 二、动作（就这些，按顺序）

| 想做 | 怎么做 |
|---|---|
| 记录一个目标 | `ticket_open(workspace, goal=主人原话, done_when?, boundary?)` |
| 现在什么情况 | `ticket_board(workspace)` —— 交付前先读它，别靠回忆 |
| 派活 | `ticket_start_payload(workspace, id)` → 把返回的参数**原样交给 bash 工具**（里面的可执行路径由宿主给出，不要自己去 `which ngu`）→ 拿到 job id 后 `ticket_start(workspace, id, job, session?)`；`--json` 的输出里有 session，记进去，后面回话要用 |
| 收结果 | 被作业结束叫醒后 `ticket_deliver(workspace, id)`；有合法 report 就是 delivered |
| 追加预算 | `ticket_grant(workspace, id, tool_timeout_secs?, max_steps?, wall_secs?, profile?)`；它重写 brief，对**下一个** worker 生效 |
| 跟 worker 说话 | `ticket_say_payload(workspace, id, text)` → 同样交给 bash 起一轮（复用它的 session）。**这是你唯一的审查手段**，细节看 `ticket-review` |
| 关掉 | `ticket_stop(workspace, id, reason)`；进程用内核对那个 job 的 `job_kill` |

三条约定：`notify:true` 意味着不要轮询、不要 sleep 等作业；工作目录每次都要传（MCP 进程看到的是包目录）；票里的 `origin` 只是"这话从哪来"的事实标注。

## 三、几条不要做

- 不要把同一目标开第二条票，除非你真的要再跑一遍（`ticket_open` 会把已有的活票附在回答里，判由你下）。
- 不要在 `text` 里代替 worker 写答案，也不要在里面讲时限和规则：时限走 `ticket_grant`，停止走 `job_kill`/`ticket_stop`。把话留在该在的通道里，worker 的历史才不会被你的元话语污染。
- 不要自己下场干活。你觉得"这我两行就改完了"的时候，正确动作是给 worker 说一句话，或者改这张票的任务书。
