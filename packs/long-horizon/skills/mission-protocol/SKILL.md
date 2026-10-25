---
name: mission-protocol
description: >
  长程任务协议:任务不是一段对话,是磁盘上的状态。目标/完成判据/下一步写进
  mission,日志只追加,细节靠 recall 按行检索而不是重读历史;每到一个里程碑
  打检查点;每推进一步在全新 worker 里跑(tick);代理之间用信箱通信。
  给超过上下文窗口几十上百倍、跨天跨周的任务用。
---

# mission-protocol —— 长程任务协议

一段对话是**工作集**,一个任务**住在磁盘上**。记住这条,下面全是它的推论。

## 什么时候开任务

**判据**:这件事的规模超过一个上下文窗口,或者你会跨天回来接着做。

- 一次问答、改一个 bug、读一篇文档 → **不要开任务**,直接干。
- "把这个仓库迁到新框架""把这批课程资料整理完""跑一个月的采集并持续改进"
  → **开任务**:`mission_start(objective, done_when, first_step)`。

`done_when` 一定要**可检查**:"测试全绿""`out/report.md` 存在且含三张表"
远好于"做得足够好"。一个判不了的完成判据等于没有。

## 状态写在哪里

```
<workspace>/missions/<id>/
  mission.json    结构化状态:status / step / advances / ticks / checkpoints / agents
  STATE.md        给人看和给模型看的叙述:目标、完成判据、决策、阻塞
  log.jsonl       只追加的事实:每次推进、检查点、信件的产出
  checkpoints/    里程碑快照(当时的 STATE.md + 任务记录)
  mailbox/        代理信箱,每人一个 jsonl
```

`STATE.md` 里 **`<!-- mission:generated -->` 到 `<!-- /mission:generated -->`
之间是自动维护的,别手改**;围栏之外(决策记录、阻塞、备注)是你的地盘,任何
工具都不会覆盖它。要改当前状态,用 `mission_advance`,不要手工编辑那段。

## 每一步的节奏

1. **开工前**:`mission_status`(几十行的报告,不是读历史)。
2. **推进一步**:你自己干——用你手上的工具(bash / read / write / web_search …)
   把当前这一步做掉。**这个包里没有"跑一步"的工具,也没有后台 worker**:
   任务不会自己动,动它的是你在这个回合里的动作。执行中发现 `next step` 写得
   不对,就按实际情况做,然后在 advance 时把 `next` 改写成真实的下一步。
3. **干完立刻记账**:`mission_advance(summary, next, status?)`。
   `next` 要写成下一个接手者能直接开工的样子;任务 running 期间这一回合不会
   因为给出答复而结束,所以你记完账就接着做下一步,不需要问用户要不要继续。
4. **里程碑打卡**:`mission_checkpoint(label)` —— 回滚点和交接点都在这。
5. **卡住了**:`mission_advance(status="blocked", summary=...)` 说明卡在哪、
   需要什么;然后停手或换路子,别在原地空转。

## 记忆怎么用

**不要**把历史读回上下文。用 `recall(query)` 搜:

- `recall("zanzibar token")` → 命中行 + 文件 + 行号,读那几行就够了;
- 记不清上次的决定 → `recall("决定")` 或搜关键词;
- 想知道某个文件怎么来的 → 搜文件名。

上下文只装**现在这一步需要的东西**;其余都在磁盘上,随用随取。这是长任务
不崩、不贵、不漂移的根本原因。

## 多代理与横向通信

- 每个代理由一个**路径**命名:`root`、`worker-crawler`、`reviewer-1`。
- `agent_post(to, body)` 投信:**信箱落盘**,收件人不在跑也收得到,它在
  **自己下一次开工时**用 `agent_inbox` 读到。
- 投信要短、要具体:"schema 在 db.rs,别动 migrations"远好于"同步一下进度"。
- 收信后先判断:是信息就用,是请求就做,是警告就改路子。**回信要用 `from`
  写清自己是谁**,否则对方不知道谁在说话。
- 横向聊天没有限制,但每一次通信都是别人上下文里的成本:值得说才说。

## 交接与重启

- 进程会死、上下文会满、人会走开。任务不会因为这些丢:所有状态在磁盘上,
  `mission_status` 随时能读出"到哪了、下一步是什么"。
- 换一个会话继续做时:先 `mission_status`,再 `recall` 补需要的细节,
  然后接着 tick。**不要**要求把上一个会话的历史搬过来。
- 交付给别人(人或另一个代理)时:`mission_checkpoint("handoff")` +
  在 STATE.md 的"备注"里写清下一步的入口,这两样就是完整的交接包。
