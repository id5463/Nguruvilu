---
name: mail
description: Read and send email from two accounts — the user's own Yandex mailbox (no provider footer, the one to use for anything a person will read) and a Tencent Agent Mail alias (one-off addresses, provider stamps every message). Use when a task involves sending, reading, searching, or checking mail.
license: MIT
---

# Skill: mail

两个真实邮箱，一套工具。**发信前先确认用的哪个账号**，因为两者对收件人的观感完全不同。

## 两个账号

| 工具前缀 | 地址 | 通道 | 特点 |
|---|---|---|---|
| `yandex_*` | `l1u.shen@yandex.com` | SMTP `smtp.yandex.ru:465` + IMAP `imap.yandex.ru:993` | 本人邮箱，**无任何提供方页脚**，收件人看到的是学生本人。给学校老师发信一律用这个 |
| `agentmail_*` | `mingzi5876@agent.qq.com` | `agently-cli`（腾讯 Agent Mail） | 腾讯会在正文末尾强制追加「此邮件由 xxx 通过 Agent Mail 自动发送」+ 退订链接，**极易被判垃圾邮件**。只在需要一次性地址收验证码时用 |

凭据：`~/.nguruvilu/mail/credentials.json`（权限 600）。
- `yandex.app_password` 是 Yandex 应用专用密码（IMAP/SMTP 类型）。
- 前提：`mail.yandex.ru` → 设置 →「Почтовые программы」里的 IMAP 开关必须勾选，否则认证报 `535 5.7.8 This user does not have access rights`。

## 怎么调

工具走 MCP 代理，只有一个入口 `call`：

```json
{"action": "call", "server": "mail", "tool": "yandex_list", "args": {"folder": "INBOX", "limit": 10}}
```

**第一步先列目录**，确认工具名和参数：

```json
{"action": "list", "server": "mail"}
```

`args` 原样转发，字段名和类型按下面各工具的说明。

## 工具

### yandex
- `yandex_status` — 查 SMTP/IMAP 认证与收件箱总数。任何操作失败先跑这个。
- `yandex_send` — 发信。参数 `to` / `subject` / `text` / `html` / `cc` / `from_name`。
- `yandex_list` — 列出最新邮件（`folder`、`limit`）。默认 INBOX。
- `yandex_search` — 按 `field`（FROM / TO / SUBJECT / TEXT）检索。**查某人是否回信就用这个**，别拉全量。
- `yandex_read` — 按 `uid` 读正文。
- `yandex_folders` — 列 IMAP 文件夹（INBOX / Sent / Spam / Trash / Drafts）。

### agentmail
- `agentmail_status` / `agentmail_whoami` — OAuth 状态、别名、额度（50 封/天）。
- `agentmail_list` / `agentmail_read` — 列 / 读。
- `agentmail_send` — **两阶段**：第一次调用返回 `confirmation_token`（`ctk_…`），带上它再调一次才真正发出。

## 发信前必须做的

1. 确认账号：给真人（老师、同学、机构）发 → `yandex_send`。不要用 agentmail。
2. 收件人地址要核实过来源，不要猜。
3. 正文先写进文件给用户过目，得到明确同意再调 `yandex_send`。
4. 涉及事实（年级、组号、课程名、日期）逐条对照原始出处，别凭印象。

## 查回信的标准流程

```
yandex_search {query: "对方邮箱", field: "FROM", limit: 10}
```
`matched: 0` 就是没回。同时把 `folder` 换成 `Spam` 再查一次——有的会进垃圾箱。

## 已知坑

- Yandex IMAP 服务端偶尔在刚连上时断连，本服务内部已重试 4 次。
- Yandex 对新设备的 SMTP 首次登录可能要求「应用专用密码」，普通密码不行。
- Agent Mail 的 `+send` 必须 `--body-file` 用**相对路径**（相对 cwd），已经是绝对路径会直接报错。本服务已处理。
- Agent Mail 应用发送后即使返回 `queued: true`，也只代表进了队列，不代表投递成功；没有退信列表接口。
