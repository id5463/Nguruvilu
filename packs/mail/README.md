# mail

两个真实邮箱装进同一个 MCP server，零第三方依赖（只用标准库）。

| 邮箱 | 地址 | 通道 |
|---|---|---|
| Yandex | `l1u.shen@yandex.com` | 直连 SMTP 465 / IMAP 993，无提供方页脚 |
| Agent Mail | `mingzi5876@agent.qq.com` | 调 `agently-cli`（腾讯 OAuth 设备码） |

## 凭据

`~/.nguruvilu/mail/credentials.json`（600）：

```json
{
 "yandex": {
  "email": "l1u.shen@yandex.com",
  "app_password": "16 位应用专用密码",
  "smtp_host": "smtp.yandex.ru", "smtp_port": 465,
  "imap_host": "imap.yandex.ru", "imap_port": 993
 },
 "agentmail": {"cli": "/home/ubuntu/.npm-global/bin/agently-cli"}
}
```

也支持环境变量 `MAIL_YANDEX_EMAIL` / `MAIL_YANDEX_PASSWORD` 覆盖。
若 credentials.json 里没有 `app_password`，会回退读 `~/.config/mailer/yandex_app_pw`（历史位置）。

## 前置条件

Yandex 侧必须先在 `mail.yandex.ru` → 设置 →「Почтовые программы」勾选 IMAP 访问，
否则 SMTP 认证报 `535 5.7.8 This user does not have access rights to this service`。

## 工具

| 工具 | 作用 |
|---|---|
| `yandex_status` | SMTP/IMAP 连通性 + 收件箱总数 |
| `yandex_send` | 发信 |
| `yandex_list` / `yandex_search` / `yandex_read` | 列 / 检索 / 读正文 |
| `yandex_folders` | 列文件夹 |
| `agentmail_status` / `agentmail_whoami` | OAuth 状态、别名、额度 |
| `agentmail_list` / `agentmail_read` / `agentmail_send` | 列 / 读 / 发（两阶段确认） |

## 安装

```bash
/home/ubuntu/Nguruvilu/target/release/ngu apply ~/.nguruvilu/packs/mail-1.0.0/assembly.yaml
# 或打成一个可迁移的包
/home/ubuntu/Nguruvilu/target/release/ngu pack ~/.nguruvilu/packs/mail-1.0.0 -o mail-1.0.0.ngupack
```
