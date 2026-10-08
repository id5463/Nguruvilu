---
name: vk
description: 操作俄罗斯社交平台 VK(VKontakte)——读取/发布动态、社群与用户信息、消息;当用户提到 VK、ВКонтакте、vk.com 或要对接 VK API 时使用
---

# VK(VKontakte)接入

两条路,按可用性选:**优先用本包 MCP 服务器提供的结构化工具**;工具不在时,
退回 `bash` 直接调 VK Open API(见文末"裸调 API")。

## 一、准备 token(一次性,用户操作)

1. 取 token(会打开浏览器完成 VK 授权):
   ```bash
   npx -y vk-mcp-server --login
   ```
   它会把 access token 打印/写好,照它说的做即可。
2. 写入用户级环境变量(Windows):
   ```bash
   setx VK_ACCESS_TOKEN "<拿到的token>"
   ```
   **然后必须重启本程序**——MCP 服务器是新进程,只有重启后才继承到这个变量。
3. 自检:
   ```bash
   npx -y vk-mcp-server --check
   ```
   它会报告 token 是否有效、以及能用哪些工具。

**token 就是密码**:只放环境变量;不写进对话、不写进文件、不出现在要贴给
别人的输出里。

## 二、工具怎么来、不来怎么办

本包(`vk`)加载时会拉起 MCP 服务器 `vk`(命令 `npx -y vk-mcp-server`)。
加载成功后,它的工具会出现在工具表里,模型可直接调用。

工具没出现时按序排查:

1. `node -v` ≥ 18、`npx -v` 可用(PATH 里要有 nodejs 目录);
2. `VK_ACCESS_TOKEN` 是否已设置、且**设置之后重启过本程序**;
3. `npx -y vk-mcp-server --check` 单独跑一遍,它会直说哪一步坏了;
4. 启动日志里找 `[pack] vk-1.0.0` 一行——`on_failure: abort` 会把失败说清楚,
   不会静默少一个工具。

## 三、裸调 API(兜底)

MCP 工具不可用、或需要它没封装的方法时,直接 HTTP:

```
GET https://api.vk.com/method/<METHOD>?v=5.199&access_token=$VK_ACCESS_TOKEN&<参数>
```

- **限速**:每 token 约 **3 req/s**。批量操作加间隔;错误码 `5`(too many
  requests)就等 1 秒重试,别立刻重发。
- **错误统一格式**:`{"error":{"code":…,"message":…}}`。
  `1117`/`permission denied` = token 缺这个权限,回第一节重新授权并勾选对应 scope;
  `1114`/`access_token invalid` = token 过期,重新 `--login`。
- **写操作先确认目标**:发帖(`wall.post`)、发消息(`messages.send`)前,
  向用户确认发到哪个社群/哪段对话,不要猜。
- 常用读方法:`users.get`、`groups.getById`、`wall.get`、`newsfeed.get`、
  `messages.getConversation`。方法总览:<https://dev.vk.com/ru/method>。

## 四、token 类型速记

- **用户 token**:读自己的信息/动态/消息,权限取决于授权时勾选的 scope。
- **社群(群组)token**:在社群管理后台生成,能替社群发帖——`wall.post` 的
  `owner_id` 为负数社群 id 时用它。
- 两类 token 都走同一个环境变量;同时需要两类时,以当前任务需要的那个为准。
