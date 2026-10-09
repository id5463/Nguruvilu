---
name: vk
description: 操作俄罗斯社交平台 VK(VKontakte)——读公开主页/墙/帖子/照片,配合社群 token 可发帖发消息;当用户提到 VK、ВКонтакте、vk.com 或要对接 VK API 时使用
---

# VK(VKontakte)接入

本机**已配置**:`VK_SERVICE_KEY` 已写入用户环境(应用 54812191 «ngu vk bridge» 的
服务密钥),MCP 服务器 `vk-api` 随本包自动拉起。先看工具表里 `mcp__vk-api__*`
在不在,在就直接用工具;不在见文末排查。

## 一、当前能力(2026-10 实测)

| 能做什么 | 靠什么 |
|---|---|
| 读公开主页/社群资料、读墙、按 id 读帖子、读照片 | ✅ 服务密钥(已配) |
| 发帖、评论、发故事、社群消息、社群管理 | ❌ 需要**社群 token**(`VK_ACCESS_TOKEN`) |
| 点赞/搜索/成员列表/资讯流/统计 | ❌ VK 不再授给新应用,任何 token 都拿不到(error 1051/28/100) |

服务器的组合规则:调用先用 `VK_ACCESS_TOKEN`(社群 token),被拒的**读**再用
`VK_SERVICE_KEY` 重试;写永远只走社群 token。所以只要把社群 token 补上,
读写就都齐了——目前只有读。

**不要用 `npx vk-mcp-server --login`**:它拿到的 VK ID token 只有公开读权限,
和服务密钥等价但多一道 OAuth,写操作一样是1051。

## 二、补社群 token(需要发帖时,一次性,约三下点击)

1. 打开你管理的社群 → 右侧菜单 **管理(Управление)**;
2. 找 **API 使用**(新版可能在 **高级/Advanced** 下)→ **访问令牌** → **创建令牌**;
3. 勾选 `wall`、`photos`、`stories`、`messages`、`manage` → 确认,复制令牌;
4. `setx VK_ACCESS_TOKEN "<令牌>"`,**重启本程序**。
   令牌永不过期、不绑机器。完成后 `npx -y vk-mcp-server --check` 应显示
   Community token。

> 确认码防不住:创建/查看密钥时 VK 会向手机推送确认码。用 ADB 读法:
> `adb shell dumpsys notification --noredact | findstr "это код"` 直接取码。

## 三、排查

1. `node -v` ≥18、`npx -v` 可用;
2. 环境变量是否在**本程序启动之前**已设置(启动后 setx 不生效,必须重启);
3. `npx -y vk-mcp-server --check` —— 它会说清是哪类 token、缺哪项权限;
4. 启动日志找 `[pack] vk-1.0.0`;`on_failure: abort` 会把装载失败说清楚。

## 四、裸调 API(兜底)

```
GET https://api.vk.com/method/<METHOD>?v=5.199&access_token=<token>&<参数>
```

- 限速约 3 req/s;错误码 5 = 太频繁,等 1 秒重试;
- 响应错误统一 `{"error":{"code","message"}}`:1117 权限不足、1051/28 token
  类型不允许、27 社群 token 被拒的读(走服务密钥)、15 数据受限;
- 写操作(发帖/发消息)前先向用户确认目标社群/对话;
- 方法总览:<https://dev.vk.com/ru/method>
