---
name: vpn
description: 管理本机 mihomo(Clash Meta)代理节点——查当前走哪个节点、测每个节点的真实延迟、切换节点。当用户问"现在走哪个节点/哪个节点快/换一个节点/VPN 能用吗/代理通不通"时使用。
---

# VPN 节点管理（vpn MCP）

管理的是**本机正在运行的 mihomo**，通过它的 REST API（`external-controller`，
默认 `127.0.0.1:9090`）读写。不修改配置文件：选节点是运行时决定，mihomo 记在内存里。

## 工具

1. **vpn_status** — 第一个该调的。回答"现在走哪个节点"、有哪些分组、mihomo 是否可达。
2. **vpn_test** — **最常用的一个**。测每个节点的真实延迟并按快慢排序。
   一次请求一个节点，4 个节点约 3–5 秒。
3. **vpn_nodes** — 列节点。`probe=false`（默认）是廉价清单，只读 API 不测速。
4. **vpn_switch** — 把分组指向某个节点。**只有 `select` 类型的分组能被手动指定**；
   `url-test` 分组自己按延迟挑，指定会报错并说明原因。
5. **vpn_groups** — 看分组结构（类型、当前选中、成员），决定切换的目标。

## 常见问法与对应动作

| 用户说 | 用哪个 |
|---|---|
| "现在走哪个节点？" | `vpn_status` |
| "哪个节点快 / 测一下" | `vpn_test` |
| "换到日本节点" | 先 `vpn_nodes` 拿准确名字，再 `vpn_switch` |
| "代理通不通 / 网络是不是断了" | `vpn_status` 看 reachable，再 `vpn_test` |
| "有几个节点" | `vpn_nodes`（不带 probe，快） |

## 必须注意的三件事

1. **节点名是带 emoji 和竖线的完整字符串**（例如 `🇯🇵日本 01 | 高级专线`）。
   `vpn_switch` 要求**精确匹配**，所以先 `vpn_nodes` 拿名字，不要自己拼。
2. **`url-test` 分组切不了**。这类分组按延迟自动选，手动指定会被拒绝并告诉你原因。
   如果你的分组是 url-test 而又想手动控制，需要在 mihomo 配置里把 `type: url-test`
   改成 `type: select`。**不要假装切换成功了** —— 工具会返回真实结果，照实说。
3. **`vpn_switch` 改的是运行时状态，不写盘**。mihomo 重启会回到配置里的默认。
   用户如果要持久，得改配置文件，那不是这些工具能做的，直说。

## 环境变量

- `MIHOMO_CONTROLLER` — API 地址，默认 `http://127.0.0.1:9090`
- `MIHOMO_SECRET` — 配置里 `secret` 的值，没设就不用填
- `MIHOMO_PROBE_URL` — 测速目标，默认 `http://www.gstatic.com/generate_204`

## 出问题时

- **`cannot reach mihomo`** — 进程没跑，或配置里没开 `external-controller`。
  这不是节点坏了，是管理面不通，回答时要说清区别。
- **`delay` 为 null 或报错** — 该节点确实不可用（超时/握手失败），不是工具坏了。
- 所有工具返回 JSON，`delay_ms` 越小越快；排前面的是当前可用的最优选择。
