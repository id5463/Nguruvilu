# vk 包 —— 接入 VK(VKontakte)

给会话装上俄罗斯社交平台 VK 的接入能力,两条腿:

- **MCP 服务器**([vk-mcp-server](https://github.com/bulatko/vk-mcp-server),
  经 `npx -y vk-mcp-server` 拉起):结构化工具直接进工具表;
- **技能** `skills/vk/SKILL.md`:token 一次性准备(`--login` → `setx` → 重启)、
  排查手册、以及裸调 VK Open API 的兜底(端点/限速/错误码/token 类型)。

## 内容文件

| 文件 | 作用 |
|---|---|
| `dsh.index.json` | 清单:引用 `mcp.json` + 内联技能 `skills/vk` |
| `mcp.json` | 服务器 `vk-api`:stdio,`npx -y vk-mcp-server`,scope session |
| `skills/vk/SKILL.md` | 使用与排查技能(按需加载,只占一行目录) |

**token 不进包**:`mcp.json` 不写任何密钥,服务器从环境继承
`VK_ACCESS_TOKEN`——和 `search.json` 只写 `apiKeyEnv` 名字是同一条规则。

## 安装

- UI:「安装…」选本目录打好的 `.dshpack`;或
- 直接把本目录放进用户包目录为 `vk-1.0.0/`(安装版会由 install 渲染
  `assembly.yaml`;离线直放需自带,见仓库存档)。

首次使用按技能 `vk` 里的三步取 token;取完重启程序,工具即出现。
