# moodle 包 —— 接入 Herzen 大学 Moodle

给会话装上学校 Moodle(moodle.herzen.spb.ru)的采集与提交能力,两条腿:

- **MCP 服务器**(`scripts/moodle_mcp.py`,Python + requests,stdio JSON-RPC,零第三方 MCP 依赖):
  课程/作业爬取与增量 diff、作业详情、附件下载、作业提交、登录态检查;
- **技能** `skills/moodle/SKILL.md`:工具选择与推荐用法、diff 语义、提交保持 dry_run 的红线。

## 内容文件

| 文件 | 作用 |
|---|---|
| `dsh.index.json` | 清单:引用 `mcp.json` + 内联脚本 + 技能 `skills/moodle` |
| `mcp.json` | 服务器 `moodle-herzen`:stdio,`python <pack>/scripts/moodle_mcp.py`,scope session |
| `scripts/moodle_mcp.py` | MCP 服务器本体 |
| `skills/moodle/SKILL.md` | 使用技能(按需加载,只占一行目录) |

**凭据不进包**:账号密码放 `~/.nguruvilu/moodle_auth.json`(`{"username","password"}`),
会话缓存 `moodle_session.json`,课程/作业缓存 `moodle_cache.json`——都在包外,服务器自动登录与续期。

**路径注意**:`mcp.json` 的 args 写的是安装后的绝对路径(mcp args 不经 pack 目录解析,原样进 spawn)。
换机器时把该行改成新机器的 `~/.nguruvilu/packs/moodle-<版本>/scripts/moodle_mcp.py` 实际位置。
