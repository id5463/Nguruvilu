# starter —— 预装起点包(五合一)

一个包,带齐开局要用的五样东西 —— 这是原来五个独立预装包
(`chinese` / `ui` / `search` / `subagent` / `computer-use`)合并后的一份:

| 内容 | 文件 | 从哪来 | 它带来什么 |
|---|---|---|---|
| 人设 | `soul.md` | chinese + computer-use | 中文、简洁、如实报告;装了驱动时附**桌面操作规矩**(截图→看→动作→再确认;有后果的动作先确认) |
| 常驻注入 | `injections.json` | chinese | 两条中文相关注入 |
| 界面文案/主题 | `look.json` | chinese | 界面全部中文化 + 判官/搜索等节的标题与提示 |
| 搜索默认值 | `search.json` | search | provider/key 环境变量名等**空缺时才填**的默认值 |
| 界面 | `index.html` | ui | 三面板界面副本 —— 窗口服务的就是它;卸载则回内核内置界面 |
| 桌面驱动声明 | `mcp.json` | computer-use | `cua-driver` MCP,按平台自动下载并校验 |
| 插件 | — | subagent + search | `delegate`(子代理)与 `search_web`(联网搜索)的**启用声明** |

清单里 `"plugins"` 声明两个、`ui`/`mcp`/四个内容字段各管各的 —— 包没有"只能装一样
东西"的限制,合并只是把五份声明写进了一份 `dsh.index.json`。

## 配置(不合并、也不随包的两件事)

- **搜索密钥**:面板"联网搜索"三栏,或 `ngu config set --search-provider … --search-api-key …`,
  或 `NGU_*` 环境变量 —— **钥匙从不进包**;没配 key 就没有 `search_web` 工具;
- **判官(判断模型)**:那是**另一个包 `judge`**(预装二号),自己装、自己卸,互不影响。

## 桌面驱动:装包时自动下载

`mcp.json` 用 `platforms` 按平台写死了上游
[trycua/cua](https://github.com/trycua/cua) 的发布地址与 sha256(版本
`cua-driver-rs-v0.28.0`),Windows / macOS / Linux 六平台全覆盖。装包时选中当平台
那一条:下载(每平台约 27–67MB,**只下这一次**)→ 按 sha256 校验 → 解包进
`files/mcp/cua-driver/`(拒绝任何逃逸路径)→ 把绝对路径写回,装载时不查 PATH。

下载落在 `~/.nguruvilu/cache/`(内容寻址),之后的安装、重装、断网全部命中缓存。

- **首次安装需要联网**;断网首跑会打印一行并**跳过本包**(其它包不受影响),联网后再跑一次即可。
- macOS 装完按上游指引授权(辅助功能 + 屏幕录制);Windows 解压即用。
- 想改用自己装进 PATH 的驱动:删掉 `mcp.json` 里的 `platforms` 整块(保留
  `"command": "cua-driver"`),重新打包安装。

装载失败时报 `[pack] failed mcp:cua-driver: …`,跳过、不影响其它内容,每次启动重报 ——
不用桌面操作就把它卸掉,别让每个会话白报一次。

## 卸载与重新加载

```bash
ngu uninstall starter           # 卸载:之后的对话不再加载,文件保留
ngu uninstall starter --delete  # 删除:连文件一起清掉
ngu pack packs/starter --out packs/starter/starter-1.0.0.dshpack   # 改完重新打包
```

- **按对话生效**:已载入的对话在回合结束时生效,不重启;
- **卸载 = 五样能力一起下线**(人设、界面、注入、搜索、桌面)。想要更细的粒度,
  用各自的开关:搜索在面板里 `off`、判官是独立的包 —— 这是合并时如实说明的取舍;
- **预装三规则不变**:卸载过不复活;改过的文件绝不覆盖;幂等。

## 从五个旧包迁移(升级时自动发生)

本包取代旧的五个预装包。装到一台**已有旧五包**的机器上时,启动逻辑会:

1. 先放置 `starter` 与 `judge`;
2. 再逐个检查旧五包 —— **只有"我们放置的、且一个字节没被改过"的**才自动退役
   (删目录 + 清记录),报告一行
   `[preinstall] chinese@1.1.0 retired: merged into starter`;
3. **改过的一律保留**,并每次启动如实提示
   `[preinstall] ui@1.0.0 kept: you have edits; it may now duplicate starter's copy` ——
   你的编辑永远优先,想合并就手工搬,想清理就 `ngu uninstall <名> --delete`。

从不被本程序放置过的(你手工装的)同样保留:没记录就无法证明"未改动",不动它。

## 许可

本包 MIT。`cua-driver` 是上游项目,按上游许可使用 —— 本包不含驱动的任何字节,
驱动在装包时从上游发布页下载,校验上游 `checksums.txt` 里的 sha256。
