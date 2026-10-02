# computer-use —— 桌面操作

给模型一个可以操作的**真实桌面**:截屏、点击、键入、读屏幕。装载后工具表里出现
一组 `mcp__cua-driver__*` 工具。

网页那条线不归这个包管:浏览器操作是独立的 [`browser-use`](../browser-use/README.md)。

## 驱动是装包时自动下载的

这个包**声明并自动携带驱动**:1.2.0 起,`mcp.json` 用 `platforms` 按平台写死了
上游 [trycua/cua](https://github.com/trycua/cua) 的发布地址与 sha256(版本
`cua-driver-rs-v0.28.0`,2026-09-11 发布),Windows / macOS / Linux 六个平台全覆盖。
装包时按 `fetch::platform_tag()`(`win-x64` `win-arm64` `linux-x64` `linux-arm64`
`mac-x64` `mac-arm64`)选中当平台那一条:

1. 下载压缩包(每平台约 27–67MB,**只下这一次**),按声明的 sha256 校验字节;
2. 解包进包目录的 `files/mcp/cua-driver/`(zip 与 tar.gz 都解,拒绝任何逃逸路径);
3. 把选中的来源与可执行文件的**绝对路径**写回,装载时直接执行
   `cua-driver(.exe) mcp`,不查 PATH。

下载落在 `~/.nguruvilu/cache/`(按内容寻址),之后的安装、重装、离线环境全部命中
缓存,**完全不再联网**。

- **首次安装需要联网**。断网首跑会打印一行 `[preinstall] ...` 并**跳过本包**
  (其它包照常放置、装载不受影响);联网后再跑一次就装上。
- macOS 装完仍要按上游指引授权(辅助功能 + 屏幕录制);Windows 解压即用。

## 手工装进 PATH 的替代写法

`platforms` 存在时装载**不查 PATH**(跑的就是解包出来的那份)。想改用自己装进
PATH 的驱动,把包里 `mcp.json` 的 `platforms` 整块去掉(保留
`"command": "cua-driver"`),重新打包安装即可 —— 此时装载按 PATH 解析 `cua-driver`。
上游手工装法见 [GitHub releases](https://github.com/trycua/cua/releases/tag/cua-driver-rs-v0.28.0):
取对应平台的压缩包,解出 `cua-driver`(`.exe`)放进 PATH,`cua-driver --version`
应输出版本号。

## 没装上(或没联网)会怎样

装载时报一条,然后**跳过 —— 不影响其它包**,每次启动都会再报一遍:

```
[pack] failed mcp:cua-driver: ...
```

不打算用桌面操作,就把它卸掉,别让每个会话都白报一次:

```bash
ngu uninstall computer-use           # 卸载:文件留着,以后可再加载
ngu uninstall computer-use --delete  # 删除:连文件一起清掉
```

## 卸载与重新加载

- **卸载**:之后的对话不再加载;已经载入的对话在回合结束时生效,不重启。
- **文件保留**:重新加载不用重装,`pack` 工具的 `load` 动作即可恢复。
- 这个包是预装的:首次运行自动放置;卸载过不会被放回来,自己改过的文件不会被
  新版本覆盖。

## 和 browser-use 的分工

| 包 | 管什么 | 驱动 |
|---|---|---|
| `computer-use`(预装) | 真实桌面:任何窗口、任何应用 | `cua-driver`(装包时自动下载) |
| `browser-use` | 网页:浏览器内的快照、点击、填表 | `chrome-devtools-mcp`(npm 包,需自装) |

1.0.0 时代这个包用的是 Playwright MCP(浏览器操作);1.1.0 起换成 cua-driver,
浏览器那条线拆去 `browser-use`;1.2.0 起驱动由装包自动获取,不再要求预装。

## 许可

本包 MIT。`cua-driver` 是上游项目,按上游许可使用;本包不含驱动的任何字节 ——
驱动在装包时从上游发布页下载,校验的是上游 `checksums.txt` 里的 sha256。
