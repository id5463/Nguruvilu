# computer-use —— 桌面操作

给模型一个可以操作的**真实桌面**:截屏、点击、键入、读屏幕。装载后工具表里出现
一组 `mcp__cua-driver__*` 工具。

网页那条线不归这个包管:浏览器操作是独立的 [`browser-use`](../browser-use/README.md)。

## 必须先装驱动

这个包**只声明、不携带驱动**。MCP 服务器是 PATH 上已装好的 `cua-driver`,装载时
执行 `cua-driver mcp`。驱动是上游 [trycua/cua](https://github.com/trycua/cua) 的
Rust 二进制,本包按版本 `cua-driver-rs-v0.28.0`(2026-09-11 发布)写成,Windows /
macOS / Linux 都有 zip。

三种装法(上游指引:[Install Cua Driver](https://cua.ai/docs/how-to-guides/driver/install)):

1. **macOS / Linux**:`install.sh` 一行安装脚本 —— `curl` 拉下来交给 bash 执行,
   地址见上游指引;
2. **Windows(PowerShell)**:`install.ps1` 一行安装脚本 —— `irm <地址> | iex`,
   地址同样见上游指引;
3. **手动**:到 [GitHub releases](https://github.com/trycua/cua/releases) 取 tag
   `cua-driver-rs-v0.28.0` 的对应平台 zip,解压出 `cua-driver`(`.exe`)放进 PATH。

装完在终端跑 `cua-driver --version` 应能输出版本号。macOS 还要按上游指引授权
(辅助功能 + 屏幕录制);Windows 装完即可用。

## 没装驱动会怎样

装载时报一条,然后**跳过 —— 不影响其它包**,每次启动都会再报一遍:

```
[pack] failed mcp:cua-driver: launching MCP server 'cua-driver' (cua-driver): ...
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
| `computer-use`(预装) | 真实桌面:任何窗口、任何应用 | `cua-driver`(上游二进制,需自装) |
| `browser-use` | 网页:浏览器内的快照、点击、填表 | `chrome-devtools-mcp`(npm 包,需自装) |

1.0.0 时代这个包用的是 Playwright MCP(浏览器操作);1.1.0 起换成 cua-driver,
浏览器那条线拆去 `browser-use`。

## 许可

本包 MIT。`cua-driver` 是上游项目,按上游许可使用;本包不含驱动的任何字节。
