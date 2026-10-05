# browser-use —— 浏览器操作

给模型一个可以操作的浏览器:打开页面、读快照、点击、填表、截图。装载后工具表里
出现一组 `mcp__chrome-devtools-mcp__*` 工具。

真桌面那条线不归这个包管:桌面操作随预装的 `starter` 包提供(`mcp.json` 声明
`cua-driver`,装包时自动下载驱动)。

## 里面是什么

| 文件 | 做什么 |
|---|---|
| `mcp.json` | 声明一个 MCP 服务器:`chrome-devtools-mcp --headless` |
| `soul.md` | 告诉模型怎么用它、边界在哪 |

**零代码。** 包里没有一个字节的程序。

## 为什么是 chrome-devtools-mcp

- **小、零依赖**:13.6 MB 的 npm 包,没有任何依赖 —— 装完就是它自己。
- **固定版本**:装法是 `npm install -g chrome-devtools-mcp@1.10.1`,`mcp.json`
  里写的是装好的命令,不是 `npx …@latest`。`@latest` 每次启动都要问一遍 npm
  registry —— 冷启动十几秒,还可能某天突然给你一个没测过的版本;固定版本把这件事
  挪到安装时决定,启动时不再联网。
- **用本机 Chrome**:Google 官方维护的 Chrome DevTools 集成,走 DevTools 协议,
  快照是可交互元素的结构化文本。

## 装

```bash
npm install -g chrome-devtools-mcp@1.10.1
```

- 需要 Node(`^20.19.0 || ^22.12.0 || >=23`);
- 需要本机有 Chromium 系浏览器(Chrome / Edge)。自动发现不了时,给服务器的
  args 加 `--executablePath <浏览器可执行文件路径>` 指定。

## 首回合要等浏览器起来

装载时浏览器还没起,第一次调用要等它启动完成,比后续轮次慢;之后整个会话复用
同一个浏览器,不用每次重开。

## 没装会怎样

装载时报一条,然后跳过,不影响其它包,每次启动都会再报一遍:

```
[pack] failed mcp:chrome-devtools-mcp: launching MCP server 'chrome-devtools-mcp' (chrome-devtools-mcp): ...
```

不打算用浏览器操作,就把它卸掉:

```bash
ngu uninstall browser-use           # 卸载:文件留着,以后可再加载
ngu uninstall browser-use --delete  # 删除:连文件一起清掉
```

## 和桌面操作的分工

| 包 | 管什么 | 驱动 |
|---|---|---|
| `browser-use`(本包) | 网页:快照读结构、点、填、截图 | `chrome-devtools-mcp`(npm 包) |
| `starter` 的桌面部分(预装) | 真实桌面:任何窗口、任何应用 | `cua-driver`(上游二进制) |

网页用 `browser-use` —— 结构化文本,快且省;要动浏览器之外的东西,才用
桌面那条线(`starter` 的 `mcp__cua-driver__*`)。

## 许可

本包 MIT。`chrome-devtools-mcp` 由 Chrome DevTools 团队维护,Apache-2.0,按上游
许可使用;本包不含它的任何字节。
