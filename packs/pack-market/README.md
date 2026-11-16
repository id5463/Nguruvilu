# pack-market — 包商城

从远程索引浏览、搜索、下载 nguruvilu 包。**只下载不安装**，装不装由你决定。

## 配置

只有一个环境变量：

```bash
export NGU_MARKET_INDEX=https://example.com/packs.json
```

或者写进包的 `assembly.yaml` 的 `env` 段：

```yaml
mcp:
- id: pack-market
  env:
    NGU_MARKET_INDEX: https://example.com/packs.json
```

没配的时候，`market_search` 会直接告诉你缺什么，不会假装成功返回空列表。

## 索引格式

```json
{
  "version": 1,
  "packs": [
    {
      "name": "skill-router",
      "version": "1.0.0",
      "description": "技能路由：按需挑出相关技能",
      "url": "https://example.com/skill-router-1.0.0.ngupack",
      "sha256": "d9be8f988b6d49076b26383fd1db34de...",
      "tags": ["tools", "skills"],
      "dependencies": { "nguruvilu": ">=0.1.0" }
    }
  ]
}
```

`sha256` 可以不写；写了就校验，不匹配会删掉下载的文件并报错——半个包比没有包更糟。

## 三个工具

| 工具 | 参数 | 返回 |
|---|---|---|
| `market_search` | `query`, `limit`（默认 10） | `[{name, version, description, url}]` |
| `market_info` | `name` | 完整条目（含 sha256、tags、依赖） |
| `market_install` | `name` | `{ok, path, sha256, matched}` |

`market_install` 下载到 `~/.nguruvilu/market/<name>-<version>.ngupack`，返回路径。
**不会自动安装**——用 `pack` 工具的 `install` 动作装它，或者自己在界面里点。

## 面板

挂在 `details-top`（设置栏上方）。一个搜索框、一个搜索按钮、结果列表，每项一个「下载」按钮。

## 为什么不当场安装

下载和安装是两件事，混在一起会让「我下错了」变成「我装错了」。

下载失败是网络问题，安装失败是包的问题，分开报错才说得清。
