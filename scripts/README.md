# 服务器运维脚本

两个脚本，都**不自动安装**，也**不会自己动服务器**。你想用的时候再装。

写好它们的原因都来自这台机器上真实发生过的事，不是预设。

---

## 1. `archive_sessions.py` —— 会话目录自动归档

### 为什么需要

`JsonlStore::list()` 会遍历 `sessions/` 下**每一个** `*.jsonl` 来生成会话列表。
这份列表在**每次重连、每次切会话、每次新建会话**时都会重建。

实测（同一台机器、同一份数据，357 个会话 / 263 MB）：

```
旧二进制  1102 ms
新二进制   199 ms      ← 修掉全量反序列化之后
```

但那个修复只解决了"每次读得慢"，没解决"文件越攒越多"。有一次我手动把 332 个
文件移进 `archive/`，`list()` 从 0.36 s 降到 0.07 s。**那是手工做的，没有任何
机制防止它再长回去。**

### 它做什么

- 把 `sessions/` 里 **7 天没动过** 的 `*.jsonl` 移进 `sessions/archive/`
- 把归档里 **30 天以上** 的 `*.jsonl` gzip 压缩
- **不删除任何东西**。`archive/` 仍在同一个 store 根下

### 实测效果（dry-run，未执行）

```
compressed 20261012-115139-dcab44.jsonl (2.4 MB -> 0.3 MB)
...
done in 0.0s: archived 0 file(s), compressed 223 (23.9 MB -> 3.4 MB), 28 still active
```

压缩比约 **7:1**。会话 JSONL 是文本，schema 重复度高，gzip 很吃这个。

### 用法

```bash
# 先看会做什么，什么都不改
python3 scripts/archive_sessions.py --dry-run

# 确认后执行
python3 scripts/archive_sessions.py

# 自定义阈值
python3 scripts/archive_sessions.py --idle-days 3 --compress-after-days 14
```

### 挂 cron（你自己决定）

```bash
# 每天凌晨 4 点跑一次
0 4 * * * /usr/bin/python3 /home/ubuntu/scripts/archive_sessions.py >> /home/ubuntu/logs/archive.log 2>&1
```

---

## 2. `health_check.py` —— 磁盘/内存/tmp 体检

### 为什么需要

三件事在这台机器上都**无声地发生过**：

| 事件 | 表现 | 为什么难查 |
|---|---|---|
| 磁盘到 86% | 只剩 5.3 GB | 不会"写入失败"，而是**会话文件丢内容**，看起来像 agent 忘事 |
| 内存耗尽 | 1.9 GB 机器跑 44 个 Chromium | 内核开始 kill 进程，第一个通常是 sshd |
| `/tmp` 涨到 3.7 GB | 全是 Chrome 残留 profile | 每个几百 MB，肉眼看不见 |

### 它做什么

- 磁盘使用率 ≥ 85% → 告警
- 内存使用率 ≥ 90% → 告警
- 列出 `/tmp` 里最大的 5 项
- `--clean-tmp` 才真的清理（默认只报告）

### 清理的安全边界

- 只动 `/tmp` 的**直接子项**
- **永不跟随符号链接**
- 只删 **3 天以上** 没动过的
- 默认 dry-run

### 实测输出

```
disk  / 30.7/39.3 GB used (74%)
mem   0.7/1.9 GB used (38%), swap 0.1/5.9 GB
tmp   377 MB under /tmp
            22.9 MB  .venv-gf
            22.8 MB  bltest
tmp   730 entr(ies) older than 3 days (189 MB) would be removed with --clean-tmp
OK: nothing above threshold (disk < 85%, memory < 90%)
```

### 用法

```bash
python3 scripts/health_check.py                    # 只报告
python3 scripts/health_check.py --clean-tmp        # 真的清 /tmp（3 天以上）
python3 scripts/health_check.py --quiet            # 只在告警时输出（适合 cron）
```

退出码：`0` 正常，`1` 有告警。cron 里可以据此决定要不要发通知。

---

## 建议的 cron 组合

```bash
# 每小时体检，只在有问题时说话
0 * * * * /usr/bin/python3 /home/ubuntu/scripts/health_check.py --quiet --clean-tmp || echo "服务器需要关注" | mail -s "ngu health" you@example.com

# 每天凌晨归档
0 4 * * * /usr/bin/python3 /home/ubuntu/scripts/archive_sessions.py >/dev/null 2>&1
```

---

## 还有一个没做成脚本的问题

**`ngu-serve.service` 曾被我的失误拖了 600 次重启循环。**

我手动启动了一个 `ngu-desktop --serve` 进程占住 8088，而 systemd 里的
`Restart=always` 每 5 秒重试一次，每次都因端口被占而失败。它一直在空转，
从外面看只是"服务有点慢"。

现在已经修复（野进程杀掉、systemd 接管、重启计数停止增长）。但**这个模式值得
留个心眼**：以后再手动起服务之前，先确认 systemd 没有在管同一个端口。

```bash
systemctl is-active ngu-serve        # 是否被托管
sudo ss -ltnp | grep 8088            # 谁占着端口
systemctl show ngu-serve -p NRestarts --value   # 重启过多少次
```

`NRestarts` 如果是个很大的数，说明有东西在反复失败。
