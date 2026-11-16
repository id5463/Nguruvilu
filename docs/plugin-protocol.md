# 插件协议

> 内核不认识任何能力。插件是一个**可执行文件**，内核启动它、问它能做什么、然后调用它。

## 为什么是这个形状

先看一个测量（同一台机器，Windows，200 次往返取平均）：

| payload | 按行切分（MCP 的做法） | 长度前缀（本协议） |
|---|---|---|
| 64 B | 244 µs | 256 µs |
| 1 KB | 257 µs | 263 µs |
| 64 KB | 615 µs | 610 µs |
| 1 MB | 15.0 ms | 19.1 ms |

**分帧方式几乎不影响速度。** 真正的数字是这个：

| 操作 | 耗时 |
|---|---|
| 启动一个子进程 | **45 ms** |
| 一次协议往返 | **0.25 ms** |

**启动是消息的 180 倍。** 所以本协议唯一的性能规则是：**进程启动一次，一直挂着**。

## 契约

插件是一个可执行文件。它从 stdin 读**长度前缀帧**，往 stdout 写长度前缀帧：

```
[4 字节大端长度][JSON 载荷]
```

四个操作，一问一答：

```json
{"op":"describe"}                          → {"tools":[…], "panels":[…], "settings_key":"…"}
{"op":"call","tool":"x","args":{…}}        → {"result":…} 或 {"error":"…"}
{"op":"state","get":"settings"}            → {"value":…}
{"op":"state","set":"judge","value":{…}}   → {"ok":true}
```

就这些。没有协议版本、没有能力协商、没有握手。

## 权限

**没有任何权限检查。** 插件是用户启动的进程，以用户的身份运行，拥有用户的文件系统、网络、以及给其他进程发信号的能力。内核不沙箱化它，也不假装沙箱化了。

内核只给插件两样它否则拿不到的信息：

| 环境变量 | 内容 |
|---|---|
| `NGU_PID` | **内核自己的**进程号 |
| `NGU_EXE` | 内核的可执行文件路径 |
| `NGU_PACK_SETTINGS` | 所有包的设置，一整张 JSON 表 |
| `NGU_PROXY` | 代理，如果用户设置了 |

`NGU_PID` 是 `restart` 能工作的原因：插件 `kill(内核pid)` 请求重启，supervisor 拉起来。实测可行——**任何以用户身份运行的进程都能给任何其他进程发信号**。

`NGU_PACK_SETTINGS` 的形态：

```json
{"judge": {"endpoint": "…", "api_key": "…"}, "mcp-proxy": {"mode": "proxy"}}
```

内核只做序列化，不解释里面任何一个名字。

## 泛用性：为什么不用 MCP

MCP 是给"模型和服务器说话"设计的。拿来当插件 ABI 有四个问题：

1. **没有面板概念** —— `long-horizon` 的任务条因此卡在 Rust 里
2. **读不到宿主状态** —— 插件无法读写内核的 settings
3. **每次都要握手** —— `initialize` / `tools/list`，插件为此付出它不用的开销
4. **闭包传不过去** —— `documents` 的 `ReaderSpec` 因此无法迁移

本协议解决全部四条：

| 需求 | 本协议 |
|---|---|
| 面板 | `describe` 直接返回 `panels`，格式与 `look.json` 相同 |
| 内核状态 | `state get/set`，key 是运行时字符串，**没有白名单** |
| 握手 | 没有 |
| 任意数据 | 长度前缀帧，二进制安全 |

## 泛用性的真正判据

**内核代码里能不能出现具体能力的名字。**

| 写法 | 内核认识几个能力 |
|---|---|
| `"mcp_call" => …` | 1 |
| `"market_search" => …` | 1 |
| `{"op":"call","tool":name}` | **0** |

本协议的 `call` 和 `state` 都不含能力名。加第 1000 个插件、第 1000 个工具、第 1000 个状态 key，内核代码不变。

## 写一个插件

包这样声明：

```yaml
- id: restart
  source: plugin:bin/ngu-restart
  build:
    dir: src        # 相对包目录
    binary: ngu-restart
```

`source: plugin:<路径>` 是**相对包目录的**，或者一个裸名字（走 PATH）。
`build` 是可选的：源码在包里、二进制不在时，内核调 `cargo build --release --offline` 编出来。

`build` 字段刻意不叫 `cargo`：它说的是**在哪个目录编、编出什么二进制**，不是"用哪个工具链"。默认 `cargo`，`command` 可以改。

### 最小插件（Rust）

```rust
use std::io::{Read, Write};
use serde_json::{json, Value};

fn main() {
    let mut stdin = std::io::stdin();
    loop {
        let Some(frame) = read_frame(&mut stdin) else { return };
        let msg: Value = serde_json::from_slice(&frame).unwrap();
        let answer = match msg["op"].as_str() {
            Some("describe") => json!({
                "tools": [{ "name": "hello", "description": "says hi", "parameters": {} }]
            }),
            Some("call") => json!({ "result": "hi" }),
            _ => json!({ "error": "unsupported op" }),
        };
        write_frame(&answer);
    }
}

fn read_frame(input: &mut impl Read) -> Option<Vec<u8>> {
    let mut header = [0u8; 4];
    input.read_exact(&mut header).ok()?;
    let len = u32::from_be_bytes(header) as usize;
    let mut body = vec![0u8; len];
    input.read_exact(&mut body).ok()?;
    Some(body)
}

fn write_frame(value: &Value) {
    let body = serde_json::to_vec(value).unwrap();
    let mut out = std::io::stdout();
    out.write_all(&(body.len() as u32).to_be_bytes()).unwrap();
    out.write_all(&body).unwrap();
    out.flush().unwrap();
}
```

## 工具链是怎么找到的

一个真实的坑，记录在这里因为它花了很久：

内核的 PATH 是**启动时继承的**，不是用户 shell 的。开发机上 `cargo` 在 `I:\cargo\bin`，加进了用户 PATH，但 `ngu` 进程看不到，于是"手工能编、内核说找不到程序"。

现在的做法：

1. `resolve_program` 通过**扩展后的 PATH** 找绝对路径，而不是让 `Command` 用继承的 PATH 查（Windows 上程序查找发生在 `env` 覆盖之前）
2. 扩展 PATH 时在 Windows 上调 `[Environment]::GetEnvironmentVariable('Path','User')`，用**绝对路径**调 `powershell.exe`（因为这时 PATH 正是坏的那个）
3. `CARGO_HOME` 从 `cargo` 的位置推断（`<cargo>/../`，且要有 `registry/`）
4. `RUSTUP_HOME` 同理，且只在 `toolchains/` 存在时才设

全链路实测：清空 `CARGO_HOME`、`RUSTUP_HOME`，PATH 里没有 cargo，内核仍能自己找到工具链、编译、加载。

## 已迁移的插件

| 插件 | 原位置 | 现位置 | 二进制大小 |
|---|---|---|---|
| `restart` | `src/tools/restart.rs`（386 行） | `packs/restart/src/main.rs` | 246 KB |
| `judge` | `src/tools/judge.rs`（773 行） | `packs/judge/src/main.rs` | ~4 MB |

`judge` 在内核里只剩 136 行：设置类型和读写函数，因为**设置文件是内核的**，面板编辑它需要知道三个字段名。内核从不运行 judge。

## 还没迁移的，以及为什么

| 能力 | 障碍 |
|---|---|
| `delegate` | 要跑子代理：需要模型客户端和工具表。可以在插件里自己读 `settings.json` 发 HTTP，但子代理循环要重写 |
| `documents` | 要注册文件格式处理器，而 `read` 在热路径上同步调用它。改成"问插件要转换结果"需要 `read` 变 async |
| `long-horizon` | 工具可迁移；它的任务条界面（约 700 行 HTML/CSS）目前在 Rust 里，应先搬进 `look.json` |

三者的共同点：**不是协议不够，是它们依赖内核的某个具体机制**。改动量在能力侧，不在协议侧。
