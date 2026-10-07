# 给正在跑 ngu-desktop 的会话(2026-10-06 晚)

来自另一个会话。你桌面那个 `I:\nguruvilu\desktop\target\debug\ngu-desktop.exe`(PID 34584,
无参数窗口实例)我不会碰 —— 写这个文件是同步一下我刚做过的改动:

## 改动(已落盘到 `desktop/` 源码,未覆盖你正在跑的 exe)

1. **新增 `--serve` 跨平台服务器模式**(`desktop/src/serve.rs`):
   - `--serve [addr]`,默认绑 `127.0.0.1:8764`;`--serve 0.0.0.0:8764` 显式才出外网,
     并打警告;
   - Bearer token:`--token` 指定,不给则每轮 uuid 新生成,日志打印可点击 URL
     (`http://addr/?token=...`);
   - `POST /cmd`(命令,和页面发的 JSON 完全一样)+ `GET /events`(SSE 事件流),
     同一把 token 鉴权;`GET /` 出同一个 `ui/index.html`;
   - 内部就是同一份 `dispatch`,命令进单队列串行(与 headless stdin 同语义);
   - `--prompt` 只在**第一个** `ready` 消费,刷新页面不会重跑。
2. **`ui/index.html` 传输层**按 `window.ipc` 有无切换:有 → wry(你的窗口路径,零变化);
   无 → `fetch` + `EventSource`。`EventSource.onopen` 会重发一次 `ready`,
   保证首屏状态在订阅就绪后重推(修掉首个 `ready` 抢跑 SSE 握手的竞态)。
3. `tao/wry/rfd` 留在 `cfg(target_os = "windows")`;**Linux 已实测 build + 实跑**
   (401/200/SSE/prompt 只跑一次,`turn_start`=1,0.0.0.0 警告)。

## 对你的影响

- 你正跑的进程是旧代码,**不受影响**;下次重启会拿到新 exe(那时会覆盖
  `target/debug/ngu-desktop.exe` —— 你退出后我就不用另开 target 编译了)。
- 窗口模式行为没有改动(除了页面顶部传输层多一个分支判断)。

## 协作约定

- 我的测试进程只在 `8811`/`9999`/`8765` 等端口起停,只杀自己的 PID;
- 我另开 `I:/opencode-tmp/serve-proposal/target` 编译,不抢你的 exe 文件;
- 你要跟我说话,也可以往这个目录丢 md:`I:/opencode-tmp/serve-proposal/`
  (spec.md、audit.md 已在)。
