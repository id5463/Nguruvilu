# dist-next — DSH 参照批次(后台 job + 插话 inbox + 完成唤醒 + 触摸适配)

最新构建 2026/10/8 19:14,**已按用户要求发布到 dist\ 与桌面副本**(旧实例经
WM_CLOSE 优雅关闭、进行中的回合先落盘)。本目录与已发布版本逐字节一致。

| 文件 | SHA256 前12位 |
|---|---|
| ngu.exe | FBCBEBCFAD89 |
| ngu-desktop.exe | 5C71DA8FD806 |
| ngu_demo_plugin.dll | EB738EFCB09B |

## 本批新增(参照 DSH 移植)

1. **后台 job**(`ctx.jobs` 的等价物)
   - `bash` 新增 `background: true` —— 命令转后台立即返回 job id,不再钉死回合
   - 新工具 `job_list` / `job_output`(支持 `wait_ms` 等待完成)/ `job_kill`
   - job 注册表:状态(运行/退出码/被杀)、输出缓冲(256KB 封顶)、
     stdout/stderr 并行排空、100ms 轮询收尾
2. **完成唤醒**(`bash notify: true` —— "程序跑完自动回来继续")
   - job 完成时由 settlement hook 唤醒发起它的会话:还在跑 → 作插话在 step 边界
     送达;已空闲 → **自动开一个新回合**把结果交回模型,无人操作
   - 归属经回合观察者 FIFO 配对(工具 start 的 notify 标志 × end 的 job id),
     会话上下文在观察者里天然可知
3. **插话 inbox**(DSH 的 steer/followup 语义)
   - 回合运行中直接回车 = 插话:消息入队,**step 边界被领取**进入下一步的模型请求
   - 迟到(最后一个边界之后)的消息由回合结束后的**级联**自动跑成后续回合
   - 发送按钮在运行中仍是红色 ■ Stop;停止会清空排队中的插话
   - serve 模式下 prompt 与 cancel 一样**绕过命令队列**(否则插话永远晚于它要引导的回合)
4. 连带修复:E2E 发现并修复了"级联 turn_start 重复画插口气泡"(steeredTexts 消费)

## 验证

- cargo:433 lib + 28 集成测试全绿(含 4 个 jobs 单测:后台存活/kill/未知 id/hook 触发)
- E2E 唤醒(隔离 NGU_HOME + mock 端点):回合 @12.8s 结束 → ping 后台 job
  @21.1s 跑完 → **shell 自动开启新回合**"Background job job-1 finished: exited…"
  → @27.1s 完成,0 错误
- E2E 插话:4ms 入队、step2 请求内含 STEER-1(mock `#6 [probe,S1,toolresult]`)、
  迟到的 STEER-2 自动级联、气泡不重复、0 错误
