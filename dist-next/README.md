# dist-next — DSH 参照批次(后台 job + 插话 inbox)

打包于 2026/10/7。按要求**未替换** dist\ 与桌面副本,未关闭任何正在运行的实例——
直接运行本目录的 ngu-desktop.exe 即可试用;确认没问题后,再决定是否发布到原位置。

| 文件 | SHA256 前12位 |
|---|---|
| ngu.exe | C225AEB5432B |
| ngu-desktop.exe | C065D209882D |
| ngu_demo_plugin.dll | EB738EFCB09B |

## 本批新增(参照 DSH 移植)

1. **后台 job**(`ctx.jobs` 的等价物)
   - `bash` 新增 `background: true` —— 命令转后台立即返回 job id,不再钉死回合
   - 新工具 `job_list` / `job_output`(支持 `wait_ms` 等待完成)/ `job_kill`
   - job 注册表:状态(运行/退出码/被杀)、输出缓冲(256KB 封顶)、
     stdout/stderr 并行排空、100ms 轮询收尾
2. **插话 inbox**(DSH 的 steer/followup 语义)
   - 回合运行中直接回车 = 插话:消息入队,**step 边界被领取**进入下一步的模型请求
   - 迟到(最后一个边界之后)的消息由回合结束后的**级联**自动跑成后续回合
   - 发送按钮在运行中仍是红色 ■ Stop;停止会清空排队中的插话
   - serve 模式下 prompt 与 cancel 一样**绕过串行命令队列**(否则插话永远晚于它要引导的回合)
3. 连带修复:E2E 发现并修复了"级联 turn_start 重复画插口气泡"(steeredTexts 消费)

## 验证

- cargo:432 lib + 28 集成测试全绿(含3个 jobs 单测、工具清单断言更新)
- E2E(隔离 NGU_HOME + mock 端点):插话 4ms 入队、step2 请求内含 STEER-1(mock 日志
  `#6 [probe,S1,toolresult]`)、迟到的 STEER-2 自动级联、气泡不重复、0 错误
