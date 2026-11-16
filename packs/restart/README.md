# restart

让代理重启它自己所在的进程。内核里的代码是 `builtin:restart`
(`src/tools/restart.rs`)，**这个包决定它存不存在**。

没有这个包时，工具表里没有 `restart` —— 内核带着那段代码但不会启用它，
这正是"能力归包、内核只给机制"的分工。

## 前提

进程得有人管。systemd 的 `Restart=always`、launchd 的 `KeepAlive`、
Docker 的 restart policy 都行；手工在终端里跑的时候退出就是退出，工具会
如实告诉你不该用而不是把你弄死。
