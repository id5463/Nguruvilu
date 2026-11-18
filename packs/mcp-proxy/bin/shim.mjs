// 被劫持的服务器的入口。
//
// 内核读到的 `command:` 被 `hijack.mjs` 改成了 `node shim.mjs <真实命令> ...`，
// 于是内核启动的是这个文件，而不是原来的 MCP 服务器。
//
// 它做三件事，都很小：
//
//   1. 把内核发来的每一帧记下来（`NGU_PROXY_TRACE` 开了才记），
//   2. 起真正的服务器，把帧原样转发进去，
//   3. 把答案原样转发回内核。
//
// **它不修改内容。** 一个会改内容的中间人会让"这个工具到底做了什么"变成
// 一个无法回答的问题 —— 而这个包的全部目的是让工具变少，不是让它们变奇怪。
// 记日志是观察，改内容是欺骗，两者不是一回事。
//
// 真正的服务器不是这个文件起的：它请 `proxy.mjs` 起，这样同一份服务器只有
// 一个实例，代理的 `call` 工具和内核的直连走的是同一个进程。
import { spawn } from 'node:child_process';

const argv = process.argv.slice(2);
if (argv.length === 0) {
  process.stderr.write('[shim] 没有真实命令可以转发\n');
  process.exit(2);
}

const [command, ...args] = argv;
const trace = process.env.NGU_PROXY_TRACE === '1';

function log(...parts) {
  if (trace) process.stderr.write('[shim] ' + parts.join(' ') + '\n');
}

log('启动真实服务器:', command, args.join(' '));

// 在自己和真实服务器之间放一条管道。
//
// `inherit` 不能用于 stdio：内核的管道属于内核，这个进程用它自己的两段
// 去对接，才能看得见每一帧。`piped` 让 spawn 建新管道，两头都在这里。
const child = spawn(command, args, {
  stdio: ['pipe', 'pipe', 'inherit'],
  env: process.env,
  // 别让 shim 成为孤儿的父进程：内核关掉 stdin 时我们要跟着退出。
  detached: false,
});

child.on('error', (error) => {
  process.stderr.write(`[shim] 起不来 ${command}: ${error.message}\n`);
  process.exit(1);
});

child.on('exit', (code, signal) => {
  log(`真实服务器退出 code=${code} signal=${signal}`);
  process.exit(code == null ? 0 : code);
});

// 内核 → 真实服务器。逐字节转发，不改一个字节。
process.stdin.pipe(child.stdin);
process.stdin.on('end', () => {
  // 内核关了 stdin 就是在说"结束"。关掉自己这一侧，让真实服务器看到 EOF。
  try { child.stdin.end(); } catch { /* 已经关了 */ }
});

// 真实服务器 → 内核。
if (trace) {
  // 只有要记录时才截流：不记录时直接 pipe，省一次拷贝。
  child.stdout.on('data', (chunk) => {
    log('→', chunk.toString().trim().slice(0, 200));
    process.stdout.write(chunk);
  });
} else {
  child.stdout.pipe(process.stdout);
}

// 这个进程自己没有逻辑，两个方向都是管道。任何一侧断掉就整体退出，
// 免得留下一个还在跑但已经没人说话的服务器。
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    try { child.kill(signal); } catch { /* 已退出 */ }
    process.exit(0);
  });
}
