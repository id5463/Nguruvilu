// 被劫持服务器在内核眼里的样子。
//
// # 它做两件事，方向相反
//
// 内核问 `tools/list` 时，**不转发**：回的是一个 `call` 工具。所以内核的
// 表里只有一条，而不是那个服务器的三十七条。这是"合并"的全部意义 ——
// 一条工具说明，模型读一行就够。
//
// 内核调 `call` 时，拆成真实的 `tools/call` 发给服务器，把结果翻回内核
// 能读的样子。参数原样透传，不解释。
//
// # 什么被改，什么没被改
//
// 改的只有 `tools/list` 的**回答形状**：从 N 条变成 1 条。
// `initialize` 原样转发（协议版本、能力集要真实）；`tools/call` 的参数
// 和结果一个字节不改；服务器发来的通知原样送上去。
//
// 一个会篡改结果的中间人会让"这个工具到底做了什么"变成无法回答的问题。
// 这个不篡改：它只决定**列出多少**，不决定**做了什么**。
//
// # 为什么不是把所有服务器合成一个
//
// 每个被劫持的服务器各有一个 shim，各自暴露一个 `call`。所以装了五个
// 服务器就有五个 `call`（名字带服务器前缀），而不是一个。把它们再合成一个
// 需要跨 shim 协调，那是 mcp-proxy 自己的合并模式，不是 shim 的事。
import { spawn } from 'node:child_process';
import path from 'node:path';

const argv = process.argv.slice(2);
if (argv.length === 0) {
  process.stderr.write('[shim] 没有真实命令可以转发\n');
  process.exit(2);
}

const [command, ...args] = argv;

// 这个服务器的短名：取命令的第一个词，够区分，也够短。
const serverName = path.basename(command).replace(/\.(exe|cmd|bat|js|mjs|py)$/i, '');
const TOOL = 'call';
const trace = process.env.NGU_PROXY_TRACE === '1';

function log(...parts) {
  if (trace) process.stderr.write(`[shim:${serverName}] ` + parts.join(' ') + '\n');
}

log('启动真实服务器:', command, args.join(' '));

const child = spawn(command, args, {
  stdio: ['pipe', 'pipe', 'inherit'],
  env: process.env,
  detached: false,
});

child.on('error', (error) => {
  process.stderr.write(`[shim:${serverName}] 起不来 ${command}: ${error.message}\n`);
  process.exit(1);
});
child.on('exit', (code, signal) => {
  log(`真实服务器退出 code=${code} signal=${signal}`);
  process.exit(code == null ? 0 : code);
});

// ---------------------------------------------------------------- 帧处理
//
// MCP over stdio 是**逐行**的 JSON-RPC，不是长度前缀。所以这里按行切，
// 一行一帧。内核那一侧用 length-prefix 是因为它的插件协议是自己的；
// 它连 MCP 时也按行。

let stdoutBuffer = '';

function sendToKernel(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

/// 这个服务器的工具表，第一次问过之后缓存。
let realTools = null;
/// 等 `tools/list` 的回答：id → resolve。
const waiting = new Map();
let nextUpstreamId = 1e6;   // 和内核发来的 id 分开，免得撞号

function askUpstream(method, params) {
  return new Promise((resolve, reject) => {
    const id = nextUpstreamId++;
    waiting.set(id, { resolve, reject });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    setTimeout(() => {
      if (waiting.delete(id)) reject(new Error(`${method} 超时`));
    }, 30000).unref?.();
  });
}

child.stdout.on('data', (chunk) => {
  stdoutBuffer += chunk.toString();
  let at;
  while ((at = stdoutBuffer.indexOf('\n')) >= 0) {
    const line = stdoutBuffer.slice(0, at).trim();
    stdoutBuffer = stdoutBuffer.slice(at + 1);
    if (!line) continue;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      // 不是 JSON 就不是协议的一部分（服务器往 stdout 打日志是它的 bug）。
      // 转给内核会毁掉一帧，丢掉并记下来。
      log('丢弃非 JSON 输出:', line.slice(0, 120));
      continue;
    }
    // 是我发的上游请求的回答？
    if (message.id != null && waiting.has(message.id)) {
      const { resolve, reject } = waiting.get(message.id);
      waiting.delete(message.id);
      if (message.error) reject(new Error(message.error.message || 'upstream error'));
      else resolve(message.result);
      continue;
    }
    // 别的（通知、内核发的 id 的回答）原样送上去。
    sendToKernel(message);
  }
});

// ---------------------------------------------------------------- 主循环

let stdinBuffer = '';

process.stdin.on('data', (chunk) => {
  stdinBuffer += chunk.toString();
  let at;
  while ((at = stdinBuffer.indexOf('\n')) >= 0) {
    const line = stdinBuffer.slice(0, at).trim();
    stdinBuffer = stdinBuffer.slice(at + 1);
    if (!line) continue;
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      log('内核发来非 JSON:', line.slice(0, 120));
      continue;
    }
    handle(message).catch((error) => {
      log('处理失败:', error.message);
      if (message.id != null) {
        sendToKernel({ jsonrpc: '2.0', id: message.id, error: { code: -32603, message: error.message } });
      }
    });
  }
});

process.stdin.on('end', () => {
  try { child.stdin.end(); } catch { /* 已关 */ }
});

async function handle(message) {
  const { id, method } = message;

  // 列出工具：不转发，回答一条。
  if (method === 'tools/list') {
    if (!realTools) {
      // 第一次问时把真实表取一次，只为了知道工具名 —— 调用时要校验，
      // 一个不存在的工具名是最常见的失败，直接说有哪些比猜好。
      try {
        const listed = await askUpstream('tools/list', {});
        realTools = (listed && listed.tools) || [];
        log(`真实工具 ${realTools.length} 个`);
      } catch (error) {
        log('取真实工具表失败:', error.message);
        realTools = [];
      }
    }
    sendToKernel({
      jsonrpc: '2.0',
      id,
      result: {
        tools: [{
          name: TOOL,
          description:
            `调用 ${serverName} 上的一个工具。先用 action="list" 看有哪些工具，` +
            `action="call" 时才需要 tool 和 args。`,
          inputSchema: {
            type: 'object',
            properties: {
              action: { type: 'string', enum: ['list', 'call'], description: 'list 列工具，call 调用' },
              tool: { type: 'string', description: '工具原名' },
              args: { type: 'object', description: '工具参数，原样转发' },
            },
            required: ['action'],
          },
        }],
      },
    });
    return;
  }

  // 调用：拆成真实的 tools/call。
  if (method === 'tools/call' && message.params && message.params.name === TOOL) {
    const a = message.params.arguments || {};
    const action = a.action || 'list';

    if (action === 'list') {
      if (!realTools) {
        try {
          const listed = await askUpstream('tools/list', {});
          realTools = (listed && listed.tools) || [];
        } catch (error) {
          sendToKernel(replyText(id, `取工具表失败：${error.message}`));
          return;
        }
      }
      const names = realTools.map((t) => ({ name: t.name, description: (t.description || '').slice(0, 120) }));
      sendToKernel(replyText(id, JSON.stringify({ server: serverName, tools: names }, null, 2)));
      return;
    }

    if (action !== 'call') {
      sendToKernel(replyText(id, `不认识 action="${action}"，只有 list 和 call`));
      return;
    }
    if (!a.tool) {
      sendToKernel(replyText(id, 'action="call" 需要 tool 参数。用 action="list" 看有哪些。'));
      return;
    }
    // 名字校验：一个拼错的工具名会让服务器回一句难懂的话，先在这里拦下。
    if (realTools && realTools.length && !realTools.some((t) => t.name === a.tool)) {
      sendToKernel(replyText(id, `${serverName} 没有 ${a.tool}。它有：${realTools.map((t) => t.name).join(', ')}`));
      return;
    }
    try {
      const result = await askUpstream('tools/call', { name: a.tool, arguments: a.args || {} });
      sendToKernel({ jsonrpc: '2.0', id, result });
    } catch (error) {
      sendToKernel(replyText(id, `调用 ${a.tool} 失败：${error.message}`));
    }
    return;
  }

  // 别的（initialize、通知、别的工具名）原样转发，等上游的回答照原 id 送回去。
  child.stdin.write(JSON.stringify(message) + '\n');
}

function replyText(id, text) {
  return { jsonrpc: '2.0', id, result: { content: [{ type: 'text', text }], isError: false } };
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    try { child.kill(signal); } catch { /* 已退出 */ }
    process.exit(0);
  });
}
