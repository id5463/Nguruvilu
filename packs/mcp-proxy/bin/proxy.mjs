#!/usr/bin/env node
// MCP 代理：一个 stdio MCP 服务器，转发给若干真实 MCP 服务器。
//
// 为什么存在：每个 MCP 都把自己的工具全暴露出来，chrome-devtools 是 37 个，
// GitHub 官方的是 46 个，光这两个就 83 条。模型光是读工具名就花掉一半注意力，
// 而这 83 条里一次对话通常只用到三五个。
//
// 三种模态：
//   full    —— 不代理，直接把每个工具按原名暴露（等于不用这个包）
//   single  —— 每个目标服务器暴露一个入口：<server>_call
//   proxy   —— 所有服务器合并成一个 call
//
// 目标清单来自 NGU_MCP_TARGETS，形如：
//   chrome-devtools=chrome-devtools-mcp --headless
//   github=github-mcp-server stdio --toolsets=repos,issues
// 用分号或换行分隔多条。等号左边是给模型看的短名字。

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { delimiter, join } from 'node:path';

/// Find the file to actually launch for a command name.
///
/// Windows does not run a `.cmd` shim through `spawn` — `chrome-devtools-mcp` is
/// a `.cmd` in npm's global bin, and spawning the bare name fails with ENOENT
/// even though it works in a shell. The kernel solves this for its own MCP
/// clients (`resolve_program`); the proxy spawns its own children, so it needs
/// the same lookup or it would only work for servers that happen to be `.exe`.
///
/// An explicit extension wins: a caller that named a file gets that file.
function resolveProgram(command) {
  if (process.platform !== 'win32') return { program: command, needsShell: false };
  const backslash = String.fromCharCode(92);
  if (command.includes('/') || command.includes(backslash)) {
    // A path the caller named: respect it, but a shim still needs a shell.
    return { program: command, needsShell: /\.(cmd|bat)$/i.test(command) };
  }
  if (/\.[a-z0-9]+$/i.test(command)) {
    return { program: command, needsShell: /\.(cmd|bat)$/i.test(command) };
  }

  const extensions = ['.exe', '.cmd', '.bat', ''];
  const dirs = (process.env.PATH || '').split(delimiter).filter(Boolean);
  for (const dir of dirs) {
    for (const extension of extensions) {
      const candidate = join(dir, command + extension);
      if (existsSync(candidate)) {
        return { program: candidate, needsShell: /\.(cmd|bat)$/i.test(candidate) };
      }
    }
  }
  return { program: command, needsShell: false };
}

// 自己的设置，从内核递过来的整张表里取自己那一段。
//
// 内核以前为这个包写死了两个变量（NGU_MCP_MODE / NGU_MCP_TARGETS），
// 那意味着内核知道自己有一个叫 mcp-proxy 的包、还知道它有两个设置项。
// 现在内核只递一整张 `{包名: 设置}` 的 JSON，包自己取——内核不认识
// 这里面任何一个名字，也就不会因为多了一个包而要改代码。
function ownSettings() {
  const raw = process.env.NGU_PACK_SETTINGS;
  if (!raw) return {};
  try {
    const all = JSON.parse(raw);
    return (all && typeof all === 'object' && all['mcp-proxy']) || {};
  } catch (e) {
    return {};
  }
}

const SETTINGS = ownSettings();
const MODE = String(SETTINGS.mode || 'proxy').trim().toLowerCase();
const TARGET_SPEC = Array.isArray(SETTINGS.targets)
  ? SETTINGS.targets.join('\n')
  : (typeof SETTINGS.targets === 'string' ? SETTINGS.targets : '');
const DEBUG = process.env.NGU_MCP_DEBUG === '1';

function log(...args) {
  // stderr 是日志通道：stdout 是协议，写一个字节进去就会毁掉这一帧。
  if (DEBUG) process.stderr.write('[mcp-proxy] ' + args.join(' ') + '\n');
}

// ---------------------------------------------------------------- 目标解析

function parseTargets(spec) {
  const out = [];
  for (const raw of spec.split(/[;\n]/)) {
    const line = raw.trim();
    if (!line) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) {
      log('跳过无法解析的目标（缺少 =）:', line);
      continue;
    }
    const name = line.slice(0, eq).trim();
    const rest = line.slice(eq + 1).trim();
    if (!name || !rest) continue;
    // 命令按空格切。引号里的空格不切——一个路径带空格的程序名很常见。
    const argv = [];
    let current = '';
    let quote = null;
    for (const ch of rest) {
      if (quote) {
        if (ch === quote) quote = null;
        else current += ch;
      } else if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === ' ') {
        if (current) { argv.push(current); current = ''; }
      } else {
        current += ch;
      }
    }
    if (current) argv.push(current);
    if (argv.length === 0) continue;
    // cwd=<dir>::<command> — two colons, because a Windows path has one in
    // its drive letter: splitting cwd=C:\x:node at the first colon yields "C"
    // and "\x:node". The single-colon form was tried first and every Windows
    // target failed with "spawn C:\Program Files\nodejs\node.exe ENOENT" — the
    // drive letter had been eaten.
    let cwd = null;
    let command = argv[0];
    const cdMatch = /^cwd=(.+?)::(.*)$/.exec(command);
    if (cdMatch) {
      cwd = cdMatch[1];
      command = cdMatch[2];
    }
    out.push({ name, command, args: argv.slice(1), cwd });
  }
  return out;
}

// ------------------------------------------------------------ 子进程会话

/// 一个真实 MCP 服务器的连接。
///
/// 自己 spawn 而不是指望内核已经起了一份：内核不知道这个包在转发给谁，
/// 而两边的连接必须由一方持有。这个包持有它，就由这个包管理生命周期。
class Session {
  constructor(target) {
    this.target = target;
    this.child = null;
    this.buffer = '';
    this.nextId = 1;
    this.pending = new Map();
    this.tools = [];
    this.toolIndex = new Map();
    this.ready = null;
  }

  start() {
    if (this.ready) return this.ready;
    this.ready = new Promise((resolve, reject) => {
      // A `.cmd` is a batch script, and `spawn` cannot execute one directly on
      // Windows — `EINVAL`, not the friendlier `ENOENT` a missing file gives.
      // npm installs every CLI as a `.cmd` shim, so this is the common case for
      // any MCP server published to npm, not an edge.
      //
      // `shell: true` would fix it by handing the whole line to cmd.exe, but it
      // also re-parses quotes and `&` in arguments. Invoking `cmd.exe /c` with an
      // argv array keeps the arguments as separate strings, so a path with a
      // space or an ampersand still arrives intact.
      const resolved = resolveProgram(this.target.command);
      const program = resolved.needsShell ? process.env.ComSpec || 'cmd.exe' : resolved.program;
      const argv = resolved.needsShell
        ? ['/c', resolved.program, ...this.target.args]
        : this.target.args;
      const child = spawn(program, argv, {
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
        windowsVerbatimArguments: resolved.needsShell,
        // A pack's own script is named relative to *its* directory. Without
        // this the child would resolve `bin/x.mjs` against the proxy's cwd and
        // fail with a module-not-found that says nothing about why.
        cwd: this.target.cwd || undefined,
      });
      this.child = child;
      let settled = false;
      const fail = (error) => {
        if (settled) return;
        settled = true;
        reject(error);
      };

      child.on('error', (error) => fail(new Error(`${this.target.name}: 启动失败 ${error.message}`)));
      child.on('exit', (code) => {
        log(`${this.target.name} 退出，code=${code}`);
        for (const [, waiter] of this.pending) {
          waiter.reject(new Error(`${this.target.name} 已退出`));
        }
        this.pending.clear();
        if (!settled) fail(new Error(`${this.target.name}: 启动即退出 code=${code}`));
      });

      child.stdout.on('data', (chunk) => {
        this.buffer += chunk.toString('utf8');
        // 一行一帧。半行留在 buffer 里等下一次。
        let at;
        while ((at = this.buffer.indexOf('\n')) !== -1) {
          const line = this.buffer.slice(0, at).trim();
          this.buffer = this.buffer.slice(at + 1);
          if (!line) continue;
          let message;
          try { message = JSON.parse(line); } catch { continue; }
          this.dispatch(message);
        }
      });
      child.stderr.on('data', (chunk) => log(`${this.target.name} stderr:`, chunk.toString().trim()));

      // 握手：initialize → tools/list
      this.request('initialize', {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'ngu-mcp-proxy', version: '1.0.0' },
      }).then(() => {
        this.notify('notifications/initialized', {});
        return this.request('tools/list', {});
      }).then((result) => {
        this.tools = (result && result.tools) || [];
        for (const tool of this.tools) this.toolIndex.set(tool.name, tool);
        log(`${this.target.name}: ${this.tools.length} 个工具`);
        settled = true;
        resolve(this);
      }).catch(fail);
    });
    return this.ready;
  }

  dispatch(message) {
    if (message.id === undefined) return; // 通知，没有等待者
    const waiter = this.pending.get(message.id);
    if (!waiter) return;
    this.pending.delete(message.id);
    if (message.error) waiter.reject(new Error(message.error.message || '未知错误'));
    else waiter.resolve(message.result);
  }

  request(method, params) {
    const id = this.nextId++;
    const frame = JSON.stringify({ jsonrpc: '2.0', id, method, params });
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${this.target.name}: ${method} 超时`));
      }, Number(process.env.NGU_MCP_TIMEOUT_MS || 60000));
      this.pending.set(id, {
        resolve: (value) => { clearTimeout(timer); resolve(value); },
        reject: (error) => { clearTimeout(timer); reject(error); },
      });
      if (!this.child || !this.child.stdin.writable) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new Error(`${this.target.name}: 进程不可写`));
        return;
      }
      this.child.stdin.write(frame + '\n');
    });
  }

  notify(method, params) {
    if (!this.child || !this.child.stdin.writable) return;
    this.child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n');
  }

  call(name, args) {
    return this.request('tools/call', { name, arguments: args || {} });
  }

  stop() {
    if (this.child) {
      try { this.child.kill(); } catch {}
    }
  }
}

// ------------------------------------------------------------ 自动发现目标
//
// 手写一份 targets 是难用的根源：装了一个新的 MCP 包，还得回来改这里，
// 而且写错了没有报错——代理只是少一个服务器。内核已经把每个包的 mcp.json
// 都放在磁盘上了，直接读它。
//
// 只读，不启动，不改任何东西：这里做的和内核装载时做的事不重叠——内核
// 不知道代理的存在，代理也不改内核。一个包自己的 mcp.json 是它公开的
// 声明，读它是读者该做的事。
//
// 自己（`mcp-proxy`）要跳过，否则会代理自己，无限递归。
const SELF_ID = 'mcp-proxy';

// 劫持：把别的包的 MCP 启动命令换成自己的 shim。
//
// 动态 import，因为 `hijack.mjs` 只在需要时才值得加载 —— 一个不劫持的
// 安装不该为它付一次模块解析。
let hijack = null;
async function loadHijack() {
  if (!hijack) hijack = await import('./hijack.mjs');
  return hijack;
}

function discoverFromPacks() {
  const home = process.env.NGU_HOME || path.join(os.homedir(), '.nguruvilu');
  const packsDir = path.join(home, 'packs');
  const found = [];
  let entries = [];
  try {
    entries = fs.readdirSync(packsDir, { withFileTypes: true });
  } catch {
    return found;   // 没有 packs 目录不是错误，只是没有可发现的
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(packsDir, entry.name);
    const file = path.join(dir, 'mcp.json');
    if (!fs.existsSync(file)) continue;
    let parsed;
    try {
      parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch {
      // 一个坏的 mcp.json 不该让代理起不来；内核装载时也会为它报错。
      continue;
    }
    for (const server of parsed.servers || []) {
      const id = String(server.id || '').trim();
      if (!id || id === SELF_ID) continue;
      // 交给代理的条目由内核对它的 via_proxy 负责；这里不管，
      // 否则会和内核的转交重复。
      if (server.via_proxy) continue;
      const command = String(server.command || '').trim();
      if (!command) continue;
      found.push({
        name: id,
        command,
        args: Array.isArray(server.args) ? server.args.map(String) : [],
        cwd: dir,
      });
    }
  }
  return found;
}

// --------------------------------------------------------------- 代理本体

// 手写的 targets 优先，自动发现补在后面：显式配置永远能覆盖发现的结果，
// 而没配置的人什么都不用做。
const manual = parseTargets(TARGET_SPEC);
let targets = [];
function refreshTargets() {
  const discovered = discoverFromPacks();
  const seen = new Set(manual.map((t) => t.name));
  const merged = manual.concat(discovered.filter((t) => !seen.has(t.name)));
  const before = targets.length;
  targets = merged;
  if (merged.length !== before) {
    log(`目标: ${merged.length} 个（手写 ${manual.length}，发现 ${merged.length - manual.length}）`);
  }
  return merged;
}
refreshTargets();

// 周期扫描。
//
// 装载顺序是包的字母序，代理叫 `mcp-proxy`，排在 `mail`、`moodle` 前面 ——
// 而那些包声明 `via_proxy: mcp-proxy` 时，代理可能还没被内核启动。内核那一
// 侧已经回退成直连（工具不丢），代价是"代理接管"这件事发生在启动之后。
//
// 所以这一侧自己看：每隔几秒重新发现一次，新出现的服务器直接补进
// `sessions`。`tools/list` 每次都读 `sessions`，所以新目标在下一轮
// `tools/list` 就可见 —— 不需要重启内核，也不需要内核知道这件事。
const RESCAN_MS = Number(process.env.NGU_PROXY_RESCAN_MS || 5000);
const rescan = setInterval(() => {
  const next = refreshTargets();
  if (next.length > sessions.size) {
    // 有新的：起它们。失败照旧记进 `failures`，不影响已有的。
    ensureSessions().catch(() => {});
  }
}, RESCAN_MS);
// 守护定时器不该把进程吊住：stdin 关闭时我们要能退出。
if (typeof rescan.unref === 'function') rescan.unref();
const sessions = new Map();
const failures = [];

async function ensureSessions() {
  // 只起还没起的：重复调用是常态（每次 tools/list 都来一遍，周期扫描也
  // 来），而起第二个同名服务器会把第一个的管道丢在地上。
  const pending = targets.filter((target) => !sessions.has(target.name));
  if (pending.length === 0) return;
  const started = await Promise.allSettled(
    pending.map(async (target) => {
      const session = new Session(target);
      sessions.set(target.name, session);
      await session.start();
      return target.name;
    })
  );
  for (let i = 0; i < started.length; i += 1) {
    if (started[i].status === 'rejected') {
      // 一个起不来不该让别的也起不来：记下来，在 tools/list 和调用时如实说。
      // 失败的不留在 `sessions` 里，下一轮扫描会再试 —— 一个包刚装好、
      // 依赖还没就位的情况是真实的。
      sessions.delete(pending[i].name);
      const reason = `${pending[i].name}: ${started[i].reason && started[i].reason.message}`;
      if (!failures.includes(reason)) failures.push(reason);
    }
  }
}

/// 每个目标一个入口时用的工具名。
function entryName(target) {
  return `${target.replace(/[^a-zA-Z0-9_]/g, '_')}_call`;
}

function toolList() {
  if (MODE === 'proxy') {
    return [{
      name: 'call',
      description:
        '调用一个 MCP 工具。用 action="list" 查看有哪些服务器和工具，' +
        'action="call" 时才需要 server 和 tool。' +
        (failures.length ? ` 注意，以下服务器没有启动：${failures.join('; ')}` : ''),
      inputSchema: {
        type: 'object',
        properties: {
          action: { type: 'string', enum: ['list', 'call', 'hijacked', 'hijack', 'restore', 'restart'],
                    description: 'list 列目录；call 调用；hijacked 看谁被劫持了；hijack 重新劫持；restore 全部还原；restart 还原后重启内核' },
          server: { type: 'string', description: '服务器短名，如 chrome-devtools' },
          tool: { type: 'string', description: '工具原名，如 navigate_page' },
          args: { type: 'object', description: '工具参数，原样转发' },
        },
        required: ['action'],
      },
    }];
  }
  if (MODE === 'single') {
    return Array.from(sessions.keys()).map((name) => ({
      name: entryName(name),
      description: `调用 ${name} 服务器上的工具。action="list" 看有哪些工具。`,
      inputSchema: {
        type: 'object',
        properties: {
          action: { type: 'string', enum: ['list', 'call'] },
          tool: { type: 'string' },
          args: { type: 'object' },
        },
        required: ['action'],
      },
    }));
  }
  // full：不代理，原样列出。等于不用这个包，但保留一个出口用于对照测试。
  const all = [];
  for (const session of sessions.values()) {
    for (const tool of session.tools) {
      all.push({
        name: `${session.target.name}__${tool.name}`,
        description: tool.description || '',
        inputSchema: tool.inputSchema || { type: 'object', properties: {} },
      });
    }
  }
  return all;
}

function listPayload(serverName) {
  const out = {};
  for (const [name, session] of sessions) {
    if (serverName && name !== serverName) continue;
    out[name] = session.tools.map((tool) => ({
      name: tool.name,
      description: (tool.description || '').slice(0, 160),
    }));
  }
  if (failures.length) out.__failed = failures;
  return out;
}

async function handleCall(args) {
  const action = args.action || 'list';
  if (action === 'list') {
    return text(JSON.stringify(listPayload(args.server), null, 2));
  }
  if (action === 'hijacked') {
    // 自己交代：改了哪些文件、原来的命令是什么。一个不肯被念出来的劫持
    // 才需要被怀疑。
    const mod = await loadHijack();
    return text(JSON.stringify(mod.hijackedReport(), null, 2));
  }
  if (action === 'hijack') {
    // 重新劫持一次（装完新包之后）。
    const mod = await loadHijack();
    const results = mod.hijackAll();
    return text(
      `劫持完成：${results.filter((r) => r.hijacked).length} 个包。\n` +
        JSON.stringify(results, null, 2) +
        "\n重启内核后生效。",
    );
  }
  if (action === 'restore') {
    const mod = await loadHijack();
    const packs = mod.restoreAll();
    return text(`已还原 ${packs.length} 个包：${packs.join(', ') || '(无)'}`);
  }

  if (action === 'restart') {
    // 重启内核。kill 是唯一的动作：systemd 的 Restart=always 负责把它拉
    // 起来，代理不需要知道它怎么被监督。
    //
    // 先还原自己改过的文件：内核重新读 assembly 时必须是原始内容，否则
    // 新的内核会读到自己写的 shim，再套一层。
    const mod = await loadHijack();
    const restored = mod.restoreAll();
    const pid = Number(process.env.NGU_KERNEL_PID || 0);
    if (!pid) {
      return text(
        "不知道内核的 pid，没法重启。已还原：" + (restored.join(", ") || "(无)") +
          "\n设 NGU_KERNEL_PID，或者用 restart 包的 restart 工具。",
      );
    }
    const delay = Number(args.delay_ms || 700);
    setTimeout(() => {
      try { process.kill(pid, "SIGTERM"); } catch { /* 已经没了 */ }
    }, delay);
    return text(
      `已还原 ${restored.length} 个包，${delay} ms 后向 pid ${pid} 发 SIGTERM。` +
        "\n内核若由 systemd 监督（Restart=always），几秒后会自己起来。",
    );
  }
  const serverName = args.server || Array.from(sessions.keys())[0];
  const session = sessions.get(serverName);
  if (!session) {
    return text(`没有 ${serverName} 这个服务器。可用：${Array.from(sessions.keys()).join(', ')}`);
  }
  if (!args.tool) return text('action="call" 需要 tool 参数');
  if (!session.toolIndex.has(args.tool)) {
    // 编造的工具名是最常见的失败，直接给出真的有哪些。
    return text(
      `${serverName} 没有 ${args.tool} 这个工具。它有：` +
      session.tools.map((t) => t.name).join(', ')
    );
  }
  const result = await session.call(args.tool, args.args || {});
  return fromMcpResult(result);
}

async function handleSingle(serverName, args) {
  const session = sessions.get(serverName);
  if (!session) return text(`服务器 ${serverName} 没有启动`);
  const action = args.action || 'list';
  if (action === 'list') return text(JSON.stringify(listPayload(serverName), null, 2));
  if (!args.tool) return text('action="call" 需要 tool 参数');
  if (!session.toolIndex.has(args.tool)) {
    return text(`${serverName} 没有 ${args.tool}。它有：` + session.tools.map((t) => t.name).join(', '));
  }
  return fromMcpResult(await session.call(args.tool, args.args || {}));
}

/// MCP 的结果原样包一层。content 已经是 MCP 的形状，直接还回去，
/// 加一层会导致模型看到两个嵌套的 content 数组。
function fromMcpResult(result) {
  if (!result) return text('(空结果)');
  if (Array.isArray(result.content)) {
    return { content: result.content, isError: Boolean(result.isError) };
  }
  return text(JSON.stringify(result));
}

function text(value) {
  return { content: [{ type: 'text', text: String(value) }] };
}

// ------------------------------------------------------------ JSON-RPC 循环

function reply(id, result) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n');
}

function replyError(id, code, message) {
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } }) + '\n');
}

async function handle(message) {
  const { id, method, params } = message;
  if (method === 'initialize') {
    return reply(id, {
      protocolVersion: '2024-11-05',
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: 'mcp-proxy', version: '1.0.0' },
    });
  }
  if (method === 'notifications/initialized') return; // 通知，不回复
  if (method === 'tools/list') {
    await ensureSessions();
    return reply(id, { tools: toolList() });
  }
  if (method === 'tools/call') {
    await ensureSessions();
    const name = params && params.name;
    const args = (params && params.arguments) || {};
    try {
      if (MODE === 'proxy') {
        if (name !== 'call') return replyError(id, -32602, `只有 call 这个工具，收到 ${name}`);
        return reply(id, await handleCall(args));
      }
      if (MODE === 'single') {
        for (const target of sessions.keys()) {
          if (entryName(target) === name) return reply(id, await handleSingle(target, args));
        }
        return replyError(id, -32602, `未知工具 ${name}`);
      }
      // full：名字是 <server>__<tool>
      const at = String(name).indexOf('__');
      if (at <= 0) return replyError(id, -32602, `工具名要形如 server__tool，收到 ${name}`);
      const serverName = name.slice(0, at);
      const toolName = name.slice(at + 2);
      const session = sessions.get(serverName);
      if (!session) return replyError(id, -32602, `没有 ${serverName} 这个服务器`);
      return reply(id, fromMcpResult(await session.call(toolName, args)));
    } catch (error) {
      return reply(id, text(`调用失败：${error.message}`));
    }
  }
  return replyError(id, -32601, `不认识的方法 ${method}`);
}

let buffer = '';
process.stdin.on('data', (chunk) => {
  buffer += chunk.toString('utf8');
  let at;
  while ((at = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, at).trim();
    buffer = buffer.slice(at + 1);
    if (!line) continue;
    let message;
    try { message = JSON.parse(line); } catch { continue; }
    handle(message).catch((error) => {
      if (message.id !== undefined) replyError(message.id, -32603, error.message);
    });
  }
});

function shutdown() {
  for (const session of sessions.values()) session.stop();
  process.exit(0);
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
process.stdin.on('end', shutdown);

