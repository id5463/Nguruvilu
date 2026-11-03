#!/usr/bin/env node
// ticket-mcp — 票的记录与派活/回话的参数生成器。
//
// 设计立场（主人的两条）：
//  1. 不给 AI 设闸门：这里不审查措辞、不判断结论够不够"硬"、不因"可能重复"而拒绝。
//     规则由工具签名表达——想改时间只有 ticket_grant，想停只有内核的 job_kill，
//     想交付只有 worker 自己写 report.json。不需要拦截，因为没有别的入口。
//  2. 验收 = 对话：主代理跟 worker 说话，叫它自己审一遍。所以本服务器只负责
//     "把事实记准、把状态从事实推出来"，不负责裁决。
//
// 保留的东西只有三样，全都不是权限：
//  - 文件形状（写临时文件再 rename，票读不出来就报是哪个文件）
//  - 状态由盘上事实推导（有合法 report → delivered；没有 → stuck；没派过活 → queued）
//  - 事件日志（一行一条，事后唯一能查的东西）
//
// worker 进程一律由内核作业机制起（bash background:true + notify:true），本服务器
// 不起进程、不叫醒、不杀——那样"结束唤醒会话"才还在。

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const PROTOCOL = '2024-11-05';
const SERVER = { name: 'ticket', version: '1.0.0' };

const say = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');
const now = () => new Date().toISOString();
const short = (text, n = 200) => {
  const one = String(text ?? '').replace(/\s+/g, ' ').trim();
  return one.length > n ? one.slice(0, n) + '…' : one;
};

// 会话工作目录：MCP 进程的 cwd 是包目录，所以路径必须由调用方给出。
const rootOf = (args) => {
  const base = process.env.TICKET_HOME || (args && args.workspace);
  if (!base) throw new Error('错误：缺 workspace（票要写在会话的工作目录里）');
  return path.resolve(base, 'tickets');
};

const goalHash = (goal) => crypto.createHash('sha256').update(String(goal)).digest('hex').slice(0, 6);
// id 里必须带时分秒：只有日期时，同一天重复开同一个目标会算出同一个 id，
// 后一张票就把前一张覆盖掉——盘上的事实被静默改写，比任何措辞问题都严重。
// 时间戳手工拼装而不是切 ISO 字符串：ISO 里那个小数点会被切进 id。
// id 里必须带时分秒：只有日期时，同一天重复开同一个目标会算出同一个 id，
// 后一张票就把前一张覆盖掉——盘上的事实被静默改写，比任何措辞问题都严重。
const stamp = () => {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
};

const newId = (goal, root) => {
  const base = `t-${stamp()}-${goalHash(goal)}`;
  let id = base;
  // 秒级时间戳 + 目标哈希在同一秒内开同一目标会算出同一个目录，那会把前一张票
  // 静默覆盖掉。所以唯一性不靠时钟精度，靠"目录已存在就换后缀"——宁可多一个
  // 后缀，也不能让盘上的事实被改写。
  while (fs.existsSync(path.join(root, id))) {
    id = `${base}-${crypto.randomBytes(2).toString('hex')}`;
  }
  return id;
};

const readJson = (file) => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw new Error(`错误：${file} 读不出来 —— ${error.message}`);
  }
};

const writeJson = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value, null, 2) + '\n');
  fs.renameSync(tmp, file);
};

const load = (root, id) => {
  const ticket = readJson(path.join(root, id, 'ticket.json'));
  if (!ticket) throw new Error(`错误：没有票 ${id}（${path.join('tickets', id, 'ticket.json')}）`);
  return ticket;
};

const record = (root, ticket, event, extra) => {
  ticket.updated_at = now();
  writeJson(path.join(root, ticket.id, 'ticket.json'), ticket);
  fs.appendFileSync(
    path.join(root, ticket.id, 'events.jsonl'),
    JSON.stringify({ at: now(), event, state: ticket.state, ...(extra || {}) }) + '\n',
  );
  return ticket;
};

const list = (root) => {
  if (!fs.existsSync(root)) return [];
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => readJson(path.join(root, entry.name, 'ticket.json')))
    .filter(Boolean)
    .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
};

// 状态从事实推导，不靠调用方声明，也不拒绝任何调用。
const derive = (ticket, facts) => {
  if (facts.closed) return 'closed';
  if (facts.report === null) return ticket.worker.job ? 'stuck' : ticket.state;
  if (facts.report.bad) return 'stuck';
  if (facts.report.state === 'blocked') return 'stuck';
  return 'delivered';
};

const elapsed = (iso) => {
  if (!iso) return '-';
  const secs = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
  if (secs < 90) return `${secs}s`;
  if (secs < 5400) return `${Math.round(secs / 60)}min`;
  return `${(secs / 3600).toFixed(1)}h`;
};

// 面板从这个标记取数（宿主把 tool_end 的 result 转给面板帧）。保持在一行内。
const marker = (root) => {
  const rows = list(root);
  const count = (state) => rows.filter((t) => t.state === state).length;
  return `<!--ticket-state ${JSON.stringify({
    open: rows.filter((t) => t.state !== 'closed').length,
    running: count('running'),
    delivered: count('delivered'),
    stuck: count('stuck'),
    closed: count('closed'),
    attention: rows.filter((t) => t.state === 'stuck').map((t) => t.id),
  })}-->`;
};

const TOOLS = [
  ['ticket_open', '开一张票：目标原文冻结进 tickets/<id>/ticket.json，并生成交给 worker 的 brief.md。同目标已有活票时照开，只附一行提示。', {
    type: 'object',
    properties: {
      workspace: { type: 'string', description: '本会话的工作目录（绝对路径）' },
      goal: { type: 'string', description: '主人的原话，整段冻结' },
      origin: { type: 'string', description: 'person 或 external；只作为事实记录，不改变任何权限' },
      where: { type: 'string', description: '请求来自哪里' },
      boundary: { type: 'string', description: '一句话：这次不要碰什么' },
      done_when: { type: 'string', description: '什么算完成' },
      budget: { type: 'object', description: '{tool_timeout_secs,max_steps,wall_secs,profile}' },
    },
    required: ['workspace', 'goal'],
  }],
  ['ticket_board', '看板：扫描 tickets/，列出每张票的状态、已跑多久、job、最后声明。要想知道现在什么情况就读它，不要靠记忆。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string', description: '只看某一张的全文' } },
    required: ['workspace'],
  }],
  ['ticket_start_payload', '返回派活该用的 bash 参数（background:true, notify:true 都带好）。起进程的是内核，不是本服务器。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' }, exe: { type: 'string', description: 'ngu 可执行文件名，默认 ngu' } },
    required: ['workspace', 'id'],
  }],
  ['ticket_start', '把派活结果记进票：job id、worker 的 session id（后续回话要用它）。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' }, job: { type: 'string' }, session: { type: 'string' } },
    required: ['workspace', 'id', 'job'],
  }],
  ['ticket_deliver', '收回：读 worker 自己写的 report.json，状态由它决定（写了就 delivered，缺字段或 blocked 就 stuck）。声明本身就是完成判定。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' } },
    required: ['workspace', 'id'],
  }],
  ['ticket_grant', '给这张票追加预算（时间/步数/墙钟/工具面 profile）并重写 brief.md，所以**下一个** worker 看得到。正在跑的进程不因此改变。', {
    type: 'object',
    properties: {
      workspace: { type: 'string' }, id: { type: 'string' },
      tool_timeout_secs: { type: 'integer' }, max_steps: { type: 'integer' },
      wall_secs: { type: 'integer' }, profile: { type: 'string' },
    },
    required: ['workspace', 'id'],
  }],
  ['ticket_say_payload', '生成对 worker 说话用的 bash 参数（复用它的 session）。主代理用它要求 worker 自查、补证据、改方向——这是验收的唯一手段。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' }, text: { type: 'string', description: '要跟 worker 说的话' }, exe: { type: 'string' } },
    required: ['workspace', 'id', 'text'],
  }],
  ['ticket_stop', '关闭这张票并记下原因。真正停进程用内核对那个 job 的 job_kill。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' }, reason: { type: 'string' } },
    required: ['workspace', 'id', 'reason'],
  }],
];

// ---------------------------------------------------------------- 实现

function briefOf(ticket, root) {
  const reportPath = path.join(root, ticket.id, 'report.json').replace(/\\/g, '/');
  const b = ticket.budget;
  return `# 任务 ${ticket.id}

## 目标（原话）
> ${ticket.goal}

${ticket.boundary ? `## 边界\n不要碰：${ticket.boundary}\n\n` : ''}${ticket.done_when ? `## 什么算完成\n${ticket.done_when}\n\n` : ''}## 交付动作
干完后把下面这段写进 \`${reportPath}\`（这是唯一的交付方式，写什么算完成由你判断）：

{
  "state": "done 或 blocked",
  "claim": "一句话说清你做成了什么",
  "artifacts": ["产物在哪：文件/分支/编号/URL"],
  "evidence": ["你跑了什么、结果如何"],
  "question": "blocked 时填：需要主代理或人决定的一件事"
}

## 预算
前台 ${b.tool_timeout_secs}s · ${b.max_steps} 步 · 墙钟 ${b.wall_secs}s · 工具面 ${b.profile}
（追加预算要由主代理改票后重开一轮，你说了不算。）
`;
}

function ticketOpen(args) {
  const root = rootOf(args);
  const goal = String(args.goal || '').trim();
  if (!goal) throw new Error('错误：goal 为空');
  const id = newId(goal, root);
  const same = list(root).filter((t) => t.goal === goal && t.state !== 'closed');
  const ticket = {
    id,
    goal,
    goal_hash: goalHash(goal),
    origin: { kind: args.origin === 'external' ? 'external' : 'person', where: short(args.where || '', 120) },
    state: 'queued',
    created_at: now(),
    updated_at: now(),
    boundary: args.boundary ? String(args.boundary) : '',
    done_when: args.done_when ? String(args.done_when) : '',
    budget: { tool_timeout_secs: 300, max_steps: 60, wall_secs: 3600, profile: 'worker-full', ...(args.budget || {}) },
    worker: { job: null, session: null, started_at: null },
    result: null,
  };
  fs.mkdirSync(path.join(root, id), { recursive: true });
  record(root, ticket, 'open', { same_goal_open: same.map((t) => t.id) });
  fs.writeFileSync(path.join(root, id, 'brief.md'), briefOf(ticket, root));
  return {
    text:
      `${id} 已开（queued）｜brief: tickets/${id}/brief.md｜预算 前台 ${ticket.budget.tool_timeout_secs}s/${ticket.budget.max_steps} 步/${ticket.budget.wall_secs}s/${ticket.budget.profile}\n` +
      (same.length ? `另有同目标的活票：${same.map((t) => `${t.id}(${t.state})`).join(', ')}\n` : '') +
      marker(root),
  };
}

function ticketBoard(args) {
  const root = rootOf(args);
  if (args.id) {
    const t = load(root, args.id);
    const events = (() => {
      try {
        return fs.readFileSync(path.join(root, t.id, 'events.jsonl'), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
      } catch {
        return [];
      }
    })();
    return { text: JSON.stringify(t, null, 2) + '\n事件：\n' + events.map((e) => `  ${e.at} ${e.event} ${e.state}`).join('\n') + '\n' + marker(root) };
  }
  const rows = list(root);
  if (!rows.length) return { text: `没有票（${root}）` };
  const lines = rows.map(
    (t) =>
      `${t.id}  ${t.state.padEnd(9)} ${elapsed(t.worker?.started_at).padStart(7)}  job=${t.worker?.job ?? '-'}  ${short(t.goal, 50)}` +
      (t.result?.claim ? `  | ${short(t.result.claim, 40)}` : ''),
  );
  return { text: lines.join('\n') + '\n' + marker(root) };
}

const pipe = (id, args, workspace, tail) => {
  // 宿主会把自己的可执行路径经 NGU_EXE 传进来：`ngu` 通常不在 bash 的 PATH 上
  // （桌面版直接从 target 目录跑就是这种情况），靠 PATH 找会白白烧掉好几步。
  const bin = args.exe || process.env.NGU_EXE || 'ngu';
  const reader = process.platform === 'win32' ? 'type' : 'cat';
  const brief = path.join(workspace, 'tickets', id, 'brief.md');
  return `${reader} "${brief}" | "${bin}" ${tail}`;
};

function ticketStartPayload(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  return {
    text: JSON.stringify({
      command: pipe(ticket.id, args, path.resolve(args.workspace), '--json --new-session'),
      background: true,
      notify: true,
    }),
  };
}

function ticketStart(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  ticket.worker = {
    job: String(args.job),
    session: args.session ? String(args.session) : ticket.worker.session,
    started_at: now(),
  };
  ticket.state = 'running';
  record(root, ticket, 'start', { job: ticket.worker.job, session: ticket.worker.session });
  return { text: `${ticket.id} running｜job ${ticket.worker.job}｜session ${ticket.worker.session || '（未记）'}\n${marker(root)}` };
}

function ticketDeliver(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  const report = readJson(path.join(root, ticket.id, 'report.json'));
  let facts;
  if (report === null) {
    facts = { report: null };
  } else {
    const missing = ['state', 'claim'].filter((key) => !report[key]);
    facts = { report: missing.length ? 'bad' : report, bad: missing.length > 0 };
  }
  ticket.state = derive(ticket, facts);
  if (report) ticket.result = report;
  record(root, ticket, 'deliver', {
    state: ticket.state,
    missing: facts.report === 'bad' ? ['state/claim 其一'] : undefined,
  });
  if (ticket.state === 'delivered') {
    return { text: `${ticket.id} delivered｜${short(report.claim, 200)}\n产物：${(report.artifacts || []).join(', ') || '-'}\n证据：${(report.evidence || []).join(' ; ') || '-'}\n${marker(root)}` };
  }
  if (report && report.state === 'blocked') {
    return { text: `${ticket.id} stuck｜worker 交回一个决定：${short(report.question || report.claim, 200)}\n${marker(root)}` };
  }
  return {
    text:
      `${ticket.id} ${ticket.state}｜` +
      (report === null ? '没有 report.json' : 'report.json 缺 state/claim') +
      `。要追问就用 ticket_say_payload。\n${marker(root)}`,
  };
}

function ticketGrant(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  const before = { ...ticket.budget };
  for (const key of ['tool_timeout_secs', 'max_steps', 'wall_secs']) {
    if (Number.isInteger(args[key]) && args[key] > 0) ticket.budget[key] = args[key];
  }
  if (args.profile) ticket.budget.profile = String(args.profile);
  record(root, ticket, 'grant', { from: before, to: ticket.budget });
  fs.writeFileSync(path.join(root, ticket.id, 'brief.md'), briefOf(ticket, root));
  return {
    text:
      `${ticket.id} 预算：前台 ${before.tool_timeout_secs}→${ticket.budget.tool_timeout_secs}s，` +
      `${before.max_steps}→${ticket.budget.max_steps} 步，墙钟 ${before.wall_secs}→${ticket.budget.wall_secs}s，profile ${before.profile}→${ticket.budget.profile}\n` +
      `brief.md 已重写；对下一个 worker 生效。\n${marker(root)}`,
  };
}

function ticketSayPayload(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  const text = String(args.text || '').trim();
  if (!text) throw new Error('错误：text 为空');
  const workspace = path.resolve(args.workspace);
  const session = ticket.worker.session;
  if (!session) {
    // 没有 session 就用任务书重开一轮：回话仍然成立，只是丢了它的记忆。
    return {
      text: JSON.stringify({
        command: pipe(ticket.id, args, workspace, '--json --new-session'),
        background: true,
        notify: true,
      }) + `\n（票里没有 worker 的 session，这条会重开一轮而不是接着上次说；要避免就 ticket_start 时带上 session。）`,
    };
  }
  const bin = args.exe || 'ngu';
  return {
    text: JSON.stringify({
      command: `"${bin}" --json -s ${session} -p ${JSON.stringify(text)}`,
      background: true,
      notify: true,
    }),
  };
}

function ticketStop(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  ticket.state = 'closed';
  record(root, ticket, 'stop', { reason: short(args.reason, 300), job: ticket.worker?.job });
  return { text: `${ticket.id} closed｜${short(args.reason, 200)}${ticket.worker?.job ? `｜进程用 job_kill 停（job ${ticket.worker.job}）` : ''}\n${marker(root)}` };
}

const HANDLERS = {
  ticket_open: ticketOpen,
  ticket_board: ticketBoard,
  ticket_start_payload: ticketStartPayload,
  ticket_start: ticketStart,
  ticket_deliver: ticketDeliver,
  ticket_grant: ticketGrant,
  ticket_say_payload: ticketSayPayload,
  ticket_stop: ticketStop,
};

function handle(msg) {
  const { id, method, params } = msg;
  if (method === 'initialize') {
    return say({
      jsonrpc: '2.0',
      id,
      result: {
        protocolVersion: PROTOCOL,
        capabilities: { tools: {} },
        serverInfo: SERVER,
        instructions: '票记事实与状态；验收靠主代理跟 worker 对话；起进程用内核作业。',
      },
    });
  }
  if (method === 'notifications/initialized' || method === 'initialized' || id === undefined) return;
  if (method === 'ping') return say({ jsonrpc: '2.0', id, result: {} });
  if (method === 'tools/list') {
    return say({ jsonrpc: '2.0', id, result: { tools: TOOLS.map(([name, description, inputSchema]) => ({ name, description, inputSchema })) } });
  }
  if (method === 'tools/call') {
    const handler = HANDLERS[params?.name];
    if (!handler) return say({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: `错误：未知工具 ${params?.name}` }], isError: true } });
    try {
      const out = handler(params?.arguments || {});
      return say({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: out.text }], isError: false } });
    } catch (error) {
      return say({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: error.message }], isError: true } });
    }
  }
  say({ jsonrpc: '2.0', id, error: { code: -32601, message: `unsupported method: ${method}` } });
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  for (;;) {
    const index = buffer.indexOf('\n');
    if (index < 0) break;
    const line = buffer.slice(0, index).trim();
    buffer = buffer.slice(index + 1);
    if (!line) continue;
    let msg;
    try { msg = JSON.parse(line); } catch { continue; }
    handle(msg);
  }
});
process.stdin.on('end', () => process.exit(0));
