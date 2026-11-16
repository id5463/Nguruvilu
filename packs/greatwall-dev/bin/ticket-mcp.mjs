#!/usr/bin/env node
// ticket-mcp — 票的记录、派活、续跑与回话。
//
// 立场（主人的）：
//  1. 不给 AI 设闸门：这里不审查措辞、不判断结论够不够"硬"、不因"可能重复"而拒绝。
//     规则由工具签名表达——想改预算只有 ticket_grant，想停只有 ticket_stop，
//     想交付只有 worker 自己写 report.json。没有别的入口，所以不需要拦截。
//  2. 验收 = 对话：主代理跟 worker 说话，叫它自己审一遍。
//  3. 两种派活：新起（干净、什么都不知道）与续跑（复用 session、记得上一轮的试错）。
//     哪个更好由主代理判断，服务器不替它选。
//  4. 只提示，不裁决：空转、连续卡住、证据不足都写进输出，让主代理看见，
//     但一律不阻止任何调用。
//
// 保留的东西只有三样，全都不是权限：
//  - 文件形状（写临时文件再 rename；票读不出来就报是哪个文件）
//  - 状态由盘上事实推导（有合法 report → delivered；没有 → stuck；没派过活 → queued）
//  - 事件日志（一行一条，事后唯一能查的东西）

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const PROTOCOL = '2024-11-05';
const SERVER = { name: 'ticket', version: '1.1.0' };

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
const stamp = () => {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
};

const newId = (goal, root) => {
  const base = `t-${stamp()}-${goalHash(goal)}`;
  // 秒级时间戳 + 目标哈希在同一秒内开同一目标会算出同一个目录，那会把前一张票
  // 静默覆盖掉。所以唯一性不靠时钟精度，靠"目录已存在就换后缀"——宁可多一个
  // 后缀，也不能让盘上的事实被改写。
  let id = base;
  while (fs.existsSync(path.join(root, id))) {
    id = `${base}-${crypto.randomBytes(2).toString('hex')}`;
  }
  return id;
};

// 读文件：解析失败返回 null 而不是抛错。
//
// 票板要扫一整排目录，其中一张票的 report.json 正被 worker 写到一半是常态
// （writeJson 先写 .tmp 再 rename，所以正式文件永远是完整的；但手工编辑、
// 磁盘满、跨版本字段不兼容都会留下坏 JSON）。一张坏票不该让整块面板报错。
const readJson = (file) => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
};

// 原子写：进程被杀在 write 和 rename 之间时，正式文件仍是上一版的完整内容。
// 临时名带 pid，两个进程同时写不同票不会互相覆盖临时文件。
const writeJson = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(value, null, 2) + '\n');
  fs.renameSync(tmp, file);
};

// 读-改-写的互斥。
//
// ticket.json 是整文件重写，两个进程同时 grant 同一张票时后写的赢，前一个的
// 状态变更就丢了——盘上的事实被静默改写。O_EXCL 创建锁文件是最便宜的原语：
// 不需要额外的依赖，崩溃留下的锁靠 mtime 超时回收。
const withLock = (ticketDir, fn) => {
  const lock = path.join(ticketDir, '.lock');
  const deadline = Date.now() + 5000;
  let fd = null;
  for (;;) {
    try {
      fd = fs.openSync(lock, 'wx');
      break;
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      // 崩溃留下的锁：主人进程已经不在了，时间也过了。10 秒比任何一次
      // 读-改-写都长，短超时只会把慢机器上的正常操作误判成死锁。
      try {
        const age = Date.now() - fs.statSync(lock).mtimeMs;
        if (age > 10_000) {
          fs.unlinkSync(lock);
          continue;
        }
      } catch {
        continue;
      }
      if (Date.now() > deadline) throw new Error(`错误：票被锁住超过 5 秒（${lock}）`);
      // 同步等一小会儿。MCP 是行协议的，这里阻塞几百毫秒没有别的东西可做。
      const until = Date.now() + 25;
      while (Date.now() < until);
    }
  }
  try {
    return fn();
  } finally {
    try { fs.closeSync(fd); } catch {}
    try { fs.unlinkSync(lock); } catch {}
  }
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

// 读-改-写一张票：锁住目录，重新读（别的进程可能刚改过），改，写回。
//
// 传入的 ticket 可能已经过期，所以锁内必须重新 load 一次——这正是加锁要
// 解决的问题，锁外读到的副本不能拿来写回。
const mutate = (root, id, fn) => withLock(path.join(root, id), () => fn(load(root, id)));

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
//
// partial 是新加的中间态：做了一半、剩下的说得清。没有它，worker 做到一半预算
// 不够时只能选 done（撒谎）或 blocked（显得失败），两样都在污染主代理的判断。
const derive = (ticket, facts) => {
  if (facts.closed) return 'closed';
  // 读不到 report：派过活就是 stuck（它跑完了却没交东西），没派过就保持原状。
  //
  // 不能回落到 ticket.state —— 上一次是 delivered 的话，这里会返回 delivered，
  // 而后面渲染交付详情时要读 report.proof，report 是 null 就整个调用崩掉。
  // 状态只能从这一次的事实推出来，不能继承上一次的结论。
  if (facts.report === null) return ticket.worker?.job ? 'stuck' : ticket.state;
  // 缺字段的判定在 facts.bad，不在 facts.report 上：report 那一栏装的是对象
  // 或字符串 'bad'，读 .bad 永远是 undefined，于是缺 state 的 report 会被
  // 当成合法交付。
  if (facts.bad) return 'stuck';
  if (facts.report.state === 'blocked') return 'stuck';
  if (facts.report.state === 'partial') return 'partial';
  return 'delivered';
};

// 一份 report 算不算"空转"：没有产物、也没有跑过任何命令。
//
// 这不是审判——worker 可能读了半小时代码才想明白。它只是把事实摆到主代理
// 面前：这一轮盘上没有留下任何可复核的东西。连续三轮都这样，多半是任务书
// 有问题或者 worker 在打转，值得主代理换个方式。
const isEmptyTurn = (report) => {
  if (!report) return false;
  const artifacts = Array.isArray(report.artifacts) ? report.artifacts : [];
  const proof = report.proof || {};
  const verified = Array.isArray(proof.verified) ? proof.verified : [];
  const legacy = Array.isArray(report.evidence) ? report.evidence : [];
  return artifacts.length === 0 && verified.length === 0 && legacy.length === 0;
};

const elapsed = (iso) => {
  if (!iso) return '-';
  const secs = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
  if (secs < 90) return `${secs}s`;
  if (secs < 5400) return `${Math.round(secs / 60)}min`;
  return `${(secs / 3600).toFixed(1)}h`;
};

// 面板从这个标记取数（宿主把 tool_end 的 result 转给面板帧）。保持在一行内。
//
// 每一行都是扁平对象，因为面板要从 HTML 注释里读 JSON：嵌套结构一旦被
// 正则截断就会静默丢字段。
const marker = (root) => {
  const rows = list(root);
  const count = (state) => rows.filter((t) => t.state === state).length;
  return `<!--ticket-state ${JSON.stringify({
    open: rows.filter((t) => t.state !== 'closed').length,
    running: count('running'),
    delivered: count('delivered'),
    partial: count('partial'),
    stuck: count('stuck'),
    closed: count('closed'),
    attention: rows.filter((t) => t.state === 'stuck' || t.state === 'partial').map((t) => t.id),
    tickets: rows
      .filter((t) => t.state !== 'closed')
      .slice(0, 12)
      .map((t) => ({
        id: t.id,
        state: t.state,
        age: elapsed(t.worker?.started_at) || '',
        goal: short(t.goal, 60),
        job: t.worker?.job || '',
        rounds: t.worker?.rounds || 0,
        empty: t.worker?.empty_streak || 0,
      })),
  })}-->`;
};

const TOOLS = [
  ['ticket_open', '开一张票：目标原文冻结进 tickets/<id>/ticket.json，并生成交给 worker 的 brief.md。同目标已有活票时照开，只附一行提示。tried/files/never_touch 这三项不是必填，但填了 worker 就不会重复你踩过的坑。', {
    type: 'object',
    properties: {
      workspace: { type: 'string', description: '本会话的工作目录（绝对路径）' },
      goal: { type: 'string', description: '主人的原话，整段冻结' },
      origin: { type: 'string', description: 'person 或 external；只作为事实记录，不改变任何权限' },
      where: { type: 'string', description: '请求来自哪里' },
      boundary: { type: 'string', description: '一句话：这次不要碰什么' },
      done_when: { type: 'string', description: '什么算完成' },
      tried: { type: 'array', items: { type: 'string' }, description: '你已经试过什么、为什么没成。worker 看到这段就不会重跑同一批命令' },
      files: { type: 'array', items: { type: 'string' }, description: '相关文件的当前状态，带行号最好：src/x.rs:27 要删这行' },
      never_touch: { type: 'array', items: { type: 'string' }, description: '具体到文件和行：这次别动 src/pack.rs:1433' },
      budget: { type: 'object', description: '{tool_timeout_secs,max_steps,wall_secs,profile}' },
    },
    required: ['workspace', 'goal'],
  }],
  ['ticket_board', '看板：扫描 tickets/，列出每张票的状态、已跑多久、job、最后声明。要想知道现在什么情况就读它，不要靠记忆。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string', description: '只看某一张的全文' } },
    required: ['workspace'],
  }],
  ['ticket_start_payload', '新起一个 worker：全新上下文，只读 brief.md，不知道上一轮发生过什么。要续跑已有的 session 用 ticket_resume_payload。返回的 bash 参数已带 background:true, notify:true；起进程的是内核，不是本服务器。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' }, exe: { type: 'string', description: 'ngu 可执行文件名，默认 ngu' } },
    required: ['workspace', 'id'],
  }],
  ['ticket_resume_payload', '续跑同一个 worker：复用它的 session，它记得上一轮读过什么、试过什么、为什么失败。这是自动续跑的手动版本——由你判断该不该继续，而不是让 worker 自己转下去。票里没有 session 时会退回新起一轮并说明。', {
    type: 'object',
    properties: {
      workspace: { type: 'string' }, id: { type: 'string' },
      text: { type: 'string', description: '这一轮要它做什么。空则只说"继续"' },
      exe: { type: 'string' },
    },
    required: ['workspace', 'id'],
  }],
  ['ticket_start', '把派活结果记进票：job id、worker 的 session id（续跑和回话都要用它）。session 从 ngu --json 的输出里取，它在第一轮跑完后才知道。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' }, job: { type: 'string' }, session: { type: 'string' } },
    required: ['workspace', 'id', 'job'],
  }],
  ['ticket_deliver', '收回：读 worker 自己写的 report.json，状态由它决定（done→delivered，partial→partial，blocked 或缺字段→stuck）。证据按 verified/observed/assumed 三级列出，哪些需要你自己复核一眼就看得出。', {
    type: 'object',
    properties: { workspace: { type: 'string' }, id: { type: 'string' } },
    required: ['workspace', 'id'],
  }],
  ['ticket_grant', '给这张票追加预算（时间/步数/墙钟/工具面 profile）、补记 tried/files/never_touch，并重写 brief.md，所以**下一个** worker 看得到。正在跑的进程不因此改变。', {
    type: 'object',
    properties: {
      workspace: { type: 'string' }, id: { type: 'string' },
      tool_timeout_secs: { type: 'integer' }, max_steps: { type: 'integer' },
      wall_secs: { type: 'integer' }, profile: { type: 'string' },
      tried: { type: 'array', items: { type: 'string' }, description: '追加到已有的 tried 后面' },
      files: { type: 'array', items: { type: 'string' } },
      never_touch: { type: 'array', items: { type: 'string' } },
    },
    required: ['workspace', 'id'],
  }],
  ['ticket_say_payload', '生成对 worker 说话用的 bash 参数（复用它的 session）。主代理用它要求 worker 自查、补证据、改方向——这是验收的唯一手段。和 ticket_resume_payload 是同一条命令，区别在意图：这个用来追问，那个用来让它接着干。', {
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

// ---------------------------------------------------------------- 任务书

// 行为准则。这不是权限，是工作标准——照 Codex /goal 的 continuation 模板改写，
// 保留七段里全部与"什么是真做完"有关的约束，去掉与它自己机制绑定的部分
// （token 记账、update_goal 调用、update_plan）。
//
// 为什么要写这么长：每一条都对应一种交付不了却看起来做完的方式。没有这一段，
// worker 有三种偷懒路径；写了只剩一种，而那一种主代理能从证据分级里看见。
const CONDUCT = `## 工作准则

**目标跨轮持续。** 这一轮做不完不代表要把目标缩小到现在能做完的部分。做了实质进展、
目标仍未达成时，把它留在未完成状态，下一轮接着做。完成永远要求目标里写的那件事
真的成立并被验证过。

**从证据出发。** 当前工作区和外部状态才是权威。之前的对话、任务书里的描述能帮你
定位，但依赖它们之前先看一眼实际状态。已有的实现可以改、可以换、可以删——
判断标准只有一条：它是否让目标更成立。

**分辨上一轮有没有进展。** 真进展改变盘上的状态、做完了活、或拿到能改变下一步的证据。
只重述状态、只写计划没有动作，都不算进展。有效等待指你确认某个具体的进程、job 或
工具句柄此刻还活着——光有对话、意图、上一次的输出、或者一个锁文件都不够。只有在
权威状态说它已经结束、或者句柄不存在了，才判定它停了；一次观察超时不是结束，
去查同一句柄或别的权威状态，不要因为超时就重开。

**别挑容易的做。** 不要因为某个改法更容易通过当前测试，就选一个更窄、更保守、更小、
只是兼容的版本。判断改动是否对路只有一条：它让目标里描述的最终状态更成立吗。
看起来有用但保留着另一个最终状态，就是不对路。

**完成之前先假定没完成。** 从目标和它引用的文件、计划、规格、issue、指令里，逐条推出
具体的要求。保持原范围，不要围绕"已经做了什么"重新定义成功。对每一条显式要求、
每个列出的条目、每个点名的产物、命令、测试、门槛、不变量和交付物：先说清什么样的证据
能证明它，再去看那个证据——文件、命令输出、测试结果、渲染产物、运行时行为。
然后判断：证明了、和完成矛盾、还是只做了一半、还是证据太弱太间接、还是根本没有。
验证的范围要配得上要求的范围，不要用一个窄检查去支撑一个宽结论。测试、清单、
绿色勾选、搜索结果，都要先确认它们确实覆盖了那条要求，才能当证据。
不确定或间接的证据一律算未达成。**审计要证明完成，而不是没能找到明显的遗留工作。**

不要拿意图、部分进展、对早先工作的记忆、或者一个听起来合理的最终答案当作完成的证明。
标记完成是一个主张：整个目标已经做完，经得起逐条追问。只有当前证据证明每一条要求
都满足、且没有必须做的工作剩下，才算完成。

**卡住的门槛。** 第一次遇到障碍时不要判定卡住。只有同一个阻塞条件连续出现至少三轮，
且你确实到了没有用户输入或外部状态变化就无法推进的地步，才算卡住。用户让一张被判
卡住的票继续之后，重新开始计数。达到门槛后不要再一边报"我仍然卡住"一边让票悬着，
直接交回 blocked。**"难"、"慢"、"不确定"、"没做完"、"需要澄清"都不是卡住的理由。**
不要因为预算快用完、或者因为你要停下来了，就把票标成完成。

**交回什么。** 做完、做了一半、卡住，都照实写进 report.json。做了一半就在 partial 里
说清剩下什么——这比谎报 done 有用得多。`;

// 报告格式。三级证据是这个包和普通"我干完了"之间唯一的区别：
// verified 是跑过命令看到结果，observed 是看见了没跑，assumed 是推测。
// 分级不是为了打分，是为了让主代理一眼看出哪几条要自己复核。
function reportSpec(reportPath) {
  return `## 交付动作

把结果写进 \`${reportPath}\`（这是唯一的交付方式）：

\`\`\`json
{
  "state": "done | partial | blocked",
  "claim": "一句话说清你做到了什么",
  "artifacts": [
    {"kind": "commit", "ref": "8d0e38c"},
    {"kind": "file", "ref": "src/tools/search.rs", "note": "已删除"},
    {"kind": "test", "ref": "cargo test --lib", "note": "736 passed"}
  ],
  "proof": {
    "verified": [{"cmd": "cargo check --offline -p nguruvilu", "exit": 0}],
    "observed": ["git status 显示 search.rs 已删除"],
    "assumed": ["服务器上的旧二进制不再需要"]
  },
  "remaining": "state 是 partial 时填：还剩什么没做",
  "question": "state 是 blocked 时填：需要主代理或人决定的一件事"
}
\`\`\`

**artifacts 是位置，不是叙述。** 主代理会去看那个位置，不会读你的形容。给不出位置
就说明这件事没有可复核的产物，那多半是没做完。

**proof 分三级，照实填。** verified 是你跑过的命令和它的退出码；observed 是你看到的
但没跑过的；assumed 是你推断的。填错等级比不填更糟——主代理会据此决定信不信你。`;
}

function listSection(title, items, empty) {
  if (!items || !items.length) return `## ${title}\n${empty}\n\n`;
  return `## ${title}\n${items.map((x) => `- ${x}`).join('\n')}\n\n`;
}

function briefOf(ticket, root) {
  const reportPath = path.join(root, ticket.id, 'report.json').replace(/\\/g, '/');
  const b = ticket.budget;
  const parts = [];

  parts.push(`# 任务 ${ticket.id}\n`);
  parts.push(`## 目标（原话）\n> ${ticket.goal}\n`);
  if (ticket.boundary) parts.push(`## 边界\n${ticket.boundary}\n`);
  if (ticket.done_when) parts.push(`## 什么算完成\n${ticket.done_when}\n`);

  // 这三节是主代理写给 worker 的"前情"。不填也照开票，但明说没记录——
  // worker 有权知道自己可能会重复主代理踩过的坑。
  parts.push(
    listSection(
      '已经试过的（别重复）',
      ticket.tried,
      '（主代理没有记录。你可能会重复它踩过的坑。）',
    ),
  );
  parts.push(
    listSection('相关文件的当前状态', ticket.files, '（主代理没有列。自己去读，别猜。）'),
  );
  if (ticket.never_touch && ticket.never_touch.length) {
    parts.push(listSection('具体别碰这些', ticket.never_touch, ''));
  }

  if (ticket.rounds && ticket.rounds.length) {
    const recent = ticket.rounds.slice(-3);
    parts.push(
      `## 前几轮交回过什么\n${recent
        .map((r) => `- 第 ${r.n} 轮：${r.state}｜${short(r.claim, 100)}`)
        .join('\n')}\n`,
    );
  }

  parts.push(CONDUCT + '\n');
  parts.push(reportSpec(reportPath) + '\n');
  parts.push(
    `## 预算\n前台 ${b.tool_timeout_secs}s · ${b.max_steps} 步 · 墙钟 ${b.wall_secs}s · 工具面 ${b.profile}\n` +
      '（追加预算要由主代理改票后重开一轮，你说了不算。）\n',
  );

  return parts.join('\n');
}

// ---------------------------------------------------------------- 实现

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
    tried: Array.isArray(args.tried) ? args.tried.map(String) : [],
    files: Array.isArray(args.files) ? args.files.map(String) : [],
    never_touch: Array.isArray(args.never_touch) ? args.never_touch.map(String) : [],
    budget: { tool_timeout_secs: 300, max_steps: 60, wall_secs: 3600, profile: 'worker-full', ...(args.budget || {}) },
    worker: { job: null, session: null, started_at: null, rounds: 0, empty_streak: 0 },
    rounds: [],
    result: null,
  };
  fs.mkdirSync(path.join(root, id), { recursive: true });
  record(root, ticket, 'open', { same_goal_open: same.map((t) => t.id) });
  fs.writeFileSync(path.join(root, id, 'brief.md'), briefOf(ticket, root));
  return {
    text:
      `${id} 已开（queued）｜brief: tickets/${id}/brief.md｜预算 前台 ${ticket.budget.tool_timeout_secs}s/${ticket.budget.max_steps} 步/${ticket.budget.wall_secs}s/${ticket.budget.profile}\n` +
      (ticket.tried.length || ticket.files.length
        ? `已带上：${ticket.tried.length} 条前情、${ticket.files.length} 个文件位置\n`
        : '⚠ 没带 tried/files —— worker 会重复你踩过的坑\n') +
      marker(root),
  };
}

function ticketBoard(args) {
  const root = rootOf(args);
  if (args.id) {
    const t = load(root, args.id);
    let events = [];
    try {
      events = fs
        .readFileSync(path.join(root, t.id, 'events.jsonl'), 'utf8')
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => { try { return JSON.parse(line); } catch { return null; } })
        .filter(Boolean);
    } catch {}
    return {
      text:
        JSON.stringify(t, null, 2) +
        '\n事件：\n' +
        events.map((e) => `  ${e.at} ${e.event} ${e.state}`).join('\n') +
        '\n' +
        marker(root),
    };
  }
  const rows = list(root);
  if (!rows.length) return { text: `没有票（${root}）` };
  const lines = rows.map((t) => {
    const rounds = t.worker?.rounds ? ` r${t.worker.rounds}` : '';
    const empty = t.worker?.empty_streak ? ` 空转${t.worker.empty_streak}` : '';
    return (
      `${t.id}  ${t.state.padEnd(9)} ${elapsed(t.worker?.started_at).padStart(7)}  job=${t.worker?.job ?? '-'}${rounds}${empty}  ${short(t.goal, 50)}` +
      (t.result?.claim ? `  | ${short(t.result.claim, 40)}` : '')
    );
  });
  return { text: lines.join('\n') + '\n' + marker(root) };
}

const pipe = (id, args, workspace, tail) => {
  // 宿主会把自己的可执行路径经 NGU_EXE 传进来：`ngu` 通常不在 bash 的 PATH 上
  // （桌面版直接从 target 目录跑就是这种情况），靠 PATH 找会白白烧掉好几步。
  const bin = args.exe || process.env.NGU_EXE || 'ngu';
  // `cat`, not `type`, even on Windows.
  //
  // The kernel's bash tool runs Git Bash on Windows (that is where POSIX
  // commands resolve and Windows paths still work), and `type` there is the
  // builtin alias for `type -p`, not the cmd.exe command that prints a file.
  // The command looked right and failed with "brief.md: not found" every time.
  const brief = path.join(workspace, 'tickets', id, 'brief.md');
  return `cat "${brief}" | "${bin}" ${tail}`;
};

function ticketStartPayload(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  return {
    text:
      JSON.stringify({
        command: pipe(ticket.id, args, path.resolve(args.workspace), '--json --new-session'),
        background: true,
        notify: true,
      }) +
      `\n（新起一个 worker：它不记得上一轮。要接着上次干就用 ticket_resume_payload。` +
      `跑完从 --json 输出里取 session_id，用 ticket_start 记进票，之后才能续跑。）`,
  };
}

function ticketResumePayload(args) {
  const root = rootOf(args);
  const ticket = load(root, args.id);
  const text = String(args.text || '继续。' ).trim() || '继续。';
  const session = ticket.worker?.session;
  const bin = args.exe || process.env.NGU_EXE || 'ngu';
  if (!session) {
    // 没有 session 就退回新起一轮。这仍然成立，只是丢了它的记忆——明说，不假装。
    return {
      text:
        JSON.stringify({
          command: pipe(ticket.id, args, path.resolve(args.workspace), '--json --new-session'),
          background: true,
          notify: true,
        }) +
        `\n（票里没有 worker 的 session，这条会新起一轮而不是接着上次：它读不到上一轮的试错。` +
        `要续跑就在 ticket_start 时带上 session。）`,
    };
  }
  return {
    text: JSON.stringify({
      command: `"${bin}" --json -s ${session} -p ${JSON.stringify(text)}`,
      background: true,
      notify: true,
    }),
  };
}

function ticketStart(args) {
  const root = rootOf(args);
  return {
    text: (() => {
      const ticket = mutate(root, args.id, (t) => {
        if (t.state === 'running' && t.worker?.job && String(t.worker.job) !== String(args.job)) {
          // 不拦，但要让人看见：两个 worker 同时在跑同一张票，后一个会覆盖前一个
          // 的 session，之后续跑就接错了进程。
          t.worker.previous_jobs = [...(t.worker.previous_jobs || []), t.worker.job];
        }
        t.worker.job = String(args.job);
        if (args.session) t.worker.session = String(args.session);
        t.worker.started_at = now();
        t.worker.job_started_at = now();
        t.worker.rounds = (t.worker.rounds || 0) + 1;
        t.state = 'running';
        record(root, t, 'start', { job: t.worker.job, session: t.worker.session, round: t.worker.rounds });
        return t;
      });
      const warn = ticket.worker.previous_jobs?.length
        ? `\n⚠ 这张票之前派过 ${ticket.worker.previous_jobs.length} 个 job（${ticket.worker.previous_jobs.join(', ')}）——确认它们已经结束，否则同一张票有两个 worker。`
        : '';
      return (
        `${ticket.id} running｜第 ${ticket.worker.rounds} 轮｜job ${ticket.worker.job}｜session ${ticket.worker.session || '（未记，无法续跑）'}` +
        warn +
        `\n${marker(root)}`
      );
    })(),
  };
}

// 这一份证据里最弱的那一级。
//
// 顺序是 assumed → observed → verified，从弱到强：只要有一条是推测，整份就不能
// 当"已验证"看；有观察没命令，就不能说"跑过了"。反过来写会把 4 条 verified
// 加 3 条 observed 的交付报成"只有观察，没有跑过命令"，主代理据此会以为 worker
// 什么都没跑，而实际上它跑了四条命令。
const levelOf = (proof, report) => {
  // 完全没有 proof 段，和"有 evidence 字符串数组的旧格式"是两回事：
  // 前者是漏填，后者是照旧规范交的。混成一个提示会让主代理以为 worker
  // 遵守了旧规范，而实际上它两样都没给。
  if (proof === undefined || proof === null) {
    const legacy = Array.isArray(report?.evidence) && report.evidence.length > 0;
    return legacy ? 'legacy' : 'none';
  }
  if (typeof proof !== 'object') return 'none';
  const v = Array.isArray(proof.verified) ? proof.verified.length : 0;
  const o = Array.isArray(proof.observed) ? proof.observed.length : 0;
  const a = Array.isArray(proof.assumed) ? proof.assumed.length : 0;
  if (!v && !o && !a) return 'none';
  if (a) return 'assumed';
  if (o) return 'observed';
  return 'verified';
};

const artifactLine = (artifacts) => {
  if (!artifacts || !artifacts.length) return '-';
  return artifacts
    .map((a) => (typeof a === 'string' ? a : [a.kind, a.ref, a.note].filter(Boolean).join(':')))
    .join(', ');
};

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

  const state = derive(ticket, facts);
  // 缺字段时 facts.report 是字符串 'bad'，坏 JSON 时 report 是 null。
  // 两者都不能当对象用：下面要读 .proof 和 .artifacts，一个字符串会让整个
  // 工具调用以 "Cannot read properties of null" 结束，主代理看到的是崩溃
  // 而不是"这个 report 有问题"。
  const usable = report && typeof report === 'object' && !Array.isArray(report);
  const empty = usable ? isEmptyTurn(report) : false;

  const done = mutate(root, args.id, (t) => {
    t.state = state;
    if (usable) t.result = report;
    if (usable) {
      const prev = t.rounds || [];
      prev.push({
        // 序号取历史条数 + 1，不取 worker.rounds：同一轮里 worker 可能改几次
        // report 再交，那几次都是"这一轮"的结果，但主代理看到的是先后几次交付，
        // 每一步都该有自己的编号，否则历史里出现三个"第 1 轮"读起来像 bug。
        n: prev.length + 1,
        at: now(),
        state: report.state || 'bad',
        claim: report.claim || '',
        empty,
      });
      t.rounds = prev.slice(-10);
    }
    // 空转连击：连续几轮盘上一个可复核的东西都没留下。只计数，不阻止。
    if (empty) {
      t.worker = t.worker || {};
      t.worker.empty_streak = (t.worker.empty_streak || 0) + 1;
    } else if (usable) {
      t.worker = t.worker || {};
      t.worker.empty_streak = 0;
    }
    record(root, t, 'deliver', { state, empty, missing: facts.report === 'bad' ? ['state/claim 其一'] : undefined });
    return t;
  });

  const streak = done.worker?.empty_streak || 0;
  const streakNote =
    streak >= 3
      ? `\n⚠ 连续 ${streak} 轮空转（没有产物、没跑过命令）。任务书可能不清楚，或者 worker 在打转——考虑用 ticket_resume_payload 换个问法，或者自己动手。`
      : '';

  if (state === 'delivered') {
    const level = levelOf(report.proof, report);
    const verified = report.proof?.verified || [];
    const assumed = report.proof?.assumed || [];
    const observed = report.proof?.observed || [];
    // 逐级报数量，不要只报"最弱的一级"。
    //
    // 一份报告常常是 4 条命令证据加 3 条观察，只说最弱那级会读成"什么都没跑"，
    // 主代理据此会要求重跑一遍——白烧一轮。弱点仍然要突出（推测必须点名），
    // 但有多少条硬的要说清楚。
    const bits = [];
    if (verified.length) bits.push(`${verified.length} 条命令证据`);
    if (observed.length) bits.push(`${observed.length} 条观察`);
    if (assumed.length) bits.push(`⚠ ${assumed.length} 条推测`);
    const levelNote =
      level === 'none'
        ? '⚠ report 里没有 proof 段'
        : level === 'legacy'
          ? '（旧格式 evidence，按推测对待）'
          : bits.join('，') +
            (assumed.length
              ? '——推测的几条需要你自己确认'
              : verified.length && !observed.length
                ? '——可直接采信'
                : '');
    return {
      text:
        `${ticket.id} delivered｜${short(report.claim, 200)}\n` +
        `产物：${artifactLine(report.artifacts)}\n` +
        (verified.length
          ? `verified：${verified.map((v) => `${v.cmd} (exit ${v.exit})`).join(' ; ')}\n`
          : '') +
        (report.evidence ? `evidence：${[].concat(report.evidence).join(' ; ')}\n` : '') +
        `证据：${levelNote}${streakNote}\n${marker(root)}`,
    };
  }

  if (state === 'partial') {
    return {
      text:
        `${ticket.id} partial｜${short(report.claim, 200)}\n` +
        `还剩：${short(report.remaining || '（worker 没说）', 200)}\n` +
        `产物：${artifactLine(report.artifacts)}\n` +
        `接着干用 ticket_resume_payload（复用它的 session），或 ticket_grant 追加预算后重开。${streakNote}\n${marker(root)}`,
    };
  }

  if (report && report.state === 'blocked') {
    return {
      text:
        `${ticket.id} stuck｜worker 交回一个决定：${short(report.question || report.claim, 200)}\n` +
        `如果它已经连续三轮报同一个理由，多半是真卡住；如果只是这一轮，用 ticket_resume_payload 换个问法。${streakNote}\n${marker(root)}`,
    };
  }

  return {
    text:
      `${ticket.id} ${state}｜` +
      (report === null ? '没有 report.json' : 'report.json 缺 state/claim') +
      `。要追问就用 ticket_say_payload。${streakNote}\n${marker(root)}`,
  };
}

function ticketGrant(args) {
  const root = rootOf(args);
  const out = mutate(root, args.id, (ticket) => {
    const before = { ...ticket.budget };
    for (const key of ['tool_timeout_secs', 'max_steps', 'wall_secs']) {
      if (Number.isInteger(args[key]) && args[key] > 0) ticket.budget[key] = args[key];
    }
    if (args.profile) ticket.budget.profile = String(args.profile);
    // 前情是累加的：主代理第二次 grant 时往往只想补一条，不该把第一次的清掉。
    for (const key of ['tried', 'files', 'never_touch']) {
      if (Array.isArray(args[key]) && args[key].length) {
        ticket[key] = [...(ticket[key] || []), ...args[key].map(String)];
      }
    }
    record(root, ticket, 'grant', { from: before, to: ticket.budget });
    fs.writeFileSync(path.join(root, ticket.id, 'brief.md'), briefOf(ticket, root));
    return { before, after: { ...ticket.budget } };
  });
  return {
    text:
      `${args.id} 预算：前台 ${out.before.tool_timeout_secs}→${out.after.tool_timeout_secs}s，` +
      `${out.before.max_steps}→${out.after.max_steps} 步，墙钟 ${out.before.wall_secs}→${out.after.wall_secs}s，profile ${out.before.profile}→${out.after.profile}\n` +
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
      text:
        JSON.stringify({
          command: pipe(ticket.id, args, workspace, '--json --new-session'),
          background: true,
          notify: true,
        }) + `\n（票里没有 worker 的 session，这条会重开一轮而不是接着上次说；要避免就 ticket_start 时带上 session。）`,
    };
  }
  const bin = args.exe || process.env.NGU_EXE || 'ngu';
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
  return {
    text: (() => {
      const ticket = mutate(root, args.id, (t) => {
        t.state = 'closed';
        record(root, t, 'stop', { reason: short(args.reason, 300), job: t.worker?.job });
        return t;
      });
      return `${ticket.id} closed｜${short(args.reason, 200)}${ticket.worker?.job ? `｜进程用 job_kill 停（job ${ticket.worker.job}）` : ''}\n${marker(root)}`;
    })(),
  };
}

const HANDLERS = {
  ticket_open: ticketOpen,
  ticket_board: ticketBoard,
  ticket_start_payload: ticketStartPayload,
  ticket_resume_payload: ticketResumePayload,
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
        instructions:
          '票记事实与状态；验收靠主代理跟 worker 对话；起进程用内核作业。' +
          '派活有两种：ticket_start_payload 新起（干净），ticket_resume_payload 续跑（记得上一轮）。',
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
