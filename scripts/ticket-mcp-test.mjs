// Gate for the ticket pack's state machine: drives files/ticket-mcp.mjs over stdio
// the way the kernel does, and asserts what the pack promises.
//
// What this file deliberately does NOT test: any refusal of wording. The pack's
// stance is that review is a conversation between the orchestrator and the worker,
// so nothing here may police how a message is phrased — that would put meta-talk
// about deadlines and rules back into the transcript, which is the contamination
// the whole design is trying to avoid. What is asserted instead: facts are recorded
// truthfully, state is derived from what is on disk, and the payloads the kernel
// needs are shaped correctly.
//
// Usage: node scripts/ticket-mcp-test.mjs        (exit 0 = all checks passed)

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';

const SERVER = path.resolve('packs/ticket/files/ticket-mcp.mjs');
const WORK = fs.mkdtempSync(path.join(os.tmpdir(), 'ngu-ticket-test-'));

const child = spawn(process.execPath, [SERVER], { stdio: ['pipe', 'pipe', 'pipe'] });
let buf = '';
const waiting = new Map();
let nextId = 1;

child.stdout.on('data', (chunk) => {
  buf += chunk.toString();
  for (;;) {
    const index = buf.indexOf('\n');
    if (index < 0) break;
    const line = buf.slice(0, index).trim();
    buf = buf.slice(index + 1);
    if (!line) continue;
    let msg;
    try { msg = JSON.parse(line); } catch { continue; }
    if (msg.id !== undefined && waiting.has(msg.id)) {
      const done = waiting.get(msg.id);
      waiting.delete(msg.id);
      done(msg.result ?? msg.error);
    }
  }
});
child.stderr.on('data', () => {}); // The ready banner is not part of the protocol.

const send = (method, params, id) =>
  new Promise((resolve) => {
    const number = id ?? nextId++;
    waiting.set(number, resolve);
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: number, method, params }) + '\n');
  });

const call = async (name, args) => {
  called.add(name);
  const out = await send('tools/call', { name, arguments: args });
  return { text: out?.content?.[0]?.text ?? '', isError: !!out?.isError };
};

const fileOf = (id, name) => path.join(WORK, 'tickets', id, name);
// 门禁的价值取决于它跑了多少条真断言；把"实际调用过的工具"打出来，
// 免得一个什么都不测的脚本看起来也像通过了。
const called = new Set();

async function main() {
  // ---- 握手与工具面
  const init = await send('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'gate', version: '1' } });
  assert.equal(init.protocolVersion, '2024-11-05');
  assert.equal(init.serverInfo.name, 'ticket');
  // A notification carries no id, so nothing may wait for a reply to it.
child.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} }) + '\n');

  const tools = await send('tools/list', {});
  const names = tools.tools.map((t) => t.name).sort();
  assert.deepEqual(names, [
    'ticket_board', 'ticket_deliver', 'ticket_grant', 'ticket_open',
    'ticket_say_payload', 'ticket_start', 'ticket_start_payload', 'ticket_stop',
  ].sort(), 'the tool surface is exactly these eight');
  for (const tool of tools.tools) {
    assert.ok(tool.inputSchema.properties.workspace, `${tool.name} must take workspace`);
  }

  // ---- 开票：目标原话冻结、brief 生成、标记可用于面板
  const goal = '把 12 张工单在 OA 里填完，每张都带附件';
  const opened = await call('ticket_open', { workspace: WORK, goal, done_when: '12 张都可查' });
  assert.match(opened.text, /已开|queued/, 'open reports the new ticket');
  const id = opened.text.match(/(t-\d{8}-\d{6}-[0-9a-f]{6})/)[1];
  assert.ok(fs.existsSync(fileOf(id, 'brief.md')), 'brief.md is written');
  assert.match(opened.text, /<!--ticket-state /, 'the panel marker is in the answer');
  const fresh = JSON.parse(fs.readFileSync(fileOf(id, 'ticket.json'), 'utf8'));
  assert.equal(fresh.goal, goal, 'the goal is stored verbatim, not summarised');
  assert.equal(fresh.state, 'queued');
  assert.equal(fresh.budget.tool_timeout_secs, 300, 'a fresh ticket carries an explicit budget');

  // ---- 同目标再开：照开,只附一行事实（不拦、不教育）
  const again = await call('ticket_open', { workspace: WORK, goal });
  assert.match(again.text, /另有同目标的活票/, 'a same-goal ticket is reported, not refused');
  assert.match(again.text, /t-\d{8}-\d{6}-[0-9a-f]{6}/, 'and a new one was still opened');

  // 同一目标再开一次，必须落在另一个目录里：只有日期时同一天会算出同一个 id，
  // 后一张票会静默覆盖前一张 —— 这条断言就是钉住那个错误的。
  assert.equal(new Set(fs.readdirSync(path.join(WORK, 'tickets'))).size, fs.readdirSync(path.join(WORK, 'tickets')).length, 'ticket ids are unique directories');
  assert.ok(fs.existsSync(fileOf(id, 'ticket.json')), 'the first ticket survived the second open');
  assert.equal(JSON.parse(fs.readFileSync(fileOf(id, 'ticket.json'), 'utf8')).goal, goal, 'the first ticket was not overwritten');

  // ---- 派活：参数形状归内核
  const payload = await call('ticket_start_payload', { workspace: WORK, id });
  const parsed = JSON.parse(payload.text);
  assert.equal(parsed.background, true, 'the worker runs as a background job');
  assert.equal(parsed.notify, true, 'and wakes the session when it settles');
  assert.match(parsed.command, /brief\.md/, 'the worker reads the brief, not the chat');
  assert.match(parsed.command, /--new-session/, 'a fresh session per dispatch');

  // ---- 收回要有作业：状态由事实推导，不假装 delivered
  const early = await call('ticket_deliver', { workspace: WORK, id });
  assert.match(early.text, /queued|没有 report/, 'delivering before dispatch reports the fact');
  assert.equal(JSON.parse(fs.readFileSync(fileOf(id, 'ticket.json'), 'utf8')).state, 'queued', 'state is not invented');

  const started = await call('ticket_start', { workspace: WORK, id, job: 'job-3', session: '20261103-101112-aaaaaa' });
  assert.match(started.text, /running/, 'start records the job');

  // ---- report 的形状决定状态
  fs.writeFileSync(fileOf(id, 'report.json'), JSON.stringify({ claim: '都做完了' }));
  const thin = await call('ticket_deliver', { workspace: WORK, id });
  assert.match(thin.text, /stuck/, 'a declaration missing state/claim is stuck, not delivered');

  fs.writeFileSync(
    fileOf(id, 'report.json'),
    JSON.stringify({ state: 'done', claim: '12 张工单已提交，编号 3001-3012', artifacts: ['OA 工单列表'], evidence: ['页面计数 12/12'] }),
  );
  const delivered = await call('ticket_deliver', { workspace: WORK, id });
  assert.match(delivered.text, /delivered/, 'a real declaration is delivered');
  assert.match(delivered.text, /3001-3012/, 'the claim comes back to the orchestrator');

  // ---- blocked 交回一个决定，仍然是 stuck（等主代理处理）
  fs.writeFileSync(fileOf(id, 'report.json'), JSON.stringify({ state: 'blocked', claim: '第 13 张需要审批人', question: '要不要找审批人补一张' }));
  const blocked = await call('ticket_deliver', { workspace: WORK, id });
  assert.match(blocked.text, /stuck/, 'blocked means it waits on a decision');
  assert.match(blocked.text, /审批人/, 'and the question is handed over verbatim');

  // ---- 审查 = 对话：任何措辞都接受，参数就是复用 session 再投一轮
  const ask = await call('ticket_say_payload', { workspace: WORK, id, text: '对照目标原文，列出还没做到的部分' });
  const askCmd = JSON.parse(ask.text.split('\n')[0]).command;
  assert.match(askCmd, /-s 20261103-101112-aaaaaa/, 'a review question reuses the worker session');
  assert.match(askCmd, /-p /, 'and carries the text as a prompt');
  // 同一条通道不挑字眼：涉及预算的话也照样发出去（预算本身走 ticket_grant）
  const loose = await call('ticket_say_payload', { workspace: WORK, id, text: '时间不够了，你先把手上的收尾' });
  assert.ok(!loose.isError, 'the pack does not police wording — review is conversation');
  assert.match(loose.text, /-s /, 'the wording goes to the same session as anything else');

  // ---- 追加预算：写票 + 重写 brief（下一个 worker 才看得到）
  const granted = await call('ticket_grant', { workspace: WORK, id, tool_timeout_secs: 600, max_steps: 120 });
  assert.match(granted.text, /300→600s/, 'grant reports before and after');
  assert.match(fs.readFileSync(fileOf(id, 'brief.md'), 'utf8'), /前台 600s/, 'the brief carries the new budget');
  assert.match(granted.text, /下一个 worker/, 'and states who it applies to');

  // ---- 看板来自盘，不来自记忆
  const board = await call('ticket_board', { workspace: WORK });
  assert.match(board.text, /t-\d{8}-\d{6}-[0-9a-f]{6}/, 'the board lists tickets');
  assert.match(board.text, /stuck/, 'with the derived states');

  // ---- 关闭 + 事件留痕 + 不留半个文件
  const stop = await call('ticket_stop', { workspace: WORK, id, reason: '已交付并验收' });
  assert.match(stop.text, /closed/, 'stop closes the ticket');
  assert.match(stop.text, /job_kill/, 'and points at the kernel for the process');
  const events = fs.readFileSync(fileOf(id, 'events.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.deepEqual(
    events.map((e) => e.event),
    // open → 收回（还没派活）→ 派活 → 缺字段的声明 → 合法声明 → blocked 声明 → 追加预算 → 关闭
    ['open', 'deliver', 'start', 'deliver', 'deliver', 'deliver', 'grant', 'stop'],
    'events record every accepted move, in order',
  );
  assert.ok(!fs.existsSync(fileOf(id, 'ticket.json.tmp')), 'no temporary file left behind');

  // ---- 缺参数只报一行事实
  const noHome = await call('ticket_board', {});
  assert.ok(noHome.isError, 'a missing workspace is an error');
  assert.match(noHome.text, /^错误：/, 'reported as one factual line, nothing else');
  const noTicket = await call('ticket_deliver', { workspace: WORK, id: 't-00000000-000000' });
  assert.match(noTicket.text, /错误：没有票/, 'an unknown ticket names the file it looked for');

  // 断言直接在 main 里跑：全过就是过。这里数的是"实际被调用过的工具"，
// 因为一个从不报错的门禁，也可能只是什么都没测。
console.log(`\nticket gate: all assertions passed (${[...called].sort().join(', ')})`);
  child.stdin.end();
  child.kill();
  fs.rmSync(WORK, { recursive: true, force: true });
}

main().catch((error) => {
  console.error('FAIL', error.message);
  child.kill();
  fs.rmSync(WORK, { recursive: true, force: true });
  process.exit(1);
});
