// End-to-end driver for the ticket pack: one real orchestrator session, dispatched
// worker, wake-up on job settlement, and a self-review question — observed through
// the serve channel the phone uses.
//
// Cost is deliberate: one prompt, one worker, and the script prints what it saw so
// a human can judge the run without re-reading a transcript. Nothing here asserts
// model behaviour (that is not reproducible in a unit test); it produces evidence.
//
// Usage: node scripts/ticket-e2e.mjs [--port 8975] [--timeout 900]

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const PORT = argOf('port', '8975');
const TIMEOUT_MS = Number(argOf('timeout', '900')) * 1000;
const ROOT = path.resolve('..');
const HOME = process.env.E2E_HOME || fs.mkdtempSync(path.join(process.env.TEMP || 'C:/temp', 'ngu-e2e-'));
const EXE = path.resolve('desktop/target/debug/ngu-desktop.exe');
const BASE = `http://127.0.0.1:${PORT}`;

const PROMPT = [
  '用 ticket 包干活，严格按下面步骤，不要额外探索：',
  '1. 先用 bash 执行 pwd，得到工作区绝对路径。',
  '2. 加载技能 ticket-protocol。',
  '3. ticket_open 开一张票，goal 用这句原话：在工作区里建 ticket-e2e.md，写三行"今天做了什么"，每行以 - 开头。',
  '   done_when：文件存在且正好三行。boundary：只准新建 ticket-e2e.md，不要动别的文件。',
  '4. ticket_start_payload 取参数，把那段 JSON 原样交给 bash 工具起 worker（不要自己去搜 ngu 在哪里，命令里的路径就是对的）；然后 ticket_start 记账，把 --json 输出里的 session 也带上。',
  '5. 等作业结束的叫醒，然后 ticket_deliver 收回。',
  '6. 用 ticket_say_payload 要求 worker 自查：对照 goal 原文列出还差什么。同样用 bash 起。',
  '7. 再 ticket_deliver 一次，然后给我一行结论：做成了什么 + 文件在哪。',
].join('\n');

const server = spawn(EXE, ['--serve', `127.0.0.1:${PORT}`, '--token', 'e2e'], {
  cwd: path.resolve('.'),
  env: { ...process.env, NGU_HOME: HOME },
  stdio: ['ignore', 'pipe', 'pipe'],
});

let serverLog = '';
server.stderr.on('data', (c) => { serverLog += c.toString(); });
server.stdout.on('data', (c) => { serverLog += c.toString(); });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitReady() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`${BASE}/`);
      if (r.status === 200) return true;
    } catch { /* not up yet */ }
    await sleep(1000);
  }
  return false;
}

const seen = { steps: 0, tools: [], errors: [], markers: 0, wakeups: 0 };

async function drive() {
  const events = await fetch(`${BASE}/events`);
  const reader = events.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answer = '';

  const done = new Promise((resolve) => {
    (async () => {
      for (;;) {
        const { value, done: ended } = await reader.read();
        if (ended) return resolve('stream-ended');
        buffer += decoder.decode(value, { stream: true });
        for (;;) {
          const index = buffer.indexOf('\n\n');
          if (index < 0) break;
          const chunk = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          const line = chunk.split('\n').find((l) => l.startsWith('data:'));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5)); } catch { continue; }
          const name = ev.ev;
          if (name === 'step') seen.steps += 1;
          if (name === 'tool_start') {
            seen.tools.push(ev.name);
            if (ev.name.startsWith('mcp__ticket__')) console.log(`  · ${ev.name} ${(ev.arguments || '').slice(0, 120)}`);
            if (ev.name === 'bash') console.log(`  · bash ${(ev.arguments || '').slice(0, 160)}`);
          }
          if (name === 'tool_end') {
            const result = String(ev.result || '');
            if (result.includes('ticket-state')) seen.markers += 1;
            if (ev.name === 'job_output' || /job-\d+ (exited|finished|completed)/i.test(result)) seen.wakeups += 1;
            if (ev.ok === false) seen.errors.push(`${ev.name}: ${result.slice(0, 160)}`);
          }
          if (name === 'text') answer += ev.delta || '';
          if (name === 'turn_end') {
            console.log('\n--- 主对话结论 ---\n' + answer.trim());
            return resolve('turn_end');
          }
        }
      }
    })().catch(() => resolve('stream-error'));
  });

  await fetch(`${BASE}/cmd`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer e2e' },
    body: JSON.stringify({ cmd: 'prompt', text: PROMPT }),
  });

  const timer = new Promise((resolve) => setTimeout(() => resolve('timeout'), TIMEOUT_MS));
  return Promise.race([done, timer]);
}

function report(outcome) {
  const unique = [...new Set(seen.tools)];
  console.log('\n=== 观测 ===');
  console.log(`步数           : ${seen.steps}`);
  console.log(`工具调用       : ${seen.tools.length}（${unique.join(', ') || '无'}）`);
  console.log(`ticket 工具次数: ${seen.tools.filter((t) => t.startsWith('mcp__ticket__')).length}`);
  console.log(`面板标记到达   : ${seen.markers} 次（tool_end 里带 ticket-state）`);
  console.log(`失败的工具调用 : ${seen.errors.length}`);
  seen.errors.slice(0, 6).forEach((e) => console.log(`   ! ${e}`));

  // The worker writes into the *session's* workspace, which for a serve process is
  // `<home>/workspaces/<session-id>` — not where this script started. So search
  // under the home too, and report what was actually produced.
  console.log(`\n临时 home      : ${HOME}`);
  const found = [];
  const notes = [];
  const walk = (dir, depth = 0) => {
    if (depth > 6) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) { walk(full, depth + 1); continue; }
      if (['ticket.json', 'report.json', 'events.jsonl', 'brief.md'].includes(entry.name)) found.push(full);
      if (entry.name === 'ticket-e2e.md') notes.push(full);
    }
  };
  walk(HOME);
  walk(path.resolve('.'));
  console.log(`票文件         : ${found.length ? found.slice(0, 10).join('\n                  ') : '（没找到）'}`);
  for (const note of notes) {
    const body = fs.readFileSync(note, 'utf8');
    console.log(`\n=== worker 交付的文件（${note}）===\n${body}`);
    console.log(`行数校验       : ${body.trim().split('\n').length} 行（要求正好 3）`);
  }
  if (!notes.length) console.log('\nticket-e2e.md   : 没找到 —— worker 没交付，或写到了别的工作区');
  console.log(`\n结果           : ${outcome}`);
  // 上一版这里什么都不印，超时就成了无解的黑箱：宿主日志的最后几行往往是唯一证据。
  if (outcome !== 'turn_end') {
    console.log('\n--- 宿主日志尾部 ---');
    console.log(serverLog.split('\n').slice(-20).join('\n'));
  }
}

const outcome = (await waitReady()) ? await drive() : 'server-did-not-start';
report(outcome);
server.kill();
