// A stand-in provider that records the exact request body and answers with a tiny
// non-streaming completion. Used to see the message order ngu actually sends — a
// 400 about system-message placement is not guessable from the assembly code, and
// reproducing it against the real gateway costs tokens for no extra information.
//
// Usage: node scripts/echo-provider.mjs [port]     (default 9099)
// Dumps each request to ./echo-requests.jsonl as {path, roles, count}.

import http from 'node:http';
import fs from 'node:fs';

const PORT = Number(process.argv[2] || 9099);
const LOG = process.env.ECHO_LOG || 'echo-requests.jsonl';

const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', (chunk) => { body += chunk; });
  req.on('end', () => {
    const response = JSON.stringify({
      id: 'echo',
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: 'echo',
      choices: [{ index: 0, message: { role: 'assistant', content: 'echoed' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    });
    if (req.url.includes('/chat/completions')) {
      let parsed = null;
      try { parsed = JSON.parse(body); } catch { /* record the raw shape anyway */ }
      const record = {
        at: new Date().toISOString(),
        path: req.url,
        count: parsed?.messages?.length ?? 0,
        roles: (parsed?.messages || []).map((m) => m.role + (typeof m.content === 'string' ? `:${m.content.slice(0, 28)}` : '')),
        tools: (parsed?.tools || []).length,
      };
      fs.appendFileSync(LOG, JSON.stringify(record) + '\n');
      console.log(JSON.stringify(record.roles));
    } else if (req.url.includes('/models')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ data: [{ id: 'echo', object: 'model' }] }));
      return;
    }
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(response) });
    res.end(response);
  });
});

server.listen(PORT, '127.0.0.1', () => console.log(`echo provider on http://127.0.0.1:${PORT}/v1  (log: ${LOG})`));
