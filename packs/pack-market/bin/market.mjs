#!/usr/bin/env node
// pack-market — stdio MCP server. Node built-ins only, no npm dependencies.
// Tools: market_search, market_info, market_install (download only, never installs).

import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
import { mkdir, writeFile, unlink, readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

const INDEX_ENV = 'NGU_MARKET_INDEX';
const NO_INDEX_MSG =
  '未配置商店索引地址：设置环境变量 NGU_MARKET_INDEX 为索引 JSON 的 URL';

const PROTOCOL_VERSION = '2024-11-05';

/* ------------------------------------------------------------------ helpers */

function textResult(text) {
  return { content: [{ type: 'text', text }] };
}

function jsonResult(value) {
  return textResult(JSON.stringify(value, null, 2));
}

class ToolError extends Error {}

function indexUrl() {
  const url = process.env[INDEX_ENV];
  if (!url || !url.trim()) throw new ToolError(NO_INDEX_MSG);
  return url.trim();
}

async function loadIndex() {
  const url = indexUrl();
  let res;
  try {
    res = await fetch(url, { headers: { accept: 'application/json' } });
  } catch (err) {
    throw new ToolError(`拉取索引失败：${url} — ${err && err.message ? err.message : err}`);
  }
  if (!res.ok) {
    throw new ToolError(`拉取索引失败：${url} — HTTP ${res.status}`);
  }
  let data;
  try {
    data = await res.json();
  } catch (err) {
    throw new ToolError(`索引不是合法 JSON：${url} — ${err && err.message ? err.message : err}`);
  }
  if (!data || !Array.isArray(data.packs)) {
    throw new ToolError(`索引格式不对：缺少 packs 数组（${url}）`);
  }
  return data.packs;
}

function findPack(packs, name) {
  const want = String(name == null ? '' : name).trim().toLowerCase();
  if (!want) throw new ToolError('缺少参数 name（包名）');
  return packs.find((p) => String(p && p.name ? p.name : '').toLowerCase() === want) || null;
}

function packSummary(p) {
  return {
    name: p.name,
    version: p.version || null,
    description: p.description || '',
    url: p.url || null,
  };
}

async function sha256File(path) {
  const buf = await readFile(path);
  return createHash('sha256').update(buf).digest('hex');
}

function marketDir() {
  return join(homedir(), '.nguruvilu', 'market');
}

function safeFileName(name, version) {
  const clean = (s) =>
    String(s == null ? '' : s)
      .replace(/[^A-Za-z0-9._-]+/g, '-')
      .replace(/^-+|-+$/g, '');
  return `${clean(name) || 'pack'}-${clean(version) || '0.0.0'}.ngupack`;
}

/* -------------------------------------------------------------------- tools */

const tools = [
  {
    name: 'market_search',
    description:
      '在远程包索引里搜索包：按 query 在 name / description / tags 中做不区分大小写的子串匹配。只返回摘要信息。',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '搜索词；留空返回前 limit 条' },
        limit: { type: 'number', description: '最多返回几条，默认 10' },
      },
      required: ['query'],
    },
  },
  {
    name: 'market_info',
    description: '查看某个包的完整索引条目（含 sha256、tags、dependencies）。',
    inputSchema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: '包名，精确匹配（不区分大小写）' },
      },
      required: ['name'],
    },
  },
  {
    name: 'market_install',
    description:
      '下载包文件到 ~/.nguruvilu/market/<name>-<version>.ngupack 并校验 sha256。只下载，不安装——是否安装由调用方决定。',
    inputSchema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: '包名，精确匹配（不区分大小写）' },
      },
      required: ['name'],
    },
  },
];

async function marketSearch(args) {
  const packs = await loadIndex();
  const query = String(args && args.query != null ? args.query : '').trim().toLowerCase();
  let limit = Number(args && args.limit != null ? args.limit : 10);
  if (!Number.isFinite(limit) || limit <= 0) limit = 10;
  limit = Math.min(Math.floor(limit), 200);

  const hits = packs.filter((p) => {
    if (!p) return false;
    if (!query) return true;
    const haystack = [p.name, p.description, Array.isArray(p.tags) ? p.tags.join(' ') : ''];
    return haystack.some((field) =>
      String(field == null ? '' : field).toLowerCase().includes(query),
    );
  });

  return jsonResult({
    query: args && args.query != null ? args.query : '',
    total: hits.length,
    returned: Math.min(hits.length, limit),
    packs: hits.slice(0, limit).map(packSummary),
  });
}

async function marketInfo(args) {
  const packs = await loadIndex();
  const pack = findPack(packs, args && args.name);
  if (!pack) {
    throw new ToolError(`索引里没有这个包：${args && args.name ? args.name : '(空)'}`);
  }
  return jsonResult(pack);
}

async function marketInstall(args) {
  const packs = await loadIndex();
  const pack = findPack(packs, args && args.name);
  if (!pack) {
    throw new ToolError(`索引里没有这个包：${args && args.name ? args.name : '(空)'}`);
  }
  if (!pack.url) throw new ToolError(`索引条目缺少 url：${pack.name}`);

  const target = join(marketDir(), safeFileName(pack.name, pack.version));
  await mkdir(marketDir(), { recursive: true });

  let res;
  try {
    res = await fetch(pack.url);
  } catch (err) {
    throw new ToolError(
      `下载失败：${pack.url} — ${err && err.message ? err.message : err}`,
    );
  }
  if (!res.ok) throw new ToolError(`下载失败：${pack.url} — HTTP ${res.status}`);

  const bytes = Buffer.from(await res.arrayBuffer());
  await writeFile(target, bytes);

  const actual = createHash('sha256').update(bytes).digest('hex');
  const expected = pack.sha256 ? String(pack.sha256).trim().toLowerCase() : null;
  const matched = expected ? actual === expected : false;

  if (expected && !matched) {
    try {
      await unlink(target);
    } catch {
      /* best effort */
    }
    throw new ToolError(
      `sha256 不匹配，已删除下载文件：${pack.name}@${pack.version || '?'} — 期望 ${expected}，实际 ${actual}`,
    );
  }

  // Re-read from disk so the reported hash is the one on disk, not just of the buffer.
  let onDisk = actual;
  try {
    onDisk = await sha256File(target);
  } catch {
    /* keep buffer hash */
  }

  return jsonResult({
    ok: true,
    path: target,
    sha256: onDisk,
    matched,
    installed: false,
    note: matched
      ? '已下载并校验通过。只下载，未安装。'
      : '索引未提供 sha256，已下载但无法校验。只下载，未安装。',
  });
}

const handlers = {
  market_search: marketSearch,
  market_info: marketInfo,
  market_install: marketInstall,
};

/* ---------------------------------------------------------------- JSON-RPC */

async function handleMessage(msg) {
  const { id, method, params } = msg || {};
  const isNotification = id === undefined || id === null;

  if (method === 'notifications/initialized' || method === 'initialized') {
    return null; // notification: never answer
  }

  if (method === 'initialize') {
    return {
      jsonrpc: '2.0',
      id,
      result: {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: 'pack-market', version: '1.0.0' },
      },
    };
  }

  if (method === 'ping') {
    return { jsonrpc: '2.0', id, result: {} };
  }

  if (method === 'tools/list') {
    return { jsonrpc: '2.0', id, result: { tools } };
  }

  if (method === 'tools/call') {
    const name = params && params.name;
    const args = (params && params.args) || (params && params.arguments) || {};
    const handler = handlers[name];
    if (!handler) {
      return {
        jsonrpc: '2.0',
        id,
        error: { code: -32602, message: `未知工具：${name}` },
      };
    }
    try {
      const result = await handler(args);
      return { jsonrpc: '2.0', id, result };
    } catch (err) {
      const message = err && err.message ? err.message : String(err);
      // Tool-level failures are reported as content + isError so the model can read them.
      return {
        jsonrpc: '2.0',
        id,
        result: { content: [{ type: 'text', text: message }], isError: true },
      };
    }
  }

  if (isNotification) return null;

  return {
    jsonrpc: '2.0',
    id,
    error: { code: -32601, message: `Method not found: ${method}` },
  };
}

function send(payload) {
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try {
    msg = JSON.parse(trimmed);
  } catch {
    send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } });
    return;
  }
  handleMessage(msg)
    .then((reply) => {
      if (reply) send(reply);
    })
    .catch((err) => {
      const id = msg && msg.id !== undefined ? msg.id : null;
      send({
        jsonrpc: '2.0',
        id,
        error: { code: -32603, message: err && err.message ? err.message : 'Internal error' },
      });
    });
});

rl.on('close', () => process.exit(0));
