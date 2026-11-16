#!/usr/bin/env node
// pack-market — stdio MCP server. Node built-ins only, no npm dependencies.
// Tools: market_search, market_info, market_install (download only, never installs).

import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
import { mkdir, writeFile, unlink, readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const INDEX_ENV = 'NGU_MARKET_INDEX';
/// The catalogue this pack publishes, used when no environment variable says
/// otherwise.
///
/// The bundled copy is still read first and still works offline — it is what
/// makes the market usable straight after install. But an entry's `url` is now
/// absolute (see `index.json`), so the bundled copy and the remote one describe
/// the same archives in the same way, and a machine that has never cloned this
/// repository can still download them.
const DEFAULT_INDEX_URL =
  'https://raw.githubusercontent.com/id5463/Nguruvilu/main/packs/pack-market/index.json';
/// Directory of the bundled index, when that is where the catalogue came from.
let bundledIndexDir = null;
/// The URL the catalogue came from, when it came over the network. A relative
/// `url` in an entry resolves against this, so a self-hosted index that names
/// `store/x.zip` keeps working wherever it is served from.
let remoteIndexBase = null;
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
  return url && url.trim() ? url.trim() : '';
}

async function loadIndex() {
  // No index configured: read the copy that ships with this pack.
  //
  // This used to refuse. That made the market a package you install and then
  // cannot use — it carries sixteen archives and an index describing them, and
  // it would not read its own files because an environment variable was unset.
  // A working default beats a correct error: the local catalogue is what most
  // sessions want, and `NGU_MARKET_INDEX` still wins when it is set.
  const url = indexUrl();
  if (!url) return loadBundledIndex();
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
  // A relative `url` is resolved against whatever served the catalogue, so a
  // self-hosted index keeps working wherever it is mounted.
  remoteIndexBase = url;
  return data.packs;
}

/// The index and archives this pack ships with.
///
/// Read from disk rather than over HTTP: they are inside the package, and a
/// marketplace that needs a web server to list what it already carries is one
/// more thing to go wrong before the first search.
async function loadBundledIndex() {
  const here = dirname(fileURLToPath(import.meta.url));
  const path = join(here, '..', 'index.json');
  // Remembered so a relative `url` in the entry can be resolved against it.
  // The archives this pack carries are named `store/x.zip`, which is meaningful
  // only relative to the index that names them — `fetch` rejects it outright
  // with "Failed to parse URL", so downloading from the bundled catalogue was
  // impossible until this was tracked.
  bundledIndexDir = dirname(path);
  let raw;
  try {
    raw = await readFile(path, 'utf8');
  } catch (err) {
    throw new ToolError(
      `没有可用的索引：未设置 ${INDEX_ENV}，自带的 ${path} 也读不到（${err.code || err.message}）`
    );
  }
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new ToolError(`自带的索引不是合法 JSON：${path}`);
  }
  if (!data || !Array.isArray(data.packs)) {
    throw new ToolError(`自带的索引缺少 packs 数组：${path}`);
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

  // A relative url means "next to the index that named it", and the bundled
  // catalogue is a file on disk. Node's `fetch` refuses `file://` outright
  // ("fetch failed", with no detail), so a local entry is read directly instead
  // of pretending to download it. Remote entries still go through `fetch`.
  const raw_url = String(pack.url);
  const is_remote = /^https?:/i.test(raw_url);

  let bytes;
  if (is_remote) {
    let res;
    try {
      res = await fetch(raw_url);
    } catch (err) {
      throw new ToolError(
        `下载失败：${raw_url} — ${err && err.message ? err.message : err}`,
      );
    }
    if (!res.ok) throw new ToolError(`下载失败：${raw_url} — HTTP ${res.status}`);
    bytes = Buffer.from(await res.arrayBuffer());
  } else if (remoteIndexBase) {
    // The catalogue came over the network but this entry is relative: resolve it
    // against the index URL rather than the filesystem.
    const resolved = new URL(raw_url, remoteIndexBase).toString();
    let res;
    try {
      res = await fetch(resolved);
    } catch (err) {
      throw new ToolError(
        `下载失败：${resolved} — ${err && err.message ? err.message : err}`,
      );
    }
    if (!res.ok) throw new ToolError(`下载失败：${resolved} — HTTP ${res.status}`);
    bytes = Buffer.from(await res.arrayBuffer());
  } else {
    // Relative to the bundled index, or absolute when the entry says so.
    const base = bundledIndexDir || process.cwd();
    const local_path = isAbsolute(raw_url) ? raw_url : join(base, raw_url);
    try {
      bytes = await readFile(local_path);
    } catch (err) {
      throw new ToolError(
        `读取失败：${local_path} — ${err.code || err.message}`
      );
    }
  }
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
