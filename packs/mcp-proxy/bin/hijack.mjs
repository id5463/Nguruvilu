// 劫持：把别的包的 MCP 服务器换成自己的 shim。
//
// # 为什么是改文件
//
// 上一轮量过：内核一启动 MCP 服务器就开一个 `tokio::spawn` 的读循环，那条
// 管道的内核读取端归它自己。第三方从 `/proc/<pid>/fd/1` 能打开（实测能读到
// 一帧），但两个读者会**瓜分**数据——第二轮起双方都拿不到东西。所以事后
// 插进一条已经在跑的管道不可行。
//
// 可行的是在它启动**之前**换掉启动命令：内核按字母序逐个读
// `assembly.yaml`，本包装好时后面的还没读。把 `command:` 换成自己的
// `shim.mjs`，内核读到的就是新命令。
//
// # 为什么能安全恢复
//
// 每个被改的文件先存一份到 `~/.nguruvilu/hijack-backup/`，本进程退出时
// （正常、SIGINT、SIGTERM）全部还原。`shim.mjs` 本身是幂等的：某次崩溃留下
// 一个被改过的 assembly，下次启动会重新备份并覆盖，不会叠加成
// `shim.mjs shim.mjs python`。
//
// # 为什么不是病毒
//
// 病毒要藏。这个不藏：`call` 工具的 `action="hijacked"` 会把整张表念出来，
// 包括每个被改文件的原始命令。一个能自己交代清楚的劫持，本质是配置管理，
// 只是发生在启动窗口里。
//
// 它也不碰内核不认识的东西：只改带 `command:` 的 MCP 条目，只改那一行和
// 紧挨着的 `args:`，别的一个字节不动（注释、顺序、缩进都保留）。
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const SELF_IDS = new Set(['mcp-proxy', 'skill-router']);

function home() {
  return process.env.NGU_HOME || path.join(os.homedir(), '.nguruvilu');
}

export function packsDir() {
  return path.join(home(), 'packs');
}

function backupDir() {
  return path.join(home(), 'hijack-backup');
}

function statePath() {
  return path.join(backupDir(), 'hijack.json');
}

/// 本进程自己的 shim，绝对路径：内核会用别的 cwd 启动它。
export function shimPath() {
  const here = path.dirname(new URL(import.meta.url).pathname);
  return path.join(here, 'shim.mjs');
}

/// 一个包里声明了哪些 MCP 服务器。
///
/// 读 `mcp.json`：那是包公开的声明，读它不修改任何东西。
function serversOf(dir) {
  const file = path.join(dir, 'mcp.json');
  if (!fs.existsSync(file)) return [];
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
    return (parsed.servers || [])
      .map((s) => ({
        id: String(s.id || '').trim(),
        command: String(s.command || '').trim(),
        args: Array.isArray(s.args) ? s.args.map(String) : [],
      }))
      .filter((s) => s.id && s.command && !SELF_IDS.has(s.id));
  } catch {
    return [];
  }
}

/// 把一个 assembly 里某条 MCP 的 `command:` 换成 shim。
///
/// 只认紧跟 `id: <target>` 的那个 `command:`：一个 assembly 可能有多个
/// 服务器，改错了等于把别人的服务器也劫持了。
function rewrite(text, targetId) {
  const lines = text.split('\n');
  const out = [];
  let inTarget = false;
  let changed = false;
  let argsInjected = false;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const idMatch = /^(\s*)-\s+id:\s*(\S+)\s*$/.exec(line) || /^(\s*)id:\s*(\S+)\s*$/.exec(line);
    if (idMatch) {
      inTarget = idMatch[2] === targetId;
      out.push(line);
      continue;
    }
    if (!inTarget) {
      out.push(line);
      continue;
    }

    const cmdMatch = /^(\s*)command:\s*(.+?)\s*$/.exec(line);
    if (cmdMatch && !changed) {
      const [, indent, real] = cmdMatch;
      // 已经是 shim 就不再套一层。
      if (real.includes('shim.mjs')) {
        out.push(line);
        continue;
      }
      out.push(`${indent}command: node`);
      // 真实命令和它的参数要落在 `args:` 里。下一页找 `args:`；找不到就插一个，
      // 位置就在这一行下面 —— YAML 允许键按任意顺序出现。
      let argsAt = -1;
      for (let j = i + 1; j < lines.length; j += 1) {
        if (/^\s*args:/.test(lines[j])) { argsAt = j; break; }
        // 到了下一条目就停：那已经不是这条服务器的字段了。
        if (/^\s*-\s+id:/.test(lines[j])) break;
        if (/^\s*\w+:/.test(lines[j]) && /^\s*args:/.test(lines[j]) === false) {
          // 别的字段，继续往下看
        }
      }
      const injected = [
        `${indent}args:`,
        `${indent}- ${JSON.stringify(shimPath())}`,
        `${indent}- ${JSON.stringify(real)}`,
      ];
      if (argsAt < 0) {
        out.push(...injected);
        argsInjected = true;
      }
      changed = true;
      continue;
    }

    if (argsInjected && /^\s*args:/.test(line)) {
      // 已经注入过 args 了，原来那条不要重复
      continue;
    }
    if (/^\s*args:/.test(line) && changed && !argsInjected) {
      const [, indent] = /^(\s*)args:/.exec(line);
      out.push(`${indent}args:`);
      out.push(`${indent}- ${JSON.stringify(shimPath())}`);
      out.push(`${indent}- ${JSON.stringify('__REAL_COMMAND__')}`);
      argsInjected = true;
      continue;
    }
    out.push(line);
  }

  // 第二遍：把占位符换成真实命令。这样两种分支（插在 command 下面、
  // 或者接在已有 args 前面）走同一段收尾逻辑。
  let result = out.join('\n');
  if (argsInjected) {
    result = result.replace(JSON.stringify('__REAL_COMMAND__'), JSON.stringify(findRealCommand(text, targetId) || ''));
  }
  return { text: result, changed };
}

function findRealCommand(text, targetId) {
  const lines = text.split('\n');
  let inTarget = false;
  for (const line of lines) {
    const idMatch = /^(\s*)-\s+id:\s*(\S+)\s*$/.exec(line) || /^(\s*)id:\s*(\S+)\s*$/.exec(line);
    if (idMatch) { inTarget = idMatch[2] === targetId; continue; }
    if (!inTarget) continue;
    const cmdMatch = /^\s*command:\s*(.+?)\s*$/.exec(line);
    if (cmdMatch) return cmdMatch[1];
  }
  return null;
}

/// 劫持一个包。返回它做了什么。
function hijackPack(packDir, servers) {
  const assembly = path.join(packDir, 'assembly.yaml');
  if (!fs.existsSync(assembly)) {
    return { pack: path.basename(packDir), skipped: 'no assembly.yaml' };
  }
  let text = fs.readFileSync(assembly, 'utf8');
  let touched = 0;
  for (const server of servers) {
    const { text: next, changed } = rewrite(text, server.id);
    if (changed) {
      text = next;
      touched += 1;
    }
  }
  if (touched === 0) {
    return { pack: path.basename(packDir), skipped: 'nothing to rewrite' };
  }
  fs.mkdirSync(backupDir(), { recursive: true });
  const backup = path.join(backupDir(), `${path.basename(packDir)}.assembly.yaml`);
  if (!fs.existsSync(backup)) fs.writeFileSync(backup, fs.readFileSync(assembly));
  fs.writeFileSync(assembly, text);
  return { pack: path.basename(packDir), hijacked: touched };
}

/// 跑一遍：所有带 MCP 的、不是自己的包都劫持。
export function hijackAll() {
  const results = [];
  let entries = [];
  try {
    entries = fs.readdirSync(packsDir(), { withFileTypes: true });
  } catch {
    return results;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(packsDir(), entry.name);
    const servers = serversOf(dir);
    if (servers.length === 0) continue;
    results.push(hijackPack(dir, servers));
  }
  try {
    fs.mkdirSync(backupDir(), { recursive: true });
    fs.writeFileSync(
      statePath(),
      JSON.stringify({ at: new Date().toISOString(), shim: shimPath(), results }, null, 2),
    );
  } catch {
    /* 状态文件是给人看的，写不了不影响劫持本身 */
  }
  return results;
}

/// 全部还原。
export function restoreAll() {
  let names = [];
  try {
    names = fs.readdirSync(backupDir()).filter((n) => n.endsWith('.assembly.yaml'));
  } catch {
    return [];
  }
  const done = [];
  for (const name of names) {
    const pack = name.replace(/\.assembly\.yaml$/, '');
    try {
      fs.copyFileSync(path.join(backupDir(), name), path.join(packsDir(), pack, 'assembly.yaml'));
      fs.unlinkSync(path.join(backupDir(), name));
      done.push(pack);
    } catch {
      /* 还原失败就留着备份，下次启动还有机会 */
    }
  }
  return done;
}

export function hijackedReport() {
  try {
    return JSON.parse(fs.readFileSync(statePath(), 'utf8'));
  } catch {
    return { results: [] };
  }
}
