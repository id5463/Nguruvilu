#!/usr/bin/env node
// skill-router — 按需挑出相关技能，而不是把全部技能塞进提示词。
//
// 主人 2026-11-09 定的方案（L6074 / L6084 / L6094）：
//   「还有一个更好的检索器，那就是大语言模型。」
//   「那。jEv 判断性模型。」
//   「我觉得把这三个都用，然后做个切换可能更好一些。」
//
// 三个检索器：
//   1. lexical  — BM25 词法。快、免费、纯本地。英文技能描述效果好。
//   2. ngram    — 字符 n-gram。中文没有空格分词，词法检索会失效，这个兜住。
//   3. jev      — 交给 JeV（判断性模型）判断「这个技能是否需要」。最准，但要钱。
//
// 为什么不写进内核：主人明确说「我感觉这个功能是不是应该做成一个包，
// 毕竟我这个核心就是用包，然后结果这个这么重，然后做个包更好一些」。
//
// 为什么技能文件该用英文写：主人的原话是「使用中文反而会消耗更多的钱，
// 然后又得不到更好的效果」。这不只是省钱——检索器靠 description 的词面匹配，
// 而英文 description 的措辞（"Use this skill when…"）本身就是给模型看的信号。

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const PROTOCOL = '2024-11-05';
const SERVER = { name: 'skill-router', version: '1.0.0' };

const say = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');

// ---------------------------------------------------------------- 技能来源
//
// 技能文件在 `~/.nguruvilu/skills/<id>/SKILL.md`，包里的是
// `<pack>/skills/<id>/SKILL.md`。两处都扫，包优先（更具体）。

function skillRoots() {
  const roots = [];
  const home = process.env.NGU_HOME || path.join(os.homedir(), '.nguruvilu');
  roots.push(path.join(home, 'skills'));
  const packsDir = path.join(home, 'packs');
  try {
    for (const name of fs.readdirSync(packsDir)) {
      const skills = path.join(packsDir, name, 'skills');
      if (fs.existsSync(skills)) roots.push(skills);
    }
  } catch {
    /* 没有 packs 目录就是没有，不是错误 */
  }
  return roots;
}

// YAML frontmatter 的最小解析。
//
// 完整 YAML 解析器为了三个字段引一个依赖不值得，但「最小」不等于「偷懒」：
// 下面处理了技能文件里实际会出现的三种写法——单行、块标量（`>` / `|`）、
// 以及带引号的值。
function parseFrontmatter(text) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(text);
  if (!match) return { meta: {}, body: text };
  const [, head, body] = match;
  const meta = {};
  const lines = head.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const kv = /^([A-Za-z][\w-]*):\s*(.*)$/.exec(lines[i]);
    if (!kv) continue;
    const [, key, rawValue] = kv;
    const value = rawValue.trim();
    // 块标量：`>` 折叠成一行，`|` 保留换行。后面的缩进行都属于它。
    if (value === '>' || value === '|') {
      const collected = [];
      while (i + 1 < lines.length && /^\s+/.test(lines[i + 1])) {
        collected.push(lines[i + 1].trim());
        i += 1;
      }
      meta[key] = value === '>' ? collected.join(' ') : collected.join('\n');
      continue;
    }
    meta[key] = value.replace(/^["']|["']$/g, '');
  }
  return { meta, body };
}

function loadSkills() {
  const skills = [];
  const seen = new Set();
  for (const root of skillRoots()) {
    let entries;
    try {
      entries = fs.readdirSync(root, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const file = path.join(root, entry.name, 'SKILL.md');
      let text;
      try {
        text = fs.readFileSync(file, 'utf8');
      } catch {
        continue;
      }
      const { meta, body } = parseFrontmatter(text);
      const id = meta.name || entry.name;
      // 同名的后到的覆盖先到的：包里的比全局的更具体，而 skillRoots()
      // 已经保证了顺序。
      if (seen.has(id)) {
        const at = skills.findIndex((s) => s.id === id);
        if (at >= 0) skills.splice(at, 1);
      }
      seen.add(id);
      skills.push({
        id,
        description: meta.description || '',
        license: meta.license || '',
        // 规则的来源之一。作者在 SKILL.md 里写 `triggers:` 就能让 regex
        // 模态认识它，不需要动代码。
        triggers: meta.triggers || '',
        compatibility: meta.compatibility || '',
        allowedTools: meta['allowed-tools'] || '',
        path: file,
        dir: path.join(root, entry.name),
        bodyLength: body.length,
        // 正文前 400 字：检索时给 jev 一点额外上下文，但不把整个文件塞进提示词。
        excerpt: body.replace(/\s+/g, ' ').slice(0, 400),
      });
    }
  }
  return skills;
}

// ---------------------------------------------------------------- 分词
//
// 词法检索要的是「词」，而中英文的「词」不是一回事。
//
// 英文按空白与标点切；中文没有空格，但检索时不该退化成整句比较——
// 所以对含 CJK 的串额外按二元字（bigram）切。这不是完美的中文分词，
// 但对「找相关技能」够用，且不需要词典。

const STOP = new Set(
  ('a an and are as at be by for from has have how in into is it its of on or that the ' +
    'their then there these this to use used using was were what when where which who will ' +
    'with you your skill skills whenever anything something').split(' ')
);

function tokenize(text) {
  const lower = String(text || '').toLowerCase();
  const tokens = [];
  // 拉丁词
  for (const word of lower.split(/[^a-z0-9_\u4e00-\u9fff]+/)) {
    if (word.length > 1 && !STOP.has(word)) tokens.push(word);
  }
  // CJK 二元字：每个相邻两字一组
  for (const run of lower.match(/[\u4e00-\u9fff]+/g) || []) {
    for (let i = 0; i + 1 < run.length; i += 1) tokens.push(run.slice(i, i + 2));
  }
  return tokens;
}

// ---------------------------------------------------------------- 1. BM25
//
// 经典词法检索。参数用常见的 k1=1.2 / b=0.75；这个语料（几十个技能）
// 远小于 BM25 调参有意义的规模，取默认值即可。

function bm25(skills, query, limit) {
  const docs = skills.map((s) => {
    const text = `${s.id} ${s.description} ${s.excerpt}`;
    const tokens = tokenize(text);
    const freq = new Map();
    for (const t of tokens) freq.set(t, (freq.get(t) || 0) + 1);
    return { skill: s, freq, length: tokens.length };
  });
  const avgLength = docs.reduce((a, d) => a + d.length, 0) / (docs.length || 1);
  const queryTerms = [...new Set(tokenize(query))];
  const N = docs.length;
  const k1 = 1.2;
  const b = 0.75;

  // 每个查询词出现在多少文档里
  const df = new Map();
  for (const term of queryTerms) {
    let count = 0;
    for (const doc of docs) if (doc.freq.has(term)) count += 1;
    df.set(term, count);
  }

  const scored = docs.map((doc) => {
    let score = 0;
    for (const term of queryTerms) {
      const f = doc.freq.get(term) || 0;
      if (f === 0) continue;
      const n = df.get(term) || 0;
      // BM25 的 idf，加 0.5 平滑，避免负值
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * doc.length) / avgLength)));
    }
    return { id: doc.skill.id, score: Math.round(score * 1000) / 1000, why: 'bm25' };
  });

  return scored
    .filter((r) => r.score > 0)
    .sort((a, b2) => b2.score - a.score)
    .slice(0, limit);
}

// ---------------------------------------------------------------- 2. 字符 n-gram
//
// 中文没有空格，词法检索会把整句当一个词。字符 n-gram（这里用 2-gram 与
// 3-gram 的并集）不依赖分词，对「中文技能描述 + 中文提问」这种组合更稳。
//
// 这也顺手解决了拼写变体：`jev` / `gEv` / `jEv` 归一化后 n-gram 高度重叠。

function charNgrams(text, n) {
  const clean = String(text || '').toLowerCase().replace(/\s+/g, '');
  const out = new Set();
  for (let i = 0; i + n <= clean.length; i += 1) out.add(clean.slice(i, i + n));
  return out;
}

function ngram(skills, query, limit) {
  const q2 = charNgrams(query, 2);
  const q3 = charNgrams(query, 3);
  if (q2.size === 0) return [];

  const scored = skills.map((skill) => {
    const text = `${skill.id} ${skill.description}`;
    const d2 = charNgrams(text, 2);
    const d3 = charNgrams(text, 3);
    let hit = 0;
    for (const g of q2) if (d2.has(g)) hit += 1;
    for (const g of q3) if (d3.has(g)) hit += 1.5;
    // 用查询自身的规模归一，长查询不会因为 gram 多而天然得高分
    const score = hit / (q2.size + q3.size * 1.5);
    return { id: skill.id, score: Math.round(score * 1000) / 1000, why: 'ngram' };
  });

  return scored
    .filter((r) => r.score > 0.02)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

// ---------------------------------------------------------------- 3. JeV
//
// JeV 是判断性模型：不做聊天，回答类型化的问题。它的 `noul` 类型就是
// 「这个技能是否需要」的概率——正是检索要的信号。
//
// 主人实测过一次（三个技能，判断全对，成本约 $0.00002）。所以这里按
// 「每个技能问一次」来设计，而不是让模型一次吐出排名：判断性模型的输出
// 是概率，概率才可比。

async function jev(skills, query, limit) {
  const endpoint = process.env.NGU_JEV_ENDPOINT
    || 'https://api.commandcode.ai/provider/v1/systemone';
  const model = process.env.NGU_JEV_MODEL || 'typesafe/jev-1.14.0';
  const key = process.env.NGU_JEV_KEY
    || process.env.CODECOMMAND_API_KEY
    || '';
  if (!key) {
    return {
      error: 'jev 未配置：需要 NGU_JEV_KEY（或 CODECOMMAND_API_KEY）环境变量',
      results: [],
    };
  }

  const asked = skills.map(async (skill) => {
    const body = {
      model,
      input: {
        question: query,
        context: `Skill: ${skill.id}\nDescription: ${skill.description}`,
      },
      // 判断性模型要的是「问句 + 物品」，不是对话。
      answers: { needed: { type: 'noul' } },
    };
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${key}`,
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(15000),
      });
      if (!response.ok) {
        return { id: skill.id, score: 0, why: `jev http ${response.status}` };
      }
      const data = await response.json();
      // 形状：{ answers: { needed: { type: "noul", noul: 0.83 } } }
      const probability = data?.answers?.needed?.noul;
      const score = typeof probability === 'number' ? probability : 0;
      return { id: skill.id, score: Math.round(score * 1000) / 1000, why: 'jev' };
    } catch (error) {
      return { id: skill.id, score: 0, why: `jev ${error.message}` };
    }
  });

  const all = await Promise.all(asked);
  return {
    results: all
      .filter((r) => r.score > 0.15)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit),
  };
}

// ---------------------------------------------------------------- 4. LLM 直选
//
// 判断性模型（jev）一次只能答一个问题，所以要 N 次；大语言模型可以一次
// 看完整个目录，权衡之后再回答。这是「更好但更贵」的那条路。
//
// 这里最容易出的错不是问得不好，而是**信了模型的回答**：它会编出不存在的
// 技能名（尤其目录里名字相似时），也会把「看起来相关」的都给出来。
// 所以返回的每个名字都必须在技能表里真的存在，否则丢掉——静默丢掉也不行，
// 要报告，因为那说明提示词里的目录和磁盘上的技能对不上。

async function llmRoute(skills, query, limit) {
  const endpoint = process.env.NGU_LLM_ENDPOINT
    || process.env.NGU_BASE_URL
    || 'https://api.commandcode.ai/provider/v1';
  const model = process.env.NGU_LLM_MODEL || process.env.NGU_MODEL || '';
  const key = process.env.NGU_LLM_KEY || process.env.NGU_API_KEY
    || process.env.CODECOMMAND_API_KEY || '';
  if (!key || !model) {
    return { error: 'llm 未配置：需要 NGU_LLM_KEY（或 NGU_API_KEY）与 NGU_LLM_MODEL', results: [] };
  }

  const catalog = skills
    .map((s) => `- ${s.id}: ${s.description.slice(0, 300)}`)
    .join('\n');

  const body = {
    model,
    temperature: 0,
    messages: [
      {
        role: 'system',
        content:
          'You select which skills to load for a request. Answer with JSON only: ' +
          '{"skills":[{"name":"<exact name from the list>","why":"<short>"}]}. ' +
          'Pick at most ' + limit + '. If none apply, return {"skills":[]}. ' +
          'Use only names that appear in the list. Do not invent names.',
      },
      {
        role: 'user',
        content: `Available skills:\n${catalog}\n\nRequest:\n${query}`,
      },
    ],
  };

  try {
    const response = await fetch(`${endpoint.replace(/\/$/, '')}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) {
      return { error: `llm http ${response.status}`, results: [] };
    }
    const data = await response.json();
    const text = data?.choices?.[0]?.message?.content || '';
    const json = /\{[\s\S]*\}/.exec(text);
    if (!json) return { error: `llm 回答不是 JSON：${text.slice(0, 120)}`, results: [] };
    let parsed;
    try {
      parsed = JSON.parse(json[0]);
    } catch (error) {
      return { error: `llm JSON 解析失败：${error.message}`, results: [] };
    }
    const known = new Map(skills.map((s) => [s.id, s]));
    const results = [];
    const invented = [];
    for (const item of parsed.skills || []) {
      const name = String(item?.name || '').trim();
      if (!name) continue;
      if (!known.has(name)) {
        invented.push(name);
        continue;
      }
      results.push({
        id: name,
        score: 1 - results.length * 0.01,
        via: ['llm'],
        why: String(item?.why || '').slice(0, 200),
        description: known.get(name).description,
        path: known.get(name).path,
      });
    }
    // 编出来的名字要报告：它意味着提示词里的目录和磁盘上的技能不一致，
    // 而那是个真问题，不是模型的坏习惯。
    return invented.length
      ? { results, notes: [`llm 提到了不存在的技能：${invented.join(', ')}`] }
      : { results };
  } catch (error) {
    return { error: `llm ${error.message}`, results: [] };
  }
}

// ---------------------------------------------------------------- 2b. regex
//
// 纯规则，不经过任何模型。Codex 的做法是把「什么时候用这个技能」写成
// 结构化提示词交给模型；这里更进一步：规则本身就能直接判。
//
// 规则从三处取，就近覆盖：
//   1. SKILL.md 的 frontmatter `triggers:`（作者自己写，最准）
//   2. skill-router 包的 rules.json（管理员为整个机器写）
//   3. 从 description 里抽 `Use when …` 从句（退路，聊胜于无）
//
// 这一层存在的理由不只是省钱：它**可预测**。jev 和 llm 都会偶发失灵，
// 而一条正则不会。要「一定要加载」的东西（比如安全协议、交付规范）
// 应当挂在规则上，而不是指望模型每次都判断对。

function rulesFor(skill, rulesFile) {
  const patterns = [];
  // 1. frontmatter triggers: 一个技能可以写多条，逗号或换行分隔
  const declared = skill.triggers || '';
  for (const part of declared.split(/[,\n]/)) {
    const p = part.trim();
    if (p) patterns.push({ pattern: p, why: 'frontmatter triggers' });
  }
  // 2. 全局规则表
  for (const entry of rulesFile.skills?.[skill.id] || []) {
    pattern_push(patterns, entry);
  }
  // 3. description 的 "Use when …" 从句
  const useWhen = /Use when ([^.]*)/i.exec(skill.description || '');
  if (useWhen) {
    // 把 "A, B, or C" 拆成多个关键词，逐个当规则
    for (const word of useWhen[1].split(/,|\bor\b/)) {
      const w = word.trim().replace(/^the\s+/i, '');
      if (w.length >= 3 && w.length <= 40) {
        patterns.push({ pattern: w, why: 'description use-when' });
      }
    }
  }
  return patterns;
}

function pattern_push(into, raw) {
  if (typeof raw === 'string') {
    into.push({ pattern: raw, why: 'rules.json' });
  } else if (raw && typeof raw === 'object' && raw.pattern) {
    into.push({ pattern: raw.pattern, why: raw.why || 'rules.json' });
  }
}

function loadRules() {
  const home = process.env.NGU_HOME || path.join(os.homedir(), '.nguruvilu');
  // 包自己的 rules.json 优先，其次全局
  const candidates = [
    path.join(home, 'packs', 'skill-router-1.0.0', 'rules.json'),
    path.join(home, 'skill-router-rules.json'),
  ];
  for (const file of candidates) {
    try {
      return JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch {
      /* 没有就是没有规则，不是错误 */
    }
  }
  return {};
}

function regexRoute(skills, query, limit) {
  const rulesFile = loadRules();
  const scored = [];
  for (const skill of skills) {
    const patterns = rulesFor(skill, rulesFile);
    const hits = [];
    for (const rule of patterns) {
      let re;
      try {
        // `/pattern/flags` 写成完整正则；否则当关键词（大小写不敏感）
        const asRegex = /^\/(.*)\/([a-z]*)$/.exec(rule.pattern);
        re = asRegex ? new RegExp(asRegex[1], asRegex[2]) : new RegExp(rule.pattern, 'i');
      } catch {
        continue; // 一条坏规则不该让整个检索失败
      }
      if (re.test(query)) hits.push(`${rule.pattern} (${rule.why})`);
    }
    if (hits.length) {
      scored.push({
        id: skill.id,
        // 命中条数越多越靠前；同分时按 id 稳定排序，保证可重复
        score: hits.length,
        via: ['regex'],
        why: hits.join(' | ').slice(0, 300),
        description: skill.description,
        path: skill.path,
      });
    }
  }
  return scored
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, limit);
}

// ---------------------------------------------------------------- 融合
//
// 三个检索器各自的排序不同，分数也不同量纲（BM25 是无上界的正数，
// ngram 是 0..1，jev 是概率）。把它们直接相加是错的——BM25 会永远赢。
//
// 用倒数排名融合（RRF）：只看名次，不看分数。这是多个异质量纲检索器
// 合并的标准做法，也正好对得上主人的「三个都用」。

function rrf(lists, k = 60) {
  const merged = new Map();
  for (const list of lists) {
    list.forEach((item, index) => {
      const score = 1 / (k + index + 1);
      const entry = merged.get(item.id) || { id: item.id, score: 0, via: [] };
      entry.score += score;
      entry.via.push(item.why);
      merged.set(item.id, entry);
    });
  }
  return [...merged.values()]
    .sort((a, b) => b.score - a.score)
    .map((e) => ({ id: e.id, score: Math.round(e.score * 10000) / 10000, via: e.via }));
}

// ---------------------------------------------------------------- 工具

const TOOLS = [
  [
    'skill_route',
    'Pick the skills relevant to a request, instead of putting every skill in the ' +
      'prompt. Three retrievers are available and can be switched or combined: ' +
      '`lexical` (BM25, local and free, best for English descriptions), ' +
      '`ngram` (character n-grams, needed for Chinese where there are no spaces), ' +
      '`jev` (a decision model that answers "is this skill needed" as a probability; ' +
      'most accurate, costs money). `hybrid` runs all three and merges by rank.',
    {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'The user request, verbatim. Do not summarise it: the retrievers ' +
            'match on the words the user actually used.',
        },
        method: {
          type: 'string',
          enum: ['full', 'regex', 'jev', 'llm', 'lexical', 'ngram', 'hybrid'],
          description:
            'Which mechanism decides what to load. Four modes, plus three ' +
            'single-retriever modes for testing them in isolation. ' +
            '`full` loads everything (no selection at all); ' +
            '`regex` applies rules without any model; ' +
            '`jev` asks a decision model for each skill whether it is needed; ' +
            '`llm` asks a language model to pick from the whole catalog at once. ' +
            'Default: whatever NGU_SKILL_MODE says, else `hybrid`.',
        },
        limit: { type: 'number', description: 'How many skills to return. Default 5.' },
      },
      required: ['query'],
    },
  ],
  [
    'skill_list',
    'List the skills that exist, with the checks the Agent Skills specification ' +
      'requires (agentskills.io/specification): `name` must be 1-64 characters of ' +
      'lowercase letters, digits and hyphens, and must not start or end with a hyphen ' +
      'or contain a doubled hyphen; `description` must be non-empty and at most 1024 ' +
      'characters. Reports violations rather than hiding them.',
    {
      type: 'object',
      properties: {
        check: {
          type: 'boolean',
          description: 'Run the specification checks and report violations. Default true.',
        },
      },
    },
  ],
];

function checkSkill(skill) {
  const problems = [];
  const name = skill.id;
  if (!name) problems.push('缺少 name');
  else {
    if (name.length > 64) problems.push(`name 超过 64 字符（${name.length}）`);
    if (!/^[a-z0-9-]+$/.test(name)) {
      problems.push('name 只能用小写字母、数字和连字符');
    }
    if (/^-|-$/.test(name)) problems.push('name 不能以连字符开头或结尾');
    if (/--/.test(name)) problems.push('name 不能有连续连字符');
  }
  const d = skill.description || '';
  if (!d.trim()) problems.push('缺少 description');
  else if (d.length > 1024) problems.push(`description 超过 1024 字符（${d.length}）`);

  // 主人 2026-11-09 的要求：技能文件用英文写。
  // 「使用中文反而会消耗更多的钱，然后又得不到更好的效果。」
  const cjk = (d.match(/[\u4e00-\u9fff]/g) || []).length;
  if (cjk > 8) {
    problems.push(
      `description 含 ${cjk} 个汉字——技能文件应当用英文写：` +
        '英文的 token 更便宜，而且 description 的措辞本身就是检索信号'
    );
  }
  return problems;
}

function handle(name, args) {
  const skills = loadSkills();

  if (name === 'skill_list') {
    const shouldCheck = args.check !== false;
    const rows = skills.map((s) => ({
      id: s.id,
      description: s.description.slice(0, 200),
      path: s.path,
      problems: shouldCheck ? checkSkill(s) : [],
    }));
    const broken = rows.filter((r) => r.problems.length > 0);
    return {
      total: skills.length,
      broken: broken.length,
      skills: rows,
    };
  }

  if (name === 'skill_route') {
    const query = String(args.query || '');
    if (!query.trim()) throw new Error('query 不能为空：检索器要匹配用户实际用的词');
    const method = args.method || 'hybrid';
    const limit = Math.max(1, Math.min(Number(args.limit) || 5, 20));

    if (method === 'lexical') {
      return { method, query, results: bm25(skills, query, limit) };
    }
    if (method === 'ngram') {
      return { method, query, results: ngram(skills, query, limit) };
    }
    if (method === 'regex') {
      return { method, query, results: regexRoute(skills, query, limit) };
    }
    if (method === 'jev') {
      return jev(skills, query, limit).then((r) => ({ method, query, ...r }));
    }
    // `full` — 不选，全给。这是这个内核一直以来的行为，保留成一种模态：
    // 技能少于十几个时它其实最好（没有检索失败的可能），而且它是个参照物，
    // 切换其他模态时能对比「漏掉了什么」。
    if (method === 'full') {
      return {
        method,
        query,
        results: skills.map((s) => ({
          id: s.id,
          score: 1,
          via: ['full'],
          description: s.description,
          path: s.path,
        })),
      };
    }
    // `llm` — 让一个大语言模型直接判断该加载哪些。
    //
    // 和 `jev` 的区别是「问法」：jev 是判断性模型，只能回答一个概率，
    // 所以每个技能都要单独问一次；llm 能一次看完整个目录并给出理由，
    // 但会幻觉出不存在的技能，所以返回的名字必须逐个核对。
    if (method === 'llm') {
      return llmRoute(skills, query, limit).then((r) => ({ method, query, ...r }));
    }

    // hybrid：三路并行，按名次融合。jev 如果没配 key，就只用两路，
    // 并在结果里说明——静默降级会让人以为 jev 参与了。
    const lexical = bm25(skills, query, limit * 2);
    const chars = ngram(skills, query, limit * 2);
    return jev(skills, query, limit * 2).then((j) => {
      const lists = [lexical, chars];
      const notes = [];
      if (j.error) {
        notes.push(j.error + '（本次只用词法与字符两路）');
      } else {
        lists.push(j.results);
      }
      const merged = rrf(lists).slice(0, limit);
      const byId = new Map(skills.map((s) => [s.id, s]));
      return {
        method,
        query,
        notes,
        results: merged.map((m) => ({
          id: m.id,
          score: m.score,
          via: m.via,
          description: (byId.get(m.id) || {}).description || '',
          path: (byId.get(m.id) || {}).path || '',
        })),
      };
    });
  }

  throw new Error(`未知工具：${name}`);
}

// ---------------------------------------------------------------- MCP

async function main() {
  let buffer = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', async (chunk) => {
    buffer += chunk;
    let at;
    while ((at = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, at).trim();
      buffer = buffer.slice(at + 1);
      if (!line) continue;
      let request;
      try {
        request = JSON.parse(line);
      } catch {
        continue;
      }
      const { id, method, params } = request;
      try {
        if (method === 'initialize') {
          say({
            jsonrpc: '2.0',
            id,
            result: {
              protocolVersion: PROTOCOL,
              capabilities: { tools: {} },
              serverInfo: SERVER,
            },
          });
        } else if (method === 'tools/list') {
          say({
            jsonrpc: '2.0',
            id,
            result: {
              tools: TOOLS.map(([name, description, inputSchema]) => ({
                name,
                description,
                inputSchema,
              })),
            },
          });
        } else if (method === 'tools/call') {
          const result = await handle(params?.name, params?.arguments || {});
          say({
            jsonrpc: '2.0',
            id,
            result: {
              content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
            },
          });
        } else {
          say({ jsonrpc: '2.0', id, error: { code: -32601, message: 'unknown method' } });
        }
      } catch (error) {
        say({
          jsonrpc: '2.0',
          id,
          result: {
            content: [{ type: 'text', text: `错误：${error.message}` }],
            isError: true,
          },
        });
      }
    }
  });
}

main();
