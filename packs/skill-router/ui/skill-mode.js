// 技能加载：一个按钮 + 多层菜单。
//
// 主人 2026-11-15：要点击按钮弹出多层菜单，不是手风琴——点一项弹下一级，
// 再点又弹下下级。菜单项可以带子菜单，也可以带一行可编辑的值。
//
// 面板是沙箱 iframe，拿不到主页面 DOM，只能 ngu.call 让宿主执行命令。
// 菜单画在 iframe 内部，靠报告高度把外框撑开。

var MODES = [
  { id: 'full',  label: '全量' },
  { id: 'regex', label: '规则' },
  { id: 'jev',   label: 'JeV' },
  { id: 'llm',   label: '大模型' }
];

// 菜单树。kind: 'mode' 选中模式 / 'field' 可编辑值 / 'act' 执行动作 / 'sep' 分隔线。
var TREE = [
  { kind: 'mode', mode: 'full', label: '全量', hint: '所有技能都进提示词' },
  { kind: 'mode', mode: 'regex', label: '规则', hint: '按 triggers 匹配，免费' },
  { kind: 'mode', mode: 'jev', label: 'JeV', hint: '判断模型给概率', children: [
    { kind: 'field', key: 'jev.endpoint', label: '端点', value: 'https://api.commandcode.ai/provider' },
    { kind: 'field', key: 'jev.model', label: '模型', value: 'typesafe/jev' },
    { kind: 'field', key: 'jev.threshold', label: '阈值', value: '0.5' },
    { kind: 'act', action: 'test', label: '测试连接' }
  ] },
  { kind: 'mode', mode: 'llm', label: '大模型', hint: '一次看完目录直选', children: [
    { kind: 'field', key: 'llm.endpoint', label: '端点', value: '' },
    { kind: 'field', key: 'llm.model', label: '模型', value: '' },
    { kind: 'field', key: 'llm.apiKey', label: '密钥', value: '', secret: true },
    { kind: 'act', action: 'test', label: '测试连接' }
  ] },
  { kind: 'sep' },
  { kind: 'act', action: 'list', label: '查看技能目录' }
];

var current = 'full';
var skills = [];
var openPath = [];
var btn = null, menuHost = null, noteEl = null;

function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function modeLabel(id) {
  for (var i = 0; i < MODES.length; i += 1) if (MODES[i].id === id) return MODES[i].label;
  return id || '全量';
}

function closeMenu() { openPath = []; paint(); }

function chooseMode(mode) {
  current = mode;
  window.ngu.call('set_skill_mode', { mode: mode }).catch(function () {});
  paint();
}

function saveField(key, value) {
  window.ngu.call('set_skill_config', { key: key, value: value }).catch(function () {});
}

function note(text) { if (noteEl) noteEl.textContent = text || ''; reportHeight(); }

function runAction(action) {
  if (action === 'test') {
    note('测试中…');
    window.ngu.call('test_skill_judge', {}).then(function (r) {
      note(r && r.ok ? ('正常 ' + (r.detail || '')) : ('失败：' + ((r && r.error) || '未知')));
    }).catch(function (e) { note('失败：' + e); });
  } else if (action === 'list') {
    note(skills.length ? (skills.length + ' 个：' + skills.join(', ')) : '没有技能');
  }
}

function renderMenu() {
  if (!menuHost) return;
  menuHost.replaceChildren();
  if (openPath.length === 0) { menuHost.hidden = true; reportHeight(); return; }
  menuHost.hidden = false;

  var items = TREE;
  for (var depth = 0; depth <= openPath.length; depth += 1) {
    var col = el('div', 'sr-col');
    for (var i = 0; i < items.length; i += 1) {
      (function (item, depth) {
        if (item.kind === 'sep') { col.appendChild(el('div', 'sr-sep')); return; }
        var row = el('div', 'sr-item');

        if (item.kind === 'mode') {
          var dot = el('span', 'sr-dot');
          if (item.mode === current) dot.classList.add('on');
          row.appendChild(dot);
        }
        row.appendChild(el('span', 'sr-label', item.label));
        if (item.hint) row.appendChild(el('span', 'sr-hint', item.hint));

        if (item.kind === 'field') {
          row.classList.add('sr-field');
          var input = el('input', 'sr-input');
          input.type = item.secret ? 'password' : 'text';
          input.value = item.value || '';
          input.placeholder = item.label;
          input.addEventListener('click', function (e) { e.stopPropagation(); });
          input.addEventListener('change', function () { saveField(item.key, this.value); });
          row.appendChild(input);
        } else {
          if (item.children && item.children.length) row.appendChild(el('span', 'sr-arrow', '>'));
          row.addEventListener('click', function (e) {
            e.stopPropagation();
            if (item.kind === 'mode') chooseMode(item.mode);
            if (item.kind === 'act') runAction(item.action);
            if (item.children && item.children.length) {
              openPath = openPath.slice(0, depth).concat([item.label]);
              paint();
            }
          });
        }
        col.appendChild(row);
      })(items[i], depth);
    }
    menuHost.appendChild(col);

    if (depth < openPath.length) {
      var want = openPath[depth], hit = null;
      for (var k = 0; k < items.length; k += 1) if (items[k].label === want) hit = items[k];
      if (!hit || !hit.children) break;
      items = hit.children;
    }
  }
  reportHeight();
}

function paint() {
  if (!btn) return;
  btn.textContent = '技能 ' + modeLabel(current) + ' ' + (openPath.length ? '▴' : '▾');
  if (!openPath.length && noteEl && !noteEl.textContent) {
    noteEl.textContent = skills.length ? (skills.length + ' 个技能') : '';
  }
  renderMenu();
}

// 报告高度。
//
// 菜单是绝对定位的：它不撑高 body，所以量 body 只能拿到按钮那一行，
// 弹出的菜单会被 iframe 裁掉——宿主给的框 40px，菜单有 175px。
// 这里量所有可见元素（包括绝对定位的菜单）的底边，取最大值。
function reportHeight() {
  if (!document.body) return;
  var bottom = 0;
  var nodes = document.body.querySelectorAll('*');
  var base = document.body.getBoundingClientRect().top;
  for (var i = 0; i < nodes.length; i += 1) {
    var el = nodes[i];
    if (el.hidden) continue;
    var style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    var r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (r.bottom - base > bottom) bottom = r.bottom - base;
  }
  var h = Math.ceil(bottom);
  if (h <= 1) h = 0;
  parent.postMessage(
    { ngu: 'panel', panelId: 'skill-mode', requestId: null,
      cmd: 'panel:resize', args: { height: h } }, '*');
}

function build() {
  var root = el('div', 'sr-root');
  btn = el('button', 'sr-btn');
  btn.type = 'button';
  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    if (openPath.length) closeMenu();
    else { openPath = ['__root__']; paint(); }
  });
  noteEl = el('span', 'sr-note');
  menuHost = el('div', 'sr-menu');
  menuHost.hidden = true;
  root.appendChild(btn);
  root.appendChild(noteEl);
  root.appendChild(menuHost);
  document.body.appendChild(root);
  document.addEventListener('click', function (e) {
    if (!root.contains(e.target)) closeMenu();
  });
}

var style = document.createElement('style');
style.textContent = [
  '.sr-root{position:relative;display:block;font:12px/1.4 system-ui,sans-serif;color:#3a2e24;}',
  '.sr-btn{display:flex;justify-content:space-between;align-items:center;width:100%;box-sizing:border-box;border:1px solid #d5cbb9;background:#fff;color:#3a2e24;font:inherit;padding:7px 10px;border-radius:7px;cursor:pointer;text-align:left;}',
  '.sr-btn:hover{background:#efe8da;}',
  '.sr-note{display:block;margin-top:4px;color:#7d7264;font-size:11px;}',
  '.sr-menu{margin-top:6px;display:flex;flex-direction:column;background:#fffdf8;border:1px solid #e2d8c6;border-radius:8px;overflow:hidden;}',
  '.sr-menu[hidden]{display:none;}',
  '.sr-col{display:flex;flex-direction:column;padding:3px;}',
  '.sr-col+.sr-col{border-top:1px solid #eee5d6;}',
  '.sr-item{display:flex;align-items:center;gap:6px;padding:5px 9px;border-radius:6px;cursor:pointer;white-space:nowrap;}',
  '.sr-item:hover{background:#f2ece0;}',
  '.sr-item.sr-field{cursor:default;}',
  '.sr-item.sr-field:hover{background:transparent;}',
  '.sr-dot{width:7px;height:7px;border-radius:50%;border:1.5px solid #bcaa92;flex:0 0 auto;}',
  '.sr-dot.on{background:#b4632a;border-color:#b4632a;}',
  '.sr-label{flex:1 1 auto;}',
  '.sr-hint{color:#9a8d7c;font-size:10px;}',
  '.sr-arrow{color:#bcaa92;font-size:10px;}',
  '.sr-sep{height:1px;background:#eee5d6;margin:4px 6px;}',
  '.sr-input{border:1px solid #ddd2bf;border-radius:5px;background:#fff;font:inherit;padding:2px 6px;width:200px;color:#3a2e24;}'
].join('');
document.head.appendChild(style);

build();
paint();

window.addEventListener('message', function (event) {
  var data = event.data;
  if (!data || data.ngu !== 'panel-event') return;
  var ev = data.event;
  if (!ev) return;
  if (ev.ev === 'status' && ev.status) {
    current = ev.status.skill_mode || current;
    skills = ev.status.skills || [];
    paint();
  }
  if (ev.ev === 'skill_mode' && ev.mode) { current = ev.mode; paint(); }
  if (ev.ev === 'skill_test') {
    note(ev.ok ? ('正常 ' + (ev.detail || '')) : ('失败：' + (ev.error || '')));
  }
});
