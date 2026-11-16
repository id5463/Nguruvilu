// mcp-proxy 面板：三模态切换 + 目标清单。
//
// 面板是沙箱 iframe，拿不到主页面的 DOM 和 token，只能通过 ngu.call 让宿主
// 执行命令。配置存在宿主的 settings 里（skill_mode 那套同款做法）。
//
// 高度按所有可见元素的底边算：菜单和列表里会有绝对定位的子元素，
// 量 body 会漏，宿主拿到的框就被裁掉了。

var MODES = [
  { id: 'proxy',  label: '合并', hint: '所有 MCP 合成一个 call 工具。工具最多的场景用这个。' },
  { id: 'single', label: '分开', hint: '每个 MCP 一个入口，形如 github_call。想按服务器分开预算时用。' },
  { id: 'full',   label: '全量', hint: '不代理，每个工具按原名暴露。只用于对照和调试。' }
];

var current = 'proxy';
var targets = [];
var openPath = [];
var btn = null, menuHost = null, noteEl = null, listBlock = null;
var listOpen = false;

function el(tag, cls, text) {
  var n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function modeLabel(id) {
  for (var i = 0; i < MODES.length; i += 1) if (MODES[i].id === id) return MODES[i].label;
  return id || '合并';
}

function reportHeight() {
  if (!document.body) return;
  var bottom = 0;
  var base = document.body.getBoundingClientRect().top;
  var nodes = document.body.querySelectorAll('*');
  for (var i = 0; i < nodes.length; i += 1) {
    var node = nodes[i];
    if (node.hidden) continue;
    var style = window.getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    var rect = node.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) continue;
    if (rect.bottom - base > bottom) bottom = rect.bottom - base;
  }
  var height = Math.ceil(bottom);
  if (height <= 1) height = 0;
  parent.postMessage(
    { ngu: 'panel', panelId: 'mcp-proxy', requestId: null,
      cmd: 'panel:resize', args: { height: height } }, '*');
}

function closeMenu() { openPath = []; paint(); }

function chooseMode(mode) {
  current = mode;
  window.ngu.call('set_mcp_proxy_mode', { mode: mode }).catch(function () {});
  paint();
}

function saveTargets(text) {
  // 每行一条：名字=命令 参数...
  var lines = String(text || '').split('\n')
    .map(function (line) { return line.trim(); })
    .filter(function (line) { return line.length > 0; });
  window.ngu.call('set_mcp_proxy_targets', { targets: lines })
    .catch(function () {});
}

function renderMenu() {
  if (!menuHost) return;
  menuHost.replaceChildren();
  if (openPath.length === 0) { menuHost.hidden = true; reportHeight(); return; }
  menuHost.hidden = false;

  var col = el('div', 'mr-col');
  for (var i = 0; i < MODES.length; i += 1) {
    (function (mode) {
      var row = el('div', 'mr-item');
      var dot = el('span', 'mr-dot');
      if (mode.id === current) dot.classList.add('on');
      row.appendChild(dot);
      row.appendChild(el('span', 'mr-label', mode.label));
      row.appendChild(el('span', 'mr-hint', mode.hint));
      row.addEventListener('click', function (e) {
        e.stopPropagation();
        chooseMode(mode.id);
        closeMenu();
      });
      col.appendChild(row);
    })(MODES[i]);
  }
  menuHost.appendChild(col);

  var sep = el('div', 'mr-sep');
  menuHost.appendChild(sep);

  var listCol = el('div', 'mr-col');
  var head = el('div', 'mr-item');
  head.appendChild(el('span', 'mr-label', '目标清单'));
  head.appendChild(el('span', 'mr-hint', targets.length + ' 个'));
  head.addEventListener('click', function (e) {
    e.stopPropagation();
    listOpen = !listOpen;
    renderList();
    reportHeight();
  });
  listCol.appendChild(head);
  menuHost.appendChild(listCol);

  renderList();
  reportHeight();
}

function renderList() {
  if (!listBlock) return;
  listBlock.replaceChildren();
  if (!listOpen) { listBlock.hidden = true; return; }
  listBlock.hidden = false;

  var area = el('textarea', 'mr-textarea');
  area.placeholder = 'chrome-devtools=chrome-devtools-mcp --headless\ngithub=/path/to/github-mcp-server stdio --toolsets=repos';
  area.value = targets.join('\n');
  area.rows = Math.max(3, Math.min(8, targets.length + 1));
  area.addEventListener('change', function () { saveTargets(this.value); });
  area.addEventListener('click', function (e) { e.stopPropagation(); });
  listBlock.appendChild(area);

  var save = el('button', 'mr-btn', '保存清单');
  save.type = 'button';
  save.addEventListener('click', function (e) {
    e.stopPropagation();
    saveTargets(area.value);
    noteEl.textContent = '已保存，下一轮生效';
    reportHeight();
  });
  listBlock.appendChild(save);
}

function paint() {
  if (!btn) return;
  btn.textContent = 'MCP ' + modeLabel(current) + ' ' + (openPath.length ? '▴' : '▾');
  if (!openPath.length && noteEl && !noteEl.textContent) {
    noteEl.textContent = targets.length ? (targets.length + ' 个目标') : '没有配置目标';
  }
  renderMenu();
}

function build() {
  var root = el('div', 'mr-root');
  btn = el('button', 'mr-btn main');
  btn.type = 'button';
  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    if (openPath.length) closeMenu();
    else { openPath = ['__root__']; paint(); }
  });
  noteEl = el('span', 'mr-note');
  menuHost = el('div', 'mr-menu');
  menuHost.hidden = true;
  listBlock = el('div', 'mr-list');
  listBlock.hidden = true;
  root.appendChild(btn);
  root.appendChild(noteEl);
  root.appendChild(menuHost);
  root.appendChild(listBlock);
  document.body.appendChild(root);
  document.addEventListener('click', function (e) {
    if (!root.contains(e.target)) closeMenu();
  });
}

var style = document.createElement('style');
style.textContent = [
  '.mr-root{position:relative;display:block;font:12px/1.4 system-ui,sans-serif;color:#3a2e24;}',
  '.mr-btn{display:flex;justify-content:space-between;align-items:center;width:100%;box-sizing:border-box;',
  'border:1px solid #d5cbb9;background:#fff;color:#3a2e24;font:inherit;padding:7px 10px;',
  'border-radius:7px;cursor:pointer;text-align:left;}',
  '.mr-btn:hover{background:#f7f1e6;}',
  '.mr-note{display:block;margin-top:4px;color:#7d7264;font-size:11px;}',
  '.mr-menu{margin-top:6px;display:flex;flex-direction:column;background:#fffdf8;',
  'border:1px solid #e2d8c6;border-radius:8px;overflow:hidden;}',
  '.mr-menu[hidden]{display:none;}',
  '.mr-col{display:flex;flex-direction:column;padding:3px;}',
  '.mr-item{display:flex;align-items:center;gap:7px;padding:6px 9px;border-radius:6px;cursor:pointer;}',
  '.mr-item:hover{background:#f2ece0;}',
  '.mr-dot{width:7px;height:7px;border-radius:50%;border:1.5px solid #bcaa92;flex:0 0 auto;}',
  '.mr-dot.on{background:#b4632a;border-color:#b4632a;}',
  '.mr-label{flex:0 0 auto;}',
  '.mr-hint{color:#9a8d7c;font-size:11px;line-height:1.3;}',
  '.mr-sep{height:1px;background:#eee5d6;margin:3px 6px;}',
  '.mr-list{padding:4px 6px 6px;}',
  '.mr-list[hidden]{display:none;}',
  '.mr-textarea{width:100%;box-sizing:border-box;border:1px solid #ddd2bf;border-radius:6px;',
  'background:#fff;font:11px/1.45 ui-monospace,Menlo,Consolas,monospace;padding:6px;color:#3a2e24;resize:vertical;}',
  '.mr-textarea:focus{outline:none;border-color:#b4632a;}',
  '.mr-list .mr-btn{margin-top:5px;justify-content:center;font-size:11px;padding:5px 8px;}'
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
    current = ev.status.mcp_proxy_mode || current;
    targets = ev.status.mcp_proxy_targets || [];
    paint();
  }
  if (ev.ev === 'mcp_proxy_mode' && ev.mode) { current = ev.mode; paint(); }
});
