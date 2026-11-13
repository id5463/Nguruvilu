// 上传进度浮层。
//
// 它挂在 above-composer 槽位上，用 CSS 把这个槽位变成固定在输入框上方的
// 浮层 —— 所以它既不吃掉页面的一行网格，也不在没有上传时占地方。
// 页面本身不知道这个控件的存在：它只把进度作为事件发出来。
let bar = null;

function ensure() {
  if (bar) return bar;
  bar = document.createElement("div");
  bar.className = "ub";
  bar.hidden = true;
  bar.innerHTML =
    '<div class="ub-row">' +
      '<span class="ub-name"></span>' +
      '<span class="ub-pct"></span>' +
      '<button type="button" class="ub-x" title="取消">×</button>' +
    '</div>' +
    '<div class="ub-track"><div class="ub-fill"></div></div>' +
    '<div class="ub-bytes"></div>';
  host.append(bar);
  return bar;
}

function fmt(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function show(p) {
  const el = ensure();
  if (p.done) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.querySelector(".ub-name").textContent = p.label || "";
  el.querySelector(".ub-pct").textContent = (p.percent || 0) + "%";
  el.querySelector(".ub-fill").style.width = (p.percent || 0) + "%";
  el.querySelector(".ub-bytes").textContent =
    fmt(p.loaded || 0) + " / " + fmt(p.total || 0);
}

// 进度只从上层面板自己的 ngu 收到；页面不拥有这个控件。
window.addEventListener("message", (event) => {
  const data = event.data;
  if (!data || data.ngu !== "upload-progress") return;
  show(data.progress || {});
});

// 首屏就建好，免得第一次上传时才发现容器不存在。
ensure();
