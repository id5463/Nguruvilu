// 自定义面板类型 "chart"。
// 包定义了"什么是一种图表面板"，别的面板只要写 "type": "chart" 就能用，
// 不用每次都带一份自己的渲染代码。这与 Patchouli 的 pageTypes 注册表同形。
const canvas = document.createElement("canvas");
canvas.width = Math.min(host.clientWidth || 260, 320);
canvas.height = 120;
canvas.style.width = "100%";
host.replaceChildren(canvas);

const ctx = canvas.getContext("2d");
if (!ctx) {
  host.textContent = "no canvas";
  return;
}
const values = Array.isArray(data) ? data : [];
if (values.length === 0) {
  host.textContent = "no data";
  return;
}
const max = Math.max(...values.map((v) => Number(v) || 0), 1);
const gap = 6;
const barW = (canvas.width - gap * (values.length + 1)) / values.length;
const style = getComputedStyle(document.documentElement);
const accent = style.getPropertyValue("--accent").trim() || "#b4632a";
const dim = style.getPropertyValue("--dim").trim() || "#7d7264";

ctx.font = "10px system-ui";
values.forEach((v, i) => {
  const n = Number(v) || 0;
  const h = Math.max(2, Math.round((n / max) * (canvas.height - 24)));
  const x = gap + i * (barW + gap);
  const y = canvas.height - h - 14;
  ctx.fillStyle = accent;
  ctx.fillRect(x, y, barW, h);
  ctx.fillStyle = dim;
  ctx.fillText(String(n), x, canvas.height - 3);
});
