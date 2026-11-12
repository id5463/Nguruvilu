// 状态栏加一个回合计数。
ngu.hook("status.rendered", (status, strip) => {
  if (!strip) return;
  let chip = document.getElementById("hook-demo-turns");
  if (!chip) {
    chip = document.createElement("span");
    chip.id = "hook-demo-turns";
    chip.className = "hook-demo-chip";
    strip.append(chip);
  }
  const n = typeof window.__hookDemoTurns === "function" ? window.__hookDemoTurns() : 0;
  chip.textContent = "回合 " + n;
});
