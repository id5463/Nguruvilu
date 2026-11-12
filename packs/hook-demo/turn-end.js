// 每轮结束后累加计数；状态栏那个 chip 由 status.rendered 画。
let turnCount = 0;
ngu.hook("turn.end", () => {
  turnCount += 1;
});
// 把计数放在一个可查询的地方，状态栏的钩子去读它。
window.__hookDemoTurns = () => turnCount;
