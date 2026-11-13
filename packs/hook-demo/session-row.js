// 每个会话行末尾加一个消息数徽章。
// 一个点一个文件：同一个脚本挂到多个点，会让其中的 ngu.hook 重复注册。
ngu.hook("session.row.end", (row, session) => {
  const badge = document.createElement("span");
  badge.className = "hook-demo-badge";
  const n = session.messages === undefined ? 0 : session.messages;
  badge.textContent = n + "条";
  badge.title = "这个会话有 " + n + " 条消息";
  row.append(badge);
});
