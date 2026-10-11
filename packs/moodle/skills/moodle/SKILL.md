---
name: moodle
description: 操作 Herzen 大学 Moodle(moodle.herzen.spb.ru)—— 自动采集全部课程的作业清单(增量 diff)、读作业详情、下载附件、提交文件。当用户提到 Moodle、作业、задание、提交、дедлайн、课程作业时使用。
---

# Moodle 采集与提交(moodle-herzen MCP)

## 认证(全自动,无需干预)
- 凭据在 `~/.nguruvilu/moodle_auth.json`,会话缓存在 `~/.nguruvilu/moodle_session.json`。
- 服务器自动:cookie 有效 → 直接用;过期 → 用凭据重新登录并回写缓存。
- 若报 `login rejected`:让用户更新 moodle_auth.json 里的密码。

## 工具与推荐用法

1. **moodle_list_assignments** — 主力工具,给"作业又多了没"的答案。
   - 默认(不传参)= **只返回 new + changed**(对照上次扫描),上下文极省。
   - `refresh=true` = 重新爬全部课程(约 1–3 分钟,全量作业重建缓存)。
   - `new_only=false` = 输出完整清单(不再爬,读缓存)。
   - `types:["assign"]` 可只要提交类作业;`course_id` 限定单课程。
2. **moodle_get_assignment** — 单个作业详情:标题、课程、截止、提交状态、老师附件 URL、描述正文。
3. **moodle_download** — 把该作业的老师附件下到本地(默认 ./moodle_downloads)。
4. **moodle_submit** — 提交文件。**默认 dry_run=true 只探测不提交**;只有用户明确说"提交"时才传 `dry_run=false`。
5. **moodle_status / moodle_list_courses** — 登录态与课程清单。

## 工作流示例
- "Moodle 有什么新作业?" → `moodle_list_assignments`(默认 diff)
- "全面重扫一次" → `moodle_list_assignments {refresh:true, new_only:false}`
- "看 X 作业并下载资料" → `moodle_get_assignment` → `moodle_download`
- "把这个文件交上去" → 用户确认后 `moodle_submit {id, files, dry_run:false}`

## 注意
- 扫描覆盖 assign/quiz/forum/workshop 四类模块(以前只扫 assign 会漏)。
- 缓存位置 `~/.nguruvilu/moodle_cache.json`;出问题可删除该文件强制全量重建。
- 提交是写操作:除用户明确指示外一律保持 dry_run。
