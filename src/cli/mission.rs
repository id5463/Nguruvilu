//! 任务条：给模型看的那五行，和怎么把它压成一行。
//!
//! `long-horizon` 包把任务落成 `mission.json`，这一层只读它在工作区里的记录、
//! 挑出重要的几行拼进提示词。它不写任务、不改状态 —— 写是包的 `mission_advance`。

use std::path::Path;

/// 一行任务文字最多留多少字符。
///
/// 任务条要短：它每轮都进提示词，一行占掉半个屏幕就不叫"条"了。
const MISSION_TEXT_CLIP: usize = 160;

/// 任务条里最多列几行。
///
/// 五行是"看得见全貌"和"不占预算"之间的取舍：更多行不会让人更清楚下一步。
const MISSION_PROMPT_ROWS: usize = 5;

/// One mission field as a single prompt-safe line.
///
/// Whitespace is flattened because a row that wraps breaks the list the block
/// is. The cut counts characters, not bytes: objectives here are Chinese as
/// often as not, and a byte budget would clip them at a third of the English
/// length — or slice a character in half, which panics.
fn clip_mission_text(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out: String = flat.chars().take(MISSION_TEXT_CLIP).collect();
    if flat.chars().nth(MISSION_TEXT_CLIP).is_some() {
        out.push('…');
    }
    out
}

/// The mission this session is working on, as a prompt block; `None` when there
/// is none.
///
/// A **mirror** of `desktop/src/state.rs::active_mission_block`, and it has to
/// stay one: the CLI cannot call the desktop crate (the dependency points the
/// other way), and the product promises the same answer in the window and in a
/// terminal. Same `<workspace>/missions/*/mission.json` layout, same
/// `active.json` pointer, same 160-character clip and five-row cap — a change to
/// one belongs in both. Drift shows up as a mission the CLI invents or forgets.
///
/// Everything here is best-effort. This runs on the turn path, and a prompt that
/// names the newest mission is better than a turn that dies over a stray byte in
/// a half-written record, so unreadable files are skipped rather than reported.
fn active_mission_block(workspace: &Path) -> Option<String> {
    let root = workspace.join("missions");
    let Ok(entries) = std::fs::read_dir(&root) else {
        return None;
    };
    // Ids are time-ordered (`m-YYYYmmdd-HHMMSS`), so sorting the directory names
    // newest-first gives the order a reader expects.
    let mut board: Vec<(String, serde_json::Value)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(path.join("mission.json")) else {
            continue;
        };
        let Ok(record) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        board.push((name, record));
    }
    if board.is_empty() {
        return None;
    }
    board.sort_by(|a, b| b.0.cmp(&a.0));

    // The row id the board reports: the record's own `id`, falling back to the
    // directory name, exactly as the desktop's board does.
    let id_of = |name: &str, record: &serde_json::Value| -> String {
        record
            .get("id")
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| name.to_string())
    };
    let field = |record: &serde_json::Value, key: &str| -> Option<String> {
        record
            .get(key)
            .and_then(|value| value.as_str())
            .map(clip_mission_text)
            .filter(|text| !text.trim().is_empty())
    };

    // The pointer the mission tools follow, so the prompt cannot name one
    // mission while the tools advance another. A pointer that is missing,
    // unreadable, or naming a record that is gone falls back to the newest.
    let pointed = std::fs::read_to_string(root.join("active.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| {
            value
                .get("mission")
                .and_then(|id| id.as_str())
                .map(str::to_string)
        });
    let index = pointed
        .as_deref()
        .and_then(|id| {
            board
                .iter()
                .position(|(name, record)| id_of(name, record) == id)
        })
        .unwrap_or(0);

    let (name, record) = &board[index];
    let status = record
        .get("status")
        .and_then(|value| value.as_str())
        .unwrap_or("?");
    let mut block =
        String::from("\n\n## 当前任务（长程任务包；状态在磁盘上，本回合开始时如实读取）\n\n");
    block.push_str(&format!(
        "- {} [{status}] {}\n",
        id_of(name, record),
        field(record, "objective").unwrap_or_else(|| "（未写目标）".to_string())
    ));
    block.push_str(&format!(
        "  下一步：{}\n",
        field(record, "step").unwrap_or_else(|| "（未设下一步）".to_string())
    ));
    block.push_str(&format!(
        "  完成判据：{}\n",
        field(record, "done_when").unwrap_or_else(|| "（未设完成判据）".to_string())
    ));
    // The rest of the board, so the model knows the active mission is not the
    // only work in flight — capped, so it never becomes the prompt.
    let mut listed = 1;
    for (position, (name, record)) in board.iter().enumerate() {
        if position == index {
            continue;
        }
        if listed >= MISSION_PROMPT_ROWS {
            break;
        }
        listed += 1;
        let status = record
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or("?");
        block.push_str(&format!(
            "- {} [{status}] {}\n",
            id_of(name, record),
            field(record, "objective").unwrap_or_else(|| "（未写目标）".to_string())
        ));
    }
    block.push_str(
        "\n任务在 running 期间不因答复结束回合：直接做下一步，做完用 mission_advance 记录（summary + next），\
         直到任务 done 或 blocked；不要问用户要不要继续。",
    );
    Some(block)
}

