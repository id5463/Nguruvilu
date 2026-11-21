//! `ngu ui`：把一个目录写成内置界面，或说默认界面在哪。
//!
//! 界面是数据（`index.html` + `ui.css` + `look.json`），不是代码，所以这一层
//! 只写目录、报它有什么。真正把资源变成响应的是 `nguruvilu::assets`。

use std::path::PathBuf;

use anyhow::{Context, Result};

fn ui_command(dir: Option<&PathBuf>, as_json: bool) -> Result<()> {
    let target = dir.cloned().unwrap_or_else(|| PathBuf::from("ui"));
    std::fs::create_dir_all(&target)
        .with_context(|| format!("creating {}", target.display()))?;

    let page = target.join("index.html");
    std::fs::write(&page, nguruvilu::ui::BUILTIN_HTML)
        .with_context(|| format!("writing {}", page.display()))?;

    // The declaration that makes the directory an interface, so `eject` gives
    // something a pack can point at rather than a loose file.
    let manifest = nguruvilu::pack::PackManifest {
        ui: Some(nguruvilu::pack::UiDecl {
            id: "my-ui".into(),
            title: "我的界面".into(),
            entry: "index.html".into(),
            // `ngu eject` hands out a complete interface, so this is the base.
            base: true,
            overlay_order: 0,
            overlays: true,
            source: None,
            sha256: None,
        }),
        ..nguruvilu::pack::PackManifest::new("my-ui", "1.0.0")
    };
    nguruvilu::pack::write_manifest(&target, &manifest)?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "dir": target.display().to_string(),
                "entry": "index.html",
                "bytes": nguruvilu::ui::BUILTIN_HTML.len(),
            }))?
        );
    } else {
        println!("wrote the built-in interface to {}", target.display());
        println!("edit index.html, then `ngu pack {}`", target.display());
        println!("the declaration in ngu.index.json points at it");
    }
    Ok(())
}
