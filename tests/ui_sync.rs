use std::path::Path;

/// The built-in interface and the `starter` pack's copy are two ends of one
/// seeding relationship: the pack was copied from the built-in, the window
/// serves the pack's copy, and the built-in is what a machine falls back to
/// when no interface pack is installed.
///
/// Two copies that drift apart mean those three stop being the same page —
/// which is exactly how a window and its fallback silently disagree. The
/// fix when this fails is one command:
///
/// ```text
/// cp desktop/ui/index.html packs/starter/index.html
/// ngu pack packs/starter --out packs/starter/starter-1.0.0.dshpack
/// ```
#[test]
fn builtin_interface_and_the_ui_pack_copy_are_identical() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let built_in = std::fs::read(root.join("desktop/ui/index.html"))
        .expect("reading the built-in interface");
    let packed = std::fs::read(root.join("packs/starter/index.html"))
        .expect("reading the starter pack's copy");

    assert_eq!(
        built_in.len(),
        packed.len(),
        "packs/starter/index.html drifted from desktop/ui/index.html (sizes differ)"
    );
    assert!(
        built_in == packed,
        "packs/starter/index.html drifted from desktop/ui/index.html — copy the \
         built-in over it and repack packs/starter"
    );
}

/// Every camelCase name the script uses as an element binding must be
/// declared.
///
/// The page keeps two forms of the same id: `id="set-base-url"` in the HTML,
/// `const setBaseUrl = …` in the script. A field added to one side only passes
/// `node --check` — a missing binding is a runtime ReferenceError, and the
/// script's top level dies at that line, taking every listener registered
/// after it: buttons that do nothing, a save that never fires. That happened
/// twice (`run_all`, then the three search fields), so it is now a test.
#[test]
fn element_bindings_used_by_the_script_are_declared() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let html = std::fs::read_to_string(root.join("desktop/ui/index.html"))
        .expect("reading the built-in interface");

    let script = html
        .split("<script>")
        .nth(1)
        .and_then(|rest| rest.split("</script>").next())
        .expect("the script block");

    // `set-base-url` → `setBaseUrl`; ids stay kebab, bindings stay camel.
    let mut missing = Vec::new();
    for id_capture in html.split("id=\"").skip(1) {
        let id = id_capture.split('"').next().unwrap_or("");
        if !id.contains('-') || !id.starts_with("set-") {
            continue;
        }
        let mut camel = String::new();
        let mut upper = false;
        for ch in id.chars() {
            if ch == '-' {
                upper = true;
            } else if upper {
                camel.extend(ch.to_uppercase());
                upper = false;
            } else {
                camel.push(ch);
            }
        }
        // Used bare somewhere in the script…
        let used = script.contains(&format!("{camel}.")) || script.contains(&format!("{camel},"))
            || script.contains(&format!("{camel} "));
        if !used {
            continue;
        }
        // …but never declared.
        let declared = script.contains(&format!("const {camel} ")) 
            || script.contains(&format!("let {camel} "));
        if !declared {
            missing.push(format!("{camel} (id \"{id}\")"));
        }
    }

    assert!(
        missing.is_empty(),
        "the script uses element bindings that are never declared: {}",
        missing.join(", ")
    );
}

/// The overlay rules a pack declares must round-trip through the manifest.
///
/// These three fields decide whether two packs can serve one interface at
/// once, which is the whole reason overlays exist. They are read by
/// `resolve_ui_stack` and nothing else, so a silent rename or a dropped
/// default would only show up as "my pack's files stopped appearing" — with no
/// error anywhere. Pinning them here makes that a failing test instead.
#[test]
fn interface_overlay_fields_survive_a_manifest_round_trip() {
    let json = r#"{
        "formatVersion": 1,
        "game": "nguruvilu",
        "name": "overlay-demo",
        "versionId": "1.0.0",
        "license": "MIT",
        "kernelVersion": "0.1.0",
        "ui": {
            "id": "ui",
            "title": "叠加示例",
            "entry": "index.html",
            "base": false,
            "overlayOrder": 7,
            "overlays": true
        }
    }"#;
    let manifest: nguruvilu::pack::PackManifest =
        serde_json::from_str(json).expect("the manifest parses");
    let ui = manifest.ui.as_ref().expect("ui is declared");

    assert!(!ui.base, "an overlay must not claim to own the document");
    assert_eq!(ui.overlay_order, 7, "overlayOrder must survive");
    assert!(ui.overlays, "overlays default on and must round-trip");

    // And the defaults, which are what an author who writes nothing gets: an
    // overlay that owns nothing can never take the page away from its base.
    let minimal = r#"{
        "formatVersion": 1,
        "game": "nguruvilu",
        "name": "minimal",
        "versionId": "1.0.0",
        "license": "MIT",
        "kernelVersion": "0.1.0",
        "ui": { "id": "ui", "entry": "index.html" }
    }"#;
    let m: nguruvilu::pack::PackManifest =
        serde_json::from_str(minimal).expect("the minimal manifest parses");
    let ui = m.ui.as_ref().expect("ui is declared");
    assert!(!ui.base, "saying nothing must not make a pack the base");
    assert_eq!(ui.overlay_order, 0, "overlayOrder defaults to 0");
    assert!(ui.overlays, "overlays default to on");
}

/// `hooks` in a look file are file names, and the interface has to offer the
/// points they name.
///
/// Two failure modes this catches, both silent before: a look file whose hooks
/// are inlined script text (unreadable, but it parses), and a pack hanging code
/// on a point the page never runs (the code loads and does nothing, forever).
#[test]
fn look_file_hooks_name_files_and_real_points() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let look_text = std::fs::read_to_string(root.join("packs/hook-demo/look.json"))
        .expect("the demo pack's look file");
    let look: serde_json::Value =
        serde_json::from_str(&look_text).expect("look.json parses");
    let hooks = look
        .get("hooks")
        .and_then(|v| v.as_object())
        .expect("the demo pack declares hooks");

    // Each value is a file that exists next to the manifest, not inline code.
    for (point, file) in hooks {
        let name = file.as_str().unwrap_or_else(|| {
            panic!("hook '{point}' must name a file, not embed its text")
        });
        assert!(
            !name.contains('\n'),
            "hook '{point}' looks like inline script; hooks name files so the \
             code stays readable and editable"
        );
        assert!(
            root.join("packs/hook-demo").join(name).is_file(),
            "hook '{point}' names {name}, which is not in the pack"
        );
    }

    // Every point a pack uses must be one the page actually runs.
    let html = std::fs::read_to_string(root.join("desktop/ui/index.html"))
        .expect("the built-in interface");
    for point in hooks.keys() {
        let needle = format!("runHook(\"{point}\"");
        assert!(
            html.contains(&needle),
            "hook point '{point}' is declared by a pack but the page never runs \
             it — the pack's code would load and do nothing"
        );
    }
}
