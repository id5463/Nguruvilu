use std::path::Path;

/// The built-in interface and the `ui` pack's copy are two ends of one
/// seeding relationship: the pack was copied from the built-in, the window
/// serves the pack's copy, and the built-in is what a machine falls back to
/// when no interface pack is installed.
///
/// Two copies that drift apart mean those three stop being the same page —
/// which is exactly how a window and its fallback silently disagree. The
/// fix when this fails is one command:
///
/// ```text
/// cp desktop/ui/index.html packs/ui/index.html
/// ngu pack packs/ui --out packs/ui/ui-1.0.0.dshpack
/// ```
#[test]
fn builtin_interface_and_the_ui_pack_copy_are_identical() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let built_in = std::fs::read(root.join("desktop/ui/index.html"))
        .expect("reading the built-in interface");
    let packed = std::fs::read(root.join("packs/ui/index.html"))
        .expect("reading the ui pack's copy");

    assert_eq!(
        built_in.len(),
        packed.len(),
        "packs/ui/index.html drifted from desktop/ui/index.html (sizes differ)"
    );
    assert!(
        built_in == packed,
        "packs/ui/index.html drifted from desktop/ui/index.html — copy the \
         built-in over it and repack packs/ui"
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
