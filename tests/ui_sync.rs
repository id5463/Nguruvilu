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
