//! Dynamic plugin integration tests.
//!
//! These load the example plugin from `examples/demo-plugin`. When it has not
//! been built they skip rather than fail, so the suite stays runnable without
//! a prior plugin build:
//!
//! ```sh
//! cargo build --manifest-path examples/demo-plugin/Cargo.toml --release
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nguruvilu::dylib::DynamicPlugin;
use nguruvilu::plugin::{Kernel, Plugin as _, PluginCtx, RealmMap, ServiceView};
use serde_json::json;

/// Path to a private copy of the built demo plugin, when it exists.
///
/// Each caller gets its own copy on purpose. A plugin's static data belongs to
/// the loaded module, so two tests loading the *same file* would share one
/// `HOST` slot and interfere. Separate copies are also exactly how hot reload
/// works: a new build lands beside the old one as a distinct module.
fn demo_library() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("demo-plugin")
        .join("target")
        .join("release");
    let name = if cfg!(windows) {
        "ngu_demo_plugin.dll"
    } else if cfg!(target_os = "macos") {
        "libngu_demo_plugin.dylib"
    } else {
        "libngu_demo_plugin.so"
    };
    let original = dir.join(name);
    if !original.is_file() {
        return None;
    }

    let private = std::env::temp_dir().join(format!("ngu-dylib-mod-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&private).ok()?;
    let copy = private.join(name);
    std::fs::copy(&original, &copy).ok()?;
    Some(copy)
}

/// A context good enough for a plugin that injects nothing.
fn bare_ctx(plugin: &str) -> PluginCtx {
    PluginCtx {
        plugin: plugin.to_string(),
        fiber: 0,
        realm: RealmMap::new(),
        services: ServiceView::default(),
        config: serde_json::Value::Null,
    }
}

#[test]
fn the_demo_plugin_reports_its_identity() {
    let Some(path) = demo_library() else {
        eprintln!("demo plugin not built; skipping");
        return;
    };

    let plugin = unsafe { DynamicPlugin::load(&path) }.expect("the plugin loads");
    assert_eq!(plugin.name(), "demo");
    assert_eq!(plugin.meta().version.as_deref(), Some("0.1.0"));
    assert_eq!(
        plugin.meta().abi_version,
        Some(nguruvilu::dylib::ABI_VERSION),
        "the plugin targets this kernel's ABI"
    );
    assert!(plugin.meta().description.is_some());
}

#[test]
fn applying_the_plugin_registers_its_tools() {
    let Some(path) = demo_library() else {
        return;
    };

    let plugin = unsafe { DynamicPlugin::load(&path) }.expect("load");
    let contributions = plugin.apply(&bare_ctx("demo")).expect("apply");

    let mut names: Vec<&str> = contributions.tools.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["demo_clock", "demo_echo", "demo_fail"]);

    // The kernel stamps ownership, so a plugin cannot claim another's tool.
    for tool in &contributions.tools {
        assert_eq!(tool.owner, "dylib:demo");
    }

    // Every registered tool carries a usable schema.
    for tool in &contributions.tools {
        assert_eq!(tool.parameters["type"], "object");
    }
}

#[test]
fn the_plugin_logs_through_the_host_callback() {
    let Some(path) = demo_library() else {
        return;
    };

    let plugin = unsafe { DynamicPlugin::load(&path) }.expect("load");
    plugin.apply(&bare_ctx("demo")).expect("apply");

    let lines = plugin.log_lines();
    assert!(
        lines.iter().any(|(_, text)| text.contains("ready")),
        "plugin log reached the host: {lines:?}"
    );
}

#[test]
fn a_tool_call_round_trips_json() {
    let Some(path) = demo_library() else {
        return;
    };

    let plugin = unsafe { DynamicPlugin::load(&path) }.expect("load");
    plugin.apply(&bare_ctx("demo")).expect("apply");

    let echoed = plugin
        .call("demo_echo", json!({ "text": "hello" }))
        .expect("echo succeeds");
    assert!(echoed.contains("hello"), "{echoed}");
    assert!(echoed.contains("5 characters"), "{echoed}");

    let clock = plugin.call("demo_clock", json!({})).expect("clock succeeds");
    assert!(clock.contains("unix time"), "{clock}");
}

#[test]
fn a_plugin_error_surfaces_as_a_tool_error() {
    let Some(path) = demo_library() else {
        return;
    };

    let plugin = unsafe { DynamicPlugin::load(&path) }.expect("load");
    plugin.apply(&bare_ctx("demo")).expect("apply");

    // A reported failure must be an error, not a successful empty result.
    let failure = plugin.call("demo_fail", json!({})).expect_err("demo_fail fails");
    assert!(format!("{failure:#}").contains("always fails"));

    // A missing argument is the plugin's own error, passed through verbatim.
    let missing = plugin
        .call("demo_echo", json!({}))
        .expect_err("missing argument");
    assert!(format!("{missing:#}").contains("requires a 'text' argument"));

    // An unknown tool is refused by the plugin, not by the host.
    let unknown = plugin
        .call("no_such_tool", json!({}))
        .expect_err("unknown tool");
    assert!(format!("{unknown:#}").contains("no tool named"));
}

#[test]
fn a_reload_does_not_duplicate_registrations() {
    let Some(path) = demo_library() else {
        return;
    };

    let plugin = unsafe { DynamicPlugin::load(&path) }.expect("load");

    let first = plugin.apply(&bare_ctx("demo")).expect("apply");
    let second = plugin.apply(&bare_ctx("demo")).expect("apply again");

    assert_eq!(
        first.tools.len(),
        second.tools.len(),
        "the second apply cleared the first run's registrations"
    );
}

#[test]
fn the_plugin_loads_into_the_kernel_and_its_tool_runs() {
    let Some(path) = demo_library() else {
        return;
    };

    let mut kernel = Kernel::new();
    let plugin = unsafe { DynamicPlugin::load(&path) }.expect("load");
    let name = plugin.name().to_string();
    kernel.define(Arc::new(plugin));

    let fiber = kernel.load(&name, RealmMap::new(), serde_json::Value::Null).expect("kernel load");

    // Its tools are in the kernel's table...
    let mut tools = kernel.tools_of(fiber);
    tools.sort();
    assert_eq!(tools, vec!["demo_clock", "demo_echo", "demo_fail"]);

    // ...and callable through the kernel, which is what the model sees.
    let out = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(kernel.tools().execute("demo_echo", r#"{"text":"via kernel"}"#))
        .expect("tool runs");
    assert!(out.contains("via kernel"), "{out}");

    // Unloading removes exactly this plugin's tools and leaves the base tools.
    kernel.unload(fiber).expect("unload");
    assert!(kernel.tools().get("demo_echo").is_none());
    assert!(kernel.tools().get("read").is_some());
}

#[test]
fn two_copies_of_one_plugin_can_be_loaded_at_once() {
    // Hot reload works by loading a new build *beside* the old one and
    // switching to it; the old library stays mapped for the process lifetime.
    // This proves the two can coexist and both stay callable.
    let Some(path) = demo_library() else {
        return;
    };

    let dir = std::env::temp_dir().join(format!("ngu-dylib-copy-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    let copy = dir.join(path.file_name().unwrap());
    std::fs::copy(&path, &copy).expect("copying the library");

    let first = unsafe { DynamicPlugin::load(&path) }.expect("first copy loads");
    let second = unsafe { DynamicPlugin::load(&copy) }.expect("second copy loads");

    assert_eq!(first.name(), second.name(), "same plugin, two builds");

    first.apply(&bare_ctx("demo")).expect("first applies");
    second.apply(&bare_ctx("demo")).expect("second applies");

    let a = first.call("demo_echo", json!({ "text": "first" })).expect("first calls");
    let b = second.call("demo_echo", json!({ "text": "second" })).expect("second calls");
    assert!(a.contains("first"), "{a}");
    assert!(b.contains("second"), "{b}");
}
#[test]
fn a_library_that_is_not_a_plugin_is_refused_with_a_useful_message() {
    let dir = std::env::temp_dir().join(format!("ngu-dylib-it-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    let fake = dir.join("not-a-plugin.dll");
    std::fs::write(&fake, b"plain text, not a shared library").unwrap();

    let error = unsafe { DynamicPlugin::load(&fake) }.expect_err("must refuse");
    let text = format!("{error:#}");
    assert!(text.contains("not-a-plugin.dll"), "{text}");
}
