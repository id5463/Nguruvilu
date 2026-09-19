//! A demonstration Nguruvilu plugin.
//!
//! It **defines the ABI itself** instead of depending on the kernel. That is
//! exactly what a third-party plugin has to do, and it is the proof that the
//! boundary really is only C types plus JSON: this crate shares no Rust types,
//! no crate version, and no build settings with the host.
//!
//! Build it, then load the resulting shared library:
//!
//! ```sh
//! cargo build --manifest-path examples/demo-plugin/Cargo.toml --release
//! ngu plugin load examples/demo-plugin/target/release/ngu_demo_plugin.dll
//! ```

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;

/// ABI revision this plugin was written against.
const ABI_VERSION: u32 = 1;

/// The host capability table, mirrored exactly from the kernel's definition.
///
/// Field order and types must match the host's `HostApi` byte for byte; that is
/// the entire ABI contract.
#[repr(C)]
pub struct HostApi {
    /// Host ABI revision.
    pub abi_version: u32,
    /// Opaque host state, echoed back to every callback.
    pub user_data: *mut c_void,
    /// Write a log line. Levels: 0 debug, 1 info, 2 warn, 3 error.
    pub log: extern "C" fn(*mut c_void, u32, *const c_char),
    /// Register a tool; returns the running count, or 0 on failure.
    pub register_tool: extern "C" fn(*mut c_void, *const c_char) -> u64,
    /// Call another tool; the result must be freed with `free_string`.
    pub call_tool: extern "C" fn(*mut c_void, *const c_char, *const c_char) -> *const c_char,
    /// Free a string the host returned.
    pub free_string: extern "C" fn(*mut c_char),
}

/// The host table, stored on init and read by every later call.
static HOST: AtomicPtr<HostApi> = AtomicPtr::new(std::ptr::null_mut());

/// Cached descriptor string, owned by the plugin and never freed.
static DESCRIPTOR: OnceLock<CString> = OnceLock::new();

const DESCRIPTOR_JSON: &str = r#"{
  "name": "demo",
  "version": "0.1.0",
  "description": "demonstration plugin: echo, clock, and a deliberate failure",
  "abi_version": 1,
  "inject": [],
  "provide": []
}"#;

/// Metadata for the host. The returned pointer stays valid for the process.
#[no_mangle]
pub extern "C" fn ngu_describe() -> *const c_char {
    DESCRIPTOR
        .get_or_init(|| CString::new(DESCRIPTOR_JSON).expect("descriptor has no NUL"))
        .as_ptr()
}

/// Receive the host table and register this plugin's tools.
///
/// Returns 0 on success; any other value makes the host refuse the plugin.
#[no_mangle]
pub extern "C" fn ngu_init(host: *const HostApi) -> i32 {
    if host.is_null() {
        return 1;
    }
    let api = unsafe { &*host };

    // Refuse a host this plugin was not written for, rather than guessing.
    if api.abi_version != ABI_VERSION {
        return 2;
    }

    HOST.store(host as *mut HostApi, Ordering::SeqCst);
    log(1, "demo plugin initializing");

    let tools = [
        r#"{
  "name": "demo_echo",
  "description": "Echo the given text back, with its length.",
  "parameters": {
    "type": "object",
    "properties": { "text": { "type": "string", "description": "text to echo" } },
    "required": ["text"]
  }
}"#,
        r#"{
  "name": "demo_clock",
  "description": "Report the host's wall-clock time as seconds since the Unix epoch.",
  "parameters": { "type": "object", "properties": {} }
}"#,
        r#"{
  "name": "demo_fail",
  "description": "Always fails. Demonstrates that a plugin error reaches the model as a tool error.",
  "parameters": { "type": "object", "properties": {} }
}"#,
    ];

    for tool in tools {
        let Ok(c_tool) = CString::new(tool) else { return 3 };
        let registered = (api.register_tool)(api.user_data, c_tool.as_ptr());
        if registered == 0 {
            log(3, "demo plugin failed to register a tool");
            return 4;
        }
    }

    log(1, "demo plugin ready");
    0
}

/// Handle one tool call: JSON in, JSON out.
///
/// The returned string is allocated by this plugin and must be released with
/// `ngu_free_string`; the host does that.
#[no_mangle]
pub extern "C" fn ngu_call(tool: *const c_char, args: *const c_char) -> *const c_char {
    if tool.is_null() || args.is_null() {
        return std::ptr::null();
    }
    let tool = unsafe { CStr::from_ptr(tool) }.to_string_lossy().to_string();
    let args_text = unsafe { CStr::from_ptr(args) }.to_string_lossy().to_string();

    let arguments: serde_json::Value =
        serde_json::from_str(&args_text).unwrap_or(serde_json::Value::Null);

    let response = match tool.as_str() {
        "demo_echo" => match arguments.get("text").and_then(|t| t.as_str()) {
            Some(text) => ok(&format!("{text} ({} characters)", text.chars().count())),
            None => err("demo_echo requires a 'text' argument"),
        },
        "demo_clock" => {
            let seconds = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            ok(&format!("unix time {seconds}"))
        }
        "demo_fail" => err("this tool always fails, on purpose"),
        other => err(&format!("demo plugin has no tool named '{other}'")),
    };

    match CString::new(response) {
        Ok(value) => value.into_raw() as *const c_char,
        Err(_) => std::ptr::null(),
    }
}

/// Release a string this plugin returned.
#[no_mangle]
pub extern "C" fn ngu_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(ptr));
    }
}

/// Called by the host when the plugin is torn down.
#[no_mangle]
pub extern "C" fn ngu_shutdown() {
    log(1, "demo plugin shutting down");
    HOST.store(std::ptr::null_mut(), Ordering::SeqCst);
}

fn ok(content: &str) -> String {
    serde_json::json!({ "ok": true, "content": content }).to_string()
}

fn err(message: &str) -> String {
    serde_json::json!({ "ok": false, "error": message }).to_string()
}

fn log(level: u32, message: &str) {
    let host = HOST.load(Ordering::SeqCst);
    if host.is_null() {
        return;
    }
    let api = unsafe { &*host };
    if let Ok(text) = CString::new(message) {
        (api.log)(api.user_data, level, text.as_ptr());
    }
}
