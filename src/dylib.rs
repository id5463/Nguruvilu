//! Dynamically loaded plugins: the smallest possible ABI, all semantics in JSON.
//!
//! # Why this shape
//!
//! Rust has no stable ABI. A plugin that exports `fn apply(Config) -> Result<()>`
//! cannot be called by a host built with a different compiler version: struct
//! layout, `String`'s representation, enum tags, trait-object vtables and
//! closures are all unspecified. So the ABI surface here is reduced to
//! **C-compatible types only** — `*const c_char` in, `*const c_char` out — and
//! every piece of meaning travels as JSON.
//!
//! That split is the whole design:
//!
//! * **ABI**: four exported functions plus one `#[repr(C)]` table of function
//!   pointers. No Rust type crosses the boundary.
//! * **API**: JSON describing the plugin, its tools, and every call's arguments
//!   and result. Versioned by [`ABI_VERSION`].
//!
//! A plugin can therefore be built by any Rust version, with any optimization
//! settings, and still load.
//!
//! # Why libraries are never unloaded
//!
//! Unloading a shared library is only safe when nothing still points into it.
//! A plugin that registered a tool has handed the host function pointers, and
//! those live inside the library's code. `dlclose` followed by any call through
//! such a pointer is a use-after-free.
//!
//! Rather than pretend that can be made safe, this module **keeps every loaded
//! library alive for the process lifetime**. Hot reload works by loading the new
//! build (a different file), switching contributions to it, and leaving the old
//! one resident. The cost is a few megabytes per update; the alternative is
//! undefined behavior.
//!
//! # Contract for plugin authors
//!
//! ```ignore
//! #[no_mangle] pub extern "C" fn ngu_describe() -> *const c_char;   // metadata JSON, static
//! #[no_mangle] pub extern "C" fn ngu_init(host: *const HostApi) -> i32;
//! #[no_mangle] pub extern "C" fn ngu_call(tool: *const c_char, args: *const c_char) -> *const c_char;
//! #[no_mangle] pub extern "C" fn ngu_free_string(ptr: *mut c_char);  // frees a ngu_call result
//! #[no_mangle] pub extern "C" fn ngu_shutdown();                     // optional
//! ```

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use libloading::Library;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::plugin::{Contributions, Plugin, PluginCtx};
use crate::tools::{ToolDef, ToolFuture};

/// ABI revision. A plugin whose `ngu_describe` does not report this value is
/// refused rather than loaded and hoped for.
pub const ABI_VERSION: u32 = 1;

/// Required exported symbol: plugin metadata as JSON.
pub const SYM_DESCRIBE: &[u8] = b"ngu_describe\0";
/// Required exported symbol: receives the host capability table.
pub const SYM_INIT: &[u8] = b"ngu_init\0";
/// Required exported symbol: every tool call, JSON in and JSON out.
pub const SYM_CALL: &[u8] = b"ngu_call\0";
/// Required exported symbol: frees a string returned by `ngu_call`.
pub const SYM_FREE: &[u8] = b"ngu_free_string\0";
/// Optional exported symbol: called when the plugin is torn down.
pub const SYM_SHUTDOWN: &[u8] = b"ngu_shutdown\0";

/// The capability table the host hands to a plugin.
///
/// Every field is a C-compatible function pointer. `user_data` is an opaque
/// host pointer passed back as the first argument of each callback; it is how a
/// stateless C function reaches the plugin instance it belongs to.
#[repr(C)]
pub struct HostApi {
    /// Must equal [`ABI_VERSION`]; a mismatch is the plugin's cue to refuse.
    pub abi_version: u32,
    /// Opaque host state, passed back to every callback.
    pub user_data: *mut c_void,
    /// Write a line to the host log. `level`: 0 = debug, 1 = info, 2 = warn, 3 = error.
    pub log: extern "C" fn(*mut c_void, u32, *const c_char),
    /// Register a tool. Returns 0 on failure, otherwise the number registered.
    pub register_tool: extern "C" fn(*mut c_void, *const c_char) -> u64,
    /// Call another tool by name. The returned string must be freed with `free_string`.
    pub call_tool: extern "C" fn(*mut c_void, *const c_char, *const c_char) -> *const c_char,
    /// Free a string the host returned.
    pub free_string: extern "C" fn(*mut c_char),
}

/// Metadata a plugin reports from `ngu_describe`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMeta {
    /// Plugin name, unique per kernel.
    pub name: String,
    /// Plugin version, for diagnostics.
    #[serde(default)]
    pub version: Option<String>,
    /// One-line description.
    #[serde(default)]
    pub description: Option<String>,
    /// Services this plugin needs before it can load.
    #[serde(default)]
    pub inject: Vec<String>,
    /// Services this plugin provides.
    ///
    /// Advisory: a dynamic plugin cannot hand the host a Rust trait object, so
    /// what it "provides" is reachable through [`DynamicPlugin::call`] rather
    /// than through the service table. The declaration still participates in
    /// ordering and dependency checks.
    #[serde(default)]
    pub provide: Vec<String>,
    /// ABI revision the plugin was built against.
    #[serde(default)]
    pub abi_version: Option<u32>,
}

/// A tool a plugin registered through `register_tool`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// Description for the model.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for arguments.
    #[serde(default)]
    pub parameters: Value,
}

/// State shared with a plugin's callbacks.
struct DynamicState {
    plugin: String,
    tools: Mutex<Vec<ToolSpec>>,
    log: Mutex<Vec<(u32, String)>>,
}

/// A plugin loaded from a shared library.
///
/// The library is held for the process lifetime; see the module documentation
/// for why unloading is not attempted.
pub struct DynamicPlugin {
    path: PathBuf,
    /// Leaked on purpose, never dropped.
    ///
    /// Dropping a Library calls dlclose. Any function pointer the host took
    /// from this plugin — a registered tool handler, a cached symbol — would
    /// then point into unmapped memory, and the next call through it would be a
    /// use-after-free. Leaking the handle is the only way to guarantee that
    /// cannot happen.
    #[allow(dead_code)]
    library: &'static Library,
    meta: PluginMeta,
    init: extern "C" fn(*const HostApi) -> i32,
    call: extern "C" fn(*const c_char, *const c_char) -> *const c_char,
    free: extern "C" fn(*mut c_char),
    shutdown: Option<extern "C" fn()>,
    state: Arc<DynamicState>,
    /// The capability table handed to the plugin.
    ///
    /// It lives here rather than on pply's stack because the plugin **keeps
    /// the pointer** for the lifetime of the load. A stack-allocated table would
    /// be freed when pply returned, leaving every later callback — including
    /// the shutdown hook — reading unmapped stack.
    host_api: Box<HostApi>,
}

// The raw pointers inside `HostApi` are only ever passed to the plugin during
// `apply`, on one thread, and the callbacks they carry are `extern "C"`.
unsafe impl Send for DynamicPlugin {}
unsafe impl Sync for DynamicPlugin {}

impl DynamicPlugin {
    /// Load a shared library and read its metadata.
    ///
    /// # Safety
    ///
    /// The caller vouches that the library is trusted: loading native code runs
    /// it in this process with this process's privileges. There is no sandbox
    /// here, by design — isolation is what the subprocess plugin path is for.
    pub unsafe fn load(path: &Path) -> Result<Self> {
        let library = Library::new(path)
            .with_context(|| format!("loading plugin library {}", path.display()))?;

        // Read the exported symbols, then keep plain function pointers. The
        // `Symbol` borrows the library; the pointer does not.
        let describe: extern "C" fn() -> *const c_char = {
            let symbol: libloading::Symbol<extern "C" fn() -> *const c_char> = library
                .get(SYM_DESCRIBE)
                .with_context(|| {
                    format!(
                        "{} does not export {}",
                        path.display(),
                        String::from_utf8_lossy(&SYM_DESCRIBE[..SYM_DESCRIBE.len() - 1])
                    )
                })?;
            *symbol
        };
        let init: extern "C" fn(*const HostApi) -> i32 = {
            let symbol: libloading::Symbol<extern "C" fn(*const HostApi) -> i32> =
                library.get(SYM_INIT).with_context(|| {
                    format!(
                        "{} does not export {}",
                        path.display(),
                        String::from_utf8_lossy(&SYM_INIT[..SYM_INIT.len() - 1])
                    )
                })?;
            *symbol
        };
        let call: extern "C" fn(*const c_char, *const c_char) -> *const c_char = {
            let symbol: libloading::Symbol<
                extern "C" fn(*const c_char, *const c_char) -> *const c_char,
            > = library.get(SYM_CALL).with_context(|| {
                format!(
                    "{} does not export {}",
                    path.display(),
                    String::from_utf8_lossy(&SYM_CALL[..SYM_CALL.len() - 1])
                )
            })?;
            *symbol
        };
        let free: extern "C" fn(*mut c_char) = {
            let symbol: libloading::Symbol<extern "C" fn(*mut c_char)> =
                library.get(SYM_FREE).with_context(|| {
                    format!(
                        "{} does not export {}",
                        path.display(),
                        String::from_utf8_lossy(&SYM_FREE[..SYM_FREE.len() - 1])
                    )
                })?;
            *symbol
        };
        let shutdown: Option<extern "C" fn()> = {
            let symbol: Option<libloading::Symbol<extern "C" fn()>> =
                library.get(SYM_SHUTDOWN).ok();
            symbol.map(|s| *s)
        };

        // Metadata is read once, at load time, so a malformed descriptor fails
        // the load instead of surfacing later as a mysterious missing tool.
        let meta_ptr = describe();
        if meta_ptr.is_null() {
            return Err(anyhow!("{} returned a null descriptor", path.display()));
        }
        let meta_text = CStr::from_ptr(meta_ptr).to_string_lossy().to_string();
        let meta: PluginMeta = serde_json::from_str(&meta_text).with_context(|| {
            format!("{} returned an unparsable descriptor: {meta_text}", path.display())
        })?;

        if meta.name.trim().is_empty() {
            return Err(anyhow!("{} reports an empty plugin name", path.display()));
        }
        match meta.abi_version {
            Some(version) if version == ABI_VERSION => {}
            Some(version) => {
                return Err(anyhow!(
                    "plugin '{}' targets ABI {version}, this kernel speaks {ABI_VERSION}",
                    meta.name
                ))
            }
            None => {
                return Err(anyhow!(
                    "plugin '{}' does not report an ABI version; rebuild it against the plugin SDK",
                    meta.name
                ))
            }
        }

        let state = Arc::new(DynamicState {
            plugin: meta.name.clone(),
            tools: Mutex::new(Vec::new()),
            log: Mutex::new(Vec::new()),
        });

        let host_api = Box::new(HostApi {
            abi_version: ABI_VERSION,
            user_data: Arc::as_ptr(&state) as *mut c_void,
            log: host_log,
            register_tool: host_register_tool,
            call_tool: host_call_tool,
            free_string: host_free_string,
        });

        Ok(Self {
            path: path.to_path_buf(),
            library: Box::leak(Box::new(library)),
            meta,
            init,
            call,
            free,
            shutdown,
            state,
            host_api,
        })
    }

    /// The library this plugin came from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reported metadata.
    pub fn meta(&self) -> &PluginMeta {
        &self.meta
    }

    /// Tools registered during the last `apply`.
    pub fn registered_tools(&self) -> Vec<ToolSpec> {
        self.state.tools.lock().map(|t| t.clone()).unwrap_or_default()
    }

    /// Lines the plugin logged during the last `apply`.
    pub fn log_lines(&self) -> Vec<(u32, String)> {
        self.state.log.lock().map(|l| l.clone()).unwrap_or_default()
    }

    /// Call one of the plugin's tools directly.
    ///
    /// This is also how a service a plugin declared is reached: the name is the
    /// service, the arguments are JSON.
    pub fn call(&self, tool: &str, arguments: Value) -> Result<String> {
        call_plugin(self.call, self.free, &self.meta.name, tool, arguments)
    }
}

impl std::fmt::Debug for DynamicPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The library handle and the capability table have no useful debug
        // form; the identity is what a caller wants to see.
        f.debug_struct("DynamicPlugin")
            .field("name", &self.meta.name)
            .field("version", &self.meta.version)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl Drop for DynamicPlugin {
    fn drop(&mut self) {
        // Ask the plugin to clean up its own state. The library itself stays
        // mapped: see the module documentation.
        if let Some(shutdown) = self.shutdown {
            shutdown();
        }
    }
}

impl Plugin for DynamicPlugin {
    fn name(&self) -> &str {
        &self.meta.name
    }

    fn inject(&self) -> Vec<String> {
        self.meta.inject.clone()
    }

    fn provide(&self) -> Vec<String> {
        self.meta.provide.clone()
    }

    fn apply(&self, _ctx: &PluginCtx) -> Result<Contributions> {
        // A reload must not see the previous run's registrations.
        if let Ok(mut tools) = self.state.tools.lock() {
            tools.clear();
        }

        // The table handed over here outlives this call: it is owned by the
        // plugin struct, not by this stack frame.
        let code = (self.init)(&*self.host_api);
        if code != 0 {
            return Err(anyhow!(
                "plugin '{}' refused to initialize (code {code})",
                self.meta.name
            ));
        }

        let specs = self.registered_tools();
        let mut contributions = Contributions::new();
        for spec in specs {
            let def = self.tool_def(spec)?;
            contributions = contributions.tool(def);
        }
        Ok(contributions)
    }
}

impl DynamicPlugin {
    /// Wrap one registered tool spec as a kernel tool.
    fn tool_def(&self, spec: ToolSpec) -> Result<ToolDef> {
        if spec.name.trim().is_empty() {
            return Err(anyhow!("plugin '{}' registered a tool with no name", self.meta.name));
        }

        let call = self.call;
        let free = self.free;
        let plugin = self.meta.name.clone();
        let tool_name = spec.name.clone();

        let parameters = if spec.parameters.is_null() {
            serde_json::json!({ "type": "object", "properties": {} })
        } else {
            spec.parameters.clone()
        };

        Ok(ToolDef::new(
            spec.name,
            if spec.description.is_empty() {
                format!("tool '{tool_name}' provided by plugin '{plugin}'")
            } else {
                spec.description
            },
            parameters,
            format!("dylib:{plugin}"),
            move |args| {
                let tool = tool_name.clone();
                let plugin = plugin.clone();
                Box::pin(async move {
                    // `ngu_call` is a synchronous C function; running it on the
                    // async scheduler would block a worker for its duration.
                    let result = tokio::task::spawn_blocking(move || {
                        call_plugin(call, free, &plugin, &tool, args)
                    })
                    .await
                    .map_err(|e| anyhow!("plugin call task failed: {e}"))??;
                    Ok(crate::tools::ToolOutput::text(result))
                }) as ToolFuture
            },
        ))
    }
}

/// Invoke a plugin tool and decode its JSON result.
fn call_plugin(
    call: extern "C" fn(*const c_char, *const c_char) -> *const c_char,
    free: extern "C" fn(*mut c_char),
    plugin: &str,
    tool: &str,
    arguments: Value,
) -> Result<String> {
    let tool_c = CString::new(tool).map_err(|_| anyhow!("tool name contains a NUL byte"))?;
    let args_c = CString::new(arguments.to_string())
        .map_err(|_| anyhow!("arguments contain a NUL byte"))?;

    let result_ptr = call(tool_c.as_ptr(), args_c.as_ptr());
    if result_ptr.is_null() {
        return Err(anyhow!("plugin '{plugin}' returned null for tool '{tool}'"));
    }

    let text = unsafe { CStr::from_ptr(result_ptr) }.to_string_lossy().to_string();
    free(result_ptr as *mut c_char);

    // A plugin reports failure as `{"ok": false, "error": "..."}`, so a tool
    // error reaches the model as a normal tool error rather than a crash.
    match serde_json::from_str::<Value>(&text) {
        Ok(value) => {
            if value.get("ok").and_then(|v| v.as_bool()) == Some(false) {
                let message = value
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("plugin reported failure without a message");
                return Err(anyhow!("{message}"));
            }
            if let Some(content) = value.get("content").and_then(|v| v.as_str()) {
                return Ok(content.to_string());
            }
            Ok(text)
        }
        // Not JSON: hand the raw text through rather than discarding output.
        Err(_) => Ok(text),
    }
}

// ---------------------------------------------------------------- callbacks

extern "C" fn host_log(user_data: *mut c_void, level: u32, message: *const c_char) {
    if user_data.is_null() || message.is_null() {
        return;
    }
    let state = unsafe { &*(user_data as *const DynamicState) };
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy().to_string();
    if let Ok(mut log) = state.log.lock() {
        log.push((level, text));
    }
}

extern "C" fn host_register_tool(user_data: *mut c_void, def_json: *const c_char) -> u64 {
    if user_data.is_null() || def_json.is_null() {
        return 0;
    }
    let state = unsafe { &*(user_data as *const DynamicState) };
    let text = unsafe { CStr::from_ptr(def_json) }.to_string_lossy().to_string();

    let Ok(spec) = serde_json::from_str::<ToolSpec>(&text) else {
        return 0;
    };
    let Ok(mut tools) = state.tools.lock() else {
        return 0;
    };
    tools.push(spec);
    tools.len() as u64
}

extern "C" fn host_call_tool(
    user_data: *mut c_void,
    name: *const c_char,
    args: *const c_char,
) -> *const c_char {
    if user_data.is_null() || name.is_null() || args.is_null() {
        return std::ptr::null();
    }
    let state = unsafe { &*(user_data as *const DynamicState) };
    let tool = unsafe { CStr::from_ptr(name) }.to_string_lossy().to_string();
    let arguments = unsafe { CStr::from_ptr(args) }.to_string_lossy().to_string();

    // A plugin calling back into the kernel is not supported in this revision:
    // it would re-enter the kernel from inside a tool call. Reported honestly
    // rather than silently returning an empty success.
    let payload = serde_json::json!({
        "ok": false,
        "error": format!(
            "plugin '{}' called host tool '{tool}' with {arguments}, but host callbacks are not supported in ABI {ABI_VERSION}",
            state.plugin
        ),
    });
    to_c_string(&payload.to_string())
}

extern "C" fn host_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(ptr));
    }
}

/// Move a Rust string into a C string the caller owns.
fn to_c_string(text: &str) -> *const c_char {
    match CString::new(text) {
        Ok(value) => value.into_raw() as *const c_char,
        Err(_) => std::ptr::null(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_callback_string_round_trips() {
        let text = "hello from the host";
        let ptr = to_c_string(text);
        assert!(!ptr.is_null());
        let read = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string();
        assert_eq!(read, text);
        host_free_string(ptr as *mut c_char);
    }

    #[test]
    fn a_string_with_a_nul_byte_is_refused_rather_than_truncated() {
        assert!(to_c_string("bad\0string").is_null());
    }

    #[test]
    fn metadata_defaults_are_permissive() {
        let meta: PluginMeta = serde_json::from_str(r#"{"name":"x"}"#).unwrap();
        assert_eq!(meta.name, "x");
        assert!(meta.inject.is_empty());
        assert!(meta.provide.is_empty());
        assert_eq!(meta.abi_version, None);
    }

    #[test]
    fn a_tool_spec_defaults_its_schema() {
        let spec: ToolSpec = serde_json::from_str(r#"{"name":"t"}"#).unwrap();
        assert_eq!(spec.name, "t");
        assert_eq!(spec.description, "");
        assert!(spec.parameters.is_null());
    }

    #[test]
    fn loading_a_missing_library_reports_the_path() {
        let result = unsafe { DynamicPlugin::load(Path::new("definitely-not-here.dll")) };
        let error = result.err().expect("load must fail");
        let text = format!("{error:#}");
        assert!(text.contains("definitely-not-here.dll"), "{text}");
    }

    #[test]
    fn loading_a_non_library_file_is_an_error() {
        let dir = std::env::temp_dir().join(format!("ngu-dylib-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("not-a-library.dll");
        std::fs::write(&fake, b"this is not a shared library").unwrap();

        let result = unsafe { DynamicPlugin::load(&fake) };
        assert!(result.is_err());
    }
}
