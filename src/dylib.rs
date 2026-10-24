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
//! That decision has two consequences this module states rather than hides:
//!
//! * **The leak is recorded.** Every handle is registered in
//!   [`leaked_libraries`] with the [`DynamicPlugin::lib_ref`] handed out for it,
//!   so "we keep these on purpose" is a fact a reload problem can be diagnosed
//!   against instead of an invisible growth curve.
//! * **A reload does not reload the module.** Loading the *same path* again
//!   hits the module the process already has mapped: on Windows `LoadLibrary`
//!   returns the existing handle and `DllMain` does not run a second time, so
//!   the plugin's statics carry over from the previous load. A reload is a
//!   fresh `ngu_init` on an old module, which is why [`DynamicPlugin::apply`]
//!   calls `ngu_shutdown` before it initializes again.
//!
//! # Lifetime contract for the capability table
//!
//! The `HostApi` table and the state behind its `user_data` pointer are
//! **process-lifetime**, exactly like the library. A plugin is entitled to keep
//! the table pointer and call back from its own threads at any later time, and
//! since the module stays mapped there is no point at which a callback can
//! become impossible — only at which a dangling pointer would be fatal. So both
//! are leaked deliberately; dropping them when `apply` returns, or when the
//! kernel drops the plugin, would break a contract the plugin cannot see.
//!
//! # Contract for plugin authors
//!
//! ```ignore
//! #[no_mangle] pub extern "C" fn ngu_abi_version() -> u32;           // optional, checked first
//! #[no_mangle] pub extern "C" fn ngu_abi_layout() -> u32;            // optional, checked first
//! #[no_mangle] pub extern "C" fn ngu_describe() -> *const c_char;    // metadata JSON, static
//! #[no_mangle] pub extern "C" fn ngu_init(host: *const HostApi) -> i32;
//! #[no_mangle] pub extern "C" fn ngu_call(tool: *const c_char, args: *const c_char) -> *const c_char;
//! #[no_mangle] pub extern "C" fn ngu_free_string(ptr: *mut c_char);  // frees a ngu_call result
//! #[no_mangle] pub extern "C" fn ngu_shutdown();                     // optional
//! ```
//!
//! The two ABI symbols are optional but strongly preferred: they are plain
//! `u32` returns, so the host can refuse a mismatched plugin **before** calling
//! anything else in it. A plugin that omits them is still checked, but only
//! through its descriptor JSON — which means the check itself is a call into
//! unknown code.

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

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
/// Optional exported symbol: the ABI revision as a plain `u32`.
///
/// This exists so the revision can be **checked before anything else in the
/// plugin is called**. `ngu_describe` is itself a call into code built by
/// somebody else; asking it to describe a table layout it may not share is
/// exactly the call that can go wrong.
pub const SYM_ABI: &[u8] = b"ngu_abi_version\0";
/// Optional exported symbol: the `HostApi` layout fingerprint as a plain `u32`.
///
/// See [`layout_fingerprint`] for what it hashes. A plugin that mirrors the
/// table itself — as every plugin must — can export this to prove its mirror
/// matches, which the ABI number alone cannot express: two builds can agree on
/// "ABI 1" and still disagree about padding or pointer width.
pub const SYM_LAYOUT: &[u8] = b"ngu_abi_layout\0";

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

/// A fingerprint of the `HostApi` layout this build expects.
///
/// [`ABI_VERSION`] says "we both mean revision 1". It cannot say "we both mean
/// the same bytes": field order and types are fixed by `repr(C)`, but padding,
/// pointer width and target triple are not, so a table compiled elsewhere can
/// put `register_tool` where this build puts `log`. This hashes the size and
/// every field's offset out of a live table, and a plugin that exports
/// [`SYM_LAYOUT`] must agree with it before any other symbol is touched.
///
/// The value is stable for one build of this crate on one target and is not a
/// protocol number: it changes when the table changes, which is the point.
pub fn layout_fingerprint() -> u32 {
    let probe = HostApi {
        abi_version: ABI_VERSION,
        user_data: std::ptr::null_mut(),
        log: host_log,
        register_tool: host_register_tool,
        call_tool: host_call_tool,
        free_string: host_free_string,
    };

    // Measured from a live table rather than written down, so this cannot drift
    // away from the definition it describes.
    let base = &probe as *const HostApi as usize;
    let probes: [usize; 7] = [
        std::mem::size_of::<HostApi>(),
        (&probe.abi_version as *const u32 as usize).wrapping_sub(base),
        (&probe.user_data as *const *mut c_void as usize).wrapping_sub(base),
        (&probe.log as *const _ as usize).wrapping_sub(base),
        (&probe.register_tool as *const _ as usize).wrapping_sub(base),
        (&probe.call_tool as *const _ as usize).wrapping_sub(base),
        (&probe.free_string as *const _ as usize).wrapping_sub(base),
    ];

    // FNV-1a, 32 bit: short, dependency-free, and stable across runs.
    let mut hash: u32 = 0x811c_9dc5;
    for value in probes {
        for byte in (value as u64).to_le_bytes() {
            hash ^= u32::from(byte);
            hash = hash.wrapping_mul(0x0100_0193);
        }
    }
    hash
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
///
/// Process-lifetime, like the library: the address of this value is handed to
/// the plugin as `HostApi::user_data` and a plugin is entitled to keep using it
/// for as long as the module is mapped — which here means for ever. It is
/// therefore kept alive by a leaked `Arc` (see [`DynamicPlugin::load`]) as well
/// as by the plugin struct, so no callback can ever find freed memory.
struct DynamicState {
    plugin: String,
    tools: Mutex<Vec<ToolSpec>>,
    log: Mutex<Vec<(u32, String)>>,
}

/// One library the process keeps mapped on purpose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeakedLibrary {
    /// Handle handed out with the plugin that loaded it, for diagnostics.
    pub lib_ref: u64,
    /// Path as it was given to the loader.
    pub path: PathBuf,
    /// How many times this exact path has been loaded in this process.
    ///
    /// More than one means a later load found the module already mapped: that
    /// is the case where `DllMain` does not run again and the plugin's statics
    /// survive from the previous load.
    pub loads: usize,
}

/// Every library this process has leaked, in load order.
static LIBRARIES: OnceLock<Mutex<Vec<LeakedLibrary>>> = OnceLock::new();
/// Serial for [`DynamicPlugin::lib_ref`].
static LIB_SERIAL: AtomicU64 = AtomicU64::new(1);

/// Record one deliberate leak and hand back its reference plus its load count.
///
/// Written down rather than commented because a leak nobody can count is
/// indistinguishable from a bug: a reload problem can be read against this list
/// instead of guessed at from memory growth.
fn record_library(path: &Path) -> (u64, usize) {
    let lib_ref = LIB_SERIAL.fetch_add(1, Ordering::Relaxed);
    let registry = LIBRARIES.get_or_init(|| Mutex::new(Vec::new()));
    let mut entries = match registry.lock() {
        Ok(entries) => entries,
        Err(poisoned) => poisoned.into_inner(),
    };
    let loads = match entries.iter_mut().find(|entry| entry.path.as_path() == path) {
        Some(existing) => {
            existing.loads += 1;
            existing.loads
        }
        None => {
            entries.push(LeakedLibrary { lib_ref, path: path.to_path_buf(), loads: 1 });
            1
        }
    };
    (lib_ref, loads)
}

/// Every library kept mapped for the process lifetime.
///
/// The list is the answer to "why does the process hold this module": each entry
/// is a plugin this kernel loaded and deliberately never unloaded.
pub fn leaked_libraries() -> Vec<LeakedLibrary> {
    match LIBRARIES.get() {
        Some(registry) => match registry.lock() {
            Ok(entries) => entries.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        },
        None => Vec::new(),
    }
}

/// How many libraries this process has loaded and kept.
pub fn leaked_library_count() -> usize {
    leaked_libraries().len()
}

/// A plugin loaded from a shared library.
///
/// The library is held for the process lifetime; see the module documentation
/// for why unloading is not attempted.
pub struct DynamicPlugin {
    path: PathBuf,
    /// Which load this is, for diagnostics on a reload.
    ///
    /// Two loads of the same path get different refs and the same mapped
    /// module; two loads of different builds get different refs and different
    /// modules. Without this the two cases are indistinguishable from outside.
    lib_ref: u64,
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
    /// The capability table handed to the plugin, leaked for its lifetime.
    ///
    /// It lives here rather than on `apply`'s stack because the plugin **keeps
    /// the pointer** for as long as its module is mapped. A stack-allocated
    /// table would be freed when `apply` returned, and a box freed when this
    /// struct dropped, leaving every later callback — including the shutdown
    /// hook, and anything a plugin thread calls — reading memory the host no
    /// longer owns. The module is never unloaded, so the table must never be
    /// freed either: the two lifetimes are the same one.
    host_api: &'static HostApi,
    /// Whether `ngu_init` has completed.
    ///
    /// `ngu_shutdown` is only called on a plugin that was actually initialized:
    /// a plugin is entitled to assume its teardown follows its setup, and the
    /// first `apply` has no teardown to run.
    started: AtomicBool,
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

        // ABI first, before any other call into the module.
        //
        // The old order called `ngu_describe` and then asked the answer whether
        // the plugin was compatible — a call into code whose table layout was
        // still unverified decided whether the layout was acceptable. Both
        // checks below are plain `u32` symbols, so a plugin that reports them is
        // refused *without anything of its own having run*. A plugin that omits
        // them falls through to the descriptor check further down, which is
        // weaker for exactly that reason and says so.
        let label = path.display().to_string();

        let declared_abi: Option<extern "C" fn() -> u32> = {
            let symbol: Option<libloading::Symbol<extern "C" fn() -> u32>> =
                library.get(SYM_ABI).ok();
            symbol.map(|raw| *raw)
        };
        if let Some(abi) = declared_abi {
            let reported = guard("ngu_abi_version", &label, || abi())?;
            if reported != ABI_VERSION {
                return Err(anyhow!(
                    "{label} targets ABI {reported}, this kernel speaks {ABI_VERSION}; refused \
                     before any call into the plugin"
                ));
            }
        }

        let declared_layout: Option<extern "C" fn() -> u32> = {
            let symbol: Option<libloading::Symbol<extern "C" fn() -> u32>> =
                library.get(SYM_LAYOUT).ok();
            symbol.map(|raw| *raw)
        };
        if let Some(layout) = declared_layout {
            let reported = guard("ngu_abi_layout", &label, || layout())?;
            let expected = layout_fingerprint();
            if reported != expected {
                return Err(anyhow!(
                    "{label} was built against HostApi layout {reported:#010x}, this kernel's is \
                     {expected:#010x}; the capability table would be read at the wrong offsets"
                ));
            }
        }

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
        let meta_ptr = guard("ngu_describe", &label, || describe())?;
        if meta_ptr.is_null() {
            return Err(anyhow!("{label} returned a null descriptor"));
        }
        let meta_text = CStr::from_ptr(meta_ptr).to_string_lossy().to_string();
        let meta: PluginMeta = serde_json::from_str(&meta_text)
            .with_context(|| format!("{label} returned an unparsable descriptor: {meta_text}"))?;

        if meta.name.trim().is_empty() {
            return Err(anyhow!("{label} reports an empty plugin name"));
        }

        // The descriptor check is the *fallback*: a plugin that exports
        // `ngu_abi_version` already proved itself before this call, and one that
        // does not can only be judged by what it says about itself.
        check_meta_abi(&meta)?;

        let state = Arc::new(DynamicState {
            plugin: meta.name.clone(),
            tools: Mutex::new(Vec::new()),
            log: Mutex::new(Vec::new()),
        });

        // Both of these now outlive the struct on purpose, and must: the plugin
        // keeps the table pointer and may call back from its own threads at any
        // later time. The library is mapped for the process lifetime, so the
        // memory these point at has to be too — see the module documentation.
        let user_data = Arc::into_raw(Arc::clone(&state)) as *mut c_void;
        let host_api: &'static HostApi = Box::leak(Box::new(HostApi {
            abi_version: ABI_VERSION,
            user_data,
            log: host_log,
            register_tool: host_register_tool,
            call_tool: host_call_tool,
            free_string: host_free_string,
        }));

        // Recorded, not merely commented: this handle is never released, and how
        // many times this path has been loaded is exactly what makes a "reload"
        // diagnosable.
        let (lib_ref, loads) = record_library(path);
        if loads > 1 {
            eprintln!(
                "[dylib] {label} is already mapped (load {loads}, ref {lib_ref}); the module's \
                 `DllMain` will not run again and its statics carry over from the previous load — \
                 a reload is a fresh `ngu_init` on the same module, not a fresh module"
            );
        }

        Ok(Self {
            path: path.to_path_buf(),
            lib_ref,
            library: Box::leak(Box::new(library)),
            meta,
            init,
            call,
            free,
            shutdown,
            state,
            host_api,
            started: AtomicBool::new(false),
        })
    }

    /// The library this plugin came from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The handle this load was registered under, for reload diagnostics.
    ///
    /// Compare two plugins' refs to tell "the same module initialized twice"
    /// (same path, different refs) from "two builds side by side" (different
    /// paths, different refs). See [`leaked_libraries`].
    pub fn lib_ref(&self) -> u64 {
        self.lib_ref
    }

    /// Whether `ngu_init` has completed and teardown is still owed.
    pub fn is_started(&self) -> bool {
        self.started.load(Ordering::SeqCst)
    }

    /// Ask the plugin to tear down, at most once per initialization.
    ///
    /// The flag is swapped rather than read so the teardown cannot run twice —
    /// `Drop` and a reload both come through here, and a plugin is entitled to
    /// assume one shutdown per init.
    fn shutdown_now(&self) {
        let Some(shutdown) = self.shutdown else {
            return;
        };
        if !self.started.swap(false, Ordering::SeqCst) {
            return;
        }
        if let Err(error) = guard("ngu_shutdown", &self.meta.name, || shutdown()) {
            // Nothing to do about it: the plugin is going away either way, and a
            // panic here is the plugin's, not the host's.
            eprintln!("[dylib] {error:#}");
        }
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
            .field("lib_ref", &self.lib_ref)
            .field("started", &self.is_started())
            .finish_non_exhaustive()
    }
}

impl Drop for DynamicPlugin {
    fn drop(&mut self) {
        // Only ask the plugin to clean up its own state, and only if it was
        // initialized. Nothing here may free the library, the capability table
        // or the shared state: all three are process-lifetime by design (see the
        // module documentation), so a plugin thread that calls back after this
        // struct is gone still finds live memory.
        self.shutdown_now();
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
        // A reload must not see the previous run's registrations. A poisoned
        // lock is an error rather than a silently skipped clear: continuing
        // would hand back a tool list nobody can reason about.
        {
            let mut tools = self.state.tools.lock().map_err(|_| {
                anyhow!(
                    "plugin '{}' left its tool list locked by a panicking thread",
                    self.meta.name
                )
            })?;
            tools.clear();
        }

        // Reset the plugin's own state first when it was running. The library
        // stays mapped for the process lifetime (see the module documentation),
        // so `ngu_init` after a reload would otherwise build on the statics the
        // previous run left behind.
        self.shutdown_now();

        // The table handed over here outlives this call: it is a `&'static`
        // owned by the process, not by this stack frame.
        let init = self.init;
        let host_api = self.host_api as *const HostApi;
        let code = guard("ngu_init", &self.meta.name, move || init(host_api))?;
        if code != 0 {
            // The plugin refused: make sure a half-initialized module does not
            // keep its own state for the next attempt.
            self.shutdown_now();
            return Err(anyhow!(
                "plugin '{}' refused to initialize (code {code})",
                self.meta.name
            ));
        }
        self.started.store(true, Ordering::SeqCst);

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

    let result_ptr = guard("ngu_call", plugin, || call(tool_c.as_ptr(), args_c.as_ptr()))?;
    if result_ptr.is_null() {
        return Err(anyhow!("plugin '{plugin}' returned null for tool '{tool}'"));
    }

    let text = unsafe { CStr::from_ptr(result_ptr) }.to_string_lossy().to_string();
    // The plugin allocated this string; freeing it is a call back into plugin
    // code, so it gets the same barrier. A failure here does not invalidate the
    // result we already copied out.
    if let Err(error) = guard("ngu_free_string", plugin, || free(result_ptr as *mut c_char)) {
        eprintln!("[dylib] {error:#}");
    }

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

// ------------------------------------------------------------ panic barriers

/// Run a call that crosses the host/plugin boundary with a panic barrier.
///
/// `AssertUnwindSafe` is required, because a plugin's interior state is by
/// definition not `UnwindSafe` — the host cannot promise it is unchanged by a
/// panic it did not cause. That is exactly why asserting it is honest rather
/// than sloppy:
///
/// * Everything a callback touches behind the boundary is the plugin's own
///   state, and a plugin that panics has already decided it cannot continue —
///   the host is not resuming an operation mid-way, it is converting a crash
///   into an error and dropping that call.
/// * The host state reachable from a plugin (`DynamicState`) is behind
///   `Mutex`es whose contents are whole values, and the kernel's own tables are
///   not reachable across this boundary at all. A panic therefore cannot leave
///   the host half-updated; a poisoned lock reports itself as an error at the
///   next use.
///
/// What this barrier cannot do, and does not pretend to: a panic that starts
/// inside an exported `extern "C"` function aborts at that boundary in the
/// plugin before any host frame sees it. Preventing that is the plugin SDK's
/// job — its exports must catch their own panics — and the barrier here covers
/// the other direction, where *host* code entered from a plugin panics and must
/// not unwind into a foreign frame.
fn guard<T>(what: &str, plugin: &str, body: impl FnOnce() -> T) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => Ok(value),
        Err(payload) => Err(anyhow!(
            "plugin '{plugin}' panicked in {what}: {}",
            panic_text(payload)
        )),
    }
}

/// The message out of a panic payload, when it has one.
fn panic_text(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic with a non-string payload".to_string()
    }
}

/// Check the ABI a plugin claims in its descriptor.
///
/// Split out from `load` so the rule can be tested without a shared library, and
/// kept as the *fallback* path: a plugin exporting `ngu_abi_version` was already
/// checked before its descriptor was read.
fn check_meta_abi(meta: &PluginMeta) -> Result<()> {
    match meta.abi_version {
        Some(version) if version == ABI_VERSION => Ok(()),
        Some(version) => Err(anyhow!(
            "plugin '{}' targets ABI {version}, this kernel speaks {ABI_VERSION}",
            meta.name
        )),
        None => Err(anyhow!(
            "plugin '{}' does not report an ABI version; rebuild it against the plugin SDK",
            meta.name
        )),
    }
}

// ---------------------------------------------------------------- callbacks

extern "C" fn host_log(user_data: *mut c_void, level: u32, message: *const c_char) {
    // Every callback is entered *from* plugin code. A panic in host code would
    // unwind into a foreign `extern "C"` frame, so it is caught here and turned
    // into a silent no-op: a lost log line is not worth the process.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        host_log_inner(user_data, level, message)
    }));
}

fn host_log_inner(user_data: *mut c_void, level: u32, message: *const c_char) {
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
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        host_register_tool_inner(user_data, def_json)
    }))
    // A refused registration is already the documented failure code.
    .unwrap_or(0)
}

fn host_register_tool_inner(user_data: *mut c_void, def_json: *const c_char) -> u64 {
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
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        host_call_tool_inner(user_data, name, args)
    })) {
        Ok(pointer) => pointer,
        // A null answer is already what this callback returns for a call it
        // cannot serve, so a panic degrades to the same reported failure.
        Err(_) => std::ptr::null(),
    }
}

fn host_call_tool_inner(
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
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        host_free_string_inner(ptr)
    }));
}

fn host_free_string_inner(ptr: *mut c_char) {
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
    fn a_panic_across_the_boundary_becomes_an_error() {
        let failure = guard("ngu_call", "demo", || -> u32 { panic!("the plugin exploded") })
            .expect_err("a panic is reported, not propagated");
        let text = format!("{failure:#}");
        assert!(text.contains("demo"), "{text}");
        assert!(text.contains("ngu_call"), "the crossing point is named: {text}");
        assert!(text.contains("exploded"), "the plugin's own message survives: {text}");
    }

    #[test]
    fn a_panic_with_a_string_payload_keeps_its_message() {
        let reason = String::from("a formatted reason");
        let failure = guard("ngu_init", "demo", move || -> () { panic!("{reason}") })
            .expect_err("panics are errors");
        assert!(format!("{failure:#}").contains("a formatted reason"));
    }

    #[test]
    fn the_metadata_abi_check_accepts_only_this_revision() {
        let mut meta: PluginMeta =
            serde_json::from_str(r#"{"name":"x","abi_version":1}"#).unwrap();
        assert!(check_meta_abi(&meta).is_ok());

        meta.abi_version = Some(ABI_VERSION + 1);
        let error = check_meta_abi(&meta).expect_err("another revision is refused");
        assert!(format!("{error:#}").contains("targets ABI"), "{error:#}");

        meta.abi_version = None;
        let error = check_meta_abi(&meta).expect_err("a silent plugin is refused");
        assert!(
            format!("{error:#}").contains("does not report an ABI version"),
            "{error:#}"
        );
    }

    #[test]
    fn the_layout_fingerprint_describes_this_build() {
        // Stable within a build...
        assert_eq!(layout_fingerprint(), layout_fingerprint());
        // ...and non-zero, so an unanswered or zeroed value is distinguishable
        // from a genuine match.
        assert_ne!(layout_fingerprint(), 0);
    }

    #[test]
    fn every_kept_library_is_counted_so_the_leak_is_diagnosable() {
        let path = Path::new("ngu-test-not-a-real-library.dll");
        let before = leaked_library_count();

        let (first_ref, first_loads) = record_library(path);
        let (second_ref, second_loads) = record_library(path);

        assert_eq!(first_loads, 1);
        assert_eq!(
            second_loads, 2,
            "a second load of the same path is recorded as a reload of one module"
        );
        assert_ne!(first_ref, second_ref, "each load gets its own handle");
        assert!(leaked_library_count() >= before + 1);

        let entry = leaked_libraries()
            .into_iter()
            .find(|entry| entry.path.as_path() == path)
            .expect("the leak is written down, not just commented");
        assert_eq!(entry.loads, 2);
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
