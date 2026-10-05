//! Cross-module integration tests over the kernel's public surface.
//!
//! Unit tests live beside the code they cover. These check the seams: an
//! assembly manifest actually producing tools, a hot reload actually changing
//! what the next turn sees, a skill actually reaching the prompt catalog.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nguruvilu::agent::TurnConfig;
use nguruvilu::assembly::Assembly;
use nguruvilu::context::{CachePolicy, InjectionEngine, InjectionEntry, PositionKind};
use nguruvilu::git::GitSnapshot;
use nguruvilu::hotreload::{
    ApplyOutcome, Change, ChangeKind, ChangePayload, Consent, ModelRoute, Runtime,
};
use nguruvilu::ledger::{Ledger, LedgerEntry};
use nguruvilu::llm::to_wire;
use nguruvilu::loader::Loader;
use nguruvilu::message::{Message, Role, ToolCall};
use nguruvilu::plugin::{Contributions, FiberState, Kernel, Plugin, PluginCtx, RealmMap};
use nguruvilu::session::{JsonlStore, Session, SessionStore};
use nguruvilu::skills::SkillRegistry;
use nguruvilu::tools::{ConflictPolicy, ToolDef, ToolFuture, ToolOutput, ToolRegistry};
use serde_json::{json, Value};

// ------------------------------------------------------------------ helpers

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ngu-it-{tag}-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn write_skill(root: &Path, id: &str, description: &str, body: &str) {
    let dir = root.join(id);
    std::fs::create_dir_all(&dir).expect("skill dir");
    std::fs::write(
        dir.join("SKILL.md"),
        format!("---\nname: {id}\ndescription: {description}\n---\n\n{body}\n"),
    )
    .expect("skill file");
}

fn empty_tool(name: &str, owner: &str) -> ToolDef {
    ToolDef::new(
        name,
        "a test tool",
        json!({ "type": "object", "properties": {} }),
        owner,
        |_| Box::pin(async { Ok(ToolOutput::text("ok")) }) as ToolFuture,
    )
}

/// A plugin that provides one service and registers one tool.
struct Contributor {
    name: String,
    service: String,
}

impl Plugin for Contributor {
    fn name(&self) -> &str {
        &self.name
    }
    fn provide(&self) -> Vec<String> {
        vec![self.service.clone()]
    }
    fn apply(&self, ctx: &PluginCtx) -> anyhow::Result<Contributions> {
        Ok(Contributions::new()
            .service(self.service.clone(), Arc::new(ctx.fiber))
            .tool(empty_tool(&format!("tool_of_{}", ctx.fiber), &self.name)))
    }
}

/// A plugin that needs a service before it can load.
struct Dependent {
    name: String,
    needs: String,
}

impl Plugin for Dependent {
    fn name(&self) -> &str {
        &self.name
    }
    fn inject(&self) -> Vec<String> {
        vec![self.needs.clone()]
    }
    fn apply(&self, ctx: &PluginCtx) -> anyhow::Result<Contributions> {
        let _ = ctx.get::<u64>(&self.needs);
        Ok(Contributions::new().tool(empty_tool(&format!("dependent_{}", ctx.fiber), &self.name)))
    }
}

// ---------------------------------------------------------------- messages

#[test]
fn the_wire_format_round_trips_a_full_tool_exchange() {
    let assistant = Message::assistant_tools(
        vec![ToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: r#"{"path":"a.txt"}"#.into(),
        }],
        Some("let me look".into()),
    );
    let wire = to_wire(&assistant);
    assert_eq!(wire["role"], "assistant");
    assert_eq!(wire["tool_calls"][0]["function"]["name"], "read");

    let result = Message::tool_result("call_1", "read", "file contents");
    let wire = to_wire(&result);
    assert_eq!(wire["role"], "tool");
    assert_eq!(wire["tool_call_id"], "call_1");
    assert_eq!(wire["content"], "file contents");
}

#[test]
fn an_assistant_turn_without_text_sends_null_content() {
    let silent = Message::assistant_tools(
        vec![ToolCall { id: "c".into(), name: "bash".into(), arguments: "{}".into() }],
        None,
    );
    assert!(to_wire(&silent)["content"].is_null());
}

// ------------------------------------------------------------ tool registry

#[test]
fn a_duplicate_tool_name_is_refused_by_default() {
    let mut tools = ToolRegistry::new();
    tools.register(empty_tool("search", "plugin-a"), ConflictPolicy::Error).unwrap();
    let error = tools
        .register(empty_tool("search", "plugin-b"), ConflictPolicy::Error)
        .expect_err("second registration must fail");
    let text = format!("{error:#}");
    assert!(text.contains("plugin-a") && text.contains("plugin-b"), "{text}");
}

#[test]
fn an_explicit_override_is_recorded() {
    let mut tools = ToolRegistry::new();
    tools.register(empty_tool("search", "a"), ConflictPolicy::Error).unwrap();
    tools.register(empty_tool("search", "b"), ConflictPolicy::Override).unwrap();
    assert_eq!(tools.owner("search"), Some("b"));
    assert_eq!(tools.overrides().len(), 1);
    assert_eq!(tools.overrides()[0].previous_owner, "a");
}

#[test]
fn the_base_tools_are_registered_and_sorted() {
    let tools = ToolRegistry::with_base_tools().unwrap();
    assert_eq!(tools.names(), vec!["bash", "edit", "read", "write"]);
    for name in tools.names() {
        assert_eq!(tools.owner(&name), Some("kernel"));
    }
}

// -------------------------------------------------------------- base tools

#[tokio::test]
async fn the_four_tools_work_together() {
    let dir = temp_dir("tools");
    let file = dir.join("note.txt");
    let tools = ToolRegistry::with_base_tools().unwrap();

    let written = tools
        .execute("write", &json!({ "path": file, "content": "alpha\n" }).to_string())
        .await
        .unwrap();
    assert!(written.text.contains("wrote"));

    let edited = tools
        .execute(
            "edit",
            &json!({ "path": file, "old_str": "alpha", "new_str": "beta" }).to_string(),
        )
        .await
        .unwrap();
    assert!(edited.text.contains("replaced 1 occurrence"));

    let read = tools
        .execute("read", &json!({ "path": file }).to_string())
        .await
        .unwrap();
    assert!(read.text.contains("beta"));
    assert!(!read.text.contains("alpha"));

    let shell = tools
        .execute("bash", &json!({ "command": "echo done" }).to_string())
        .await
        .unwrap();
    assert!(shell.text.contains("exit code: 0"));
    assert!(shell.text.contains("done"));
}

#[tokio::test]
async fn bash_works_with_posix_syntax_on_every_platform() {
    // Models write POSIX commands by default; shell resolution has to make
    // that work rather than hand back a syntax error.
    let dir = temp_dir("bash-posix");
    std::fs::write(dir.join("marker.txt"), "x").unwrap();

    let tools = ToolRegistry::with_base_tools().unwrap();
    let out = tools
        .execute(
            "bash",
            &json!({ "command": "ls", "cwd": dir.to_string_lossy() }).to_string(),
        )
        .await
        .unwrap();
    assert!(out.text.contains("exit code: 0"), "{}", out.text);
    assert!(out.text.contains("marker.txt"), "{}", out.text);
}

// ------------------------------------------------------------ plugin kernel

#[test]
fn a_plugin_loads_its_service_and_tool_into_a_realm() {
    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Contributor {
        name: "contrib".into(),
        service: "thing".into(),
    }));

    let realm = RealmMap::new().with_default(kernel.new_realm());
    let fiber = kernel.load("contrib", realm.clone(), Value::Null).unwrap();

    assert!(kernel.service_view(realm).has("thing"));
    assert!(!kernel.service_view(RealmMap::new()).has("thing"));
    assert_eq!(kernel.tools_of(fiber).len(), 1);
}

#[test]
fn provenance_tracks_which_pack_asked_and_survives_a_reload() {
    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Contributor {
        name: "contrib".into(),
        service: "thing".into(),
    }));

    // A load with no pack behind it stays bare — that is kernel code.
    let bare = kernel.load("contrib", RealmMap::new(), Value::Null).unwrap();
    assert!(kernel.plugin_origins().is_empty());
    kernel.unload(bare).unwrap();

    // A pack's assembly names itself…
    kernel
        .load_owned(
            "contrib",
            RealmMap::new(),
            Value::Null,
            Some("subagent".into()),
        )
        .unwrap();
    assert_eq!(
        kernel.plugin_origins().get("contrib").map(String::as_str),
        Some("subagent")
    );

    // …re-applying (the configuration axis) must not lose it…
    kernel.reload_plugin("contrib").unwrap();
    assert_eq!(
        kernel.plugin_origins().get("contrib").map(String::as_str),
        Some("subagent"),
        "a reload keeps the provenance it was loaded with"
    );

    // …and unloading forgets it: the pack is not there anymore.
    let current = kernel
        .fibers()
        .iter()
        .find(|fiber| fiber.plugin == "contrib")
        .map(|fiber| fiber.id)
        .expect("the fiber is loaded");
    kernel.unload(current).unwrap();
    assert!(kernel.plugin_origins().is_empty());
}

#[test]
fn a_dependent_plugin_activates_when_its_dependency_appears() {
    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Dependent {
        name: "consumer".into(),
        needs: "thing".into(),
    }));
    kernel.define(Arc::new(Contributor {
        name: "provider".into(),
        service: "thing".into(),
    }));

    let consumer = kernel.load("consumer", RealmMap::new(), Value::Null).unwrap();
    assert_eq!(kernel.tools_of(consumer).len(), 0, "nothing yet: no dependency");

    kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
    let events = kernel.refresh().unwrap();

    assert!(events.iter().any(|e| e.action == "activated"), "{events:?}");
    assert_eq!(kernel.tools_of(consumer).len(), 1, "the tool arrived");
}

#[test]
fn replacing_a_dependency_rewires_dependents_without_a_manual_reload() {
    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Dependent {
        name: "consumer".into(),
        needs: "thing".into(),
    }));
    kernel.define(Arc::new(Contributor {
        name: "provider".into(),
        service: "thing".into(),
    }));

    let consumer = kernel.load("consumer", RealmMap::new(), Value::Null).unwrap();
    let provider = kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
    let events = kernel.refresh().unwrap();
    assert!(
        events.iter().any(|e| e.fiber == consumer && e.action == "activated"),
        "{events:?}"
    );
    assert_eq!(kernel.tools_of(consumer).len(), 1);

    // The dependency goes away: the dependent is deactivated, not destroyed.
    kernel.unload(provider).unwrap();
    assert_eq!(
        kernel.tools_of(consumer).len(),
        0,
        "its tool went away with the dependency"
    );
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Pending);

    // It comes back by itself when the dependency returns — no manual reload.
    kernel.load("provider", RealmMap::new(), Value::Null).unwrap();
    let events = kernel.refresh().unwrap();
    assert!(
        events.iter().any(|e| e.fiber == consumer && e.action == "activated"),
        "the dependent reactivated on its own: {events:?}"
    );
    assert_eq!(kernel.tools_of(consumer).len(), 1, "and its tool returned");
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);
}

#[test]
fn unloading_a_plugin_removes_its_service_and_tool() {
    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Contributor {
        name: "contrib".into(),
        service: "thing".into(),
    }));

    let fiber = kernel.load("contrib", RealmMap::new(), Value::Null).unwrap();
    let tool = kernel.tools_of(fiber)[0].clone();
    assert!(kernel.tools().get(&tool).is_some());

    kernel.unload(fiber).unwrap();
    assert!(kernel.tools().get(&tool).is_none());
    assert!(!kernel.service_view(RealmMap::new()).has("thing"));
    // Base tools survive.
    assert!(kernel.tools().get("read").is_some());
}

// ------------------------------------------------------------------ skills

#[test]
fn skills_scan_and_build_a_catalog_without_leaking_bodies() {
    let root = temp_dir("skills");
    write_skill(&root, "pdf-tools", "Read PDFs", "Run pdftotext first.");
    write_skill(&root, "csv-tools", "Work with CSV", "Use xsv.");

    let mut registry = SkillRegistry::with_roots([root]);
    let report = registry.scan().unwrap();
    assert_eq!(report.loaded, 2);

    let catalog = registry.catalog().expect("catalog");
    assert!(catalog.contains("pdf-tools: Read PDFs"));
    assert!(catalog.contains("csv-tools: Work with CSV"));
    // The standing prompt carries descriptions only; bodies load on demand.
    assert!(!catalog.contains("pdftotext"));
}

// ------------------------------------------------------ assembly + loader

#[tokio::test]
async fn an_assembly_manifest_loads_plugins_and_skills_together() {
    let dir = temp_dir("assembly");
    let skills_dir = dir.join("skills");
    write_skill(&skills_dir, "pdf-tools", "Read PDFs", "Run pdftotext.");

    let manifest = format!(
        r#"
version: 1
name: integration-pack
defaults:
  scope: session
  on_failure: abort
stages:
  - name: foundation
    plugins:
      - id: contrib
        source: "builtin:contrib"
        order: 10
  - name: extensions
    skills:
      - id: pdf
        source: "{}"
        order: 10
"#,
        skills_dir.display().to_string().replace('\\', "/")
    );

    let plan = Assembly::parse(&manifest).unwrap().plan("linux").unwrap();
    assert_eq!(plan.steps.len(), 2);

    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Contributor {
        name: "contrib".into(),
        service: "thing".into(),
    }));

    let ledger_path = dir.join("installed.json");
    let mut loader = Loader::new(kernel, ledger_path.clone(), "integration-pack");
    let report = loader.apply(&plan).await.unwrap();

    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.loaded.len(), 2, "{report:?}");

    // The plugin landed in an isolated realm (session scope).
    assert_eq!(loader.kernel().service_list().len(), 1);
    assert!(!loader.kernel().service_view(RealmMap::new()).has("thing"));

    // The skill landed and its tool appeared.
    assert_eq!(loader.skills().len(), 1);
    assert!(loader.kernel().tools().get("skill").is_some());

    // The ledger recorded both, so a second apply is cheap.
    let ledger = Ledger::load(&ledger_path).unwrap();
    assert!(ledger.has("plugin", "contrib", None));
    assert!(ledger.has("skill", "pdf", None));

    loader.shutdown().await;
}

#[tokio::test]
async fn a_failing_entry_with_skip_policy_does_not_stop_the_rest() {
    let dir = temp_dir("assembly-skip");
    let manifest = r#"
version: 1
stages:
  - name: mixed
    plugins:
      - id: broken
        source: "builtin:missing"
        on_failure: skip
        order: 10
      - id: good
        source: "builtin:contrib"
        order: 20
"#;
    let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();

    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Contributor {
        name: "contrib".into(),
        service: "thing".into(),
    }));

    let mut loader = Loader::new(kernel, dir.join("installed.json"), "pack");
    let report = loader.apply(&plan).await.unwrap();

    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].id, "broken");
    assert_eq!(report.loaded, vec!["plugin:good"]);
    assert!(loader.kernel().service_list().len() == 1, "the good one landed");

    loader.shutdown().await;
}

#[tokio::test]
async fn abort_policy_stops_at_the_first_failure() {
    let dir = temp_dir("assembly-abort");
    let manifest = r#"
version: 1
stages:
  - name: mixed
    plugins:
      - id: broken
        source: "builtin:missing"
        order: 10
      - id: good
        source: "builtin:contrib"
        order: 20
"#;
    let plan = Assembly::parse(manifest).unwrap().plan("linux").unwrap();

    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Contributor {
        name: "contrib".into(),
        service: "thing".into(),
    }));

    let mut loader = Loader::new(kernel, dir.join("installed.json"), "pack");
    let error = loader.apply(&plan).await.expect_err("abort on failure");
    assert!(format!("{error:#}").contains("broken"));
    assert!(loader.kernel().service_list().is_empty(), "nothing after the failure ran");

    loader.shutdown().await;
}

// --------------------------------------------------------------- hot reload

#[test]
fn a_session_change_applies_and_a_global_one_waits() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut runtime = Runtime::new(tools);

    // Session scope: immediate.
    let outcome = runtime
        .apply(Change::session("s1", ChangePayload::Persona("terse".into())))
        .unwrap();
    assert!(matches!(outcome, ApplyOutcome::Applied(_)));

    // Global scope: consent required, nothing changed yet.
    let outcome = runtime
        .apply(Change::global(ChangePayload::AddSkillRoot("/skills".into())))
        .unwrap();
    let pending = match outcome {
        ApplyOutcome::NeedsConsent(p) => p,
        other => panic!("expected consent, got {other:?}"),
    };
    assert!(runtime.state().skill_roots.is_empty());

    runtime.approve(&pending.id).unwrap();
    assert_eq!(runtime.state().skill_roots, vec!["/skills".to_string()]);
}

#[test]
fn a_snapshot_is_stable_for_the_turn_that_took_it() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut runtime = Runtime::new(tools);

    runtime
        .apply(Change::session("s1", ChangePayload::Persona("first".into())))
        .unwrap();
    let frozen = runtime.snapshot();

    runtime
        .apply(Change::session("s1", ChangePayload::Persona("second".into())))
        .unwrap();

    assert_eq!(frozen.persona, "first", "the taken snapshot did not move");
    assert_eq!(runtime.snapshot().persona, "second", "a new turn sees the change");
}

#[test]
fn a_denied_change_kind_never_reaches_the_runtime() {
    let tools = Arc::new(ToolRegistry::with_base_tools().unwrap());
    let mut runtime = Runtime::new(tools);
    runtime.set_consent(ChangeKind::ModelRoute, Consent::Deny);

    let outcome = runtime
        .apply(Change::global(ChangePayload::ModelRoute(ModelRoute {
            model: "forbidden".into(),
            ..Default::default()
        })))
        .unwrap();

    assert!(matches!(outcome, ApplyOutcome::Denied(_)));
    assert_ne!(runtime.state().model_route.model, "forbidden");
    assert!(runtime.history().is_empty());
}

// ------------------------------------------------------- context injection

#[test]
fn the_injection_engine_budgets_and_places_fragments() {
    let mut engine = InjectionEngine::new();
    engine.budget_percent = 25;
    engine.add(InjectionEntry::constant("rule", "standing rule").at(PositionKind::Prefix));
    engine.add(InjectionEntry::triggered(
        "pdf",
        "use pdftotext",
        vec!["pdf".into()],
    ));

    let conversation = vec![Message::user("please read this pdf")];
    let injection = engine.inject(&conversation, 8000, CachePolicy::Balanced);

    assert_eq!(injection.prefix, vec!["standing rule"]);
    assert_eq!(injection.tail, vec!["use pdftotext"]);
    assert_eq!(injection.activated.len(), 2);
    assert!(!injection.overflowed);
}

#[test]
fn cache_first_moves_prefix_injections_into_the_history() {
    let mut engine = InjectionEngine::new();
    engine.add(InjectionEntry::constant("rule", "standing rule").at(PositionKind::Prefix));

    let injection = engine.inject(&[Message::user("hi")], 8000, CachePolicy::CacheFirst);
    assert!(injection.prefix.is_empty(), "the prefix was left alone");
    assert_eq!(injection.tail, vec!["standing rule"], "but the fragment still landed");
    assert_eq!(injection.relocated, vec!["rule"]);
}

// ------------------------------------------------------------ persistence

#[test]
fn a_session_survives_a_round_trip_through_storage() {
    let dir = temp_dir("session");
    let store = JsonlStore::open(&dir).unwrap();

    let mut session = Session::new(Some("test-model".into()));
    session.messages.push(Message::user("first"));
    store.create(&session).unwrap();
    store
        .append(
            &session.id,
            &[Message::assistant("answer"), Message::user("second")],
        )
        .unwrap();

    let loaded = store.load(&session.id).unwrap().expect("loads");
    assert_eq!(loaded.messages.len(), 3);
    assert_eq!(loaded.messages[0].text(), "first");
    assert_eq!(loaded.messages[2].text(), "second");
    assert_eq!(loaded.model.as_deref(), Some("test-model"));

    let listed = store.list().unwrap();
    assert_eq!(listed[0].message_count, 3);
    assert_eq!(listed[0].title.as_deref(), Some("first"));
}

#[test]
fn a_ledger_entry_round_trips() {
    let dir = temp_dir("ledger");
    let path = dir.join("installed.json");

    let mut ledger = Ledger::new();
    ledger.record(LedgerEntry {
        kind: "mcp".into(),
        id: "github".into(),
        source: "stdio:npx".into(),
        sha1: None,
        path: None,
        scope: "session".into(),
        pack: "pack".into(),
        installed_at: "2026-01-01T00:00:00.000Z".into(),
    });
    ledger.save(&path).unwrap();

    let reloaded = Ledger::load(&path).unwrap();
    assert!(reloaded.has("mcp", "github", None));
    assert_eq!(reloaded.of_kind("mcp").len(), 1);
}

// --------------------------------------------------------------- snapshots

#[test]
fn a_workspace_snapshot_can_be_restored() {
    let dir = temp_dir("snapshot");
    let git = GitSnapshot::open(&dir).unwrap();
    if !git.is_available() {
        return; // git missing: the service degrades rather than failing
    }

    std::fs::write(dir.join("f.txt"), "good").unwrap();
    let good = git.snapshot("good").unwrap().expect("commit");

    std::fs::write(dir.join("f.txt"), "broken").unwrap();
    git.snapshot("broken").unwrap();

    git.restore(&good.commit).unwrap();
    assert_eq!(std::fs::read_to_string(dir.join("f.txt")).unwrap(), "good");
}

// ------------------------------------------------------------- turn config

#[test]
fn turn_settings_carry_the_skills_catalog_into_the_prompt() {
    // The CLI composes persona, base prompt, and catalog; this checks the
    // composition rule the loop depends on.
    struct Fixed {
        base: String,
        catalog: String,
    }

    impl TurnConfig for Fixed {
        fn settings(&self) -> nguruvilu::agent::TurnSettings {
            nguruvilu::agent::TurnSettings {
                system_prompt: format!("{}\n\n{}", self.base, self.catalog),
                tools: Arc::new(ToolRegistry::with_base_tools().unwrap()),
                model: "m".into(),
                version: 1,
                policy: Arc::new(nguruvilu::window::DefaultContextPolicy::default()),
                injection: Arc::new(InjectionEngine::load_default()),
                cache_policy: CachePolicy::default(),
            }
        }
    }

    let config = Fixed {
        base: "base prompt".into(),
        catalog: "Available skills:\n- pdf-tools: Read PDFs".into(),
    };
    let settings = config.settings();
    assert!(settings.system_prompt.contains("base prompt"));
    assert!(settings.system_prompt.contains("pdf-tools"));
    assert!(settings.tools.get("read").is_some());
}

// -------------------------------------------------------------- message log

#[test]
fn a_stored_transcript_reconstructs_the_conversation() {
    let dir = temp_dir("transcript");
    let store = JsonlStore::open(&dir).unwrap();

    let mut session = Session::new(None);
    session.messages.push(Message::user("read a.txt"));
    store.create(&session).unwrap();

    let assistant = Message::assistant_tools(
        vec![ToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: r#"{"path":"a.txt"}"#.into(),
        }],
        None,
    );
    let result = Message::tool_result("c1", "read", "contents");
    store.append(&session.id, &[assistant, result]).unwrap();

    let loaded = store.load(&session.id).unwrap().unwrap();
    assert_eq!(loaded.messages.len(), 3);
    assert_eq!(loaded.messages[1].role, Role::Assistant);
    assert_eq!(loaded.messages[1].tool_calls[0].name, "read");
    assert_eq!(loaded.messages[2].role, Role::Tool);
    assert_eq!(loaded.messages[2].tool_call_id.as_deref(), Some("c1"));
}

// ------------------------------------------------------------------- events

/// A plugin's observer runs while the fiber is loaded and stops at unload:
/// the disposer goes through `Contributions::effect`, and effects are undone
/// by calling — never by dropping.
#[test]
fn a_plugin_observes_events_until_it_unloads() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Watcher {
        seen: Arc<AtomicUsize>,
    }
    impl Plugin for Watcher {
        fn name(&self) -> &str {
            "watcher"
        }
        fn apply(&self, ctx: &PluginCtx) -> anyhow::Result<Contributions> {
            let seen = Arc::clone(&self.seen);
            let dispose = ctx.observe("tool.end", move |_, _| {
                seen.fetch_add(1, Ordering::Relaxed);
            });
            Ok(Contributions::new().effect(dispose))
        }
    }

    let mut kernel = Kernel::new();
    let seen = Arc::new(AtomicUsize::new(0));
    kernel.define(Arc::new(Watcher {
        seen: Arc::clone(&seen),
    }));
    let fiber = kernel
        .load_owned(
            "watcher",
            RealmMap::new(),
            Value::Null,
            Some("watcher-pack".into()),
        )
        .unwrap();

    kernel
        .events()
        .emit("tool.end", &json!({ "tool": "read" }));
    assert_eq!(seen.load(Ordering::Relaxed), 1, "the watcher heard it");

    kernel.unload(fiber).unwrap();
    kernel.events().emit("tool.end", &json!({ "tool": "read" }));
    assert_eq!(
        seen.load(Ordering::Relaxed),
        1,
        "unload unsubscribed: effects are undone, not dropped"
    );
}

/// The tool table is the single funnel, so `tool.start` / `tool.end` fire for
/// every caller — with truncated payloads and an honest `ok` flag either way.
#[tokio::test]
async fn executing_a_tool_emits_start_and_end_with_the_verdict() {
    use std::sync::Mutex;

    let mut registry = ToolRegistry::new();
    registry
        .register(
            ToolDef::new("ok-tool", "returns", json!({ "type": "object" }), "test", |_args| {
                Box::pin(async { Ok(ToolOutput::text("fine")) }) as ToolFuture
            }),
            ConflictPolicy::Error,
        )
        .unwrap();
    registry
        .register(
            ToolDef::new("broken", "fails", json!({ "type": "object" }), "test", |_args| {
                Box::pin(async { Err(std::io::Error::other("boom").into()) }) as ToolFuture
            }),
            ConflictPolicy::Error,
        )
        .unwrap();

    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&seen);
    let _listen = registry.events().subscribe_all(move |name, payload| {
        record.lock().unwrap().push((
            name.to_string(),
            payload["ok"].as_bool(),
            payload["tool"].as_str().unwrap_or("").to_string(),
        ));
    });

    registry.execute("ok-tool", "{}").await.unwrap();
    registry.execute("broken", "{}").await.unwrap_err();

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.iter().map(|(name, _, _)| name.as_str()).collect::<Vec<_>>(),
        [
            nguruvilu::events::TOOL_START,
            nguruvilu::events::TOOL_END,
            nguruvilu::events::TOOL_START,
            nguruvilu::events::TOOL_END,
        ]
    );
    assert_eq!(seen[1].1, Some(true), "ok-tool: ok=true");
    assert_eq!(seen[1].2, "ok-tool");
    assert_eq!(seen[3].1, Some(false), "broken: ok=false");
    assert_eq!(seen[3].2, "broken");
}
