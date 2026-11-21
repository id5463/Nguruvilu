use crate::tools::ToolDef;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::ctx::{Contributions, PluginCtx};
use super::fiber::{FiberState, ReleaseOutcome};
use super::kernel::{Kernel, Plugin};
use super::realm::{RealmId, RealmMap};
use super::service::ServiceViewError;

struct Named {
    name: String,
    inject: Vec<String>,
    provide: Vec<String>,
    body: Arc<dyn Fn(&PluginCtx) -> Result<Contributions> + Send + Sync>,
}

impl Named {
    fn new(
        name: &str,
        body: impl Fn(&PluginCtx) -> Result<Contributions> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            inject: Vec::new(),
            provide: Vec::new(),
            body: Arc::new(body),
        })
    }

    fn with_deps(name: &str, inject: &[&str], provide: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            inject: inject.iter().map(|s| s.to_string()).collect(),
            provide: provide.iter().map(|s| s.to_string()).collect(),
            body: Arc::new(|_| Ok(Contributions::new())),
        })
    }
}

impl Plugin for Named {
    fn name(&self) -> &str {
        &self.name
    }
    fn inject(&self) -> Vec<String> {
        self.inject.clone()
    }
    fn provide(&self) -> Vec<String> {
        self.provide.clone()
    }
    fn apply(&self, ctx: &PluginCtx) -> Result<Contributions> {
        (self.body)(ctx)
    }
}

fn empty_tool(name: &str) -> ToolDef {
    ToolDef::new(
        name,
        "d",
        serde_json::json!({ "type": "object" }),
        "ignored-by-kernel",
        |_| Box::pin(async { Ok(crate::tools::ToolOutput::default()) }) as crate::tools::ToolFuture,
    )
}

#[test]
fn isolated_realms_keep_same_named_services_apart() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("counter", Arc::new(1u32)))
    }));

    let realm_a = RealmMap::new().with_isolated("counter", kernel.new_realm());
    let realm_b = RealmMap::new().with_isolated("counter", kernel.new_realm());

    kernel
        .load("provider", realm_a.clone(), Value::Null)
        .unwrap();
    kernel
        .load("provider", realm_b.clone(), Value::Null)
        .unwrap();

    // Same service name, two realms, two independent instances.
    assert!(kernel.service_view(realm_a).get::<u32>("counter").is_some());
    assert!(kernel.service_view(realm_b).get::<u32>("counter").is_some());
    assert_eq!(kernel.service_list().len(), 2);

    // The root realm sees neither.
    assert!(!kernel.service_view(RealmMap::new()).has("counter"));
}

#[test]
fn duplicate_service_in_one_realm_fails_loud() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("first", |_| {
        Ok(Contributions::new().service("thing", Arc::new(1u8)))
    }));
    kernel.define(Named::new("second", |_| {
        Ok(Contributions::new().service("thing", Arc::new(2u8)))
    }));

    kernel.load("first", RealmMap::new(), Value::Null).unwrap();

    // The clash is caught at install time even though `second` never
    // declared the service it provides — and it is reported as a failed
    // load, not as a load that succeeded and then did nothing.
    let error = kernel
        .load("second", RealmMap::new(), Value::Null)
        .expect_err("a service clash fails the load");
    let text = format!("{error:#}");
    assert!(text.contains("already provided"), "{text}");
    assert!(
        text.contains("first"),
        "the existing owner is named: {text}"
    );

    // The reason stays on the fiber, so a panel can show why.
    let state = kernel
        .fibers()
        .into_iter()
        .find(|fiber| fiber.plugin == "second")
        .map(|fiber| fiber.state);
    assert_eq!(state, Some(FiberState::Failed));

    // The first provider is untouched.
    assert_eq!(
        *kernel
            .service_view(RealmMap::new())
            .get::<u8>("thing")
            .unwrap(),
        1
    );
}

#[test]
fn a_declared_provider_clash_is_rejected_before_apply() {
    let mut kernel = Kernel::new();
    kernel.define(Named::with_deps("declarer", &[], &["shared"]));
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("shared", Arc::new(1u8)))
    }));

    kernel
        .load("provider", RealmMap::new(), Value::Null)
        .unwrap();
    let error = kernel
        .load("declarer", RealmMap::new(), Value::Null)
        .expect_err("declared clash is refused up front");
    assert!(format!("{error:#}").contains("already provided"));
}

#[test]
fn a_plugin_waits_for_its_dependency_then_activates() {
    let mut kernel = Kernel::new();
    kernel.define(Named::with_deps("consumer", &["thing"], &[]));
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("thing", Arc::new(7u8)))
    }));

    let consumer = kernel
        .load("consumer", RealmMap::new(), Value::Null)
        .unwrap();
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Pending);

    kernel
        .load("provider", RealmMap::new(), Value::Null)
        .unwrap();
    let events = kernel.refresh().unwrap();

    assert!(events.iter().any(|e| e.action == "activated"), "{events:?}");
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);
}

#[test]
fn replacing_a_dependency_reloads_the_dependent_automatically() {
    let mut kernel = Kernel::new();
    kernel.define(Named::with_deps("consumer", &["thing"], &[]));
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("thing", Arc::new(1u8)))
    }));

    let consumer = kernel
        .load("consumer", RealmMap::new(), Value::Null)
        .unwrap();
    let provider = kernel
        .load("provider", RealmMap::new(), Value::Null)
        .unwrap();
    kernel.refresh().unwrap();
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);
    let epoch_before = kernel.fiber(consumer).unwrap().epoch_label();

    // Replace the provider: the consumer's epoch must change and the kernel
    // must reconcile it without being asked.
    kernel.unload(provider).unwrap();
    kernel
        .load("provider", RealmMap::new(), Value::Null)
        .unwrap();
    let events = kernel.refresh().unwrap();

    assert!(
        events.iter().any(|e| e.fiber == consumer),
        "the dependent was reconciled: {events:?}"
    );
    let epoch_after = kernel.fiber(consumer).unwrap().epoch_label();
    assert_ne!(
        epoch_before, epoch_after,
        "epoch tracks dependency identity"
    );
    assert_eq!(
        kernel.fiber(consumer).unwrap().state,
        FiberState::Active,
        "and it is active again"
    );
}

#[test]
fn a_fiber_survives_its_dependency_disappearing() {
    let mut kernel = Kernel::new();
    kernel.define(Named::with_deps("consumer", &["thing"], &[]));
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("thing", Arc::new(1u8)))
    }));

    let consumer = kernel
        .load("consumer", RealmMap::new(), Value::Null)
        .unwrap();
    let provider = kernel
        .load("provider", RealmMap::new(), Value::Null)
        .unwrap();
    kernel.refresh().unwrap();
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);

    // The dependency goes away: the consumer is deactivated, not destroyed.
    kernel.unload(provider).unwrap();
    let fiber = kernel.fiber(consumer).expect("fiber still exists");
    assert_eq!(fiber.state, FiberState::Pending);
    assert_eq!(fiber.epoch_label(), "inactive");

    // ...and it comes back on its own when the dependency returns.
    kernel
        .load("provider", RealmMap::new(), Value::Null)
        .unwrap();
    let events = kernel.refresh().unwrap();
    assert!(events
        .iter()
        .any(|e| e.fiber == consumer && e.action == "activated"));
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);
}

#[test]
fn unloading_runs_effects_in_reverse_order() {
    let mut kernel = Kernel::new();
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    let order_in = Arc::clone(&order);

    kernel.define(Named::new("resourceful", move |_| {
        let a = Arc::clone(&order_in);
        let b = Arc::clone(&order_in);
        let c = Arc::clone(&order_in);
        Ok(Contributions::new()
            .effect(move || a.lock().unwrap().push("first"))
            .effect(move || b.lock().unwrap().push("second"))
            .effect(move || c.lock().unwrap().push("third")))
    }));

    let fiber = kernel
        .load("resourceful", RealmMap::new(), Value::Null)
        .unwrap();
    kernel.unload(fiber).unwrap();

    assert_eq!(
        *order.lock().unwrap(),
        vec!["third", "second", "first"],
        "disposers run last-registered-first"
    );
}

#[test]
fn unloading_removes_services_and_tools() {
    let mut kernel = Kernel::new();
    kernel.install_bootstrap().unwrap();
    kernel.define(Named::new("contributor", |_| {
        Ok(Contributions::new()
            .service("svc", Arc::new(1u8))
            .tool(empty_tool("plugin_tool")))
    }));

    let fiber = kernel
        .load("contributor", RealmMap::new(), Value::Null)
        .unwrap();
    assert!(kernel.service_view(RealmMap::new()).has("svc"));
    assert!(kernel.tools().get("plugin_tool").is_some());
    assert_eq!(kernel.tools_of(fiber), vec!["plugin_tool".to_string()]);
    // The kernel stamps ownership itself.
    assert_eq!(kernel.tools().owner("plugin_tool"), Some("contributor"));

    kernel.unload(fiber).unwrap();
    assert!(!kernel.service_view(RealmMap::new()).has("svc"));
    assert!(kernel.tools().get("plugin_tool").is_none());
    // Base tools are untouched.
    assert!(kernel.tools().get("read").is_some());
}

#[test]
fn two_instances_of_one_plugin_do_not_remove_each_others_tools() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("multi", |ctx| {
        Ok(Contributions::new().tool(empty_tool(&format!("tool_of_{}", ctx.fiber))))
    }));

    let first = kernel.load("multi", RealmMap::new(), Value::Null).unwrap();
    let second = kernel.load("multi", RealmMap::new(), Value::Null).unwrap();
    let first_tool = format!("tool_of_{first}");
    let second_tool = format!("tool_of_{second}");
    assert!(kernel.tools().get(&first_tool).is_some());
    assert!(kernel.tools().get(&second_tool).is_some());

    // Unloading one instance must not touch the other's tool.
    kernel.unload(first).unwrap();
    assert!(kernel.tools().get(&first_tool).is_none());
    assert!(
        kernel.tools().get(&second_tool).is_some(),
        "the sibling instance keeps its tool"
    );
}

#[test]
fn a_failing_load_rolls_back_partial_installation() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("ok", |_| {
        Ok(Contributions::new().tool(empty_tool("shared")))
    }));
    kernel.define(Named::new("clash", |_| {
        Ok(Contributions::new()
            .service("side", Arc::new(9u8))
            .tool(empty_tool("shared")))
    }));

    let first = kernel.load("ok", RealmMap::new(), Value::Null).unwrap();
    assert!(kernel.tools().get("shared").is_some());

    kernel
        .load("clash", RealmMap::new(), Value::Null)
        .expect_err("the clashing load fails");
    // The failed plugin left nothing behind.
    assert!(!kernel.service_view(RealmMap::new()).has("side"));
    assert_eq!(kernel.fiber(first).unwrap().state, FiberState::Active);
    assert_eq!(kernel.tools().owner("shared"), Some("ok"));
}

#[test]
fn a_service_clash_rolls_back_the_whole_contribution_set() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("owner", |_| {
        Ok(Contributions::new().service("taken", Arc::new(1u8)))
    }));
    kernel.define(Named::new("latecomer", |_| {
        Ok(Contributions::new()
            .tool(empty_tool("latecomer_tool"))
            .service("taken", Arc::new(2u8)))
    }));

    kernel.load("owner", RealmMap::new(), Value::Null).unwrap();
    kernel
        .load("latecomer", RealmMap::new(), Value::Null)
        .expect_err("the service clash fails the load");

    // Validation happens before installation, so the tool never appeared.
    assert!(kernel.tools().get("latecomer_tool").is_none());
}

#[test]
fn realm_map_reports_isolated_names() {
    let realm = RealmId(42);
    let map = RealmMap::new().with_isolated("compaction", realm);
    assert_eq!(map.realm_of("compaction"), realm);
    assert_eq!(map.realm_of("other"), RealmId::root());
    assert_eq!(map.isolated_names(), vec!["compaction"]);
}

#[test]
fn version_advances_on_change() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("p", |_| {
        Ok(Contributions::new().service("s", Arc::new(1u8)))
    }));
    let before = kernel.version;
    let fiber = kernel.load("p", RealmMap::new(), Value::Null).unwrap();
    assert!(kernel.version > before, "loading bumps the version");
    let after_load = kernel.version;
    kernel.unload(fiber).unwrap();
    assert!(kernel.version > after_load, "unloading bumps the version");
}

#[test]
fn a_failing_plugin_records_its_error() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("broken", |_| Err(anyhow!("cannot start"))));

    // The load reports the failure — a plugin that never ran is not a
    // success — and the reason is readable both ways.
    let error = kernel
        .load("broken", RealmMap::new(), Value::Null)
        .expect_err("a plugin that cannot apply fails its load");
    assert!(format!("{error:#}").contains("cannot start"), "{error:#}");

    let fiber = kernel
        .fibers()
        .into_iter()
        .find(|fiber| fiber.plugin == "broken")
        .expect("the failed fiber stays for diagnostics");
    assert_eq!(fiber.state, FiberState::Failed);
    assert!(fiber.error.as_deref().unwrap().contains("cannot start"));
}

#[test]
fn a_later_plugins_label_wins_and_unloading_puts_the_earlier_back() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("first", |_| {
        Ok(Contributions::new().strings([("send".into(), "Send".into())]))
    }));
    kernel.define(Named::new("second", |_| {
        Ok(Contributions::new().strings([
            ("send".into(), "发送".into()),
            ("save".into(), "保存".into()),
        ]))
    }));
    kernel.load("first", RealmMap::new(), Value::Null).unwrap();
    let second = kernel.load("second", RealmMap::new(), Value::Null).unwrap();

    let strings = kernel.resolved_strings();
    assert_eq!(
        strings.get("send").map(String::as_str),
        Some("发送"),
        "the pack installed later decides"
    );
    assert_eq!(strings.get("save").map(String::as_str), Some("保存"));

    kernel.unload(second).unwrap();
    let strings = kernel.resolved_strings();
    assert_eq!(
        strings.get("send").map(String::as_str),
        Some("Send"),
        "unloading a pack must take its words with it"
    );
    assert!(!strings.contains_key("save"));
}

/// A contributed reader exists exactly while its fiber does: registration
/// rides the fiber's effects, so unload unregisters (the registry's
/// promise — 读不了就装包 means the pack's arrival *and* departure both
/// show in what `read` can open).
#[test]
fn a_plugin_reader_is_registered_on_load_and_removed_on_unload() {
    use crate::tools::readers::{ReadAnswer, ReaderSpec};
    use std::path::Path;

    let plugin = Named::new("format", |_| {
        Ok(Contributions::new().reader(ReaderSpec {
            id: "test-fmt".into(),
            claims: Arc::new(|bytes: &[u8], _path: &Path| bytes.starts_with(b"FMT")),
            convert: Arc::new(|bytes: &[u8], _path: &Path| {
                crate::tools::readers::ReadOutcome::Text(
                    String::from_utf8_lossy(&bytes[3..]).into_owned(),
                )
            }),
        }))
    });

    let mut kernel = Kernel::new();
    kernel.define(plugin);
    let fiber = kernel.load("format", RealmMap::new(), Value::Null).unwrap();

    let claimed = crate::tools::readers::convert(b"FMT readable", Path::new("x.bin"));
    assert!(
        matches!(claimed, ReadAnswer::Converted(ref text) if text == " readable"),
        "the pack taught read its format"
    );
    assert!(
        crate::tools::readers::installed()
            .iter()
            .any(|(id, owner)| id == "test-fmt" && owner == "format"),
        "with ownership stamped by the kernel, not the plugin"
    );

    kernel.unload(fiber).unwrap();
    assert!(
        matches!(
            crate::tools::readers::convert(b"FMT readable", Path::new("x.bin")),
            ReadAnswer::Unclaimed
        ),
        "unload took the reader: registrations are undone, not dropped"
    );
}

#[test]
fn an_instance_is_shared_by_its_sessions_and_dies_with_the_last_one() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("thing", Arc::new(1u8)))
    }));

    let realm = RealmMap::new().with_default(kernel.new_realm());
    let first = kernel
        .load_for_session("alice", "provider", realm.clone(), Value::Null)
        .unwrap();
    let second = kernel
        .load_for_session("bob", "provider", realm.clone(), Value::Null)
        .unwrap();
    assert_eq!(
        first, second,
        "one instance per (plugin, realm), not one per session"
    );
    assert_eq!(
        kernel.users_of(first),
        vec!["alice".to_string(), "bob".to_string()],
        "the instance knows which sessions hold it"
    );

    // The session that leaves first must not take the capability with it.
    assert_eq!(
        kernel.release("alice", first).unwrap(),
        ReleaseOutcome::StillUsed(1)
    );
    assert!(kernel.service_view(realm.clone()).has("thing"));
    assert_eq!(kernel.fiber(first).unwrap().state, FiberState::Active);

    // The last one does.
    assert_eq!(
        kernel.release("bob", first).unwrap(),
        ReleaseOutcome::Unloaded
    );
    assert!(!kernel.service_view(realm).has("thing"));
    assert!(kernel.fiber(first).is_none());
    assert_eq!(
        kernel.release("bob", first).unwrap(),
        ReleaseOutcome::Absent,
        "a second release is not an error"
    );
}

#[test]
fn switching_sessions_releases_only_the_other_sessions_claims() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("mine", |_| {
        Ok(Contributions::new().service("mine", Arc::new(1u8)))
    }));
    kernel.define(Named::new("theirs", |_| {
        Ok(Contributions::new().service("theirs", Arc::new(2u8)))
    }));

    let mine = kernel
        .load_for_session("alice", "mine", RealmMap::new(), Value::Null)
        .unwrap();
    kernel
        .load_for_session("bob", "theirs", RealmMap::new(), Value::Null)
        .unwrap();

    let dropped = kernel.unload_all_except("alice").unwrap();
    assert_eq!(
        dropped.len(),
        1,
        "only the other session's instance went: {dropped:?}"
    );
    assert!(kernel.service_view(RealmMap::new()).has("mine"));
    assert!(!kernel.service_view(RealmMap::new()).has("theirs"));
    assert_eq!(kernel.fiber(mine).unwrap().state, FiberState::Active);
}

#[test]
fn a_reload_that_cannot_apply_keeps_the_running_version() {
    let mut kernel = Kernel::new();
    let healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let flag = Arc::clone(&healthy);
    kernel.define(Named::new("flaky", move |_| {
        if flag.load(Ordering::SeqCst) {
            Ok(Contributions::new().tool(empty_tool("flaky_tool")))
        } else {
            Err(anyhow!("the replacement does not build"))
        }
    }));

    let fiber = kernel.load("flaky", RealmMap::new(), Value::Null).unwrap();
    assert!(kernel.tools().get("flaky_tool").is_some());

    healthy.store(false, Ordering::SeqCst);
    let error = kernel.reload(fiber).expect_err("the replacement fails");
    assert!(format!("{error:#}").contains("does not build"), "{error:#}");

    // Building first and swapping second is what keeps this true: the old
    // version is still installed, still owned, and still callable.
    assert!(kernel.tools().get("flaky_tool").is_some());
    assert_eq!(kernel.tools().owner("flaky_tool"), Some("flaky"));
    assert_eq!(kernel.tools_of(fiber), vec!["flaky_tool".to_string()]);
    assert_eq!(kernel.fiber(fiber).unwrap().state, FiberState::Active);
}

#[test]
fn a_failed_reload_in_a_batch_rolls_back_every_fiber() {
    let mut kernel = Kernel::new();
    let stubborn = kernel.new_realm();
    let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&armed);
    kernel.define(Named::new("plugin", move |ctx| {
        if flag.load(Ordering::SeqCst) && ctx.realm.default_realm() == stubborn {
            return Err(anyhow!("this realm refuses to re-apply"));
        }
        Ok(Contributions::new().tool(empty_tool(&format!("tool_{}", ctx.fiber))))
    }));

    let realm_a = RealmMap::new().with_default(kernel.new_realm());
    let realm_b = RealmMap::new().with_default(stubborn);
    let a = kernel.load("plugin", realm_a, Value::Null).unwrap();
    let b = kernel.load("plugin", realm_b, Value::Null).unwrap();

    armed.store(true, Ordering::SeqCst);
    let error = kernel
        .reload_plugin("plugin")
        .expect_err("the batch cannot complete");
    assert!(
        format!("{error:#}").contains("refuses to re-apply"),
        "{error:#}"
    );

    // Every target was staged before any was touched, so neither fiber lost
    // anything: no half-old, half-new table.
    assert!(kernel.tools().get(&format!("tool_{a}")).is_some());
    assert!(kernel.tools().get(&format!("tool_{b}")).is_some());
    assert_eq!(kernel.fiber(a).unwrap().state, FiberState::Active);
    assert_eq!(kernel.fiber(b).unwrap().state, FiberState::Active);
}

#[test]
fn a_panicking_disposer_does_not_stop_the_others() {
    let ran: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let tally = Arc::clone(&ran);

    let mut kernel = Kernel::new();
    kernel.define(Named::new("messy", move |_| {
        // Cloned per apply: a `Fn` plugin body may not move its own captures
        // into the disposers it hands back.
        let first = Arc::clone(&tally);
        let second = Arc::clone(&tally);
        Ok(Contributions::new()
            // Registered first, so it runs last.
            .effect(move || {
                first.fetch_add(1, Ordering::SeqCst);
            })
            .effect(move || {
                second.fetch_add(1, Ordering::SeqCst);
                panic!("this disposer is deliberately broken");
            }))
    }));

    let fiber = kernel.load("messy", RealmMap::new(), Value::Null).unwrap();
    kernel.unload(fiber).unwrap();

    assert_eq!(
        ran.load(Ordering::SeqCst),
        2,
        "both disposers ran despite one of them panicking"
    );
    assert!(
        kernel.fiber(fiber).is_none(),
        "teardown finished: the fiber is gone, not stranded half-torn-down"
    );
}

#[test]
fn a_service_view_goes_stale_when_the_table_changes() {
    let mut kernel = Kernel::new();
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("thing", Arc::new(7u8)))
    }));

    let fiber = kernel
        .load("provider", RealmMap::new(), Value::Null)
        .unwrap();
    let view = kernel.service_view(RealmMap::new());
    assert_eq!(*view.get::<u8>("thing").unwrap(), 7);
    assert!(view.is_current());

    // Unloading changes the table. The view taken before must not keep
    // serving a service that is no longer registered, and must say why.
    kernel.unload(fiber).unwrap();
    assert!(!view.is_current());
    match view.try_get::<u8>("thing") {
        Err(ServiceViewError::Stale { .. }) => {}
        other => panic!("expected a stale view, got {other:?}"),
    }
    assert!(view.get::<u8>("thing").is_none());
    assert!(!view.has("thing"));

    // A fresh view is current again, and reports the truth.
    let fresh = kernel.service_view(RealmMap::new());
    assert!(fresh.is_current());
    assert!(!fresh.has("thing"));
}

#[test]
fn a_dependency_reload_reapplies_its_dependent_with_the_same_configuration() {
    struct Recorder {
        seen: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Plugin for Recorder {
        fn name(&self) -> &str {
            "consumer"
        }
        fn inject(&self) -> Vec<String> {
            vec!["thing".to_string()]
        }
        fn apply(&self, ctx: &PluginCtx) -> Result<Contributions> {
            self.seen.lock().unwrap().push(ctx.config.to_string());
            Ok(Contributions::new())
        }
    }

    let seen: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));

    let mut kernel = Kernel::new();
    kernel.define(Arc::new(Recorder {
        seen: Arc::clone(&seen),
    }));
    kernel.define(Named::new("provider", |_| {
        Ok(Contributions::new().service("thing", Arc::new(1u8)))
    }));

    let realm = RealmMap::new();
    let consumer = kernel
        .load(
            "consumer",
            realm.clone(),
            serde_json::json!({ "answer": 42 }),
        )
        .unwrap();
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Pending);

    kernel.load("provider", realm.clone(), Value::Null).unwrap();
    kernel.refresh().unwrap();
    assert_eq!(kernel.fiber(consumer).unwrap().state, FiberState::Active);

    // Replace the dependency: the kernel rebuilds the dependent on its own,
    // and must rebuild it with the configuration the host gave it rather
    // than with `Null`.
    let provider = kernel.fiber_of("provider", &realm).unwrap();
    kernel.unload(provider).unwrap();
    kernel.load("provider", realm.clone(), Value::Null).unwrap();
    kernel.refresh().unwrap();

    let configs = seen.lock().unwrap().clone();
    assert!(
        configs.len() >= 2,
        "the consumer was re-applied: {configs:?}"
    );
    assert!(
        configs.iter().all(|config| config.contains("42")),
        "configuration survived the automatic reload: {configs:?}"
    );
}
