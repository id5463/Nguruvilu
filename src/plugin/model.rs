//! 内核握着的状态：一个发生了什么的记录，以及内核本身的结构。
//!
//! `Kernel` 的字段全部私有：外面的代码通过方法读它，因为每个字段旁边都有一条
//! 不变量（哪个 fiber 贡献了哪些工具、版本号什么时候动）。

use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::collections::BTreeMap;

use crate::plugin::fiber::Fiber;
use crate::plugin::contract::Plugin;
use crate::plugin::realm::RealmId;
use crate::plugin::service::ServiceTable;
use crate::tools::ToolRegistry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReloadEvent {
    /// Fiber affected.
    pub fiber: u64,
    /// Plugin name.
    pub plugin: String,
    /// `activated`, `reloaded`, `deactivated`, `failed`, `rebuild-failed`, or
    /// `truncated`.
    pub action: String,
    /// Why it happened.
    pub reason: String,
}

/// The plugin kernel: fibers, services, permissions.
pub struct Kernel {
    plugins: BTreeMap<String, Arc<dyn Plugin>>,
    fibers: BTreeMap<u64, Fiber>,
    /// Shared with every live [`ServiceView`]; see [`ServiceTable`].
    services: Arc<ServiceTable>,
    /// Tools contributed by each fiber, so unload removes exactly its own.
    fiber_tools: BTreeMap<u64, Vec<String>>,
    tools: ToolRegistry,
    counter: Arc<AtomicU64>,
    realm_counter: Arc<AtomicU64>,
    /// Serial for service instances, so an epoch changes when a provider is
    /// re-applied even though its fiber id stays the same.
    instance_counter: u64,
    /// Bumped on every change, so callers can tell snapshots apart.
    pub version: u64,
}
