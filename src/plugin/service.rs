//! The service table and the read-only view a plugin sees it through.
//!
//! [`ServiceView`] is a view, not a copy: it holds the kernel's live table by
//! `Arc` plus the generation it was taken at, so a view taken before a change
//! reports [`ServiceViewError::Stale`] rather than serving a value that is no
//! longer registered.

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::realm::{RealmId, RealmMap};

/// Why a service lookup through a [`ServiceView`] did not produce a service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceViewError {
    /// The kernel's service table moved on after this view was taken.
    ///
    /// The view is a reference into a table that is still live, so its numbers
    /// are honest: what it can no longer promise is that the value it would
    /// return is still registered. Answering anyway would be the bug — a plugin
    /// whose service was unregistered must not keep serving from a snapshot.
    Stale {
        /// Generation the view was taken at.
        view: u64,
        /// Generation the kernel is at now.
        current: u64,
    },
    /// Nothing of that name is registered in this view's realm.
    Missing {
        /// Service name asked for.
        name: String,
        /// Realm the view resolves it in.
        realm: RealmId,
    },
    /// The name is registered, but holds something other than the requested
    /// type.
    WrongType {
        /// Service name asked for.
        name: String,
        /// Rust type the caller asked for.
        expected: &'static str,
    },
}

impl std::fmt::Display for ServiceViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stale { view, current } => write!(
                f,
                "this service view was taken at table generation {view} and the kernel is now at \
                 {current}; the service it names may have been unregistered since"
            ),
            Self::Missing { name, realm } => {
                write!(f, "no service named '{name}' is registered in {realm}")
            }
            Self::WrongType { name, expected } => {
                write!(f, "service '{name}' is not a {expected}")
            }
        }
    }
}

impl std::error::Error for ServiceViewError {}

/// A read-only, realm-scoped view of the service table.
///
/// The view holds the kernel's live table behind an `Arc` plus the generation it
/// was taken at — it is not a copy. Two consequences, both of them the point:
///
/// * Handing a plugin a view is O(1). The previous implementation cloned the
///   whole table on every activation *and* on every lookup of a view.
/// * A view never keeps an unregistered service alive. When the kernel changes
///   its table the generation moves, and [`ServiceView::try_get`] reports
///   [`ServiceViewError::Stale`] instead of serving a value that is no longer
///   registered.
#[derive(Clone)]
pub struct ServiceView {
    table: Arc<ServiceTable>,
    /// Table generation this view was taken at.
    generation: u64,
    realm: RealmMap,
}

impl Default for ServiceView {
    fn default() -> Self {
        let table = Arc::new(ServiceTable::default());
        let generation = table.generation.load(Ordering::Relaxed);
        Self {
            table,
            generation,
            realm: RealmMap::new(),
        }
    }
}

impl ServiceView {
    pub(crate) fn of(table: Arc<ServiceTable>, realm: RealmMap) -> Self {
        let generation = table.generation.load(Ordering::Relaxed);
        Self {
            table,
            generation,
            realm,
        }
    }

    /// Look up a service in this view's realm, or fail with a reason.
    pub fn try_get<T: Any + Send + Sync>(
        &self,
        name: &str,
    ) -> std::result::Result<Arc<T>, ServiceViewError> {
        let current = self.table.generation.load(Ordering::Relaxed);
        if current != self.generation {
            return Err(ServiceViewError::Stale {
                view: self.generation,
                current,
            });
        }
        let realm = self.realm.realm_of(name);
        let key = (name.to_string(), realm);
        let found = self
            .table
            .read()
            .get(&key)
            .map(|entry| Arc::clone(&entry.value));
        match found {
            Some(value) => value
                .downcast::<T>()
                .map_err(|_| ServiceViewError::WrongType {
                    name: name.to_string(),
                    expected: std::any::type_name::<T>(),
                }),
            None => Err(ServiceViewError::Missing {
                name: name.to_string(),
                realm,
            }),
        }
    }

    /// Look up a service in this view's realm.
    ///
    /// `None` covers all three ways that can fail — stale view, unregistered
    /// name, wrong type — because the callers that want a value only care that
    /// they did not get one. Callers that need to tell the cases apart use
    /// [`ServiceView::try_get`].
    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        self.try_get::<T>(name).ok()
    }

    /// Whether a service exists in this view's realm.
    ///
    /// A stale view reports nothing: it cannot say what is there any more.
    pub fn has(&self, name: &str) -> bool {
        if !self.is_current() {
            return false;
        }
        let key = (name.to_string(), self.realm.realm_of(name));
        self.table.read().contains_key(&key)
    }

    /// Names visible in this view's realm.
    pub fn names(&self) -> Vec<String> {
        if !self.is_current() {
            return Vec::new();
        }
        self.table
            .read()
            .keys()
            .filter(|(name, realm)| self.realm.realm_of(name) == *realm)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// The generation this view was taken at.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether the kernel's table has changed since this view was taken.
    pub fn is_current(&self) -> bool {
        self.table.generation.load(Ordering::Relaxed) == self.generation
    }
}

/// One registered service instance.
#[derive(Clone)]
pub(crate) struct ServiceEntry {
    pub(crate) value: Arc<dyn Any + Send + Sync>,
    /// Fiber that provided it; the epoch input for dependents.
    pub(crate) fiber: u64,
    /// Plugin that provided it.
    pub(crate) owner: String,
    /// Serial of this exact instance.
    ///
    /// Part of the epoch, so re-applying a provider — same fiber, new value —
    /// still invalidates its dependents. Without it a reloaded provider would
    /// keep its fiber id and its dependents would never be rebuilt against the
    /// new instance.
    pub(crate) serial: u64,
}

/// The kernel's live service table, shared with every [`ServiceView`] by `Arc`.
#[derive(Default)]
pub(crate) struct ServiceTable {
    pub(crate) entries: RwLock<BTreeMap<(String, RealmId), ServiceEntry>>,
    /// Bumped on every change; a view taken at an older value is stale.
    pub(crate) generation: AtomicU64,
}

impl ServiceTable {
    /// The table for reading.
    ///
    /// A poisoned lock is treated as usable: the map holds only whole entries
    /// and has no invariant a half-finished write could break, so refusing to
    /// read it would turn one plugin's panic into a kernel-wide outage. The
    /// panic itself is still reported where it happened.
    pub(crate) fn read(&self) -> RwLockReadGuard<'_, BTreeMap<(String, RealmId), ServiceEntry>> {
        self.entries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The table for writing.
    ///
    /// Poison policy is [`ServiceTable::read`]'s.
    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, BTreeMap<(String, RealmId), ServiceEntry>> {
        self.entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
