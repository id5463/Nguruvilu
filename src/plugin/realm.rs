//! Isolation realms: which realm a service name resolves in.
//!
//! [`RealmId`] names one realm, and [`RealmMap`] says which realm each service
//! lives in. A child context clones the map and overrides one name, which is
//! what makes two sessions own independent instances of the same service.

use std::collections::BTreeMap;

/// Identifies one isolation realm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RealmId(pub(crate) u64);

impl std::fmt::Display for RealmId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "realm-{}", self.0)
    }
}

impl RealmId {
    /// The realm every context starts in.
    pub fn root() -> Self {
        RealmId(0)
    }
}

/// Service name → realm, plus the realm used for names not listed.
///
/// The default realm is what makes session isolation work without enumerating
/// every service a plugin will provide: a session-scoped context simply
/// defaults to its own realm, so everything that plugin contributes is
/// isolated without the loader having to guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealmMap {
    named: BTreeMap<String, RealmId>,
    default: RealmId,
}

impl Default for RealmMap {
    fn default() -> Self {
        Self {
            named: BTreeMap::new(),
            default: RealmId::root(),
        }
    }
}

impl RealmMap {
    /// An empty map: every name resolves in the root realm.
    pub fn new() -> Self {
        Self::default()
    }

    /// The realm a service name resolves to.
    pub fn realm_of(&self, name: &str) -> RealmId {
        self.named.get(name).copied().unwrap_or(self.default)
    }

    /// A copy with `name` moved into `realm`.
    pub fn with_isolated(&self, name: &str, realm: RealmId) -> Self {
        let mut next = self.clone();
        next.named.insert(name.to_string(), realm);
        next
    }

    /// A copy where every unlisted name resolves in `realm`.
    pub fn with_default(&self, realm: RealmId) -> Self {
        let mut next = self.clone();
        next.default = realm;
        next
    }

    /// The realm unlisted names resolve to.
    pub fn default_realm(&self) -> RealmId {
        self.default
    }

    /// Names explicitly placed outside the default realm.
    pub fn isolated_names(&self) -> Vec<&str> {
        self.named
            .iter()
            .filter(|(_, realm)| **realm != self.default)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}
