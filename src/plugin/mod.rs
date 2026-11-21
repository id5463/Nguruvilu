//! The plugin mechanism: isolation, dependency tracking, reversible effects.
//!
//! Design follows Cordis (`vendor/cordis/src/`, MIT) but is rewritten for Rust's
//! ownership model, which changes the shape of the mechanism:
//!
//! * `realm` — [`RealmMap`]: which realm each service name resolves in, so two
//!   sessions can each own an independent instance of the same service.
//! * `service` — [`ServiceTable`] and the [`ServiceView`] a plugin reads it
//!   through.
//! * `ctx` — [`PluginCtx`]: what a plugin sees and may change, and the
//!   [`Contributions`] it returns instead of mutating the kernel.
//! * `fiber` — [`Fiber`]: one instance, its [`Epoch`], its users, its effects.
//! * `kernel` — [`Kernel`] and the [`Plugin`] contract it runs.
//! * `seam` — [`Seam`]: a value a plugin may rewrite on its way through.
//!
//! A plugin never mutates the kernel directly: `apply` returns a
//! [`Contributions`] value that the kernel installs, which sidesteps the borrow
//! conflicts an in-place API would create and makes rollback possible.

mod ctx;
mod fiber;
mod kernel;
mod realm;
mod seam;
mod service;

pub use ctx::{Contributions, Disposer, PluginCtx, StateHandle};
pub use fiber::{Fiber, FiberState, ReleaseOutcome};
pub use kernel::{Kernel, Plugin, ReloadEvent};
pub use realm::{RealmId, RealmMap};
pub use seam::{Seam, ToolCallView, TurnContext};
pub use service::{ServiceView, ServiceViewError};
