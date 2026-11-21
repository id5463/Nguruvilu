//! The seam between the loop and whatever a pack wants done for a turn.
//!
//! [`PluginHooks`] is a trait object: the loop asks its questions and never
//! learns that a kernel exists behind them.

/// What the agent loop asks plugins, without knowing anything about them.
///
/// Three questions: what to add to the prompt, what to drop from the tool
/// table, and whether to refuse a call. The kernel answers them; the loop only
/// relays. This is the seam that means a new capability does not need a new
/// field on any struct — the plugin implements a method and the loop already
/// asks.
pub trait PluginHooks: Send + Sync {
    /// Extra prompt text for this turn, in load order.
    fn prompt_additions(&self, input: &str, model: &str) -> Vec<String>;
    /// Which of these tool names survive for this turn.
    ///
    /// Names rather than definitions: the registry owns the handlers, and
    /// handing them out to be rebuilt invites a plugin to drop one by accident.
    /// Returning `None` keeps the table whole.
    fn filter_tools(&self, input: &str, model: &str, names: Vec<String>) -> Option<Vec<String>>;
    /// The first refusal for one call, if any.
    fn refuse_call(&self, input: &str, model: &str, name: &str, arguments: &str) -> Option<String>;
    /// News that a call finished.
    fn note_call(&self, input: &str, model: &str, name: &str, ok: bool);
    /// Pass one piece of data through every plugin that wants it.
    fn intercept(&self, input: &str, model: &str, seam: crate::plugin::Seam)
        -> crate::plugin::Seam;
}
