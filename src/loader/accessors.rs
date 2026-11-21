//! 造一个加载器，以及把它握着的东西露出来。
//!
//! 这一组不加载任何东西：它们只是构造与读取。分开是因为"换一个加载器怎么
//! 建起来"和"三类条目怎么装进去"是两个不同的替换点。

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::ledger::Ledger;
use crate::mcp::McpClient;
use crate::plugin::Kernel;
use crate::skills::SkillRegistry;

use super::model::{LazyMcp, Loader};

impl Loader {
    pub fn new(kernel: Kernel, ledger_path: PathBuf, pack: impl Into<String>) -> Self {
        let ledger = Ledger::load(&ledger_path).unwrap_or_default();
        Self {
            kernel,
            skills: SkillRegistry::new(),
            mcp_clients: Vec::new(),
            ledger,
            ledger_path,
            pack: pack.into(),
            processes: Vec::new(),
            lazy_mcp: Vec::new(),
            pack_dir: None,
        }
    }

    /// Build a loader with the default skill roots already configured.
    pub fn with_default_skills(mut self) -> Self {
        for root in SkillRegistry::default_roots() {
            self.skills.add_root(root);
        }
        self
    }

    /// Treat `dir` as the pack's own directory, so relative sources resolve
    /// against it.
    ///
    /// Always stored absolute. `ngu apply assembly.yaml` passes `file.parent()`,
    /// which is `""` for a bare file name — and `absolute("")` is this process's
    /// cwd, so a plugin's binary became `<cwd>/bin/x` while `Spec::cwd` was set
    /// to the same directory. The child then resolved `<cwd>/bin/x` a second
    /// time *after* the `chdir`, and `bin/x` inside `bin/` does not exist:
    /// ENOENT, from a file that is plainly there.
    ///
    /// Measured: `cd packs/delegate && ngu apply assembly.yaml` failed while
    /// `ngu apply packs/delegate/assembly.yaml` from the parent succeeded.
    pub fn with_pack_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.pack_dir = Some(absolute(&dir.into()));
        self
    }

    /// The pack these steps come from, by manifest name.
    ///
    /// `None` when the directory has no manifest (a bare assembly): nothing
    /// points at a pack because there is none.
    fn origin(&self) -> Option<String> {
        self.pack_dir
            .as_ref()
            .and_then(|dir| crate::pack::read_manifest(dir).ok())
            .map(|manifest| manifest.name)
    }

    /// The kernel.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// The kernel, mutably.
    pub fn kernel_mut(&mut self) -> &mut Kernel {
        &mut self.kernel
    }

    /// The skill registry.
    pub fn skills(&self) -> &SkillRegistry {
        &self.skills
    }

    /// The ledger.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Connected MCP servers.
    pub fn mcp_servers(&self) -> Vec<&str> {
        self.mcp_clients.iter().map(|c| c.server()).collect()
    }

    /// Rescan the skill roots.
    pub fn rescan_skills(&mut self) -> Result<usize> {
        let report = self.skills.scan()?;
        // A rescan is usually triggered because a skill appeared or vanished,
        // so the paths it could not use are exactly what the user needs to see;
        // the return value alone would report "0 loaded" with no reason.
        if !report.skipped.is_empty() || !report.collisions.is_empty() {
            eprintln!("[skills] {}", report.summary());
        }
        Ok(report.loaded)
    }

}
