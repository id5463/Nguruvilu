//! 内核的全部可读状态：一棵 JSON 树 + 一堆活对象。
//!
//! # 为什么分成 `tree` 和 `live` 两半
//!
//! 状态里有两类东西，形状完全不同：
//!
//! - **数据**：`settings.model`、`messages`、`turn`、`usage`……它们的共同点是
//!   "能变成 JSON"。这些东西适合放在一棵树里，因为树可以被 `snapshot()`
//!   整个拿出来、可以按路径寻址、可以原样塞进协议。
//! - **活对象**：`Arc<ToolRegistry>`、`Arc<dyn ContextPolicy>`、注入源……
//!   它们的价值恰恰在于 **身份**：多个调用方共享同一份、内部带锁、带
//!   后台任务、带不是 `serde` 能表达的内部状态。
//!
//! # 为什么不给活对象做序列化
//!
//! 把 `Arc<ToolRegistry>` 压成 JSON 再读出来，得到的是一个**副本**，
//! 不是原来那个对象：之后对副本的注册，原持有者看不见；里面的锁、句柄、
//! 后台任务全都丢了。这就叫"丢共享语义"——看起来存取成功，实际语义已经
//! 不是同一份东西了。
//!
//! 所以活对象只存 `Arc`（克隆一次就是加一次引用计数，仍是同一份），
//! 取出时按类型 `downcast` 回原来的 `T`。**访问入口是路由，不等于要把对象
//! 压成 JSON**：路由内部拿 `Arc` 干活，序列化的只有参数和返回值。
//!
//! # 版本号
//!
//! 每次 [`Catalog::write`] 都 `fetch_add(1)` 并返回新值。这唯一的用途是
//! 给乐观并发用：调用方记下版本号，回头写的时候比对，被改过就重来。
//! 它是**单调递增的整数，不是时间戳**，不要在别的地方赋予它含义。

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use serde_json::{Map, Value};

/// 内核的可读状态容器。
///
/// `tree` 和 `live` 各有一把独立的读写锁；两把锁从不嵌套获取，所以不会死锁。
pub struct Catalog {
    /// 能 JSON 化的一切。默认是一棵空的 `Value::Object`。
    tree: RwLock<Value>,
    /// 活对象：按名字存 `Arc`，取出时 downcast 回原类型。
    live: RwLock<HashMap<String, Arc<dyn Any + Send + Sync>>>,
    /// 每次写 +1。给乐观并发用。
    version: AtomicU64,
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

impl Catalog {
    /// 建一个空目录：树是空对象，没有活对象，版本号 0。
    pub fn new() -> Self {
        Self {
            tree: RwLock::new(Value::Object(Map::new())),
            live: RwLock::new(HashMap::new()),
            version: AtomicU64::new(0),
        }
    }

    /// 按路径读，路径用点分：`read("settings.model")`。
    ///
    /// 路径不存在返回 `Value::Null`，**不是错误**——调用方不需要区分
    /// "没设过"和"设成了 null"，两种情况默认值都该生效。
    /// 路径上有一层不是对象（比如 `messages` 是数组），也返回 `Null`。
    /// 空路径返回整棵树。
    pub fn read(&self, path: &str) -> Value {
        let segs = split_path(path);
        let guard = self.tree.read().unwrap_or_else(|e| e.into_inner());
        read_at(&guard, &segs)
    }

    /// 按路径写，自动创建中间对象。返回写入后的版本号。
    ///
    /// 写 `settings.model = "x"` 时 `settings` 不存在就建一个空对象；
    /// 中间那层存在但不是对象（比如是个字符串），**直接覆盖成对象**——
    /// 路径是结构声明，写下去就该是这个形状。空路径替换整棵树。
    ///
    /// 写入后树立刻对读者可见：读锁和写锁保证不会读到"写了一半"的树。
    pub fn write(&self, path: &str, v: Value) -> u64 {
        let segs = split_path(path);
        {
            let mut guard = self.tree.write().unwrap_or_else(|e| e.into_inner());
            write_at(&mut guard, &segs, v);
        }
        // 先放写锁再抬版本号：拿版本号的人不应该看到"版本已涨、数据还没好"。
        self.version.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// 整棵树的快照。
    ///
    /// 深拷贝，拿出来的东西之后不会被任何写影响；也正因为是拷贝，
    /// 大树上要慎用（`messages` 很长时一次快照就是一次全量克隆）。
    pub fn snapshot(&self) -> Value {
        let guard = self.tree.read().unwrap_or_else(|e| e.into_inner());
        guard.clone()
    }

    /// 当前的版本号。只增不减，初值 0。
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    /// 放一个活对象。
    ///
    /// 存的是 `Arc<T>`，所以放进去之后**外面那份和里面那份是同一个 T**：
    /// 对外面那份的修改，从这里取出来的调用方都看得见。同名覆盖，旧的
    /// 若没有别的引用就会被释放。
    ///
    /// 要求 `Send + Sync + 'static` 是因为这个容器要被多个任务共享。
    pub fn put_live<T: Any + Send + Sync + 'static>(&self, name: &str, obj: T) {
        let arc: Arc<dyn Any + Send + Sync> = Arc::new(obj);
        let mut guard = self.live.write().unwrap_or_else(|e| e.into_inner());
        guard.insert(name.to_string(), arc);
    }

    /// 取活对象。名字不存在、或存着的类型不是 `T`，都返回 `None`。
    ///
    /// 返回的是 `Arc<T>`：克隆它只是加引用计数，仍是同一份对象。
    pub fn get_live<T: Any + Send + Sync + 'static>(&self, name: &str) -> Option<Arc<T>> {
        let arc = {
            let guard = self.live.read().unwrap_or_else(|e| e.into_inner());
            guard.get(name)?.clone()
        };
        // downcast 失败时把 Arc 还回来，这里不需要，丢掉即可。
        Arc::downcast::<T>(arc).ok()
    }

    /// 这个名字下有没有活对象（不看类型）。
    pub fn has_live(&self, name: &str) -> bool {
        let guard = self.live.read().unwrap_or_else(|e| e.into_inner());
        guard.contains_key(name)
    }

    /// 所有活对象的名字，排好序返回（HashMap 顺序不定，排序才有可重复的观测）。
    pub fn live_names(&self) -> Vec<String> {
        let mut names: Vec<String> = {
            let guard = self.live.read().unwrap_or_else(|e| e.into_inner());
            guard.keys().cloned().collect()
        };
        names.sort();
        names
    }

    /// 列出一层的子键，路径为空时列根。不是对象（或不存在）返回空表。
    pub fn keys(&self, path: &str) -> Vec<String> {
        let segs = split_path(path);
        let guard = self.tree.read().unwrap_or_else(|e| e.into_inner());
        match read_ref(&guard, &segs) {
            Some(Value::Object(m)) => {
                let mut ks: Vec<String> = m.keys().cloned().collect();
                ks.sort();
                ks
            }
            _ => Vec::new(),
        }
    }
}

/// `a.b.c` → `["a", "b", "c"]`。空段（`..`、首尾点）丢掉，空路径得空表。
fn split_path(path: &str) -> Vec<String> {
    path.split('.')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// 只读地顺着路径走一层。走不通就 `None`（和 `Null` 区分开，`keys` 要用）。
fn read_ref<'a>(cur: &'a Value, segs: &[String]) -> Option<&'a Value> {
    match segs.split_first() {
        None => Some(cur),
        Some((head, rest)) => match cur.as_object().and_then(|m| m.get(head)) {
            Some(child) => read_ref(child, rest),
            None => None,
        },
    }
}

/// 顺着路径读出值（克隆一份）。中间缺一层就是 `Null`。
fn read_at(cur: &Value, segs: &[String]) -> Value {
    match segs.split_first() {
        None => cur.clone(),
        Some((head, rest)) => match cur.as_object().and_then(|m| m.get(head)) {
            Some(child) => read_at(child, rest),
            None => Value::Null,
        },
    }
}

/// 顺着路径写下去，缺的中间层补成空对象，非对象的中间层覆盖成空对象。
fn write_at(cur: &mut Value, segs: &[String], v: Value) {
    let (head, rest) = match segs.split_first() {
        Some((h, r)) => (h, r),
        None => {
            *cur = v;
            return;
        }
    };
    if !cur.is_object() {
        *cur = Value::Object(Map::new());
    }
    let map = match cur.as_object_mut() {
        Some(m) => m,
        None => return, // 上面刚保证过是对象，走不到这里
    };
    let child = map.entry(head.clone()).or_insert(Value::Null);
    write_at(child, rest, v);
}

