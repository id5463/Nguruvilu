//! 名字 → 处理链：把"内核里写死的一次调用"变成"表里查得到的一串处理函数"。
//!
//! # 为什么是"链"而不是"一个函数"
//!
//! 如果表里一个名字只对应一个函数，那么"换掉默认行为"就等于重写它：
//! 想在一个已有实现前面加一句判断，也得把原实现整段抄过来。这做不到
//! "可替换"，只做到"可重写"。
//!
//! 所以名字对应的是 **一条链**（`Vec<Handler>`）。链尾是内核的默认实现，
//! 后注册的插到链头。一个处理函数拿到参数之后有两种选择：
//!
//! - 自己处理完直接返回 —— 这就是覆盖；
//! - 判断一下，再把控制权交回去 —— 这就是"只加一句，其余照旧"。
//!
//! 交回去的动作就是 [`Chain::next`]：给定当前是第几层，调第 i 层及之后。
//! 这样"拦一下"的代码只需要 5 行，不需要知道下面那一层长什么样。
//!
//! # 数据形状只有一个
//!
//! 参数和返回值都是 `serde_json::Value`。链上的每一层都只认这一个形状，
//! 于是任意两层之间可以互换、可以跨进程、可以序列化进协议。代价是类型
//! 检查从编译期挪到了运行期的取值处，换来的是"注册"这件事在运行期可做。
//!
//! # 锁不会跨 await
//!
//! `call` 在读锁里只做一件事：把链**克隆**出来（`Arc` 克隆，很便宜），
//! 然后立刻放锁，再 await。处理函数执行期间表是可以被改的，这既避免了
//! 死锁，也避免了持锁跨 await 的 `Send` 问题。

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

/// 链上流动的唯一数据形状。
pub type Args = serde_json::Value;

/// 处理函数的返回：已装箱的 future。
///
/// 装 `Send` 是因为它要能在 tokio 的任务里被 poll；装 `Box` 是因为
/// trait object 必须有确定的大小。
pub type Fut = Pin<Box<dyn Future<Output = Result<Args, String>> + Send>>;

/// 处理函数：吃一份参数和整条链，吐出 future。
///
/// 第二个参数（[`Chain`]）是回退用的把手 —— 处理函数想"让下面那层继续干活"
/// 就调 `chain.next(i + 1, args)`。注意传进来的 `Chain` 永远是整条链，
/// 不是剩下的那段；"我现在在第几层"由调用方自己带（见 [`Chain::next`]）。
pub type Handler = Arc<dyn Fn(Args, Chain) -> Fut + Send + Sync>;

/// 一条链的只读视图，交给处理函数用来回退。
///
/// 内部是 `Arc<Vec<Handler>>`：克隆一个 `Chain` 只是加一次引用计数，
/// 所以每层都能随便拿走一份。
#[derive(Clone, Default)]
pub struct Chain(Arc<Vec<Handler>>);

impl Chain {
    /// 从一串处理函数建一条链。第 0 个是链头。
    pub fn new(handlers: Vec<Handler>) -> Self {
        Chain(Arc::new(handlers))
    }

    /// 链的总层数。
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// 空链：这个名字下面没有任何实现。
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// 取第 `i` 层上的处理函数（不执行）。
    pub fn get(&self, i: usize) -> Option<Handler> {
        self.0.get(i).cloned()
    }

    /// 调链里第 `i` 条及之后。
    ///
    /// 是不是"及之后"由被调的那一层自己决定：它要么算完返回（覆盖），
    /// 要么在内部再调 `next(i + 1, args)`（回退）。
    ///
    /// 越界就是"没人实现"—— 返回 `Err`，而不是返回一个空对象。返回空对象
    /// 会让"我这一层处理完了"和"根本没有下一层"看起来一样，上层分不出来。
    pub async fn next(&self, i: usize, a: Args) -> Result<Args, String> {
        match self.0.get(i) {
            Some(h) => h(a, self.clone()).await,
            None => Err(format!("no route at layer {i}")),
        }
    }

    /// 同 [`Chain::next`]，但不立刻执行，而是把 future 交出去。
    ///
    /// 有些处理函数想让回退成为**最后一句**，用 `chain.skip(i + 1, a).await`
    /// 比写 `Box::pin(async move { ... })` 干净。这里直接返回已经装好箱的
    /// `Fut`，可以原样吐给上层。
    pub fn skip(&self, i: usize, a: Args) -> Fut {
        let chain = self.clone();
        Box::pin(async move { chain.next(i, a).await })
    }
}

/// 名字 → 处理链的表。
///
/// 表本身很薄：注册是往链头插，调用是从链头走。所有"聪明"的地方都在
/// [`Chain::next`] 的交还语义上。
pub struct Dispatch {
    table: RwLock<HashMap<String, Vec<Handler>>>,
}

impl Default for Dispatch {
    fn default() -> Self {
        Self::new()
    }
}

impl Dispatch {
    /// 空表。
    pub fn new() -> Self {
        Dispatch {
            table: RwLock::new(HashMap::new()),
        }
    }

    /// 插到链头：覆盖默认，但仍能 `next(1, args)` 回退。
    ///
    /// # 为什么是插链头，不是覆盖
    ///
    /// 直接覆盖（`map.insert(name, vec![h])`）会把原来那层**扔掉**。那样
    /// 一来，"我只想在默认实现前面加一句判断"就必须把默认实现整个复制
    /// 一份 —— 复制出来的那份不会跟着内核升级，等于分叉。
    ///
    /// 插链头保留了下面那层：新来的处理函数可以自己处理完（效果上就是
    /// 覆盖），也可以调 `next(1, args)` 把活交回去（效果上就是环绕）。
    /// 覆盖和环绕用同一个动作表达，注册方不需要先声明自己是哪一种。
    ///
    /// 同名的多次注册会一层层往上叠，**后注册的在更前面**，也就是通常
    /// 希望的那样：越晚插进来的越先看到调用。
    pub fn register(&self, name: &str, h: Handler) {
        let mut table = match self.table.write() {
            Ok(t) => t,
            // 锁中毒：说明之前有人在持锁时 panic 了。表只是 `Vec<Handler>`，
            // 不参与任何不变量，继续用是安全的（后面每层都会拿到自己的参数）。
            Err(p) => p.into_inner(),
        };
        table.entry(name.to_string()).or_default().insert(0, h);
    }

    /// 从链头执行。
    ///
    /// 名字不存在时返回 `Err(format!("no route {name}"))` —— 调用方据此
    /// 判断"这件事内核不认"，而不是拿到一个伪装成成功的空值。
    pub async fn call(&self, name: &str, a: Args) -> Result<Args, String> {
        // 读锁只在克隆链的时候持有；链是 `Arc`，克隆很便宜。
        let chain = {
            let table = match self.table.read() {
                Ok(t) => t,
                Err(p) => p.into_inner(),
            };
            match table.get(name) {
                Some(v) => Chain::new(v.clone()),
                None => return Err(format!("no route {name}")),
            }
        };
        // 锁已经放掉，这里再 await，处理函数里可以放心再注册新名字。
        chain.next(0, a).await
    }

    /// 不执行，只把第 0 层取出来（要给别的执行器跑时用）。
    pub fn head(&self, name: &str) -> Option<Handler> {
        let table = match self.table.read() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        table.get(name).and_then(|v| v.first().cloned())
    }

    /// 整条链（克隆出来，之后随便改表也不影响它）。
    pub fn chain(&self, name: &str) -> Option<Chain> {
        let table = match self.table.read() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        table.get(name).map(|v| Chain::new(v.clone()))
    }

    /// 全部名字，带层数，形如 `"a.b (2 层)"`。
    ///
    /// 排序过，所以两次调用的输出可以直接 diff —— 排查"某个名字是从哪一次
    /// 注册开始多出来的"时有用。
    pub fn names(&self) -> Vec<String> {
        let table = match self.table.read() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        let mut out: Vec<String> = table
            .iter()
            .map(|(k, v)| format!("{k} ({} 层)", v.len()))
            .collect();
        out.sort();
        out
    }

    /// 这个名字有没有人注册过。
    pub fn has(&self, name: &str) -> bool {
        let table = match self.table.read() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        table.get(name).map(|v| !v.is_empty()).unwrap_or(false)
    }

    /// 这个名字下面叠了几层。没有就是 0。
    pub fn layers(&self, name: &str) -> usize {
        let table = match self.table.read() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        table.get(name).map(|v| v.len()).unwrap_or(0)
    }

    /// 拆掉某一层（按层级位置，不是按内容）。返回是否真的拆掉了。
    ///
    /// 给"包被卸载"这类场景用：注册上去的东西总得有办法拿下来，否则
    /// 一次热更新之后链会越叠越长、旧代码一直被持有。拆完一层之后，
    /// 上面各层的下标都会往前挪一格 —— 这跟热更新期间正在执行的那次调用
    /// 冲突，所以只在没有调用在飞的时候用。
    pub fn remove_layer(&self, name: &str, i: usize) -> bool {
        let mut table = match self.table.write() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        match table.get_mut(name) {
            Some(v) if i < v.len() => {
                v.remove(i);
                if v.is_empty() {
                    table.remove(name);
                }
                true
            }
            _ => false,
        }
    }

    /// 整条删掉一个名字。
    pub fn forget(&self, name: &str) -> bool {
        let mut table = match self.table.write() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        table.remove(name).is_some()
    }

    /// 名字总个数。
    pub fn len(&self) -> usize {
        let table = match self.table.read() {
            Ok(t) => t,
            Err(p) => p.into_inner(),
        };
        table.len()
    }

    /// 表是空的。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

