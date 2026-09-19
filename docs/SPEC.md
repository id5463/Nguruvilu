# Nguruvilu 规格

本文档是 Nguruvilu 的**唯一权威规格**。所有实现必须服从本文档;与本文档冲突的代码是错的。

**Nguruvilu** 是独立 agent 内核,CLI 名 `ngu`,语言 **Rust**。它与 dshd 家族是内核与发行版的关系:内核独立演进,dshd 是使用方之一。

## 架构层次

```
┌─ 内核(Rust)─────────────────────────────────┐
│  会话 / 循环 / 模型(仅 OpenAI)/ 工具表        │
│  四个基础工具:read / write / edit / bash       │
│  插件机制:隔离域 / 依赖 / 热重载 / 撤销        │
└───────────────────────────────────────────────┘
                      ↕ 扩展点(见 1.4)
┌─ 动态加载层(Yellow)─────────────────────────┐
│  职责:动态加载、卸载、热替换                  │
│  分发格式:整合包(.dshpack)                   │
└───────────────────────────────────────────────┘
                      ↕ 加载
┌─ 可加载对象(三者平级)────────────────────────┐
│     插件        │       MCP       │    技能    │
└───────────────────────────────────────────────┘
```

**三个要点:**

1. **Yellow 是动态加载层**——架构中的一层,负责"把东西动态加载进来",本身不是插件。
2. **插件、MCP、技能三者平级**——都是可被 Yellow 加载的对象,彼此没有从属关系。
3. **内核不含上述任何对象**——内核只提供加载所需的扩展点。

---

## 1. 内核边界

### 1.1 内核包含什么

内核是**不可再分割的最小可运行 agent**:

| 组成 | 职责 |
|---|---|
| **会话** | 消息历史、持久化、恢复、多会话管理 |
| **循环** | agent loop:模型 → 工具 → 结果 → 模型,直到结束 |
| **模型调用** | 只做 OpenAI 兼容格式;流式、工具调用、用量统计 |
| **工具表** | 可变注册表:注册、注销、查找、schema 导出 |
| **四个基础工具** | `read` / `write` / `edit` / `bash` |
| **扩展点** | 插件注册工具、提供服务、事件钩子、会话操作、创建 agent、模型路由 |
| **插件机制** | 加载、隔离、依赖、热重载、卸载(见第 2 节) |

### 1.2 内核不包含什么

**以下全部由动态加载层加载,永不进内核。** 可加载对象分三类,三者**平级**:

**插件**(工具型 / 服务型):
- 搜索(web search / fetch)
- computer use(截图、鼠标键盘)
- 子代理(subagent / fork)
- 手机远程控制
- MCP 客户端(实现加载 MCP 的插件)
- 压缩(compaction)
- 审批与权限
- 沙箱
- TUI / Web UI

**MCP**:MCP 服务器(stdio / streamable-http)

**技能**:skills

**动态加载层本身(Yellow)也不在内核里**——内核只提供它所需的扩展点(见 1.4)。

### 1.3 四个基础工具

保留四个而非只留 `bash`,理由:**专用工具省 token 且更安全**。

| 工具 | 参数 | 说明 |
|---|---|---|
| `read` | `path`, `start_line?`, `end_line?` | 按行范围读取,返回带行号内容。比 `cat` 省大量 token |
| `write` | `path`, `content` | 覆盖写入 |
| `edit` | `path`, `old_str`, `new_str` | 字符串替换。比 `sed` 可靠(不依赖行号) |
| `bash` | `command`, `timeout_ms?` | 执行任意 shell 命令,兜底一切 |

设计参考:Pi 的四个工具合计 443 行。

### 1.4 内核扩展点

内核虽小,但必须暴露以下接口,否则插件无法工作:

```rust
// 工具注册
kernel.tools().register(def)?;          // def: ToolDef { name, description, parameters, execute }
kernel.tools().unregister(name);

// 服务提供与消费(配合隔离,见 2.2)
kernel.services().provide::<T>(name, impl)?;
kernel.services().get::<T>(name) -> Option<Arc<T>>;

// 生命周期事件(返回 disposer,自动登记为 effect)
kernel.events().on(Event::TurnStart, handler)?;

// 会话操作(子代理插件需要)
kernel.sessions().create(opts).await?;
kernel.sessions().list();
kernel.sessions().get(id);
kernel.sessions().fork(id).await?;

// agent 创建(子代理插件需要)
kernel.agent().spawn(scope).await?;

// 模型路由(所有插件可能需要)
kernel.llm().route();
kernel.llm().set_route(route).await?;

// 热加载入口(见第 3 节)
kernel.apply_change(patch).await?;
```

事件清单:`TurnStart` / `TurnEnd` / `ToolBefore` / `ToolAfter` / `SessionCreate` / `PluginLoaded` / `RuntimeChanged`。

---

## 2. 插件机制

设计来源:Cordis(`vendor/cordis/src/`,MIT)。**Rust 实现只能参考设计、重写代码**——Cordis 是 TypeScript,Rust 的所有权模型让实现方式不同(反而更干净)。

### 2.1 插件形态

```rust
pub trait Plugin: Send + Sync + 'static {
    fn name(&self) -> &str { "unnamed" }
    /// 依赖的服务:不全则等待(PENDING)
    fn inject(&self) -> &[&str] { &[] }
    /// 自己提供的服务
    fn provide(&self) -> &[&str] { &[] }
    /// 配置校验
    fn validate(&self, config: &Value) -> Result<()> { Ok(()) }
    /// 插件主体
    fn apply(&self, ctx: &Context, config: Value) -> Result<()>;
}
```

### 2.2 服务与隔离(realm)

**要求:同一进程内,不同会话的服务实例互相隔离。**

隔离映射用"可覆盖的 map"表达,而非 JS 的原型链:

```rust
/// 服务名 → realm 标识。子上下文 clone 父级映射,可覆盖单个服务名。
#[derive(Clone, Default)]
pub struct RealmMap(HashMap<String, RealmId>);

impl Context {
    /// 为某个服务名开一个新的隔离域
    pub fn isolate(&self, name: &str) -> Context {
        let mut map = self.realm_map.clone();
        map.0.insert(name.to_string(), RealmId::new());
        self.extend_with(map)
    }
}
```

服务表按 `(服务名, realm 标识)` 存放:

```rust
services: HashMap<(String, RealmId), Arc<dyn Any + Send + Sync>>,
```

查找时先算 realm 标识,再取实例:

```rust
fn service_key(ctx: &Context, name: &str) -> (String, RealmId) {
    let realm = ctx.realm_map.0.get(name).copied().unwrap_or_else(|| ctx.root_realm(name));
    (name.to_string(), realm)
}
```

于是:会话 A 的 `compaction` 与会话 B 的 `compaction` 是**两个独立实例**,各自持有游标、统计、策略,互不冲突。

**作用**:整合包 A 与整合包 B 可在同一进程共存,各自拥有独立的同名服务。

### 2.3 epoch:依赖变化自动重载

**这是"全部热加载"的引擎。** 插件不靠"手动卸载再加载",靠 epoch:

```rust
fn refresh(&mut self) {
    let mut epoch = String::new();
    for name in self.inject.iter() {
        match self.store.get(name) {
            // 缺依赖 → 不激活
            None => { self.set_epoch(Epoch::Inactive); return }
            // 把每个依赖的 fiber id 拼进 epoch
            Some(impl_) => epoch.push_str(&format!(":{}", impl_.fiber_id)),
        }
    }
    // epoch 变了 → 自动卸载旧的 + 加载新的
    self.set_epoch(Epoch::Active(epoch));
}
```

- 依赖的服务被替换 → 新实例 fiber id 不同 → epoch 变 → **自动重载**
- 依赖消失 → epoch 变 Inactive → 自动卸载,进入 Pending 等待
- 依赖重新出现 → 自动激活

**任何服务的增删替换,沿依赖链自动传播,不需要手写传播代码。**

### 2.4 effect:可撤销副作用

**所有副作用必须通过 effect 登记**,这是热卸载不残留的唯一保证:

```rust
// execute 立即执行,返回的 disposer 被收集
ctx.effect(|ctx| {
    let handle = ctx.spawn_timer(Duration::from_secs(10), tick);
    move || { handle.cancel(); }        // disposer
})?;
```

- 返回的 disposer 被收集到当前 fiber
- fiber 卸载时**逆序**执行所有 disposer(支持异步,卸载等待完成)
- 注册服务、监听事件、注册工具、定时器——**全部走 effect**

Rust 的 RAII 让这件事比 JS 更干净:disposer 是 `Box<dyn FnOnce()>` 或实现 `Drop` 的守卫对象,所有权明确。

### 2.5 参考什么、砍什么

| 参考 Cordis | 说明 |
|---|---|
| `isolate()` + 隔离映射 | 核心思路 + 查找逻辑 |
| Fiber 的 epoch 机制 | 热加载引擎 |
| effect 撤销体系 | 卸载安全底线 |
| `provide` / `get` / `notify` | 服务注册与依赖传播 |

| 砍掉 | 理由 |
|---|---|
| Loader(从 cordis.yml 读清单) | 用 Yellow 的包清单代替 |
| `intercept` 拦截配置 | 无多插件配置合并需求 |
| HMR 诊断栈 | 第三方插件开发体验,用不到 |
| 装饰器 | `inject` 数组足够 |
| 平面划分(host plane / agent plane) | 单进程单用户场景不需要 |

**预估规模:600~900 行 Rust**(比 TS 版略多,因为要显式处理所有权与生命周期)。

### 2.6 权限接口:内核只提供检查点

**内核不设访问权限,但把"权限检查"做成接口供插件使用。**

```
内核提供:权限检查点(接口)
   ↓
插件实现:沙箱插件 / 只读模式插件 / 审批插件 / 企业策略插件
   ↓
聚合规则:保守优先(deny wins)
```

```rust
// 内核侧:只提供注册与检查,不含任何策略
kernel.permissions().register_provider(provider);   // 插件注册
kernel.permissions().check(action, ctx).await       // 内核在关键点调用
```

**检查点位置**:工具执行前、文件读写前、命令执行前、网络请求前、热加载应用前。

**聚合必须是"收集式 + deny wins"**:

- **收集式**:顺序无关,只是聚合
- **deny wins**:任何 provider 说 deny,结果就是 deny;全都没有意见才放行

**这是硬规则**:把权限检查做成管道式(顺序相关)会让插件加载顺序影响安全结论,是致命缺陷。

### 2.7 冲突处理:默认 fail loud

冲突有三种形态,分别处理:

| 形态 | 处理 |
|---|---|
| **服务名冲突** | 同隔离域内第二次 `provide` 同名服务 → **报错**;不同域互不干扰 |
| **工具名冲突** | 按贡献声明的策略处理(见下) |
| **资源竞争** | 共享池 + 引用计数(见 5.4) |

工具注册的冲突策略:

```rust
kernel.tools().register(ToolDef {
    name: "search",
    conflict: Conflict::Error,        // 默认:冲突即报错
    // Conflict::Override,            // 覆盖,但记录被覆盖者
    // Conflict::Priority(10),        // 高优先级胜出
    // Conflict::Namespaced,          // 自动加插件前缀
});
```

**默认必须是 `Error`**。静默覆盖是插件系统最危险的失败模式:一个插件悄悄替换了另一个插件的安全检查,不留任何痕迹。

**不用"默认加前缀"的原因**:工具名会进入模型上下文,`websearch__search` 这类前缀污染 prompt、浪费 token。命名空间只在显式要求时使用。

**冲突必须可查询**:

```rust
kernel.tools().owners("search")      // 谁注册了 search
kernel.tools().overrides()           // 哪些被覆盖了、被谁
kernel.services().conflicts()        // 服务注册冲突记录
```

### 2.8 顺序控制:加载顺序与执行顺序

**加载顺序 = 依赖图自动排序**,优先级明确:

| 优先级 | 机制 | 用途 |
|---|---|---|
| 1(最高) | `after` / `before` / `inject` | 有真实依赖必须遵守 |
| 2 | `stages` 阶段 | 粗粒度分组 |
| 3 | `order` 数值 | 阶段内精细排序 |
| 4(兜底) | 注册顺序 | 稳定排序,保证可复现 |

- 依赖关系**压倒一切**——`order` 再小也不能违反 `after`
- **循环依赖 → 报错**,并指出环
- 同层 `order` 相同 → 按名称排序(保证确定性)

**执行顺序按场景区分,且必须分清两类语义**:

| 场景 | 机制 | 语义 |
|---|---|---|
| 事件监听器 | 优先级 + 注册顺序(稳定排序) | 收集式 |
| 提示词段 | `order` 数值(先 order 升序,同号按名称) | 管道式 |
| 工具调用前后钩子 | 显式 stage 数值 | 管道式 |
| 权限检查 | **顺序无关**,deny wins | 收集式 |
| 工具同名冲突 | 冲突策略决定 | — |

**管道式(pipeline)**:有明确先后,前一个的输出是后一个的输入——顺序**有语义**,必须可控。

**收集式(collection)**:顺序无关,只是聚合——顺序**无意义**,不得依赖。

**分不清这两类就会出 bug**:把权限检查做成管道式,插件加载顺序就会影响安全结论。

---

## 3. 热加载

### 3.1 统一入口

所有热加载走一个函数,保证可校验、可回滚、可审计:

```rust
pub async fn apply_change(&self, patch: Patch) -> Result<()> {
    let backup = self.snapshot_state();
    match self.try_apply(&patch).await {
        Ok(()) => {
            self.bump_version();
            self.emit(Event::RuntimeChanged(patch.clone()));
            if patch.persist { self.save(&patch)?; }
            Ok(())
        }
        Err(e) => {
            self.restore(backup);       // 失败回滚,旧配置原样保留
            Err(e)
        }
    }
}
```

**事务性要求**:新配置失败不得破坏旧配置。典型场景——热加载新 MCP 但连不上,必须"**先建后换**":新连接成功才替换,失败保留旧的继续跑。

### 3.2 每轮快照

避免竞态的关键。循环每轮开始读一次快照:

```rust
async fn run_turn(&self, session: &Session, user_msg: Message) -> Result<()> {
    let snap = self.runtime.snapshot();      // 本轮冻结
    let tools = snap.tools();
    let route = snap.model_route();
    // 整轮使用 snap,不受并发热加载影响
    ...
}
```

**一轮之内配置不变,轮与轮之间可变。**

### 3.3 作用域

```rust
kernel.apply_change(Patch::new(..).scope(Scope::Session(session_id))).await?;  // 默认
kernel.apply_change(Patch::new(..).scope(Scope::Global)).await?;
```

| 作用域 | 默认策略 | 说明 |
|---|---|---|
| `Session` | 无需同意 | 整合包加载走这里,包 A 不污染包 B |
| `Global` | **需要用户同意** | 影响所有会话,含以后新建的 |

### 3.4 同意逻辑(策略表)

```rust
policy: HashMap<ChangeKind, Consent>   // Consent = AutoAllow | Ask | Deny
```

| 变更类别 | 默认策略 |
|---|---|
| 会话级任何变更 | `AutoAllow`(不打扰) |
| 全局加技能 / 加 MCP | `Ask` |
| 全局换模型路由(自己换 API) | `Ask`(用户可改 Auto) |
| 全局改人设 / 提示词 | `Ask` |
| 全局降权限 / 关沙箱 | **`Deny`**(不可自动同意) |

**`Ask` 的交互**:

```
Agent 请求:全局添加 MCP 服务器「xxx」

影响:所有会话
缓存:会破坏当前前缀(约 12,400 tokens 需重新计费)

  [ 立即生效(牺牲缓存) ]   [ 下个会话生效(保留缓存) ]   [ 取消 ]
  [ ] 以后这类变更都自动立即生效
```

勾选"以后自动"→ 该类别加入 `AutoAllow`,写入用户设置。

---

## 4. 缓存策略

### 4.1 原则

> **缓存只是成本,不是约束。任何热加载都必须能立即生效,绝不因缓存而拒绝变更。**

```
能力层:任何变更都能立即生效   ← 永不阻塞
优化层:尽量少破坏前缀缓存     ← 尽力而为,可配置,可覆盖
```

### 4.2 三档可配置

| 档位 | 行为 | 适合 |
|---|---|---|
| **新鲜优先** `Freshness` | 立即改写 prompt,不管缓存 | 干活要紧 |
| **平衡** `Balanced`(默认) | 能追加就追加,结构性变更才改写 | 默认 |
| **省钱优先** `CacheFirst` | 一切走追加,易变内容推后 | 长对话、成本敏感 |

**第二层:按类别覆盖**

```yaml
cache_policy:
  default: balanced
  by_change:
    model-route: freshness      # 换 API 必须立刻生效
    skill: balanced
    persona: freshness
    mcp: balanced
```

**第三层:单次强制覆盖**

```rust
apply_change(patch.cache_policy(CachePolicy::ForceFresh)).await?;  // 这次不管缓存
apply_change(patch.cache_policy(CachePolicy::Defer)).await?;       // 攒到下个会话生效
```

### 4.3 追加式 vs 改写式

KV cache 是**前缀缓存**:前缀中任一 token 变化,从该位置起全部未命中。

**改写式**(无 `in-history` 能力时):
```
[0] system: 新提示词     ← 变了 → 整个请求都不同 → 全部未命中
[1..N] 历史消息
[N+1] user: 新消息
```

**追加式**(声明 `system_prompt_update: "in-history"` 时):
```
[0] system: 旧提示词     ← 不变,缓存命中
[1..N] 历史消息           ← 不变,缓存命中
[N+1] system: 新提示词    ← 新增
[N+2] user: 新消息        ← 新增
```
→ **直到历史末尾的前缀仍可复用**

来源:DSH `packages/core/system-prompt` 与 `packages/core/agent-loop` 的决策规则。

### 4.4 prompt 分区(与档位无关,直接采用)

```
[ 稳定区 ]  harness 身份 → persona 前缀 → 工具 schema → 前置指令
[ 易变区 ]  persona 后缀 → 源码路径 → Web URL → 环境信息 → 热加载配置
```

**稳定的放前面,易变的放后面。** 于是环境变化不影响可复用前缀。

### 4.5 动态上下文走 user 消息

每轮变化的运行时信息(时间、cwd、环境)**不塞进 system prompt**,而是作为 **user 角色消息**插入历史。system prompt 保持稳定。

### 4.6 必须处理的技术前提

"追加式"依赖模型/API 支持**会话历史中的 system 消息**:

- **要探测**:能否在 messages 中间插 system 消息并被正确理解
- **要可配置**:模型目录里带能力标记(`system_prompt_update: "in-history"`)
- **要有回退**:探测失败或报错时自动退回改写式,**不能因此让请求失败**
- **显式声明,不猜**

### 4.7 物理限制(诚实记录)

换 provider / model 时,**缓存必然从头开始**——不同模型的缓存空间不共享。换 API 这个动作本身就该被理解为"牺牲一次缓存"。

### 4.8 上下文注入引擎

机制参考 SillyTavern 的 World Info(`I:\SillyTavern\public\scripts\world-info.js`,5605 行),按 agent 场景改造。它解决的问题是:**在有限上下文里,动态决定注入什么内容、注入到哪里。**

#### 条目结构

每个可注入条目带以下控制维度:

| 维度 | 作用 |
|---|---|
| `constant` | 常驻:不靠触发条件,始终激活 |
| `triggers` | 触发条件(关键词 / 路径 / 工具名) |
| `selective` | 多条件逻辑组合 |
| `position` | 注入位置(见下) |
| `depth` | 注入深度(历史第 n 层) |
| `role` | 以什么角色注入(system / user / assistant) |
| `order` | 排序 |
| `group` / `group_weight` | 分组竞争 |
| `scan_depth` | 只在最近 n 条消息里匹配 |
| `delay_until_recursion` | 延迟到第 n 层递归才可用 |
| `prevent_recursion` | 本条目不触发递归 |
| `ignore_budget` | 不受预算限制 |
| `enabled` | 启用开关 |

#### 预算机制:按上下文百分比

```
budget = round(budget_percent × max_context / 100)      // 默认 25%
if budget_cap > 0 and budget > budget_cap: budget = budget_cap
```

- 预算是**上下文的百分比**——自动适应不同模型的窗口大小
- 另有绝对上限 `cap`
- 累计内容超预算 → **停止激活新条目**(已激活的保留),而非硬切断
- `ignore_budget` 条目**仍可激活**(关键内容可突破预算,如安全策略)

#### 扫描状态机

```
INITIAL         → 在最近 scan_depth 条消息里匹配
RECURSION       → 把新激活条目的内容再扫一遍,可能触发更多条目
MIN_ACTIVATIONS → 激活数不足时,加深扫描范围再来一轮
NONE            → 结束
```

**递归激活**是本机制的核心价值:激活的内容成为新的匹配源。对 agent 的用途是**上下文自我扩展**——读到某个文件后自动注入相关文档或技能。

#### 插入位置(与缓存策略联动)

```
prefix              提示词前缀区(破坏缓存)
history_head        历史开头
at_depth(n)         历史第 n 层 ← 关键:不破坏前缀
history_tail        历史末尾(追加,不破坏前缀)
```

**`at_depth` + `role` 是保持前缀稳定的主要手段**:把动态内容作为消息插入历史深处,而不是改写前缀。

**与第 4 章档位联动**:

| 缓存档位 | 注入到前缀区 | 注入到历史深处 |
|---|---|---|
| `Freshness` | 允许 | 允许 |
| `Balanced` | 仅在无历史深处位置可用时 | 优先 |
| `CacheFirst` | **拒绝**,改用历史深处 | 允许 |

#### 分组竞争

同组条目**只选一个**(按 `group_weight` 加权)。对 agent 的用途:**多个相似技能只加载一个**,省预算、避免重复。

#### 与 SillyTavern 的差异(必须改造的地方)

| SillyTavern | Nguruvilu | 理由 |
|---|---|---|
| `probability` 随机激活 | **移除**,改为确定性排序 | 随机会让同一输入产生不同上下文,破坏可复现性,且与缓存策略冲突(每次激活不同内容 → 前缀不稳定) |
| 内嵌世界书(角色卡) | 不采用 | 内嵌不利于去重与版本管理;走 Yellow 的引用式设计 |
| 面向角色扮演 | 面向任务 | 触发条件从"关键词"扩展到"路径 / 工具名 / 任务状态" |

---

## 5. 多会话、多设备与隔离

### 5.1 三层隔离

```
第一层  进程隔离      每台电脑 / 每个用户 → 一个进程(天然隔离,零成本)
第二层  会话配置隔离  基线配置 + 会话覆盖层
第三层  服务实例隔离  realm(见 2.2)
```

### 5.2 会话配置分层

```rust
/// 进程级基线(所有会话共享的起点)
pub struct BaseConfig {
    tools: ToolSet,
    mcp: McpSet,
    skill_dirs: HashSet<PathBuf>,
    model_route: ModelRoute,
    persona: String,
}

/// 会话级覆盖(整合包加载到这里,不碰基线)
pub struct Overlay {
    tools: Delta<ToolDef>,
    mcp: Delta<McpSpec>,
    skill_dirs: Delta<PathBuf>,
    model_route: Option<ModelRoute>,
    persona: Option<String>,
}

/// 生效配置 = 基线 + 覆盖
impl Session {
    pub fn resolve(&self, base: &BaseConfig) -> ResolvedConfig {
        ResolvedConfig {
            tools: base.tools.merged(&self.overlay.tools),
            mcp: base.mcp.merged(&self.overlay.mcp),
            ..
        }
    }
}
```

### 5.3 多设备与多标签

**同一部手机可开多个会话(多标签)。**

```
手机(Blue)
 ├─ 标签 1:整合包 A 的会话(带 MCP X)
 ├─ 标签 2:整合包 B 的会话(带 MCP Y)
 └─ 标签 3:普通会话
```

需要:
- **客户端**:会话列表 / 标签切换 UI
- **服务端**:会话列表 API + 每会话独立配置(天然契合 5.2)
- **隧道**:多会话共用一条隧道,靠 `session_id` 区分(现有帧多路复用已支持)

### 5.4 共享资源池

同一进程内,相同资源**只建一份**,按引用计数复用:

```rust
mcp_pool: HashMap<SpecHash, Pooled<McpConn>>,
// Pooled 持有 conn + HashSet<SessionId>;会话加入/离开调整计数;空集才真正关闭
```

模型客户端同理:按 `(provider, base_url, api_key)` 复用。

---

## 6. 模型层

### 6.1 只支持 OpenAI 格式

**砍掉多 provider 适配。** 对比:Pi 的 `ai` 包 179 文件 / 22.5k 行几乎全在适配 Anthropic / Google / Mistral / Azure / Bedrock 的差异。只做 OpenAI 格式可压缩到几百行。

### 6.2 中立消息格式

**存储用自有中立格式,请求时转换**(参考 Pi 的 `convertToLlm`,只在 LLM 调用边界转换一次)。

这是**中途换模型的前提**——历史消息不能绑定某家 provider 的格式。

```rust
pub enum Message {
    System { text: String },
    User { parts: Vec<Part> },
    Assistant { parts: Vec<Part>, tool_calls: Vec<ToolCall> },
    ToolResult { call_id: String, content: String },
}

/// 只在请求边界转换一次
fn to_openai(messages: &[Message]) -> Vec<OpenAiMessage> { .. }
```

### 6.3 能力声明

模型目录携带能力标记,显式声明不猜:

```rust
pub struct ModelInfo {
    pub id: String,
    pub system_prompt_update: Option<SystemPromptUpdate>,  // 存在时值必须精确匹配
    pub context_window: usize,
    pub max_output_tokens: usize,
}
```

---

## 7. 性能硬要求

**依据:瓶颈不在模型而在框架。** 实测数据——单次 LLM 调用约 800ms,若 agent 总共 15 秒,模型只占约 **5%**;**工具执行占 agent 总请求时间的 35~61%**;上下文膨胀与多步复合造成的延迟常超过模型本身。

因此以下为**硬要求**,不是"优化建议":

### 7.1 工具必须并行调度

模型一轮请求的多个**独立**工具必须同时执行,不得串行等待。串行 5 个工具 × 1 秒 = 5 秒;并行 = 1 秒。

Rust 用 `tokio::join!` / `JoinSet` 实现真并发。

### 7.2 上下文必须增量构建

**不得每轮重新序列化整个历史。** 缓存已构建的请求前缀,只追加新消息。

配套要求:中立消息格式 + 借用式序列化(见 7.4)。

### 7.3 流式输出不得经中间缓冲

模型 token 一到就交给消费端。禁止:批量刷新、跨进程中转、等待完整响应。

### 7.4 零拷贝序列化

序列化优先用借用(`&str` / `Cow`)而非克隆;大工具结果避免深拷贝。

### 7.5 无 GC 停顿

Rust 天然满足。**不得引入会带来全局停顿的运行时**(这条同时排除了"内核用脚本引擎实现"的方案——脚本引擎只能作为插件层)。

### 7.6 连接复用

HTTP 客户端必须复用连接(keep-alive / 连接池),不得每请求重新握手。

### 7.7 工具结果流式处理

大输出(如 `bash` 产生 10MB 日志)必须**边收边处理**(截断/落盘/摘要),不得等全部收完再处理。

### 7.8 prompt 前缀稳定

保持前缀逐字节稳定(见 4.4),使服务端 KV cache 可复用——**服务端少算,首 token 更快**。

### 7.9 每轮开销可测量

内核必须暴露每轮的时间分解:TTFT、模型生成、工具执行、上下文构建、框架自身。**无法测量的开销无法优化。**

### 7.10 每轮必经路径不得经过脚本引擎

循环、上下文构建、模型调用、流式解析、消息存储、**四个基础工具**必须在 Rust 内核内完成。脚本引擎只服务插件层。

**理由**:脚本引擎(QuickJS)的解释开销在 IO 场景下可忽略(被网络与磁盘延迟掩盖),但在每轮必经路径上会累积。分层保证性能敏感路径零脚本开销。

**验证方式**:见 7.9 的时间分解——"框架自身"项中不得包含脚本引擎执行时间。

---

## 8. 插件规划

### 8.1 插件运行时:三层架构

```
┌─ 内核(Rust 原生)────────────────────────────┐
│  会话 / 循环 / 模型(OpenAI) / 工具表          │
│  四个基础工具 read/write/edit/bash            │
│  插件机制:隔离域 + 依赖 + 热重载 + 撤销        │
└──────────────────────────────────────────────┘
        ↕ 宿主接口
┌─ 插件运行时(嵌入脚本引擎)────────────────────┐
│  轻量工具插件:搜索、todo、提示词处理、技能      │
│  → 热加载天然(重新加载脚本即可,无 ABI 问题)   │
└──────────────────────────────────────────────┘
        ↕ JSON-RPC over stdio
┌─ 子进程插件(任何语言)───────────────────────┐
│  computer use、MCP 客户端、手机远程控制        │
│  → 隔离最好、崩溃不影响内核、可用 JS 写        │
└──────────────────────────────────────────────┘
```

**分层的理由:**

1. **轻量插件走脚本** → 热加载**天然成立**(重新加载文件即可),完全绕开 Rust 的 ABI/卸载难题
2. **重活/外部服务走子进程** → computer use、MCP、远程控制本来就需要系统访问或长连接,子进程隔离最好,且可用任意语言写
3. **内核保持纯 Rust** → 单文件、快、稳

**脚本引擎:QuickJS**(`rquickjs`,约 1MB,可静态编译进二进制)。

选它的理由:

1. **热加载天然成立**——重新加载脚本文件即可,无 Rust 的 ABI 与卸载难题
2. **现有 dshd 的 JS 代码可直接搬为插件**(White 隧道、文件盒、Yellow 逻辑)
3. **Pi / DSH 的插件设计可借鉴**(它们也是 JS/TS)
4. 体积小,符合单文件可执行的目标

**性能边界由 7.10 保证**:每轮必经路径全在 Rust 内,脚本只服务插件层。

### 8.2 两类插件

| 类型 | 特征 | 例子 |
|---|---|---|
| **工具型** | 只给模型加工具 | 搜索、computer use、todo |
| **服务型** | 自己提供服务/长连接,可被其他插件依赖 | 手机远程控制、MCP 客户端、压缩器 |

服务型插件依赖 1.4 的"提供服务"扩展点。

### 8.3 首批插件

| 插件 | 类型 | 可行性 | 要点 |
|---|---|---|---|
| **搜索** | 工具型 | 完美 | 纯 HTTP,零障碍 |
| **压缩** | 服务型 | 可行 | 每轮核心路径,需注意 7.2 的增量构建 |
| **手机远程控制** | **服务型** | 可行 | 需内核支持插件提供网络服务;复用现有 White 隧道 |
| **子代理** | 工具型 + 服务依赖 | 可行 | 需内核暴露"创建 agent / 会话"接口 |
| **computer use** | 工具型 | 可行 | 截图 + 鼠标键盘;走子进程插件,跨平台分别处理 |

---

## 9. 工作区 git 自动快照

**内核提供 git 快照服务**(不是 git 全局配置):

```
每个工作区 = 一个独立 git 仓库
每次工具改动前 → 自动快照
每次工具改动后 → 自动提交
会话 ↔ commit 对应 → 可回滚到任意一轮
```

三个用途:
1. **安全网**:agent 改错能回滚
2. **审计**:哪一轮改了什么
3. **"自己改自己"的前提**:热加载改配置/代码前先提交

**不使用全局 git 钩子**——全局配置会影响所有项目,且"何时提交"只有内核知道。

---

## 10. 动态加载层(Yellow)

**Yellow 是架构中的一层,不是插件。** 它承担"把东西动态加载进来"这一职责。

### 10.1 职责

| 职责 | 说明 |
|---|---|
| **加载** | 插件 / MCP / 技能(三者平级,见 10.2) |
| **卸载** | 运行时移除;副作用由 effect 保证清理(见 2.4) |
| **热替换** | 同一对象的新版本替换旧版本,依赖方经 epoch 自动重载(见 2.3) |
| **分发** | 整合包(`.dshpack`)格式 |

### 10.2 可加载对象(平级)

| 对象 | 内容 | 来源 |
|---|---|---|
| **插件** | 工具型 / 服务型插件 | npm / git / 本地 / 整合包 |
| **MCP** | MCP 服务器(stdio / streamable-http) | 整合包声明 |
| **技能** | skills 目录 | 整合包引用 + sha1 校验 |

三者**没有从属关系**:技能不是插件的子集,MCP 也不是插件的一种。它们都是 Yellow 的加载对象。

### 10.3 整合包(`.dshpack`)

整合包是一个 **zip 归档**,含三部分:

| 文件 | 作用 |
|---|---|
| `dsh.index.json` | 身份:名称、版本、协议、构建时的内核版本、依赖范围 |
| `assembly.yaml` | 装配:加载什么、什么顺序(见 10.4) |
| 其余文件 | 装配引用的内容(技能目录、插件载荷等) |

**容器选择 zip 而非 tar**:任何平台上的任何普通工具都能打开它、查看内容、取出单个文件,不必依赖本内核。条目使用 deflate 压缩,并保留文件模式,所以包里带的脚本在解包后仍可执行。

**身份与装配分离是有意的**:`dsh.index.json` 说明"这个包是什么",`assembly.yaml` 说明"它加载什么、什么顺序"。前者可以在不执行任何加载逻辑的前提下检查来源与协议。

**命令**:

```
ngu pack <目录> [--out <文件>]      # 目录须含 dsh.index.json
ngu verify <文件>                   # 校验清单合法性、是否含装配、协议是否声明
ngu install <文件> [--into <目录>]  # 安装到 <packs>/<name>-<version>/
ngu packs                           # 列出已安装
```

安装目录带版本号,所以同一个包的两个版本共存而不互相覆盖。解包逐条进行,并**拒绝逃逸目标目录的条目路径**;打包时排除 `.git`、`target`、`node_modules`。

**与旧格式的关系**:理念保留(引用清单、哈希去重、台账、版本兼容声明、协议标注),容器从 tar 换成 zip。

### 10.4 装配清单:`assembly.yaml`

整合包必须能声明"**怎么装配**"——顺序、配置、依赖、失败策略。平铺的插件列表不够,因此新增 `assembly.yaml`,取代原来的 `plugins.json`,并**统一管理插件 / MCP / 技能**(三者平级)。

```yaml
version: 1

# 默认策略(可被单个条目覆盖)
defaults:
  scope: session          # session | global
  conflict: error         # error | override | priority(n) | namespaced
  on_failure: abort       # abort | skip | retry
  load: eager             # eager | lazy

# 加载阶段:按阶段顺序执行;阶段内按 依赖 → order → 注册顺序
stages:
  - name: foundation
    plugins:
      - id: fs-tools
        source: "npm:@dshd/fs-tools@^1.2"
        order: 10
      - id: shell-tools
        source: "npm:@dshd/shell-tools@^1.0"
        order: 20

  - name: services
    plugins:
      - id: compaction
        source: "npm:@dshd/compaction@^0.5"
        order: 10
        config:
          threshold_tokens: 150000
          keep_recent_turns: 20
      - id: mcp-client
        source: "npm:@dshd/mcp-client@^1.0"
        order: 20
        after: [compaction]

  - name: extensions
    plugins:
      - id: search
        source: "npm:@dshd/search@^2.0"
        order: 10
        on_failure: skip
      - id: computer-use
        source: "subprocess:./plugins/computer-use.js"
        order: 20
        on_failure: skip
        platform: [win32, darwin]
    mcp:
      - id: github-mcp
        transport: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-github"]
        order: 10
        after: [mcp-client]
    skills:
      - id: pdf-tools
        source: "github:owner/repo@skills/pdf@v1"
        sha1: "…"
        order: 10

teardown:
  reverse_stages: true
```

**条目字段**:

| 字段 | 作用 |
|---|---|
| `source` | 来源:npm / git / 本地 / 子进程 / 包内 |
| `order` | 阶段内排序 |
| `after` / `before` | 显式依赖顺序 |
| `config` | 条目配置 |
| `scope` | `session`(默认)或 `global`(需用户同意) |
| `conflict` | 同名冲突策略 |
| `on_failure` | `abort` / `skip` / `retry` |
| `load` | `eager` / `lazy` |
| `platform` | 平台限定 |
| `requires` | 版本/能力约束 |

**与包内其他文件的关系**:

| 文件 | 变化 |
|---|---|
| `dsh.index.json` | 保留(包身份、版本、协议、依赖声明) |
| `plugins.json` | **被 `assembly.yaml` 取代** |
| `mcp.json` | 保留,但也可在 `assembly.yaml` 中声明(需要顺序) |
| `soul.md` / `models.json` / `patches.yaml` / `presets/` | 保留不变 |

### 10.5 对接点(内核需要提供)

| # | Yellow 现在调用 | 内核需要提供 |
|---|---|---|
| 1 | `dsh plugin add <pkg>` | `kernel.load_plugin(name)` |
| 2 | `agentPreset.read` | `kernel.read_preset(id)` |
| 3 | `agentPreset.select` | `kernel.apply_preset(session_id, id)` |
| 4 | `session.create` | `kernel.sessions().create(opts)` |
| 5 | `session.fork` | `kernel.sessions().fork(id)` |
| 6 | `dsh-skill-filesystem` 的 `customSkillDirs` | `kernel.register_skill_dir(dir)` |
| 7 | `dsh-mcp-client` 挂载行 | `kernel.add_mcp_server(spec)` |
| 8 | `cordis.patch.yml` 补丁合并 | `kernel.patch_composition(rows)` |
| 9 | `settings.yaml` 的 `llm-pi-ai` 段 | `kernel.set_model_route(route)` |

### 10.6 改造方向

从"写配置文件 + 建新会话"改成"调 `apply_change` 热加载"(作用域默认 `Session`)。

---

## 11. 待定问题

**已定**:插件运行时 = QuickJS(8.1);会话存储 = JSONL + trait 接口;压缩 = 纯插件;语言 = Rust。

**仍待定**:

1. **异步运行时**:默认 tokio(除非有特殊约束)
2. **手机端(Blue)改造范围**:多标签 UI 由谁实现
3. **computer use 的跨平台方案**:原生模块选型与子进程协议
4. **Yellow 重构范围**:现有 JS 实现是搬为插件,还是用 Rust 重写加载层

---

## 12. CLI 接口

`ngu` 是内核的参考前端,同时是**其他 agent 调用内核的契约**。

### 12.1 三种模式

| 模式 | 触发 | 用途 |
|---|---|---|
| **交互** | 无 `-p`,stdin 是终端 | 人工使用(REPL) |
| **打印** | `-p "<prompt>"`,或 stdin 非终端 | 脚本、管道 |
| **JSON** | 追加 `--json` | **其他 agent 程序化调用** |

```bash
ngu -p "列出当前目录"                      # 打印模式
echo "总结这个文件" | ngu --json           # 管道 + JSON
ngu --session <id> -p "继续"               # 继续会话
ngu sessions                               # 列出会话
ngu models                                 # 列出可用模型
```

### 12.2 环境变量

| 变量 | 作用 |
|---|---|
| `NGU_API_KEY`(或 `OPENAI_API_KEY`) | API key |
| `NGU_BASE_URL`(或 `OPENAI_BASE_URL`) | API base(含版本段) |
| `NGU_MODEL` | 模型 id |
| `NGU_HOME` | 会话存储根(默认 `<cwd>/.nguruvilu`) |
| `NGU_SHELL` | `bash` 工具的 shell(默认 Windows 用 PowerShell,其余用 `sh`) |

### 12.3 JSON 输出契约

其他 agent 依赖这个结构,**字段只增不减**:

```json
{
  "ok": true,
  "session_id": "20260918-120000-ab12cd",
  "text": "最终回答",
  "steps": 3,
  "tool_calls": 5,
  "usage": { "input": 1234, "output": 567, "cached": 900 },
  "timing": { "model_ms": 2100, "tools_ms": 480 },
  "messages": [ /* 本轮新增的中立格式消息 */ ]
}
```

- `ok` 为 `false` 时进程以非零码退出,错误走 stderr
- `timing` 是性能硬要求 7.9 的落点:调用方可以据此判断时间花在模型还是工具上

### 12.4 被调用时的约定

- **stdout 只放结果**:进度、工具轨迹一律走 stderr,保证管道可用
- **退出码**:0 成功,1 失败
- **会话可续**:`--session <id>` 复用历史,`--new-session` 强制新开

---

## 13. 实现状态

截至当前提交,规格中的以下部分已经落地并有测试覆盖(125 个测试:100 单元 + 25 集成):

| 规格章节 | 状态 | 落点 |
|---|---|---|
| 1 内核边界 | ✅ | `tools/`(四工具 + 注册表)、`agent.rs`、`session.rs`、`llm.rs` |
| 2.2 隔离域 | ✅ | `plugin.rs`:`RealmMap`(含默认 realm)、`(name, realm)` 服务键 |
| 2.3 epoch | ✅ | `plugin.rs`:`refresh()` 沿依赖图自动重载;依赖消失时 fiber 保留为 Pending |
| 2.4 effect | ✅ | `plugin.rs`:disposer 逆序执行,卸载清理服务与工具 |
| 2.6 权限接口 | ✅ | `plugin.rs`:`PermissionStack`,收集式 + deny-wins |
| 2.7 冲突处理 | ✅ | `tools/mod.rs` + `plugin.rs`:默认 fail loud,覆盖记录可查 |
| 2.8 顺序控制 | ✅ | `assembly.rs`:依赖拓扑 → 阶段 → order → 声明顺序 |
| 3 热加载 | ✅ | `hotreload.rs`:`apply_change`、每轮快照、作用域、同意策略表 |
| 4 缓存策略 | ✅ | `hotreload.rs`:`CachePolicy` 三档;`context.rs` 按档位决定注入位置 |
| 4.8 上下文注入引擎 | ✅ | `context.rs`:预算、触发、深度注入、分组竞争、递归激活(无随机) |
| 5 多会话与隔离 | ✅ | `plugin.rs` 隔离域 + `loader.rs` 按 scope 选 realm |
| 6 模型层 | ✅ | `llm.rs` 仅 OpenAI 格式;`message.rs` 中立格式 + 边界转换 |
| 7 性能硬要求 | ✅ | 并行调度(JoinSet)、增量请求构建、流式直连、时间分解 |
| 8 插件规划 | ⏳ | 运行时分层已定(内核 Rust + 脚本 + 子进程);QuickJS 接入待做 |
| 9 git 快照 | ✅ | `git.rs`:快照、历史、回滚(含删除快照后新增的文件) |
| 10 动态加载层 | ✅ | `assembly.rs` + `ledger.rs` + `loader.rs` + `mcp.rs`(技能/MCP/插件平级) |
| 12 CLI | ✅ | `main.rs`:三运行模式 + 8 个管理子命令 |

**尚未实现**(诚实记录):

- **QuickJS 插件运行时**:插件目前必须是内核内定义的(编译期)。脚本插件热加载是下一步。
- **子进程插件协议**:computer use 等重活插件需要它;MCP 已经走通同一条路。
- **streamable-http MCP**:声明可解析,但只连 stdio;HTTP 传输会明确报告"未实现"而非静默跳过。
- **手机端多标签会话**:服务端模型已就绪(会话级隔离 + 作用域),客户端未做。

---
## 附:实现顺序

1. **内核骨架**:会话 + 循环 + 模型(OpenAI)+ 工具表 + 4 工具
2. **中立消息格式 + 请求转换**(换 API 的前提,必须一开始就有)
3. **性能骨架**:并行工具调度 + 增量上下文 + 流式直连(第 7 节是硬要求,不能事后补)
4. **插件机制**:隔离域 + epoch + effect(600~900 行)
5. **`apply_change` + 每轮快照 + 作用域 + 同意逻辑**
6. **缓存策略三档 + prompt 分区**
7. **会话配置分层 + 共享资源池**
8. **工作区 git 快照服务**
9. **动态加载层(Yellow)**:加载 / 卸载 / 热替换 + QuickJS 接入 + 子进程协议
10. **可加载对象接入**:插件(搜索 → 压缩 → 手机远程 → 子代理 → computer use)、MCP、技能
11. **整合包(`.dshpack`)分发与安装**
12. **UI 与远程**(复用现有隧道 / TUI)
