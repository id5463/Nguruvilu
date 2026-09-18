# Nguruvilu 规格

本文档是 Nguruvilu 的**唯一权威规格**。所有实现必须服从本文档;与本文档冲突的代码是错的。

**Nguruvilu** 是独立 agent 内核,CLI 名 `ngu`,语言 **Rust**。它与 dshd 家族是内核与发行版的关系:内核独立演进,dshd 是使用方之一。

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

**以下全部是插件,永不进内核**:

- 搜索(web search / fetch)
- computer use(截图、鼠标键盘)
- 子代理(subagent / fork)
- 手机远程控制
- MCP 客户端
- 技能(skills)
- **压缩(compaction)** —— 纯插件
- 审批与权限
- 沙箱
- TUI / Web UI
- 整合包(Yellow)

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

**脚本引擎选型**:待定(见第 11 节)。倾向 QuickJS(`rquickjs`,约 1MB,可静态编译),因为现有 dshd 的 JS 代码可直接复用为插件,且 Pi/DSH 的插件设计可借鉴。

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

## 10. Yellow 整合包对接

Yellow 的理念与包格式**原样保留**(纯引用清单、sha1 去重、台账、版本兼容声明、协议标注),只改**落地层**。

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

**改造方向**:从"写配置文件 + 建新会话"改成"调 `apply_change` 热加载"(作用域默认 `Session`)。

---

## 11. 待定问题

1. **插件运行时脚本引擎**:QuickJS / Rhai / Lua / 纯子进程(见 8.1)
2. **会话持久化格式**:JSONL / SQLite(vendored)/ JSONL + trait 接口
3. **异步运行时**:默认 tokio(除非有特殊约束)
4. **手机端(Blue)改造范围**:多标签 UI 由谁实现
5. **computer use 的跨平台方案**:原生模块选型与子进程协议

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
9. **插件运行时**:脚本引擎接入 + 子进程协议
10. **插件**:搜索 → 压缩 → 手机远程 → 子代理 → computer use
11. **Yellow 对接改造**
12. **UI 与远程**(复用现有隧道 / TUI)
