# dshd-core 规格

本文档是 dshd-core 的**唯一权威规格**。所有实现必须服从本文档;与本文档冲突的代码是错的。

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
- 压缩(compaction)——**待定,见第 10 节**
- 审批与权限
- 沙箱
- TUI / Web UI
- 整合包(Yellow)

### 1.3 四个基础工具

保留四个而非只留 `bash`,理由:**专用工具省 token 且更安全**。

| 工具 | 参数 | 说明 |
|---|---|---|
| `read` | `path`, `startLine?`, `endLine?` | 按行范围读取,返回带行号内容。比 `cat` 省大量 token |
| `write` | `path`, `content` | 覆盖写入 |
| `edit` | `path`, `oldStr`, `newStr` | 字符串替换。比 `sed` 可靠(不依赖行号) |
| `bash` | `command`, `timeout?` | 执行任意 shell 命令,兜底一切 |

实现参考:Pi 的四个工具合计 443 行(`read` 138 / `write` 41 / `edit` 131 / `bash` 133)。

### 1.4 内核扩展点

内核虽小,但必须暴露以下接口,否则插件无法工作:

```js
// 工具注册
kernel.tools.register(def)              // def: { name, description, parameters, execute }
kernel.tools.unregister(name)

// 服务提供与消费(配合隔离,见 2.2)
kernel.services.provide(name, impl)
kernel.services.get(name)               // 跨隔离域查找,找不到返回 undefined

// 生命周期事件
kernel.events.on(event, handler)        // 返回 disposer
// 事件:turn:start / turn:end / tool:before / tool:after / session:create / plugin:loaded ...

// 会话操作(子代理插件需要)
kernel.sessions.create(opts) / list() / get(id) / fork(id)

// agent 创建(子代理插件需要)
kernel.agent.spawn(scope)

// 模型路由(所有插件可能需要)
kernel.llm.route() / kernel.llm.setRoute(route)

// 热加载入口(见第 3 节)
kernel.applyChange(patch)
```

---

## 2. 插件机制

机制来源:Cordis(`vendor/cordis/src/`,MIT)。**抄精简版,不抄全套。**

### 2.1 插件形态

```js
// 三种形态等价,内部统一 resolve 出 callback
export function apply(ctx, config) { ... }        // 函数
export default class X { constructor(ctx, config) {} }  // 类
export default { apply(ctx, config) { ... } }     // 对象

// 元数据
export const name = 'my-plugin'
export const inject = ['someService']             // 依赖的服务:不全则等待
export const provide = ['myService']              // 自己提供的服务
export const Config = schema                       // 配置校验(可选)
```

### 2.2 服务与隔离(realm)——解法乙

**要求:同一进程内,不同会话的服务实例互相隔离。**

实现方式(抄 Cordis,核心极小):

```js
// 隔离映射:服务名 → scope label(Symbol)
// 用原型链表达:子上下文继承父级映射,可覆盖某个服务名
function isolate(ctx, name, label) {
  const shadow = Object.create(ctx[ISOLATE])
  shadow[name] = label ?? Symbol(name)
  return extend(ctx, { [ISOLATE]: shadow })
}

// 同域判断:一个服务实例只在同标签下可见
function sameRealm(ctxA, ctxB, name) {
  return ctxA[ISOLATE][name] === ctxB[ISOLATE][name]
}
```

服务注册时,key 不是字符串名,而是**隔离映射算出的 symbol**:

```js
function serviceKey(ctx, name) {
  ctx.root[ISOLATE][name] ??= Symbol(name)
  return ctx[ISOLATE][name]
}
```

于是:会话 A 的 `compaction` 与会话 B 的 `compaction` 是**两个独立实例**,各自持有游标、统计、策略,互不冲突。

**作用**:整合包 A 与整合包 B 可在同一进程共存,各自拥有独立的同名服务。

### 2.3 epoch:依赖变化自动重载

**这是"全部热加载"的引擎。** 插件不靠"手动卸载再加载",靠 epoch:

```js
function refresh(fiber) {
  let epoch = ''
  for (const name of Object.keys(fiber.inject)) {
    const impl = fiber.store[name]
    if (!impl) { epoch = INACTIVE; break }   // 缺依赖 → 不激活
    epoch += ':' + impl.fiber.uid            // 把每个依赖的 fiber id 拼进 epoch
  }
  setEpoch(fiber, epoch)                      // epoch 变了 → 自动卸载旧的 + 加载新的
}
```

- 依赖的服务被替换 → 新实例 fiber id 不同 → epoch 变 → **自动重载**
- 依赖消失 → epoch 变 INACTIVE → 自动卸载,进入 PENDING 等待
- 依赖重新出现 → 自动激活

**任何服务的增删替换,沿依赖链自动传播,不需要手写传播代码。**

### 2.4 effect:可撤销副作用

**所有副作用必须通过 effect 登记**,这是热卸载不残留的唯一保证:

```js
ctx.effect(() => {
  const timer = setInterval(...)
  return () => clearInterval(timer)     // disposer
}, 'label')
```

- `execute` 立即执行,返回的 disposer 被收集
- fiber 卸载时**逆序**执行所有 disposer(支持异步,卸载等待完成)
- 注册服务、监听事件、注册工具、定时器——**全部走 effect**

### 2.5 抄什么、砍什么

| 从 Cordis 抄 | 说明 |
|---|---|
| `isolate()` + 隔离映射 | 核心 4 行 + 查找逻辑 |
| Fiber 的 epoch 机制 | 热加载引擎 |
| effect 撤销体系 | 卸载安全底线 |
| `provide` / `get` / `notify` | 服务注册与依赖传播 |

| 砍掉 | 理由 |
|---|---|
| Loader(从 cordis.yml 读清单) | 用 Yellow 的包清单代替 |
| `intercept` 拦截配置 | 无多插件配置合并需求 |
| HMR 诊断栈(`getOuterStack`、`EffectMeta`) | 第三方插件开发体验,用不到 |
| `@Inject` 装饰器 | `inject` 数组足够 |
| 平面划分(host plane / agent plane) | 单进程单用户场景不需要 |

**预估精简版规模:500~700 行。**

---

## 3. 热加载

### 3.1 统一入口

所有热加载走一个函数,保证可校验、可回滚、可审计:

```js
async function applyChange(patch) {
  const backup = snapshotState()
  try {
    validate(patch)                  // 校验
    await apply(patch)               // 应用(可能异步:连 MCP)
    runtime.version++                // 版本号
    emit('runtime/changed', patch)   // 通知 UI / 日志
    if (patch.persist) save(patch)   // 可选落盘
  } catch (e) {
    restore(backup)                  // 失败回滚,旧配置原样保留
    throw e
  }
}
```

**事务性要求**:新配置失败不得破坏旧配置。典型场景——热加载新 MCP 但连不上,必须"先建后换":新连接成功才替换,失败保留旧的继续跑。

### 3.2 每轮快照

避免竞态的关键。循环每轮开始读一次快照:

```js
async function runTurn(session, userMsg) {
  const snap = runtime.snapshot()        // 本轮冻结
  const tools = [...snap.tools.values()]
  const route = snap.modelRoute
  // 整轮使用 snap,不受并发热加载影响
}
```

**一轮之内配置不变,轮与轮之间可变。**

### 3.3 作用域

```js
applyChange({ scope: 'session', id, patch })   // 只对该会话生效(默认)
applyChange({ scope: 'global',      patch })   // 对所有会话生效
```

| 作用域 | 默认策略 | 说明 |
|---|---|---|
| `session` | 无需同意 | 整合包加载走这里,包 A 不污染包 B |
| `global` | **需要用户同意** | 影响所有会话,含以后新建的 |

### 3.4 同意逻辑(策略表)

```js
policy[patch.type] → 'auto-allow' | 'ask' | 'deny'
```

| 变更类别 | 默认策略 |
|---|---|
| 会话级任何变更 | `auto-allow`(不打扰) |
| 全局加技能 / 加 MCP | `ask` |
| 全局换模型路由(自己换 API) | `ask`(用户可改 auto) |
| 全局改人设 / 提示词 | `ask` |
| 全局降权限 / 关沙箱 | **`deny`**(不可自动同意) |

**`ask` 的交互**:

```
Agent 请求:全局添加 MCP 服务器「xxx」

影响:所有会话
缓存:会破坏当前前缀(约 12,400 tokens 需重新计费)

  [ 立即生效(牺牲缓存) ]   [ 下个会话生效(保留缓存) ]   [ 取消 ]
  [ ] 以后这类变更都自动立即生效
```

勾选"以后自动"→ 该类别加入 `auto-allow`,写入用户设置。

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
| **新鲜优先** `freshness` | 立即改写 prompt,不管缓存 | 干活要紧 |
| **平衡** `balanced`(默认) | 能追加就追加,结构性变更才改写 | 默认 |
| **省钱优先** `cache-first` | 一切走追加,易变内容推后 | 长对话、成本敏感 |

**第二层:按类别覆盖**

```yaml
cachePolicy:
  default: balanced
  byChange:
    model-route: freshness      # 换 API 必须立刻生效
    skill: balanced
    persona: freshness
    mcp: balanced
```

**第三层:单次强制覆盖**

```js
applyChange({ ..., cachePolicy: 'force-fresh' })   // 这次不管缓存
applyChange({ ..., cachePolicy: 'defer' })         // 攒到下个会话生效
```

### 4.3 追加式 vs 改写式

KV cache 是**前缀缓存**:前缀中任一 token 变化,从该位置起全部未命中。

**改写式**(无 `in-history` 能力时):
```
[0] system: 新提示词     ← 变了 → 整个请求都不同 → 全部未命中
[1..N] 历史消息
[N+1] user: 新消息
```

**追加式**(声明 `systemPromptUpdate: 'in-history'` 时):
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
- **要可配置**:模型目录里带能力标记(`systemPromptUpdate: 'in-history'`)
- **要有回退**:探测失败或报错时自动退回改写式,**不能因此让请求失败**
- **显式声明,不猜**(照抄 DSH 的做法)

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

```js
// 进程级基线(所有会话共享的起点)
const base = {
  tools: Map, mcp: Map, skillDirs: Set, modelRoute: {}, persona: '',
}

// 会话级覆盖(整合包加载到这里,不碰基线)
const session = {
  id,
  overlay: {
    tools:     { add: Map, remove: Set },
    mcp:       { add: Map, remove: Set },
    skillDirs: { add: Set, remove: Set },
    modelRoute: {...},
    persona:    '...',
  },
}

// 生效配置 = 基线 + 覆盖
function resolve(session) {
  return {
    tools: merge(base.tools, session.overlay.tools),
    mcp:   merge(base.mcp,   session.overlay.mcp),
    ...
  }
}
```

### 5.3 多设备与多标签(方案 C)

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
- **隧道**:多会话共用一条隧道,靠 `sessionId` 区分(现有帧多路复用已支持)

### 5.4 共享资源池

同一进程内,相同资源**只建一份**,按引用计数复用:

```js
mcpPool: Map<specHash, { conn, refCount: Set<sessionId> }>
// 会话加入 → refCount.add(id);离开 → refCount.delete(id);空集才真正关闭
```

模型客户端同理:按 `(provider, baseURL, apiKey)` 复用。

---

## 6. 模型层

### 6.1 只支持 OpenAI 格式

**砍掉多 provider 适配。** 参考对比:Pi 的 `ai` 包 179 文件 / 22.5k 行几乎全在适配 Anthropic / Google / Mistral / Azure / Bedrock 的差异。只做 OpenAI 格式可将其压缩到几百行。

### 6.2 中立消息格式

**存储用自有中立格式,请求时按当前 provider 转换**(参考 Pi 的 `convertToLlm`,只在 LLM 调用边界转换一次)。

这是**中途换模型的前提**——历史消息不能绑定某家 provider 的格式。

### 6.3 能力声明

模型目录携带能力标记,显式声明不猜:

```js
{
  id: 'xxx',
  systemPromptUpdate: 'in-history',   // 可选;存在时值必须精确匹配
  contextWindow: 128000,
  maxOutputTokens: 8192,
}
```

---

## 7. 插件规划

### 7.1 两类插件

| 类型 | 特征 | 例子 |
|---|---|---|
| **工具型** | 只给模型加工具 | 搜索、computer use、todo |
| **服务型** | 自己提供服务/长连接,可被其他插件依赖 | 手机远程控制、MCP 客户端、压缩器 |

服务型插件依赖 1.4 的"提供服务"扩展点。

### 7.2 四类首批插件

| 插件 | 类型 | 可行性 | 要点 |
|---|---|---|---|
| **搜索** | 工具型 | 完美 | 纯 HTTP,零障碍 |
| **computer use** | 工具型 | 可行 | 截图 + 鼠标键盘;依赖原生模块,跨平台分别处理 |
| **子代理** | 工具型 + 服务依赖 | 可行 | 需内核暴露"创建 agent / 会话"接口 |
| **手机远程控制** | **服务型** | 可行 | 需内核支持插件提供网络服务;复用现有 White 隧道 |

---

## 8. 工作区 git 自动快照

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

## 9. Yellow 整合包对接

Yellow 的理念与包格式**原样保留**(纯引用清单、sha1 去重、台账、版本兼容声明、协议标注),只改**落地层**。

| # | Yellow 现在调用 | 内核需要提供 |
|---|---|---|
| 1 | `dsh plugin add <pkg>` | `kernel.loadPlugin(name)` |
| 2 | `agentPreset.read` | `kernel.readPreset(id)` |
| 3 | `agentPreset.select` | `kernel.applyPreset(sessionId, id)` |
| 4 | `session.create` | `kernel.sessions.create({preset})` |
| 5 | `session.fork` | `kernel.sessions.fork(id)` |
| 6 | `dsh-skill-filesystem` 的 `customSkillDirs` | `kernel.registerSkillDir(dir)` |
| 7 | `dsh-mcp-client` 挂载行 | `kernel.addMCPServer(spec)` |
| 8 | `cordis.patch.yml` 补丁合并 | `kernel.patchComposition(rows)` |
| 9 | `settings.yaml` 的 `llm-pi-ai` 段 | `kernel.setModelRoute(route)` |

**改造方向**:从"写配置文件 + 建新会话"改成"调 `applyChange` 热加载"(作用域默认 `session`)。

---

## 10. 待定问题

1. **压缩(compaction)是否进内核?**
   - 进:长对话立即可用,但内核变大
   - 不进:作为服务型插件,但它是"每轮都要用"的核心路径
   - 倾向:**作为内置服务**,但通过插件机制挂载(可替换实现)
2. **会话持久化格式**:JSONL / SQLite / 两者?
3. **手机端(Blue)改造范围**:多标签 UI 由谁实现
4. **computer use 的跨平台方案**:原生模块选型
5. **内核语言**:TypeScript(与参考代码一致)还是其他

---

## 附:实现顺序

1. **内核骨架**:会话 + 循环 + 模型(OpenAI)+ 工具表 + 4 工具
2. **中立消息格式 + `convertToLlm`**(换 API 的前提,必须一开始就有)
3. **插件机制**:`isolate` + epoch + effect(500~700 行)
4. **`applyChange` + 每轮快照 + 作用域 + 同意逻辑**
5. **缓存策略三档 + prompt 分区**
6. **会话配置分层 + 共享资源池**
7. **工作区 git 快照服务**
8. **插件**:搜索 → 手机远程 → 子代理 → computer use
9. **Yellow 对接改造**
10. **UI 与远程**(复用现有隧道 / TUI)
