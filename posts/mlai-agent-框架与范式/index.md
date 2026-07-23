# AI Agent 框架与范式


## 1. 先建立正确的概念分层

面试中最容易混淆的是 **Agent 范式、系统架构和开发框架**。

| 层次 | 回答的问题 | 典型例子 |
|---|---|---|
| Agent 范式 | Agent 如何思考、行动、纠错 | ReAct、Plan-and-Execute、Reflexion、LATS、RAISE |
| 系统架构 | 多个组件或 Agent 如何分工、通信 | Supervisor-Worker、Planner-Executor、Debate、Swarm、状态图 |
| 开发框架 | 用什么工程工具把上述设计实现出来 | LangGraph、AutoGen、AgentScope、CAMEL |
| 基础能力 | Agent 可以依赖哪些底层能力 | Model、Tool、Memory、RAG、Guardrail、Evaluation |

一句话概括：

&gt; **ReAct 是决策范式，Supervisor-Worker 是协作架构，LangGraph 是编排框架，Tool Calling 是模型与工具交互的接口能力。**

同一种范式可以由不同框架实现，同一个框架也可以实现多种范式。例如，可以使用 LangGraph 实现 ReAct、Plan-and-Execute、Reflection 或多智能体工作流。

```mermaid
flowchart TB
    B[&#34;基础能力：Model / Tool / Memory / RAG&#34;] --&gt; P[&#34;Agent 范式：ReAct / LATS / RAISE&#34;]
    P --&gt; A[&#34;系统架构：单 Agent / 多 Agent / Workflow&#34;]
    A --&gt; F[&#34;开发框架：LangGraph / AutoGen / AgentScope / CAMEL&#34;]
    F --&gt; R[&#34;生产运行时：状态、持久化、观测、权限、评测&#34;]
```

---

## 2. 为什么需要 Agent 框架

不用框架也可以手写一个 Agent 循环：

```python
while not finished:
    action = llm.decide(state, tools)
    observation = execute(action)
    state = update(state, action, observation)
```

但从 Demo 走向生产后，需要解决的远不只是“让模型调用一次工具”。框架的价值主要有以下四点。

### 2.1 封装与抽象

框架通常会抽象出：

- Agent；
- Message；
- Model Client；
- Tool；
- Memory；
- State；
- Node / Edge；
- Runtime。

开发者不必重复实现消息格式、工具注册、模型适配、循环调度和异常处理。

### 2.2 解耦与可扩展

把模型、工具、记忆和编排逻辑解耦后，可以独立替换：

```text
Model：OpenAI / Claude / Qwen / 自建模型
Tool：搜索 / 数据库 / 浏览器 / 业务 API
Memory：对话窗口 / 摘要 / 向量库 / 结构化存储
Workflow：ReAct / DAG / 状态图 / 多智能体协作
```

这能够降低模型迁移、工具扩展和业务流程变化的成本。

### 2.3 标准化状态管理

复杂 Agent 不是一个普通聊天接口，而是一个有状态、长生命周期的任务系统。框架需要帮助管理：

- 当前任务进行到哪一步；
- 已调用过哪些工具；
- 哪些中间结果需要保留；
- 失败后从哪里恢复；
- 人工审批后如何继续；
- 多个 Agent 如何共享或隔离状态；
- 如何避免并发写入冲突和重复执行。

生产环境尤其关注 **Checkpoint、持久化、幂等、重试、超时、取消和恢复**。

### 2.4 可观测性、调试与评测

框架可以统一记录：

```text
trace_id / step_id
模型及 Prompt 版本
节点输入和输出
工具名称、参数和结果
Token、延迟和费用
错误类型与重试次数
人工介入点
最终答案和引用证据
```

这使开发者能够区分：模型决策错误、工具路由错误、参数错误、工具执行失败、状态污染、停止条件错误或答案生成错误。

### 2.5 面试中的高级补充

&gt; 框架的核心价值不是减少几行调用模型的代码，而是把不稳定的模型能力放入一个可控制、可恢复、可观测、可评测的工程运行时中。

同时也要指出，框架不是越重越好：

- 单轮问答或固定两三步流程，普通函数和显式代码往往更简单；
- 只有当任务存在动态分支、循环、持久化、人工介入或多智能体协作时，才更值得引入完整框架；
- 抽象层越多，调试成本、版本迁移成本和供应商绑定风险也越高。

---

## 3. 图片中的四种 Agent 框架

&gt; 注意：框架迭代较快，具体 API 可能变化。面试中应重点讲设计思想、适用场景和取舍，不要只背 API。

### 3.1 总览对比

| 框架 | 核心抽象 | 主要优势 | 典型场景 | 主要代价 |
|---|---|---|---|---|
| AutoGen | 会话式 Agent、消息和多 Agent 对话 | 多角色协作自然，适合对话驱动的任务分解 | 代码生成与评审、研究团队、模拟产品开发流程 | 对话轮数易膨胀，终止条件和成本控制较难 |
| AgentScope | 消息传递、Agent Runtime、分布式执行 | 工程化和部署能力较强，适合大规模多 Agent 应用 | 企业级多 Agent、分布式任务、复杂业务系统 | 系统概念较多，简单场景可能偏重 |
| CAMEL | Role Playing、Communicative Agents | 角色设定清晰，较容易构造自主协作和研究实验 | 创意生成、专家协作、任务模拟、合成数据 | 角色可能漂移，协作结果依赖 Prompt 与终止设计 |
| LangGraph | State、Node、Edge、Graph | 分支、循环、持久化和人工介入表达清晰 | 复杂工作流、Reflection、长任务、Human-in-the-loop | 需要显式设计状态和图，前期建模成本较高 |

### 3.2 AutoGen：以对话组织多智能体协作

核心思想是让不同职责的 Agent 通过消息对话推进任务，例如：

```text
User Proxy → Planner → Coder → Reviewer → Executor
                     ↑                   ↓
                     └──── 修改反馈 ─────┘
```

适合：

- 任务天然可以表达成多个角色之间的讨论；
- 希望快速构建代码生成、评审、研究和专家协作原型；
- 不同 Agent 使用不同模型、工具或系统提示词。

需要重点治理：

- 谁拥有最终决策权；
- 消息应该发给谁；
- 何时结束对话；
- 如何避免两个 Agent 相互附和或无限争论；
- 工具执行和危险操作由谁审批。

### 3.3 AgentScope：面向工程化的消息驱动多 Agent

图片中将其核心概念概括为“消息传递”。更完整地说，它强调 Agent 之间的消息通信、运行时管理和可扩展部署。

适合：

- 需要构建较复杂、可运维的多 Agent 应用；
- Agent 数量较多或存在分布式执行需求；
- 对日志、监控、消息路由和工程扩展性要求较高。

面试时不要只说“支持分布式”，还要说明分布式带来的新问题：

- 消息顺序和重复消费；
- 超时与部分失败；
- 状态一致性；
- 任务重试的幂等性；
- 跨 Agent Trace 关联；
- 共享记忆的并发读写。

### 3.4 CAMEL：通过角色扮演驱动自主协作

CAMEL 关注 Role Playing。通常先定义多个角色、目标和协作规则，再让 Agent 通过对话完成任务。

例如：

```text
AI 用户：提出需求、约束和验收标准
AI 助手：分解任务、执行并交付
Critic：检查事实、逻辑和格式
```

适合：

- 探索性问题；
- 创意生成；
- 模拟某领域的专家协作；
- 多 Agent 研究和合成数据生成。

主要风险是 **Role Confusion / Role Drift**：

- Agent 忘记自己的职责；
- 两个角色开始做相同的事；
- 角色目标与全局任务冲突；
- 对话很热闹，但没有产生可验证的任务进展。

解决思路包括角色权限约束、结构化交接协议、共享任务板、明确验收标准和最大轮数。

### 3.5 LangGraph：用状态图表达复杂 Agent 工作流

LangGraph 的核心不是“画图”，而是把 Agent 明确建模为：

```text
共享状态 State
&#43; 执行节点 Node
&#43; 路由和条件边 Edge
&#43; Checkpoint / Interrupt
```

示例：

```mermaid
flowchart LR
    S[&#34;用户请求&#34;] --&gt; P[&#34;Planner&#34;]
    P --&gt; E[&#34;Executor&#34;]
    E --&gt; C[&#34;Critic&#34;]
    C --&gt;|&#34;通过&#34;| F[&#34;Final&#34;]
    C --&gt;|&#34;需修改&#34;| E
    E --&gt;|&#34;高风险操作&#34;| H[&#34;人工审批&#34;]
    H --&gt; E
```

适合：

- 存在循环和条件分支；
- 需要持久化长任务；
- 需要人工审批或暂停恢复；
- 需要精确控制每一步，而不是让多 Agent 自由聊天；
- 希望显式实现 Reflection、Planner-Executor 或 Supervisor 架构。

图片中“天然支持循环和条件分支、精准控制复杂工作流”的判断是 LangGraph 最重要的面试关键词。

---

## 4. 四种框架应该怎么选

不要回答“哪个框架最好”，而要先问任务的主矛盾是什么。

| 需求 | 优先考虑 | 原因 |
|---|---|---|
| 多角色以对话方式协作 | AutoGen / CAMEL | 对话与角色是一级抽象 |
| 探索角色扮演、专家协作和模拟 | CAMEL | Role Playing 设计突出 |
| 大规模、多 Agent、强调部署和运行时 | AgentScope | 工程化消息机制与扩展性更重要 |
| 强状态、复杂分支、循环、持久化 | LangGraph | 状态图更可控、更容易恢复 |
| 固定且简单的业务流程 | 普通代码 / Workflow | 不一定需要完整 Agent 框架 |
| 高风险生产操作 | 显式 Workflow &#43; 审批 | 确定性和权限边界比自主性更重要 |

选型时至少评估以下维度：

1. **控制流**：线性、DAG、循环，还是开放式对话？
2. **状态**：是否跨轮、跨会话、跨进程持久化？
3. **通信**：共享状态、点对点消息，还是发布订阅？
4. **可靠性**：是否需要重试、恢复、幂等、补偿事务？
5. **人工介入**：哪些动作必须审批？
6. **观测与评测**：能否获得完整 Trace、Token、延迟与结果指标？
7. **生态集成**：模型、工具、MCP、数据库和部署环境是否匹配？
8. **团队成本**：学习曲线、API 稳定性、测试难度和供应商绑定。

### 选型回答模板

&gt; 我不会先按框架热度选型，而会先判断控制流和状态复杂度。如果是开放式多角色对话，我会考虑 AutoGen 或 CAMEL；如果是强状态、分支、循环、人工审批和断点恢复，我更倾向 LangGraph；如果强调大规模多 Agent 的消息通信和部署，会评估 AgentScope。对于固定流程，我会优先使用普通代码或工作流，避免为了 Agent 而 Agent。

---

## 5. 除了 ReAct，还有哪些 Agent 范式

### 5.1 ReAct：边推理、边行动、边观察

```text
Thought → Action → Observation → Thought → ... → Finish
```

优势是实现简单、能动态适应环境；不足是单路径局部决策容易陷入局部最优，且多轮调用的延迟和 Token 成本较高。

### 5.2 Plan-and-Execute：先规划，再执行

```mermaid
flowchart LR
    Q[&#34;目标&#34;] --&gt; P[&#34;生成全局计划&#34;]
    P --&gt; E[&#34;逐步执行&#34;]
    E --&gt; V[&#34;检查结果&#34;]
    V --&gt;|&#34;计划仍有效&#34;| E
    V --&gt;|&#34;环境变化或失败&#34;| R[&#34;重新规划&#34;]
    R --&gt; E
```

适合长任务、步骤依赖明确的场景。与 ReAct 相比：

- ReAct 更像短视的闭环控制，每一步根据最新观察决定下一步；
- Plan-and-Execute 先建立全局结构，执行器再完成局部步骤；
- 生产中经常组合为“Planner 生成计划，Executor 在每一步内部使用 ReAct”。

风险是初始计划可能基于错误假设，因此需要进度检查和动态重规划。

### 5.3 Reflection / Reflexion / Self-Refine：生成后复盘纠错

基本流程：

```text
Draft / Trajectory → Critique → Revision → Evaluation
                              ↑              ↓
                              └── 未通过 ────┘
```

它不是简单地让模型说一句“请再检查”，而应提供明确的评判标准和可验证反馈，例如：

- 单元测试是否通过；
- 引用能否支撑结论；
- SQL 是否只读；
- 输出是否满足 Schema；
- 任务目标是否全部覆盖。

优点是能提高复杂生成任务的质量；缺点是模型既当运动员又当裁判时可能看不出自己的盲点，还会增加调用成本。

### 5.4 Tree of Thoughts：探索多条推理路径

CoT 通常沿一条推理路径向前，而 Tree of Thoughts 会生成多个候选思路，并对候选进行评分、剪枝、回溯或继续展开。

```text
                初始问题
             /     |      \
          思路 A  思路 B   思路 C
           / \      |       / \
         A1  A2     B1     C1  C2
```

适合：搜索空间较大、早期选择会显著影响最终结果、存在明确评价标准的任务。

代价是模型调用次数和搜索成本明显高于单路径 ReAct。

### 5.5 LATS：Language Agent Tree Search

LATS 将 **Reasoning、Acting、Planning、环境反馈、价值评估和自我反思** 统一到树搜索中，论文实现借鉴了 Monte Carlo Tree Search。

与 ReAct 的关键区别：

| ReAct | LATS |
|---|---|
| 通常只维护一条当前轨迹 | 同时探索多条候选轨迹 |
| 当前动作失败后在原路径上修补 | 可以回溯并转向其他分支 |
| 依赖局部下一步判断 | 使用价值评估指导全局搜索 |
| 成本较低、实现简单 | 搜索质量更高，但调用成本大 |

适合代码生成、交互式问答、网页操作、数学问题等有环境反馈或评价函数的复杂任务。

面试中的一句话：

&gt; ReAct 是单轨迹的在线决策，LATS 则把 Agent 的轨迹当作搜索树，通过环境反馈、价值函数和反思选择更优路径，从而降低单路径决策陷入局部最优的风险。

### 5.6 RAISE：带双层记忆的 ReAct 增强架构

RAISE 全称为 **Reasoning and Acting through Scratchpad and Examples**。它面向多轮对话 Agent，在 ReAct 基础上引入类似人类短期记忆和长期记忆的双组件记忆系统。

可以抽象为：

```mermaid
flowchart LR
    U[&#34;当前输入&#34;] --&gt; W[&#34;短期记忆 / Scratchpad&#34;]
    L[&#34;长期记忆 / Examples&#34;] --&gt; W
    W --&gt; D[&#34;Reason &#43; Act&#34;]
    D --&gt; O[&#34;回复或行动&#34;]
    O --&gt; M[&#34;筛选、总结、写入记忆&#34;]
    M --&gt; L
```

它重点解决的是长对话中的上下文连续性、角色一致性和经验复用。

需要注意：

- Memory 不等于把全部聊天记录塞回上下文；
- 长期记忆需要解决写入、检索、更新、冲突、遗忘和权限问题；
- 错误记忆会让 Agent 在未来持续犯错；
- 角色设定和检索到的示例冲突时，可能出现 Role Confusion。

### 5.7 RAG / Memory-Augmented Agent

RAG 本身通常不是完整 Agent 范式，而是一种外部知识增强能力。只有当模型能够自主决定“是否检索、检索什么、如何验证、是否继续检索”时，才形成 Agentic RAG。

典型流程：

```text
Query Analysis
→ Query Rewrite / Decomposition
→ Retrieval
→ Rerank
→ Evidence Check
→ 必要时再次检索
→ Answer with Citations
```

### 5.8 Workflow / State Machine

并不是所有问题都应该交给自由决策的 Agent。对于财务审批、工单流转、订单操作等强约束任务，更适合：

- 用确定性状态机定义主流程；
- 只在文本理解、信息抽取、候选生成等局部节点使用 LLM；
- 关键操作通过规则校验和人工审批。

这是大厂生产落地中非常重要的观点：

&gt; 自主性应该按风险逐步开放。能用确定性代码解决的部分，不要全部交给模型决定。

---

## 6. 常见多智能体架构

### 6.1 Supervisor-Worker

Supervisor 负责拆解任务、分派 Worker、汇总结果和判断结束。

```mermaid
flowchart TB
    S[&#34;Supervisor&#34;] --&gt; R[&#34;Researcher&#34;]
    S --&gt; C[&#34;Coder&#34;]
    S --&gt; A[&#34;Analyst&#34;]
    R --&gt; S
    C --&gt; S
    A --&gt; S
```

优点是职责和控制权清晰；风险是 Supervisor 成为性能瓶颈和单点故障。

### 6.2 Planner-Executor-Critic

- Planner：分解任务并维护计划；
- Executor：调用工具完成步骤；
- Critic：根据标准检查质量；
- 必要时返回 Planner 重规划。

它适合代码开发、Deep Research、数据分析等需要“计划—执行—验收”的任务。

### 6.3 Debate / Generator-Critic

多个 Agent 提出不同候选或互相质疑，最后由 Judge 汇总。

适合高价值决策和复杂推理，但必须防止：

- 多个 Agent 使用相同模型和上下文，产生高度相关的错误；
- 对话轮数增加却没有新增证据；
- Judge 只偏好表达更自信的答案。

### 6.4 Blackboard / Shared Memory

多个 Agent 不直接长时间对话，而是读写共享任务板：

```text
任务列表 / 当前状态 / 中间证据 / 待验证结论 / 最终产物
```

优点是减少点对点通信复杂度；难点是并发修改、冲突解决、数据权限和脏状态清理。

### 6.5 Handoff / Swarm

当前 Agent 根据任务类型把控制权移交给更专业的 Agent，例如售前 Agent 将退款问题移交给售后 Agent。

关键不是“有很多 Agent”，而是明确：

- 路由条件；
- 交接上下文；
- 权限边界；
- 控制权归属；
- 失败时的回退路径。

---

## 7. 面试题：除了 ReAct，还有什么新范式或者架构

### 7.1 90 秒标准回答

&gt; 我会先区分范式、架构和框架。ReAct 是单 Agent 的推理—行动范式，不是具体开发框架。除了 ReAct，单 Agent 侧常见的有 Plan-and-Execute、Reflection/Reflexion、Tree of Thoughts、LATS 和带长期记忆的 RAISE。Plan-and-Execute 适合长任务，先做全局规划再执行；Reflection 通过 Critic 或环境反馈反复修正；LATS 把多条 Agent 轨迹组织成树搜索，能回溯并减少局部最优；RAISE 在 ReAct 上加入短期和长期记忆，更关注长对话连续性。系统架构侧还有 Supervisor-Worker、Planner-Executor-Critic、Debate、共享黑板和 Handoff。工程实现可以使用 LangGraph、AutoGen、AgentScope 或 CAMEL。生产选型的关键不是范式越新越好，而是根据任务的不确定性、搜索空间、时延成本、风险等级和是否存在可验证反馈来选择。

### 7.2 追问：LATS 为什么可能比 ReAct 好

回答要点：

1. ReAct 通常只沿一条路径连续决策；
2. 一旦早期动作选错，后续容易在错误路径上修补；
3. LATS 保留多个候选分支，可以评估、剪枝和回溯；
4. 它结合环境反馈、价值判断和反思，搜索更全面；
5. 代价是调用次数、延迟和 Token 显著增加；
6. 如果没有可信评价函数，树搜索也可能只是昂贵地探索错误方向。

### 7.3 追问：Memory 怎么设计

可以按以下维度回答：

| 类型 | 内容 | 生命周期 | 常见实现 |
|---|---|---|---|
| Working Memory | 当前目标、计划、最近观察、中间变量 | 单次任务 | State / Context |
| Episodic Memory | 历史任务、成功或失败轨迹 | 跨任务 | 向量库 &#43; 元数据 |
| Semantic Memory | 用户事实、业务知识、稳定偏好 | 长期 | KV / 数据库 / 知识图谱 |
| Procedural Memory | 工具使用方式、策略、规则 | 长期 | Prompt、Skill、Policy |

完整的 Memory Pipeline 应包括：

```text
Write → Filter → Summarize → Store → Retrieve → Rerank
      → Conflict Resolution → Update / Forget → Access Control
```

高分回答：

&gt; 记忆系统的难点不是存储，而是决定什么值得写入、什么时候检索、如何解决冲突和过期信息，以及如何防止错误或恶意内容长期污染 Agent。

### 7.4 追问：什么时候不应该用多 Agent

- 一个强模型配合工具即可完成；
- 任务无法自然分解；
- Agent 之间没有信息或能力差异；
- 时延和成本要求严格；
- 缺乏可靠的汇总和验收机制；
- 多 Agent 只是在重复生成相似文本。

多 Agent 的价值应来自 **上下文隔离、角色专业化、并行执行、权限隔离或相互验证**，而不是来自 Agent 数量本身。

---

## 8. 生产级 Agent 的核心模块

一个真正可上线的 Agent 系统通常包含：

```mermaid
flowchart TB
    U[&#34;用户 / 上游系统&#34;] --&gt; G[&#34;Gateway：鉴权、限流、会话&#34;]
    G --&gt; O[&#34;Orchestrator：状态与控制流&#34;]
    O --&gt; M[&#34;Model Router&#34;]
    O --&gt; T[&#34;Tool Runtime&#34;]
    O --&gt; R[&#34;Memory / RAG&#34;]
    T --&gt; P[&#34;权限、Schema、沙箱、审批&#34;]
    O --&gt; C[&#34;Checkpoint / Queue&#34;]
    O --&gt; V[&#34;Tracing / Evaluation&#34;]
    V --&gt; D[&#34;数据集、指标、回归测试&#34;]
```

### 8.1 Tool Runtime

- JSON Schema 参数校验；
- 工具权限最小化；
- 超时、重试、熔断和限流；
- 读写工具分级；
- 高风险操作二次确认；
- 幂等键与重复执行保护；
- 工具结果长度限制和内容清洗。

### 8.2 状态和持久化

- Task State 与 Chat History 分离；
- 每个节点执行前后保存 Checkpoint；
- 失败后从安全位置恢复；
- 长任务支持暂停、取消和人工接管；
- 状态迁移需要版本管理。

### 8.3 Guardrail 与安全

- 防止直接和间接 Prompt Injection；
- 工具返回内容按不可信数据处理；
- 敏感信息脱敏；
- 用户、Agent、工具分别做权限校验；
- 输出做事实、格式、合规和风险检查；
- 删除、付款、发信等动作进入审批流。

### 8.4 评测体系

至少同时评估：

| 维度 | 示例指标 |
|---|---|
| 最终效果 | Task Success、准确率、人工评分 |
| 过程质量 | 工具选择准确率、参数正确率、步骤完成率 |
| 事实可靠性 | Citation Correctness、Groundedness |
| 效率 | Token、模型调用次数、工具调用次数、P95 延迟、成本 |
| 稳定性 | 超时率、重试率、循环率、恢复成功率 |
| 安全性 | 越权率、危险动作拦截率、注入攻击成功率 |

只看最终答案会掩盖过程问题。例如 Agent 可能最终答对，但调用了错误工具、泄露了数据或进行了大量无效搜索。

---

## 9. 容易失分的说法

### 错误 1：LangGraph 是一种新的推理范式

更准确的说法：LangGraph 是状态图编排框架，可以承载多种推理范式和多 Agent 架构。

### 错误 2：Function Calling 就是 Agent

Function Calling 只是结构化表达工具调用的接口。Agent 还需要状态、循环、规划、反馈、停止条件和运行时治理。

### 错误 3：Memory 就是向量数据库

向量数据库只是长期记忆的一种检索实现。Memory 还涉及写入策略、摘要、冲突、遗忘、权限和生命周期。

### 错误 4：多 Agent 一定比单 Agent 强

如果 Agent 之间没有差异化能力和有效协作机制，多 Agent 只会增加延迟、成本和错误传播路径。

### 错误 5：框架自动解决了可靠性问题

框架提供机制，但重试、幂等、权限、评测、回滚和停止条件仍然需要业务侧明确设计。

### 错误 6：把完整思维链作为可解释性

生产系统更应记录结构化的计划摘要、Action、Observation、引用、状态变化和评测结果，而不是依赖展示模型的完整内部推理。

---

## 10. 一页速记

### 为什么需要 Agent 框架

```text
封装抽象
&#43; Model / Tool / Memory 解耦
&#43; 状态与流程标准化
&#43; 可观测、调试、评测
&#43; 持久化、恢复和人工介入
```

### 四个框架关键词

```text
AutoGen    = 对话式多 Agent
AgentScope = 消息驱动与工程化运行时
CAMEL      = Role Playing
LangGraph  = State Graph &#43; 分支/循环/持久化
```

### ReAct 之外的范式

```text
Plan-and-Execute = 先全局规划，再执行与重规划
Reflection       = 生成、批评、修改
ToT              = 多候选推理树
LATS             = Agent 轨迹树搜索 &#43; 环境反馈 &#43; 价值评估
RAISE            = ReAct &#43; 短期/长期记忆
```

### 多 Agent 架构

```text
Supervisor-Worker
Planner-Executor-Critic
Debate / Generator-Critic
Blackboard / Shared Memory
Handoff / Swarm
```

### 生产落地关键词

```text
State / Checkpoint / Idempotency
Timeout / Retry / Circuit Breaker
Permission / Sandbox / Human Approval
Tracing / Evaluation / Regression
Prompt Injection / Memory Poisoning
Latency / Token / Cost
```

---

## 11. 参考资料

- ReAct: Synergizing Reasoning and Acting in Language Models  
  &lt;https://arxiv.org/abs/2210.03629&gt;
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models  
  &lt;https://arxiv.org/abs/2305.10601&gt;
- Reflexion: Language Agents with Verbal Reinforcement Learning  
  &lt;https://arxiv.org/abs/2303.11366&gt;
- Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models  
  &lt;https://arxiv.org/abs/2310.04406&gt;
- From LLM to Conversational Agent: A Memory Enhanced Architecture with Fine-Tuning of Large Language Models（RAISE）  
  &lt;https://arxiv.org/abs/2401.02777&gt;
- AutoGen  
  &lt;https://github.com/microsoft/autogen&gt;
- AgentScope  
  &lt;https://github.com/agentscope-ai/agentscope&gt;
- CAMEL  
  &lt;https://github.com/camel-ai/camel&gt;
- LangGraph  
  &lt;https://github.com/langchain-ai/langgraph&gt;


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlai-agent-%E6%A1%86%E6%9E%B6%E4%B8%8E%E8%8C%83%E5%BC%8F/  

