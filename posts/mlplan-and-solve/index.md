# Plan-and-Solve


## 1. 一句话理解 Plan-and-Solve

Plan-and-Solve 是一种把复杂任务拆成两个阶段的 Agent 范式：

1. **规划阶段**：把原始问题分解成有顺序、可执行的子任务；
2. **执行阶段**：逐步执行子任务，并把前序结果作为后续步骤的上下文；
3. **可选的重规划阶段**：当计划不可执行、外部环境变化或中间结果与预期不一致时，调整剩余计划。

```text
直接问答：      Question ───────────────→ Answer

Plan-and-Solve：Question → Plan → Step 1 → Step 2 → ... → Final Answer
                              ↑                 │
                              └──── Replan ─────┘
```

它解决的核心问题不是“让模型输出更长的思维过程”，而是：

&gt; 将复杂问题显式分解，降低单次推理负担，让执行过程更可控、可观察、可纠错。

---

## 2. 抽象模型与架构

### 2.1 规划阶段

设用户问题为 `q`，规划器 `π` 生成包含 `n` 个子任务的计划：

```text
P = π(q) = [p₁, p₂, ..., pₙ]
```

好的子任务通常满足：

- **可执行**：不是“分析一下”，而是能产生明确结果；
- **顺序正确**：后续步骤依赖的数据必须已经生成；
- **粒度适中**：不能粗到仍然无法执行，也不能细到造成大量无意义调用；
- **覆盖完整**：所有步骤组合起来能够回答原始问题；
- **终止明确**：最后需要有汇总、验证或交付步骤。

### 2.2 执行阶段

第 `i` 步的执行结果不仅依赖当前子任务，还可能依赖原始问题、完整计划和历史结果：

```text
sᵢ = τ(q, P, pᵢ, s₁, s₂, ..., sᵢ₋₁)
```

其中：

- `τ` 是执行器；
- `pᵢ` 是当前步骤；
- `s₁ ... sᵢ₋₁` 是已完成步骤的结果。

Notebook 中正是通过 `history` 字符串，把前序结果传给下一次模型调用。

### 2.3 完整的工程化结构

```mermaid
flowchart LR
    U[&#34;User Request&#34;] --&gt; P[&#34;Planner&#34;]
    P --&gt; PL[&#34;Structured Plan&#34;]
    PL --&gt; E[&#34;Task Executor&#34;]
    E --&gt; T[&#34;Tools / External Environment&#34;]
    T --&gt; E
    E --&gt; V[&#34;Verifier / Evaluator&#34;]
    V --&gt;|&#34;通过&#34;| F[&#34;Finalizer&#34;]
    V --&gt;|&#34;失败、信息不足或环境变化&#34;| R[&#34;Replanner&#34;]
    R --&gt; PL
    F --&gt; A[&#34;Final Answer&#34;]
```

课程图片展示的是带 `Replan` 的完整 Plan-and-Solve 闭环；当前 Notebook 实际实现的是简化版：

```text
Planner → Executor（串行执行）→ 返回最后一步结果
```

它**没有真正实现 Replanner、工具调用、结果校验和最终答案合成器**。这是理解代码时最重要的边界。

---

## 3. Notebook 整体结构

| 模块 | 主要职责 | 输入 | 输出 |
|---|---|---|---|
| `LLMProxy` | 封装兼容 OpenAI 接口的模型调用 | `messages` | 模型文本 |
| `Planner` | 将复杂问题拆成步骤列表 | 原始问题 | `list[str]` |
| `Executor` | 串行执行每个步骤 | 问题、计划、历史 | 最后一步文本 |
| `PlanAndSolveAgent` | 编排规划和执行流程 | 原始问题 | 打印最终结果 |

调用链如下：

```mermaid
sequenceDiagram
    participant U as User
    participant A as PlanAndSolveAgent
    participant P as Planner
    participant E as Executor
    participant L as LLMProxy

    U-&gt;&gt;A: run(question)
    A-&gt;&gt;P: plan(question)
    P-&gt;&gt;L: think(planner_messages)
    L--&gt;&gt;P: Python 列表文本
    P--&gt;&gt;A: plan: list[str]

    loop 对每个步骤
        A-&gt;&gt;E: execute(question, plan)
        E-&gt;&gt;L: think(question &#43; plan &#43; history &#43; current_step)
        L--&gt;&gt;E: 当前步骤结果
        E-&gt;&gt;E: 追加到 history
    end

    E--&gt;&gt;A: final_answer
    A--&gt;&gt;U: 打印最终答案
```

---

## 4. `LLMProxy`：模型访问层

`llm_client.py` 负责把底层模型服务封装为统一接口：

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=temperature,
    stream=True,
)
```

主要特点：

- 从参数或 `.env` 加载 `MODEL_ID`、`API_KEY`、`BASE_URL` 和 `TIMEOUT`；
- 使用兼容 OpenAI Chat Completions 的接口；
- 默认 `temperature=0`，减少计划格式和答案的随机漂移；
- 使用流式响应，同时收集完整文本；
- 模型调用失败时捕获异常并返回 `None`。

这一层只负责“请求模型”，并不知道 Planner、Executor 或 Agent 的业务语义，符合基础的分层思想。

### 面试中的工程追问

当前异常处理策略偏弱：`think()` 失败后返回 `None`，上层通过 `or &#34;&#34;` 把它变成空字符串。Planner 会解析失败并终止，但 Executor 会把空结果写入历史并继续执行。

生产环境更合理的做法是：

- 区分超时、限流、服务端错误和不可重试错误；
- 对可重试错误使用指数退避与抖动；
- 为每次调用设置 request ID、耗时、Token 数和模型版本；
- 达到重试上限后把步骤标记为 `FAILED`，交给 Replanner 决定重试、替代或终止；
- 不要用空字符串掩盖底层失败。

---

## 5. Planner 源码拆解

### 5.1 Planner Prompt

Planner 的任务约束是：

```text
复杂问题
  ↓
多个简单、独立、可执行、按逻辑顺序排列的子任务
  ↓
Python list[str]
```

关键提示词包括：

- 设定“AI 规划专家”角色；
- 要求子任务简单且可执行；
- 要求严格按逻辑顺序；
- 强制输出被 ` ```python ` 包裹的 Python 列表。

示例问题生成的计划为：

```python
[
    &#34;计算周二卖出的苹果数量：15 * 2&#34;,
    &#34;根据周二的结果，计算周三卖出的苹果数量：周二的数量 - 5&#34;,
    &#34;将周一、周二和周三卖出的苹果数量相加得到总销售量&#34;,
]
```

这份计划体现出显式依赖关系：

```text
周一数量 15
   ↓
计算周二 30
   ↓
计算周三 25
   ↓
汇总 15 &#43; 30 &#43; 25 = 70
```

### 5.2 输出解析

```python
plan_str = response_text.split(&#34;```python&#34;)[1].split(&#34;```&#34;)[0].strip()
plan = ast.literal_eval(plan_str)
```

`ast.literal_eval` 相比直接使用 `eval` 安全得多，因为它只解析 Python 字面量，不会执行任意代码。

但这套解析仍然脆弱：

- 模型漏写 Markdown 代码块会触发 `IndexError`；
- 输出 `json` 代码块而非 `python` 代码块会失败；
- 列表前后多输出一个代码块，可能截取错误；
- 只检查“是不是列表”，没有检查元素是否都是非空字符串；
- 没有限制最大步骤数；
- 没有表达步骤 ID、依赖关系、工具、完成标准或执行状态。

### 5.3 更适合生产环境的计划结构

不要把协议建立在 Markdown 文本切割上。优先使用原生 Structured Output / JSON Schema：

```json
{
  &#34;goal&#34;: &#34;计算三天苹果总销量&#34;,
  &#34;steps&#34;: [
    {
      &#34;id&#34;: &#34;step_1&#34;,
      &#34;description&#34;: &#34;计算周二销量&#34;,
      &#34;dependencies&#34;: [],
      &#34;tool&#34;: &#34;calculator&#34;,
      &#34;success_criteria&#34;: &#34;得到一个非负整数&#34;,
      &#34;status&#34;: &#34;pending&#34;
    },
    {
      &#34;id&#34;: &#34;step_2&#34;,
      &#34;description&#34;: &#34;计算周三销量&#34;,
      &#34;dependencies&#34;: [&#34;step_1&#34;],
      &#34;tool&#34;: &#34;calculator&#34;,
      &#34;success_criteria&#34;: &#34;得到一个非负整数&#34;,
      &#34;status&#34;: &#34;pending&#34;
    }
  ]
}
```

结构化计划的价值：

- 可以校验字段和类型；
- 可以表示 DAG 依赖和并行执行；
- 可以单独重试失败步骤；
- 可以持久化、恢复和审计；
- 可以计算步骤级成功率与成本；
- 更容易接入工作流引擎。

---

## 6. Executor 源码拆解

### 6.1 执行器获得的上下文

每一步 Prompt 包含：

```text
原始问题 &#43; 完整计划 &#43; 历史步骤与结果 &#43; 当前步骤
```

对应代码：

```python
prompt = EXECUTOR_PROMPT_TEMPLATE.format(
    question=question,
    plan=plan,
    history=history if history else &#34;无&#34;,
    current_step=step,
)
```

这使模型在执行当前步骤时仍能看到全局目标，也能消费前序结果。

### 6.2 串行执行与状态累积

```python
for i, step in enumerate(plan, 1):
    response_text = self.llm_client.think(messages=messages) or &#34;&#34;
    history &#43;= f&#34;步骤 {i}: {step}\n结果: {response_text}\n\n&#34;
    final_answer = response_text
```

当前实现有三个重要特征：

1. **严格串行**：每一步完成后才执行下一步；
2. **全量历史回填**：之前所有步骤和结果都进入下一步 Prompt；
3. **最后一步即最终答案**：循环结束时直接返回最后一次模型输出。

### 6.3 “最后一步结果”不一定等于“最终答案”

当前代码隐含了一个很强的假设：Planner 一定会把最后一步设计为最终汇总。

如果计划是：

```python
[&#34;搜索资料 A&#34;, &#34;搜索资料 B&#34;, &#34;比较 A 和 B&#34;]
```

最后一步可能只给出比较结果，却未按用户要求形成完整报告、附带来源或遵守输出格式。

更稳健的架构应增加 `Finalizer`：

```text
步骤执行结果集合
    ↓
Finalizer：重新对齐原始问题、证据和交付格式
    ↓
最终答案
```

Finalizer 至少需要检查：

- 是否覆盖用户的所有子问题；
- 是否引用了必要证据；
- 是否满足格式、语言和长度要求；
- 是否存在中间结论互相冲突；
- 是否把未验证内容误写成事实。

### 6.4 历史状态的局限

当前 `history` 是普通字符串，适合教学，但不适合复杂任务：

- 无法可靠查询某一步的状态；
- 无法区分模型答案、工具观察、错误信息和人工反馈；
- 上下文会随步骤数线性增长；
- 某一步输出过长时容易超过上下文窗口；
- 外部网页内容直接拼接时会引入 Prompt Injection 风险；
- 失败后无法从检查点恢复。

更合理的状态可以定义为：

```python
class StepResult(BaseModel):
    step_id: str
    status: Literal[&#34;pending&#34;, &#34;running&#34;, &#34;succeeded&#34;, &#34;failed&#34;, &#34;skipped&#34;]
    output: Any | None
    error: str | None
    evidence: list[str]
    started_at: datetime | None
    finished_at: datetime | None
    retry_count: int = 0
```

---

## 7. Agent 编排逻辑

`PlanAndSolveAgent` 负责组合 Planner 和 Executor：

```python
plan = self.planner.plan(question)
if not plan:
    return
final_answer = self.executor.execute(question, plan)
```

它体现了最基础的 Agent Runtime 思想：

- LLM 不是完整的 Agent；
- Planner 和 Executor 是由 LLM 驱动的能力模块；
- Python 宿主程序负责状态、控制流、错误处理和终止条件；
- 真正的 Agent 是“模型 &#43; Prompt &#43; 状态 &#43; 工具 &#43; 控制循环”的组合。

当前 `run()` 只打印结果，没有返回值。这会降低可组合性和可测试性。工程中通常应返回结构化运行结果：

```python
class AgentResult(BaseModel):
    run_id: str
    status: str
    plan: Plan
    step_results: list[StepResult]
    final_answer: str | None
    usage: Usage
```

---

## 8. 示例运行全过程

原始问题：

&gt; 周一卖出 15 个苹果；周二是周一的两倍；周三比周二少 5 个。三天共卖出多少？

执行轨迹：

| 阶段 | 当前任务 | 可用历史 | 输出 |
|---|---|---|---|
| Plan | 拆解问题 | 原始问题 | 3 个步骤 |
| Step 1 | `15 × 2` | 无 | 周二 `30` |
| Step 2 | `周二 - 5` | Step 1 = 30 | 周三 `25` |
| Step 3 | 汇总三天销量 | Step 1 = 30，Step 2 = 25 | `70` |

最终结果：

```text
15 &#43; 30 &#43; 25 = 70
```

这个例子可以验证控制流，但不足以证明 Agent 已经具备生产能力：它没有调用计算器工具、没有校验结果，也没有触发动态重规划。对于如此简单的问题，三次执行调用加一次规划调用还会明显增加成本和延迟。

---

## 9. Plan-and-Solve 的优势

结合课程图片与当前代码，可以把优势总结为四点。

### 9.1 提升复杂任务的可靠性

复杂问题被拆成小步骤后，每次模型只处理局部目标，能够减少：

- 漏掉中间条件；
- 计算步骤跳跃；
- 长任务中目标漂移；
- 一次性生成导致的前后矛盾。

但要注意：**分解只是在统计上提升可靠性，不代表必然正确**。错误计划仍可能被执行器“忠实地执行到底”。

### 9.2 更强的任务分解能力

规划器可以把大任务映射成子任务序列或 DAG，符合工程领域的分治思想：

```text
复杂目标 → 子目标 → 可执行动作 → 局部结果 → 最终交付
```

如果步骤之间没有依赖，还可以并行执行，提高吞吐量。

### 9.3 更好的可观察性与可调试性

相比直接问答，Plan-and-Solve 能记录：

- 生成了什么计划；
- 当前执行到哪一步；
- 每一步产生了什么结果；
- 哪一步失败；
- 是否发生重试或重规划；
- 每一步消耗了多少 Token 和时间。

这使开发者可以定位问题究竟来自 Planner、Executor、Tool，还是 Finalizer。

### 9.4 更灵活的资源分配

不同阶段可以使用不同模型和工具：

- Planner：使用推理能力更强的模型；
- 简单步骤：使用更便宜、更快的模型；
- 数学问题：调用计算器或代码执行器；
- 检索步骤：调用搜索、数据库或 RAG；
- Finalizer：使用擅长长文本组织的模型。

这是一种典型的模型路由与成本优化方式。

---

## 10. 劣势与风险

### 10.1 高度依赖规划质量

Plan-and-Solve 的典型风险是级联错误：

```text
错误理解问题
   ↓
生成错误或不完整计划
   ↓
执行器逐步放大错误
   ↓
得到看似完整但方向错误的答案
```

因此不能只评估最终答案，还要评估：计划覆盖率、依赖正确率和步骤可执行率。

### 10.2 状态管理复杂

引入 Replan 后，需要管理：

- 已完成、进行中、失败和跳过的步骤；
- 新旧计划之间的映射；
- 已产生结果是否仍然有效；
- 是否需要回滚有副作用的操作；
- 多个并行步骤之间的数据一致性；
- 断点恢复、幂等和重复执行。

对于“发邮件、退款、改数据库”等有副作用的动作，重试绝不能只是再次调用一次，必须有幂等键、审批或补偿机制。

### 10.3 成本和延迟增加

假设规划调用一次，执行 `n` 个步骤，最终合成一次，则至少需要：

```text
模型调用次数 ≈ 1 &#43; n &#43; 1
```

如果加入验证、失败重试和重规划，调用次数会继续增长。因此应设置：

- 最大步骤数；
- 最大重规划次数；
- 最大 Token / 金额预算；
- 单步骤和全局超时；
- 简单问题跳过规划的路由策略。

### 10.4 计划可能在执行前就已过时

对于动态环境，一次性生成完整计划可能不现实。例如网页结构变化、库存变化、搜索结果不足、工具权限受限。此时纯开环的 Plan-and-Solve 不如带观察反馈的闭环 Agent。

---

## 11. 适用与不适用场景

### 11.1 适用场景

课程图片强调了两类典型任务：

1. **多步数学或逻辑应用题**：步骤依赖清晰，适合先分解再计算；
2. **整合多个信息源的报告撰写**：先规划调研维度，再分别检索、比较和汇总。

还包括：

- Deep Research；
- 代码仓库分析与分阶段修改；
- 数据分析报告；
- 旅行或项目方案制定；
- 多系统业务流程；
- 有明确阶段和验收标准的长任务。

### 11.2 不适用或收益较低的场景

- 简单事实问答；
- 一步即可完成的格式转换；
- 极低延迟对话；
- 环境高度动态、每一步都必须立即观察后再决定下一步；
- 高风险且不可逆、但系统又缺少审批和事务保障的操作。

面试中可以回答：

&gt; 我不会默认所有请求都走 Plan-and-Solve。工程上应先做复杂度路由：简单请求直接回答，需要频繁环境反馈的任务走 ReAct，结构清晰的长任务走 Plan-and-Solve，高风险写操作再叠加审批、幂等和补偿机制。

---

## 12. Plan-and-Solve、ReAct、Reflection 的区别

| 范式 | 核心机制 | 计划生成时机 | 环境反馈 | 主要优势 | 典型场景 |
|---|---|---|---|---|---|
| Direct / CoT | 一次性回答或内部逐步推理 | 无显式计划 | 通常没有 | 快、成本低 | 简单问答、短推理 |
| Plan-and-Solve | 先规划，再逐步执行 | 执行前生成较完整计划 | 可选 | 全局结构强、可观察 | 长流程、报告、多步计算 |
| ReAct | Thought → Action → Observation 循环 | 每轮决定下一步 | 强 | 适应动态环境 | 搜索、网页操作、工具交互 |
| Reflection | 生成后自我批评与修正 | 可在执行后追加 | 可选 | 改善答案质量 | 代码审查、写作、结果纠错 |

三者并不互斥，一个成熟 Agent 可以组合为：

```text
Plan
  ↓
每个步骤内部使用 ReAct 调工具
  ↓
Verifier / Reflection 检查结果
  ↓
必要时 Replan
  ↓
Finalizer
```

---

## 13. 当前实现与“完整 Plan-and-Solve”的差距

| 能力 | 当前 Notebook | 生产级方向 |
|---|---|---|
| 计划格式 | Markdown 中的 Python 列表 | JSON Schema / Typed Plan |
| 步骤依赖 | 依赖列表顺序隐式表达 | 显式 DAG dependencies |
| 执行方式 | 全部串行 | 按依赖并行 &#43; 并发限制 |
| 工具调用 | 无 | 原生 Tool Calling &#43; 权限控制 |
| 重规划 | 无 | 基于明确触发条件修改剩余计划 |
| 步骤校验 | 无 | success criteria &#43; verifier |
| 失败处理 | 空字符串或终止 | 分类重试、降级、替代、人工介入 |
| 最终答案 | 最后一步输出 | 独立 Finalizer 汇总 |
| 状态 | 普通字符串 | 类型化状态 &#43; 持久化 checkpoint |
| 可观测性 | `print` | trace、日志、指标、Token 与成本 |
| 安全 | 无专门设计 | 工具白名单、参数校验、审批、沙箱 |
| 预算控制 | 无 | 步骤、Token、时间、金额上限 |

如果面试官问“这份代码算不算 Agent”，可以回答：

&gt; 它是一个最小化的 Agent 编排示例，已经包含目标分解、状态传递和控制流，但更准确地说是一个 LLM 驱动的串行工作流。因为它缺少工具—环境交互、动态决策、重规划、验证和持久化状态，自主性与鲁棒性仍然有限。

---

## 14. Replan 应该如何设计

### 14.1 触发条件

不要让模型每一步都随意重规划，否则计划会频繁漂移。可以在以下条件触发：

- 当前步骤调用工具失败且重试耗尽；
- 当前步骤的前置条件不成立；
- Verifier 判断结果未达到 `success_criteria`；
- 获得的新信息推翻了计划假设；
- 用户中途修改目标；
- 剩余预算不足，需要缩减任务；
- 检测到计划循环、重复或不可执行。

### 14.2 重规划原则

- 已验证且仍有效的步骤结果尽量复用；
- 默认只修改未完成的计划后缀；
- 记录修改原因和新旧步骤映射；
- 限制最大重规划次数；
- 有副作用的已执行步骤不能简单“删除”，需要补偿或人工审批；
- 重规划后再次进行结构和依赖校验。

### 14.3 简化伪代码

```python
state = create_initial_state(question)
state.plan = planner.create_plan(state)

while not state.finished:
    step = scheduler.next_ready_step(state.plan)

    if step is None:
        break

    result = executor.run(step, state)
    verdict = verifier.check(step, result, state)

    if verdict.passed:
        state.mark_succeeded(step, result)
    elif verdict.retryable and step.retry_count &lt; MAX_RETRIES:
        state.mark_for_retry(step, verdict.reason)
    else:
        state.plan = replanner.revise_remaining_plan(
            state=state,
            failed_step=step,
            reason=verdict.reason,
        )

return finalizer.compose(state)
```

---

## 15. 如何把 Notebook 改造成更可靠的 Agent

建议按以下优先级演进。

### 第一阶段：让接口可测试

- `run()` 返回 `AgentResult`，不要只 `print`；
- 将日志和业务结果分离；
- 为 Planner 和 Executor 注入可替换的模型客户端；
- 使用固定假模型编写单元测试；
- 校验空计划、超长计划、非法步骤和模型超时。

### 第二阶段：结构化协议

- 使用 JSON Schema / Pydantic 定义计划和步骤结果；
- 给每个步骤增加 ID、依赖、状态和完成标准；
- 限制最大步骤数与每步输出长度；
- 使用原生 Tool Calling，避免自由文本解析动作。

### 第三阶段：闭环控制

- 增加 Verifier；
- 增加 Replanner；
- 增加独立 Finalizer；
- 对失败分类处理：retry、fallback、replan、human-in-the-loop、abort；
- 对简单任务增加 fast path，跳过规划。

### 第四阶段：生产治理

- 持久化 checkpoint，支持断点恢复；
- 增加全链路 trace 和步骤级指标；
- 对工具设置权限、超时、沙箱和审批；
- 为外部写操作提供幂等键与补偿机制；
- 设置 Token、时间、调用次数和金额预算；
- 做 Prompt Injection、越权调用和敏感信息泄漏防护。

---

## 16. 测试与评估指标

### 16.1 不要只看最终答案准确率

Plan-and-Solve 至少要分层评估：

| 层级 | 指标示例 |
|---|---|
| Planner | 计划覆盖率、步骤可执行率、依赖正确率、冗余步骤率 |
| Executor | 单步骤成功率、工具调用成功率、重试率、平均延迟 |
| Replanner | 重规划成功率、无效重规划率、平均重规划次数 |
| Finalizer | 需求覆盖率、事实一致性、引用完整率、格式遵循率 |
| 系统整体 | 任务完成率、端到端延迟、Token/成本、人工接管率 |

### 16.2 测试集应覆盖

- 简单任务：验证是否能走 fast path；
- 顺序依赖任务：前一步输出是后一步输入；
- 可并行任务：验证 DAG 调度；
- 工具超时、空结果和错误结果；
- 计划缺步骤、重复步骤或循环依赖；
- 中途修改用户目标；
- 上下文超长；
- 外部内容包含恶意指令；
- 有副作用动作的重复执行；
- Agent 在预算耗尽前能否安全终止。

---

## 17. 高频面试题与参考回答

### Q1：Plan-and-Solve 和 Chain-of-Thought 有什么区别？

CoT 主要通过中间推理改善单次回答；Plan-and-Solve 把“规划”和“执行”拆成显式阶段，并由宿主程序维护步骤、历史和控制流。后者更容易加入工具、重试、并行、状态持久化和可观测性，更接近 Agent Runtime。

### Q2：Plan-and-Solve 为什么可能比直接回答更可靠？

它降低了单次推理复杂度，并让每个局部结果可检查。但可靠性提升不是无条件的：如果 Planner 漏步骤或依赖错误，执行器会产生级联错误，所以还需要计划校验、步骤验证和 Replan。

### Q3：为什么不能把所有任务都先规划？

规划会增加一次模型调用，分步执行还会增加延迟和 Token。对于简单问答，规划收益可能小于成本，因此需要复杂度分类和 fast path。

### Q4：什么时候用 Plan-and-Solve，什么时候用 ReAct？

任务结构相对稳定、可以提前拆解时使用 Plan-and-Solve；环境不确定、下一步强依赖实时 Observation 时使用 ReAct。工程上常把二者结合：先生成高层计划，每个步骤内部用 ReAct 完成工具交互。

### Q5：如何防止错误计划一路执行到底？

在执行前做计划结构校验；为每个步骤定义完成标准；执行后使用程序规则或独立 Verifier 验证；失败时按策略重试、替代工具或 Replan；最终由 Finalizer 重新对齐原始问题。

### Q6：如何处理上下文越来越长？

不要无限拼接字符串。使用结构化状态保存完整记录，只向模型选择性提供当前步骤所需的依赖结果；对长结果做摘要或存入外部存储，通过引用检索；设置上下文和输出预算。

### Q7：如何支持并行执行？

把计划从线性列表升级为 DAG。调度器选择所有依赖已完成的 ready steps，在并发上限内执行；汇总结果后再解锁后继节点。并行前要检查工具限流、共享资源和写操作冲突。

### Q8：Planner 和 Executor 应该用同一个模型吗？

不一定。Planner 更需要全局推理和任务分解能力，Executor 可能只需完成局部任务。可以用强模型规划、便宜模型执行简单步骤，并按任务难度动态路由。最终选择要通过质量、延迟和成本评测确定。

### Q9：为什么当前代码的 `ast.literal_eval` 仍不够好？

它解决了直接 `eval` 的代码执行风险，但没有解决输出协议脆弱、字段缺失、类型约束不足和格式漂移。生产环境应优先使用模型原生的 Structured Output / JSON Schema，并在宿主程序中做严格校验。

### Q10：如何保证失败重试不会重复产生副作用？

读操作可以相对安全地重试；写操作必须使用幂等键、执行前状态检查、事务或补偿机制。高风险操作还应在真正提交前加入人工审批，不能仅依赖 LLM 判断。

### Q11：如何判断该触发 Replan 还是重试？

暂时性基础设施错误，如超时、限流，优先重试；当前工具不可用但目标仍有效，可以切换工具；前提被推翻、步骤不可执行或目标变化时才重规划。需要把错误分类交给确定性策略，而不是完全让模型自由决定。

### Q12：这份 Notebook 最大的三个问题是什么？

1. 计划解析依赖 Markdown 文本切割，协议脆弱；
2. 没有 Verifier 和 Replanner，执行过程是开环的；
3. 直接把最后一步当最终答案，缺少独立的结果合成与需求对齐。

---

## 18. 面试中的 30 秒回答模板

&gt; Plan-and-Solve 是一种先规划、后执行的 Agent 范式。Planner 先把复杂目标拆成有依赖的可执行步骤，Executor 再结合原始问题、完整计划和历史结果逐步完成任务。它相比直接问答更适合长任务，优势是任务分解、可观察性和资源调度能力更强；主要风险是规划错误会级联传播，状态、成本和延迟也更复杂。生产实现中我会用结构化计划代替文本列表，引入步骤级验证、失败分类、Replan、Finalizer、预算控制和 checkpoint；对于动态环境，还会让每个步骤内部使用 ReAct 调工具。

---

## 19. 面试时可以主动指出的代码亮点与不足

### 亮点

- Planner 与 Executor 职责分离，结构清晰；
- `temperature=0` 有利于协议稳定；
- `ast.literal_eval` 避免了直接 `eval`；
- Executor 始终保留原始问题和完整计划，能减少局部执行时的目标漂移；
- 用历史结果建立了最小的步骤依赖传递机制。

### 不足

- `List`、`Dict` 等导入未充分使用，接口类型仍较松散；
- Planner 只返回 `list[str]`，无法表达依赖、状态和完成标准；
- Prompt 全部放在单条 `user` 消息中，固定规则更适合放在 `system`；
- Executor 没有工具，数学计算仍由语言模型生成；
- 调用失败后可能把空字符串当作步骤结果继续执行；
- 没有最大步骤数、重试次数、Token 与时间预算；
- 没有 Replan，与课程图片中的完整架构存在差距；
- 没有 Finalizer，最后一步输出被直接当成最终答案；
- `run()` 只打印不返回，不利于测试和复用；
- 全量历史重复进入 Prompt，任务变长后成本较高。

---

## 20. 最终记忆框架

可以用下面的“六件套”记住 Plan-and-Solve：

```text
1. Plan       —— 如何分解目标？
2. State      —— 如何保存步骤、依赖和结果？
3. Execute    —— 如何调用模型、工具或外部系统？
4. Verify     —— 如何判断当前步骤真的完成了？
5. Replan     —— 失败或环境变化后如何调整？
6. Finalize   —— 如何重新对齐用户需求并生成最终交付？
```

对应大厂面试最常考的四个工程维度：

```text
可靠性：结构化输出、验证、重试、Replan
效率：任务路由、模型路由、并行、上下文压缩
安全性：工具权限、参数校验、幂等、审批、沙箱
可运营：Trace、指标、评测、成本、人工接管
```

一句话收尾：

&gt; Plan-and-Solve 的重点不只是“先列计划再回答”，而是把复杂任务变成一个可调度、可验证、可恢复、可治理的执行过程。


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlplan-and-solve/  

