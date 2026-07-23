# Reflection


## 1. 一句话理解 Reflection

**Reflection 是让 Agent 对自己上一轮的输出或执行轨迹进行评估，把失败原因和改进建议转化为语言反馈，再用反馈指导下一轮生成。**

普通的一次性生成是：

```text
Task → LLM → Answer
```

Reflection 增加了一个质量改进闭环：

```text
Task → Actor 生成候选结果
            ↓
       Evaluator 评审
            ↓
     Reflection 形成反馈
            ↓
       Actor 根据反馈重写
            ↓
       满足标准或达到迭代上限
```

它的关键不只是“多调用几次模型”，而是让每次额外调用承担明确职责：

- **Actor**：产生或修改解决方案；
- **Evaluator**：判断当前方案有什么问题；
- **Self-reflection**：把评估结果提炼成可执行的改进建议；
- **Memory**：保存历史尝试、失败经验和反馈；
- **Stop condition**：判断何时结束迭代。

面试时可以用下面这句话概括：

&gt; Reflection 用测试、规则、环境结果或模型评审构造反馈信号，并通过“生成—评估—修正”的闭环提升复杂任务的最终成功率。它主要优化的是推理时的搜索与纠错过程，通常不修改模型参数。

---

## 2. Reflection、Self-Reflection 与 Reflexion

这些词在课程、论文和工程项目中经常混用，面试中最好主动区分。

### 2.1 Reflection

Reflection 是一个宽泛的 Agent 设计范式：对当前答案或执行过程进行复盘，再迭代改进。

反馈可以来自：

- 同一个 LLM 的自我批评；
- 另一个 LLM Judge；
- 单元测试、编译器、静态分析器；
- 搜索、数据库或其他工具的返回结果；
- 人类反馈；
- 业务规则和奖励函数。

### 2.2 Self-Reflection

Self-Reflection 强调由模型阅读自己的输出并生成反思内容。Actor 和 Evaluator 可以复用同一个底座模型，只是使用不同 Prompt，也可以使用两个独立模型。

### 2.3 Reflexion

Reflexion 通常特指一种更具体的思路：把环境奖励、执行轨迹和评估反馈转化为**自然语言反思**，再将反思保存到情景记忆中，供后续尝试使用。

因此，图片中的 `Trajectory（短期记忆）`、`Evaluator`、`Self-reflection`、`Experience（长期记忆）` 更接近完整的 Reflexion 式架构；本 Notebook 的名称虽然是 Reflection，但实现的是一个简化版的语言反馈循环。

&gt; 不必在面试中纠结名词边界。更重要的是讲清楚：反馈从哪里来、如何进入下一轮上下文、记忆保留多久、怎样判断真的变好了。

---

## 3. 图片中的完整 Reflection 架构

第一张图可以抽象为以下闭环：

```mermaid
flowchart LR
    T[&#34;Task / 当前任务&#34;] --&gt; A[&#34;Actor&#34;]
    X[&#34;Experience&lt;br/&gt;长期经验&#34;] --&gt; A
    M[&#34;Trajectory&lt;br/&gt;短期轨迹&#34;] --&gt; A
    A --&gt; AC[&#34;Action&#34;]
    AC --&gt; ENV[&#34;Environment&#34;]
    ENV --&gt; OB[&#34;Observation&#34;]
    OB --&gt; M
    M --&gt; E[&#34;Evaluator&#34;]
    OB --&gt; E
    E --&gt;|&#34;内部 Feedback&#34;| R[&#34;Self-reflection&#34;]
    H[&#34;外部 Feedback&lt;br/&gt;测试 / 人类 / 环境奖励&#34;] --&gt; R
    R --&gt; X
    X --&gt; A
```

### 3.1 Actor：执行者

Actor 根据任务、当前轨迹和历史经验决定下一步动作。动作可以是：

- 生成或修改代码；
- 调用工具；
- 查询资料；
- 与环境交互；
- 输出最终答案。

在 Notebook 中，Actor 不是一个单独的 Python 类，而是同一个 LLM 在两类 Prompt 下扮演的角色：

- `INITIAL_PROMPT_TEMPLATE`：生成初始代码；
- `REFINE_PROMPT_TEMPLATE`：根据反馈重写代码。

### 3.2 Trajectory：短期轨迹

Trajectory 保存当前任务中的中间过程，例如：

```text
第 0 轮代码
→ 第 1 轮评审反馈
→ 第 1 轮优化代码
→ 第 2 轮评审反馈
→ 第 2 轮优化代码
```

它解决的是当前一次运行中的上下文连续性问题，通常随着任务结束而清空。

Notebook 中的 `Memory.records` 承担了这一角色，但主流程实际上只读取了：

- 最近一次执行结果：`get_last_execution()`；
- 最新一轮评审反馈：局部变量 `feedback`。

虽然类中实现了 `get_trajectory()`，但 `ReflectionAgent.run()` 并没有调用它，因此完整历史轨迹没有真正进入后续 Prompt。

### 3.3 Evaluator：评估器

Evaluator 判断当前方案的正确性、效率、安全性或任务完成度。图片同时展示了两类反馈：

- **内部反馈**：LLM Judge、规则判断、模型自评；
- **外部反馈**：环境奖励、单元测试、人类评审、线上指标。

Notebook 的 Evaluator 是 `REFLECT_PROMPT_TEMPLATE` 驱动的同一个 LLM，评估维度被限定为**算法效率**。

### 3.4 Self-reflection：形成可复用的语言经验

Evaluator 的原始信号不一定能直接帮助 Actor。例如，测试只会告诉我们：

```text
test_large_input: timeout
```

Self-reflection 需要进一步将它转化为：

```text
当前实现对每个整数使用试除法，整体约为 O(n√n)，大输入超时。
下一轮应改用埃拉托斯特尼筛法，将时间复杂度降到 O(n log log n)。
```

Notebook 中“评估”和“形成反思”没有拆成两个步骤，`REFLECT_PROMPT_TEMPLATE` 一次完成了问题诊断与改进建议生成。

### 3.5 Experience：长期经验

长期经验应跨任务或跨会话保存，例如：

```text
经验：涉及区间素数枚举时，先根据 n 的规模比较试除法、普通筛和分段筛。
经验：代码生成后必须去除 Markdown 围栏再编译和测试。
```

完整系统通常会将经验持久化到数据库、向量库或结构化经验表中，并在新任务开始时检索相关经验。

Notebook 中的 `Memory` 只存在于单个 `ReflectionAgent` 实例的内存中，也没有检索和持久化，所以它更准确地说是**短期情景记忆**，不是图片中的长期 Experience。

---

## 4. 用数学形式描述 Reflection

设：

- 任务为 `x`；
- 第 `t` 轮候选答案为 `y_t`；
- 执行轨迹为 `τ_t`；
- 第 `t` 轮评估反馈为 `f_t`；
- 可检索经验为 `m_t`。

初始生成：

```text
y₀ = Actor(x, m₀)
```

评估与反思：

```text
fₜ = Evaluator(x, yₜ, τₜ, external_feedbackₜ)
```

经验更新：

```text
mₜ₊₁ = MemoryUpdate(mₜ, fₜ, τₜ)
```

下一轮优化：

```text
yₜ₊₁ = Actor(x, yₜ, fₜ, mₜ₊₁)
```

终止条件：

```text
stop = pass(yₜ) or t &gt;= max_iterations or budget_exhausted
```

Notebook 实际实现得更简单：

```text
y₀ = LLM(initial_prompt(task))
fₜ = LLM(reflect_prompt(task, yₜ))
yₜ₊₁ = LLM(refine_prompt(task, yₜ, fₜ))
```

如果反馈文本包含“无需改进”，则提前停止。

---

## 5. Notebook 整体结构

代码由三个部分组成：

```mermaid
flowchart TD
    C[&#34;LLMProxy&lt;br/&gt;模型调用层&#34;] --&gt; RA[&#34;ReflectionAgent&lt;br/&gt;循环编排层&#34;]
    RA --&gt; MEM[&#34;Memory&lt;br/&gt;轨迹存储层&#34;]
    RA --&gt; P1[&#34;Initial Prompt&lt;br/&gt;初始生成&#34;]
    RA --&gt; P2[&#34;Reflect Prompt&lt;br/&gt;算法评审&#34;]
    RA --&gt; P3[&#34;Refine Prompt&lt;br/&gt;根据反馈重写&#34;]
    P1 --&gt; C
    P2 --&gt; C
    P3 --&gt; C
```

| 组件 | 职责 | Notebook 对应代码 |
|---|---|---|
| 模型访问层 | 调用兼容 OpenAI 接口的模型并收集流式响应 | `LLMProxy.think()` |
| 记忆层 | 保存每一轮代码和评审反馈 | `Memory` |
| Actor | 初始生成与优化代码 | Initial / Refine Prompt |
| Evaluator | 分析算法复杂度并给出改进建议 | Reflect Prompt |
| Orchestrator | 控制迭代、停止和返回结果 | `ReflectionAgent.run()` |

这里体现了一种常见的 Agent 工程分层：

```text
模型能力（LLM）
    ↑
角色与协议（Prompt）
    ↑
状态与记忆（Memory）
    ↑
流程编排（Agent Loop）
```

---

## 6. `Memory` 源码拆解

### 6.1 数据结构

```python
self.records: List[Dict[str, Any]] = []
```

每条记录形如：

```python
{&#34;type&#34;: &#34;execution&#34;, &#34;content&#34;: &#34;生成的代码&#34;}
{&#34;type&#34;: &#34;reflection&#34;, &#34;content&#34;: &#34;评审反馈&#34;}
```

优点是实现简单，能够按产生顺序保存多轮结果。缺点是结构太弱：

- `record_type` 只是自由字符串，容易拼错；
- 没有轮次、时间、任务 ID、模型、Token、得分等元数据；
- `content` 全是自由文本，难以做结构化统计；
- 没有容量限制和上下文压缩；
- 没有持久化、索引、召回和去重机制。

生产环境更适合使用 `dataclass` 或 Pydantic Model：

```python
class ReflectionRecord(BaseModel):
    task_id: str
    iteration: int
    record_type: Literal[&#34;execution&#34;, &#34;evaluation&#34;, &#34;reflection&#34;]
    content: str
    score: float | None = None
    created_at: datetime
```

### 6.2 `add_record()`

```python
def add_record(self, record_type: str, content: str):
    self.records.append({&#34;type&#34;: record_type, &#34;content&#34;: content})
```

作用是将本轮代码或反馈追加到轨迹末尾，时间复杂度通常是 `O(1)`。

可改进点：

- 校验 `record_type`；
- 禁止空内容；
- 写入轮次与质量评分；
- 对敏感信息做脱敏；
- 设置最大保留轮数；
- 支持数据库持久化。

### 6.3 `get_trajectory()`

该方法把全部记录拼成一段 Prompt 上下文：

```text
--- 上一轮尝试（代码）---
...

--- 评审员反馈 ---
...
```

但当前主流程没有调用它，这是理解本例时最容易忽略的一点。

如果直接将完整轨迹注入 Prompt，会获得更多上下文，但也会带来：

- Token 成本随轮次增长；
- 早期错误持续污染上下文；
- 模型可能重复已失败方案；
- 长上下文中关键信息被淹没。

更好的做法通常是保留最近若干轮原文，并把更早轨迹压缩成结构化摘要。

### 6.4 `get_last_execution()`

```python
for record in reversed(self.records):
    if record[&#34;type&#34;] == &#34;execution&#34;:
        return record[&#34;content&#34;]
```

从后向前找到最近一次代码。在最坏情况下时间复杂度是 `O(r)`，`r` 为记录数；本例记录很少，可以忽略。

返回值注解写成 `str`，但找不到记录时会返回 `None`，更严谨的类型应是：

```python
def get_last_execution(self) -&gt; Optional[str]:
```

---

## 7. 三类 Prompt 的职责

### 7.1 Initial Prompt：建立初始解

```text
角色：资深 Python 程序员
任务：根据要求编写函数
约束：完整签名、文档字符串、PEP 8
输出协议：只输出代码
```

它的优点是任务边界清晰，缺点是“只输出代码”仍然无法保证模型不添加 Markdown 代码围栏。Notebook 的真实输出确实包含了：

````text
```python
...
```
````

如果后面要把输出交给编译器，必须先解析代码块或使用结构化输出，而不能直接 `exec()` 原始文本。

### 7.2 Reflect Prompt：聚焦算法效率

反思 Prompt 明确要求：

- 分析时间复杂度；
- 找到算法瓶颈；
- 判断是否有算法层面的更优方案；
- 给出具体可执行的建议；
- 只有达到算法最优时才能回答“无需改进”。

这种写法比泛泛地说“请检查代码”更有效，因为它指定了评估维度和输出目标。

但评估范围过窄，只关注性能，可能遗漏：

- 功能正确性；
- 边界条件；
- 安全风险；
- 可读性和可维护性；
- API 契约；
- 测试覆盖率；
- 资源限制；
- 是否真正落实上一轮建议。

### 7.3 Refine Prompt：把反馈变成新解

Refine Prompt 同时提供：

- 原始任务；
- 上一轮代码；
- 评审反馈；
- 新代码的格式约束。

这使 Actor 不必重新从零推理，而是围绕明确问题做定向修改。

不过它没有要求：

- 逐条落实反馈；
- 保持已有正确行为不回退；
- 输出修改说明或变更清单；
- 通过测试后才能返回；
- 当反馈本身错误时进行质疑。

生产系统可让 Actor 返回结构化结果：

```json
{
  &#34;code&#34;: &#34;...&#34;,
  &#34;changes&#34;: [&#34;使用筛法替代逐数试除&#34;],
  &#34;unresolved_risks&#34;: [],
  &#34;expected_complexity&#34;: &#34;O(n log log n)&#34;
}
```

---

## 8. `ReflectionAgent.run()` 执行流程

### 8.1 初始执行

```python
initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
initial_code = self._get_llm_response(initial_prompt)
self.memory.add_record(&#34;execution&#34;, initial_code)
```

此时模型扮演 Actor，生成第一版代码并写入短期记忆。

### 8.2 反思循环

每轮依次执行：

```text
取最近代码
→ 调用 Evaluator 生成反馈
→ 保存反馈
→ 检查是否提前停止
→ 根据反馈生成新代码
→ 保存新代码
```

对应伪代码：

```python
candidate = actor.initial(task)

for iteration in range(max_iterations):
    feedback = evaluator.review(task, candidate)
    if should_stop(feedback):
        break
    candidate = actor.refine(task, candidate, feedback)

return candidate
```

### 8.3 停止机制

Notebook 使用字符串匹配：

```python
if &#34;无需改进&#34; in feedback or &#34;no need for improvement&#34; in feedback.lower():
    break
```

这是一个适合教学但不稳健的实现：

- 反馈可能说“并非无需改进”，仍包含目标短语；
- 模型可能用“已经最优”“可以停止”等其他表达；
- 模型可能一边说“无需改进”，一边又列出严重问题；
- 没有客观验证当前代码是否正确；
- 停止权完全交给生成式模型，结果不够可控。

更稳健的做法是让评估器返回结构化字段：

```json
{
  &#34;passed&#34;: false,
  &#34;score&#34;: 0.72,
  &#34;issues&#34;: [&#34;大输入性能不足&#34;],
  &#34;suggestions&#34;: [&#34;改用埃拉托斯特尼筛法&#34;]
}
```

然后由宿主程序基于 `passed`、测试结果、质量阈值和预算共同决定是否停止。

### 8.4 模型调用次数与成本

设最大迭代轮数为 `k`。

如果每轮都反思并优化，调用次数为：

```text
1 次初始生成 &#43; k 次反思 &#43; k 次优化 = 1 &#43; 2k
```

本例 `max_iterations=2`，最多调用模型 `5` 次。

如果第 `r` 次反思后判断停止，则调用次数为：

```text
1 次初始生成 &#43; r 次反思 &#43; (r - 1) 次优化 = 2r
```

因此 Reflection 的收益必须与模型成本、端到端延迟和上下文长度一起评估。

---

## 9. Notebook 运行结果分析

示例任务是：

```text
编写一个 Python 函数，找出 1 到 n 之间所有的素数。
```

### 9.1 初始方案：逐数试除

初始代码对 `2...n` 中的每个整数，都尝试除到它的平方根：

```python
for num in range(2, n &#43; 1):
    for i in range(2, int(num ** 0.5) &#43; 1):
        ...
```

粗略时间复杂度为：

```text
O(n√n)
```

空间复杂度除结果列表外约为 `O(1)`。

### 9.2 第一轮反思：改用埃拉托斯特尼筛法

Evaluator 指出逐个试除存在重复计算，并建议一次性标记合数：

```text
初始状态：2...n 都可能是素数
从 2 开始，将其倍数标为合数
再处理下一个仍为素数的数
只需筛到 √n
```

改进后的复杂度：

```text
时间复杂度：O(n log log n)
空间复杂度：O(n)
```

这是一次有效的**算法级优化**，不是常数级代码微调。

### 9.3 第二轮反思：只存储奇数

第二轮 Evaluator 又指出：

- 普通筛需要 `O(n)` 布尔空间；
- 偶数除了 `2` 外都不可能是素数；
- 可以只存储奇数，减少约一半空间和标记工作；
- 超大范围还可进一步考虑分段筛，提高缓存局部性并降低内存峰值。

最终生成代码实现了“只处理奇数”的筛法，但**没有真正实现反馈中提到的分段筛法**。

这个现象非常适合在面试中说明：

&gt; LLM 给出改进建议不等于 Actor 已完整落实建议。系统必须在优化后重新执行测试和评估，检查建议覆盖率与质量是否真的提升。

### 9.4 最后一版为什么仍不能直接称为“已验证最优”

Notebook 在第二轮 Refine 后直接返回代码，没有再评审最终版本。也就是说：

```text
第 2 轮反馈 → 第 2 轮重写 → 直接返回
```

缺少：

```text
编译 → 单元测试 → 性能测试 → 最终评估
```

因此它只能说明“模型根据反馈生成了看起来更好的代码”，不能证明：

- 所有边界条件正确；
- 没有引入回归；
- 性能确实优于上一版；
- 分段筛建议已落实；
- 输出可以被 Python 直接执行。

---

## 10. 图片中的成本、收益与适用场景

第二张图总结了 Reflection 的主要工程权衡。

### 10.1 核心收益

#### 解决方案质量跃迁

对于存在明确改进空间的任务，Reflection 能从“可用解”逐步搜索到“更优解”。Notebook 中从 `O(n√n)` 的逐数试除改进到 `O(n log log n)` 的筛法，就是典型案例。

#### 鲁棒性与可靠性增强

当反馈来自可执行测试、编译器或业务规则时，Agent 可以发现一次生成中的错误并定向修复，降低随机失败概率。

但需要注意：如果只有同一个模型自评，没有客观验证，Actor 与 Evaluator 可能共享盲区，可靠性提升有限。

### 10.2 主要成本

#### 模型调用开销增加

每轮通常至少增加一次评估调用和一次重写调用。本例两轮最多从一次模型调用上升到五次。

#### 端到端延迟显著提高

Reflection 是串行依赖链：必须等当前候选结果产生后才能评审，再等反馈产生后才能重写。即使单次模型很快，多轮累计也会显著提高 P95/P99 延迟。

#### Prompt 工程和流程复杂度上升

系统需要额外设计：

- 评估标准；
- 反馈格式；
- 终止条件；
- 记忆管理；
- 回归检测；
- Token 与时间预算；
- 失败恢复与观测指标。

### 10.3 适用场景

- 关键业务代码、SQL、配置或技术报告生成；
- 数学、算法、科研等复杂逻辑推演；
- 需要深度分析与多步规划的决策系统；
- 有编译器、测试集、模拟器或规则引擎提供客观反馈的任务；
- 单次错误成本高于额外推理成本的任务。

### 10.4 不适用或需要谨慎的场景

- 极低延迟的在线请求；
- 简单、确定、一次生成已足够的任务；
- 调用成本严格受限的高 QPS 场景；
- 没有可验证标准，Evaluator 只能凭主观语言打分的任务；
- 错误反馈可能把正确结果“改坏”的场景。

一个实用决策原则是：

```text
预期质量收益 × 错误成本
是否大于
额外模型成本 &#43; 延迟成本 &#43; 系统复杂度
```

---

## 11. 当前 Notebook 与完整架构的对应关系

| 图片中的组件 | Notebook 是否实现 | 代码对应 | 说明 |
|---|---:|---|---|
| Actor | 是 | Initial / Refine Prompt | 同一个 LLM 扮演生成者 |
| Evaluator | 部分 | Reflect Prompt | 只做语言层面的算法评审 |
| Self-reflection | 部分 | Reflect Prompt 输出 | 评估与反思未拆分 |
| Trajectory 短期记忆 | 部分 | `Memory.records` | 保存了记录，但完整轨迹未注入 Prompt |
| Experience 长期记忆 | 否 | 无 | 不持久化，也不跨任务检索 |
| Action | 弱实现 | 输出代码 | 只是生成文本，没有真正执行 |
| Environment | 否 | 无 | 没有 Python 沙箱、测试框架或工具环境 |
| Observation | 否 | 无 | 没有编译、测试、性能数据等观察结果 |
| 外部 Feedback | 否 | 无 | 没有测试、人类或业务规则反馈 |
| 终止策略 | 简化实现 | 关键词匹配 &#43; 最大轮数 | 缺少结构化评分和预算控制 |

因此，更准确的评价是：

&gt; 该 Notebook 展示了 Reflection 的最小语言闭环，适合理解 Prompt 角色分工和迭代控制，但还不是一个具备真实环境反馈、长期经验和可靠验证的生产级 Reflection Agent。

---

## 12. 当前实现的优点

- 结构简单，执行、反思、优化三阶段清晰；
- Actor 与 Evaluator 的角色约束明确；
- Reflection Prompt 聚焦算法复杂度，反馈目标具体；
- Memory 把执行结果与反馈分类型保存；
- 支持 `max_iterations`，避免无限循环；
- 支持“无需改进”提前退出；
- 模型访问层与 Agent 编排层分离；
- `temperature=0` 有助于降低格式和结论漂移；
- 示例任务能直观展示算法质量从试除法到筛法的提升。

---

## 13. 当前实现的关键问题

### 13.1 没有真实执行，Reflection 缺少事实基础

代码被生成后没有：

- 去除 Markdown 代码围栏；
- AST 解析或编译；
- 单元测试；
- 随机测试或性质测试；
- 性能基准；
- 资源限制；
- 安全沙箱。

Evaluator 只能阅读代码并“猜测”它是否正确，容易出现自信但错误的反馈。

### 13.2 Actor 和 Evaluator 共享模型盲区

Notebook 使用同一个 `llm_client` 完成生成和评审。这样成本和实现复杂度较低，但可能产生相关性偏差：生成者没发现的问题，评审者也可能继续忽略。

改进方向：

- Actor 与 Critic 使用不同 Prompt；
- 使用不同模型或不同采样结果；
- 引入确定性测试和规则；
- 多个 Evaluator 分别检查正确性、性能、安全性；
- 通过加权或投票聚合结论。

### 13.3 反馈是自由文本，机器难以可靠消费

自由文本难以稳定判断：

- 是否通过；
- 问题严重度；
- 需要修改哪些位置；
- 评分是否提升；
- 是否已经收敛。

应优先使用 JSON Schema、Pydantic 或原生 Structured Outputs。

### 13.4 停止条件脆弱

通过关键词判断“无需改进”容易误触发，也没有：

- 分数阈值；
- 连续无提升检测；
- Token 预算；
- 时间预算；
- 成本预算；
- 重复方案检测；
- 最终强制验证。

### 13.5 `get_trajectory()` 没有进入主流程

Memory 保存了全轨迹，但 Actor 和 Evaluator 只看到最近代码和最新反馈，早期失败经验没有发挥作用。

另一方面，简单地把全部历史塞回 Prompt 也不是最佳方案。合理设计应是：

```text
最近 1～2 轮原始内容
&#43; 更早历史的压缩摘要
&#43; 与当前任务相关的长期经验检索结果
```

### 13.6 最后一轮优化没有再验证

循环耗尽时，最后动作总是 Refine，之后直接返回。这会导致最终答案没有经过 Evaluator 检查。

更合理的状态机应确保结束前至少执行一次最终验证：

```text
GENERATE → EXECUTE → EVALUATE
                    ├─ PASS → FINISH
                    └─ FAIL → REFLECT → REFINE → EXECUTE
```

### 13.7 错误处理不足

`LLMProxy.think()` 异常时返回 `None`，`_get_llm_response()` 将其变成空字符串。后续流程仍可能继续评审和优化空内容，使真正错误被掩盖。

生产环境应区分：

- 可重试的限流和网络错误；
- 不可重试的鉴权错误；
- 内容为空；
- 格式解析失败；
- 上下文超限；
- 模型拒答。

并配套指数退避、最大重试次数、降级模型和错误状态记录。

### 13.8 Prompt Injection 与代码安全

如果任务描述来自不可信用户，恶意内容可能诱导 Evaluator 忽略规则；如果自动执行生成代码，还可能访问文件、网络、环境变量或启动子进程。

必须将生成代码放入隔离环境，并限制：

- CPU、内存和运行时间；
- 文件系统访问；
- 网络访问；
- 系统调用；
- 可导入模块；
- 输出大小。

---

## 14. 生产级 Reflection Agent 如何升级

### 14.1 推荐架构

```mermaid
flowchart TD
    U[&#34;Task&#34;] --&gt; OR[&#34;Orchestrator&#34;]
    OR --&gt; MR[&#34;检索相关经验&#34;]
    MR --&gt; A[&#34;Actor 生成候选方案&#34;]
    A --&gt; PARSE[&#34;结构化解析 / 代码提取&#34;]
    PARSE --&gt; SB[&#34;Sandbox 执行&#34;]
    SB --&gt; TEST[&#34;单元测试 / 静态检查 / Benchmark&#34;]
    TEST --&gt; EV[&#34;Evaluator 聚合评分&#34;]
    EV --&gt;|&#34;达到阈值&#34;| FIN[&#34;最终验证并返回&#34;]
    EV --&gt;|&#34;未达到阈值&#34;| REF[&#34;Reflection 生成改进计划&#34;]
    REF --&gt; MW[&#34;写入短期轨迹 / 可选长期经验&#34;]
    MW --&gt; A
    OR --&gt; BUDGET[&#34;轮次 / Token / 时间 / 成本预算&#34;]
    BUDGET --&gt; EV
```

### 14.2 把“生成代码”升级为“生成—执行—观察”

代码任务最重要的反馈不是 LLM 的感觉，而是工具结果：

```text
代码生成
→ AST/compile 检查
→ 单元测试
→ 随机或性质测试
→ 性能基准
→ 将 stdout、stderr、失败用例和耗时反馈给 Reflector
```

这样图片中的 `Action → Environment → Observation` 才真正落地。

### 14.3 多维 Evaluator

可以把评估拆成多个维度：

| 评估器 | 证据 | 示例指标 |
|---|---|---|
| Correctness | 单元测试、性质测试 | pass rate |
| Efficiency | Benchmark、复杂度分析 | latency、memory |
| Security | SAST、依赖扫描、沙箱日志 | high-risk issue count |
| Style | Ruff、Black、类型检查 | lint error count |
| LLM Judge | 任务满足度、解释质量 | rubric score |

最终分数可表示为：

```text
Score = w₁·Correctness &#43; w₂·Efficiency &#43; w₃·Security &#43; w₄·Quality
```

代码任务通常应把 Correctness 设为硬门槛，而不是允许其他维度用高分抵消错误结果。

### 14.4 结构化 Reflection

推荐输出字段：

```json
{
  &#34;passed&#34;: false,
  &#34;score&#34;: 0.65,
  &#34;root_causes&#34;: [
    &#34;对每个候选数重复进行试除，导致大输入超时&#34;
  ],
  &#34;evidence&#34;: [
    &#34;benchmark_n_1000000 exceeded 2 seconds&#34;
  ],
  &#34;repair_plan&#34;: [
    &#34;改用埃拉托斯特尼筛法&#34;,
    &#34;增加 n &lt; 2 的边界测试&#34;
  ],
  &#34;must_preserve&#34;: [
    &#34;返回 1 到 n 的所有素数，包含 n 本身&#34;
  ]
}
```

反思应尽量包含“根因—证据—修复方案—不可回退行为”，而不只是宽泛评价。

### 14.5 更合理的终止策略

可组合以下条件：

```python
should_stop = (
    all_required_tests_pass
    and score &gt;= quality_threshold
    and no_critical_security_issue
) or iteration &gt;= max_iterations \
  or token_used &gt;= token_budget \
  or elapsed_seconds &gt;= time_budget \
  or no_improvement_rounds &gt;= patience
```

即使因为预算终止，也应明确标注结果是：

- `SUCCESS`：通过所有硬标准；
- `BEST_EFFORT`：达到预算上限，返回当前最好候选；
- `FAILED`：没有安全或可用候选；
- `NEEDS_HUMAN_REVIEW`：存在高风险不确定性。

### 14.6 候选集与回滚

不能默认新一轮一定优于旧一轮。应保存每一版候选及分数：

```text
v0: score 0.55
v1: score 0.88
v2: score 0.81  ← 发生回退
```

最终返回历史最高分且通过硬约束的版本，而不是机械返回最后一版。

### 14.7 长期经验的写入与检索

不是每条反思都值得进入长期记忆。可以只保存：

- 多次出现的失败模式；
- 被客观测试验证有效的修复策略；
- 高价值任务中的人工确认经验；
- 能被抽象复用、且不包含敏感数据的知识。

经验记录可以结构化为：

```text
适用条件 &#43; 失败模式 &#43; 根因 &#43; 修复策略 &#43; 验证证据 &#43; 置信度
```

检索时同时考虑任务语义、代码语言、错误类型和适用条件，避免把不相关经验强行注入 Prompt。

---

## 15. Reflection 与其他 Agent 范式的区别

### 15.1 Reflection vs. Chain-of-Thought

- CoT 主要展开单次调用内部的推理步骤；
- Reflection 对已经得到的候选结果进行跨轮复盘和修正；
- CoT 不一定有评估信号，Reflection 强调反馈闭环；
- 工程上通常不依赖暴露完整思维链，而使用简洁的结论、证据和结构化改进计划。

### 15.2 Reflection vs. ReAct

- ReAct 强调 `Reasoning → Action → Observation`，重点是与外部工具和环境交互；
- Reflection 强调 `Execute → Evaluate → Reflect → Refine`，重点是复盘和提升已有方案；
- ReAct 的 Observation 可以成为 Reflection 的外部反馈；
- 两者经常组合：ReAct 完成一轮工具执行，Reflection 再对整段轨迹复盘。

组合流程：

```text
ReAct 执行任务
→ 收集 Thought / Action / Observation 轨迹
→ Evaluator 判断失败原因
→ Reflection 总结经验
→ 下一次 ReAct 使用经验重新规划
```

### 15.3 Reflection vs. Plan-and-Solve

- Plan-and-Solve 先拆解任务，再按计划执行；
- Reflection 在候选结果产生后评估和修正；
- 规划解决“下一步做什么”，反思解决“刚才哪里做错了、下一轮怎样更好”；
- 复杂 Agent 可以同时具备规划、执行、反思和重规划。

### 15.4 Reflection vs. 多 Agent Debate

- Reflection 常见形态是 Actor—Critic 串行迭代；
- Debate 让多个 Agent 对不同观点进行辩论或互审；
- Debate 能降低单一路径偏差，但成本更高；
- 多个 Agent 如果使用同一模型和相似 Prompt，也未必真正具备多样性。

### 15.5 Reflection vs. 强化学习

- Reflection 常在推理时通过语言反馈改变上下文，不更新模型参数；
- 强化学习通过奖励优化策略参数；
- Reflection 的“奖励”可以是文本形式，适合快速工程实现；
- 它不能替代训练阶段的能力提升，模型能力上限、上下文上限和固有偏差仍然存在。

---

## 16. 如何评价 Reflection 是否真的有效

不能只展示一个“优化后看起来更好”的案例。实验设计至少应包含：

### 16.1 基线组

- 单次直接生成；
- 单次生成 &#43; 更强 Prompt；
- Reflection 1 轮、2 轮、3 轮；
- 不同 Evaluator 或不同反馈来源。

### 16.2 核心指标

| 指标 | 含义 |
|---|---|
| Task success rate | 最终任务成功率 |
| pass@1 / pass@k | 代码或推理任务通过率 |
| Quality delta | 每轮质量分变化 |
| Regression rate | 优化后反而变差的比例 |
| Average iterations | 平均收敛轮数 |
| Token / cost per success | 每次成功任务的成本 |
| End-to-end latency | 端到端延迟及 P95/P99 |
| Stop precision | 模型判断“无需改进”时真正通过的比例 |

### 16.3 消融实验

可以分别去掉：

- 外部测试反馈；
- 完整轨迹；
- 长期经验；
- 独立 Evaluator；
- 结构化输出；
- 最终验证。

观察每个组件对成功率、成本和延迟的真实贡献。

### 16.4 防止评测污染

- 评测测试集不能原样出现在 Prompt；
- LLM Judge 与生成模型最好不要完全同源；
- 主观评价应使用清晰 Rubric；
- 关键任务应以确定性规则或真实业务结果为主；
- 保存每轮输入、输出、工具结果、模型版本和参数，保证可复现。

---

## 17. 面试高频问题与参考回答

### Q1：Reflection Agent 的本质是什么？

&gt; 本质是推理时的闭环优化。Actor 先生成候选方案，Evaluator 根据任务标准和环境反馈定位问题，Reflector 将问题转成可执行的语言经验，Actor 再基于经验重写，直到通过标准或耗尽预算。它通常通过更新上下文而不是更新模型参数来提升结果。

### Q2：为什么 Reflection 能提升效果？

&gt; 一次生成容易受采样误差、局部推理错误和遗漏约束影响。Reflection 把最终任务拆成生成与评估两个角色，并将失败信号重新注入上下文，相当于在推理时做迭代搜索。效果提升的前提是反馈足够准确、具体且能够被 Actor 执行。

### Q3：Actor 和 Evaluator 可以是同一个模型吗？

&gt; 可以，本 Notebook 就是同一个模型通过不同 Prompt 扮演两个角色，成本低、实现简单。但它们可能共享盲区和偏差。高可靠场景应优先引入测试、规则、工具结果等客观反馈，必要时使用不同模型或多个评估器做交叉验证。

### Q4：Reflection 和让模型“再想一想”有什么区别？

&gt; “再想一想”缺少明确协议，往往只是增加输出长度。完整 Reflection 应包含候选结果、可验证评估标准、具体失败证据、结构化改进建议、记忆更新和终止策略，是一个由宿主程序编排的状态循环。

### Q5：如何防止无限反思？

&gt; 同时设置最大轮数、Token、时间和成本预算；要求质量分达到阈值；检测连续多轮无提升、重复方案和振荡；每轮保留最高分候选；预算耗尽时返回明确状态，而不是继续循环。

### Q6：Reflection 最大的问题是什么？

&gt; 主要问题是成本和延迟增加，以及自评反馈不一定可靠。如果 Evaluator 没有外部事实依据，Agent 可能把正确答案改错，或在错误方向上反复自洽。因此工程重点不是简单增加轮数，而是设计高质量反馈和可验证的停止条件。

### Q7：如何将 Notebook 改造成生产系统？

&gt; 我会增加代码提取与结构化解析、沙箱执行、单元测试和 Benchmark，把 Observation 交给多维 Evaluator；评估结果采用结构化 Schema；记录每版候选及分数，返回历史最佳版本；增加预算、重试、追踪和安全限制；最后再建设经验证的长期经验库。

### Q8：为什么 `get_trajectory()` 没有被使用是一个问题？

&gt; 因为虽然系统保存了多轮记录，但后续模型只看到最近代码与最新反馈，历史失败经验没有真正参与决策。不过直接注入全部轨迹也会造成上下文膨胀，更合理的是近期原文加历史摘要，再检索少量相关长期经验。

### Q9：如何设计评估器？

&gt; 先把任务成功标准拆成硬约束和软指标。硬约束用测试、编译器、规则引擎验证；软指标用带 Rubric 的 LLM Judge 评分。评估器输出 `passed`、分数、失败证据、根因、修复计划和严重度。关键决策不能只依赖一段自由文本。

### Q10：为什么最后一轮必须再验证？

&gt; 因为 Refine 本身也可能引入新错误。如果循环以“重写”结束而没有重新执行和评估，系统无法证明最终版本比上一版更好。正确状态机应该以验证通过结束，而不是以模型生成结束。

### Q11：长期记忆应该保存什么？

&gt; 保存经过客观验证、可跨任务复用的失败模式和修复策略，并附适用条件、证据、置信度和版本信息。不能把每条模型自评都写入长期记忆，否则错误经验会持续污染后续任务。

### Q12：如何处理“越改越差”？

&gt; 对每版候选做相同测试和评分，保存版本历史，设置硬约束与回归测试，最终选择历史最佳可行解；如果连续多轮无提升就提前停止。Actor 还应接收 `must_preserve` 列表，避免修复一个问题时破坏已有正确行为。

---

## 18. 面试中的项目讲述模板

可以按“问题—方案—权衡—改进”四段式回答：

&gt; 我实现了一个面向代码优化的 Reflection Agent。Actor 首先根据任务生成代码，Evaluator 重点分析算法复杂度并输出改进建议，Memory 保存每轮代码和反馈，Orchestrator 控制最多两轮反思与重写。示例中，模型将素数枚举从约 `O(n√n)` 的试除法优化为 `O(n log log n)` 的筛法，并进一步使用奇数压缩降低空间开销。
&gt;
&gt; 这个实现能清楚展示生成—评估—修正闭环，但我不会把它直接称为生产级 Agent。当前反馈完全来自同一个 LLM，没有执行代码或运行测试；`get_trajectory()` 未进入 Prompt；长期经验没有持久化；停止条件依赖关键词；最后一次重写后也没有最终验证。
&gt;
&gt; 如果工程化，我会接入沙箱、单元测试、静态检查和 Benchmark，用结构化 Evaluator 聚合客观证据；保存每版候选与得分并支持回滚；设置轮次、Token、时间和成本预算；只把经过验证、可复用的反思写入长期经验库。
&gt;
&gt; 这类范式适合错误成本高、结果可验证的复杂任务，但不适合所有请求默认开启。线上应先做任务路由，只对高难度或初次验证失败的请求启动 Reflection，以控制延迟和成本。

---

## 19. 一页速记

### 核心闭环

```text
Execute → Evaluate → Reflect → Refine → Verify
```

### 五个关键角色

```text
Actor &#43; Evaluator &#43; Reflection &#43; Memory &#43; Environment
```

### Notebook 的实现

```text
Initial Prompt → 初始代码
Reflect Prompt → 算法反馈
Refine Prompt → 优化代码
Memory → 保存 execution / reflection
max_iterations → 限制循环
```

### 示例中的质量提升

```text
逐数试除：O(n√n)
→ 埃拉托斯特尼筛：O(n log log n)
→ 只存奇数：复杂度量级不变，常数和空间约减半
```

### 三个最大收益

- 复杂任务的解质量可能实现跃迁；
- 能基于失败反馈进行定向纠错；
- 可把验证过的失败经验用于后续任务。

### 三个主要成本

- 模型调用次数和 Token 成本增加；
- 串行迭代导致端到端延迟上升；
- 评估、记忆、终止和安全机制更复杂。

### 当前代码的五个关键缺口

- 没有真实环境执行和 Observation；
- 没有单元测试或 Benchmark；
- `get_trajectory()` 没有被使用；
- 没有真正的长期 Experience；
- 最后一轮 Refine 后没有最终验证。

### 面试收尾句

&gt; Reflection 的上限不由“反思轮数”决定，而由反馈质量决定。高质量系统要把模型自评升级为基于测试、工具和业务规则的可验证反馈，并用预算控制、最佳候选保留和最终验证保证收益大于成本。



---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlreflection/  

