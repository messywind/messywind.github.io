# ReAct


## 1. 一句话理解 ReAct

**ReAct = Reasoning（推理/规划）&#43; Acting（调用工具与环境交互）**。

模型不是收到问题后直接生成最终答案，而是在一个循环中交替完成：

1. 根据问题和历史信息判断下一步；
2. 选择并调用工具；
3. 读取工具返回的观察结果；
4. 基于新证据继续规划、纠错或结束任务。

典型轨迹为：

```text
Question
  ↓
Thought → Action → Observation
             ↑          ↓
             └── Thought → Action → Observation
                                ↓
                         Finish[Final Answer]
```

ReAct 的关键价值不是“让模型多想几步”，而是让模型能够将生成式推理和外部世界中的真实行动结合起来，通过环境反馈修正后续决策。

ReAct 来自 Shunyu Yao 等人的论文 **ReAct: Synergizing Reasoning and Acting in Language Models**（ICLR 2023）。论文中的核心观察是：推理轨迹有助于模型规划和更新行动，外部行动获得的真实反馈又能反过来约束推理，二者交替比单独推理或单独行动更适合交互式任务。

---

## 2. ReAct 解决了什么问题

单纯依赖 LLM 参数知识直接作答，会遇到以下问题：

- 知识可能过时，例如“最新手机是什么”；
- 模型可能不知道某个企业内部数据；
- 数学计算、数据库查询等任务仅靠语言生成不够可靠；
- 一次性生成缺少环境反馈，出错后无法动态调整；
- 复杂任务需要多步执行，不能只输出一段文本。

ReAct 将 Agent 拆成两个相互促进的部分：

| 部分 | 作用 | 本例对应内容 |
|---|---|---|
| Reasoning | 分析当前状态、规划下一步、判断是否结束 | `Thought:` |
| Acting | 选择工具并传入参数，或者输出最终答案 | `Action: Search[...]` / `Finish[...]` |
| Environment Feedback | 工具或外部环境返回的结果 | `Observation:` |

因此，ReAct 本质上可以看作一个由 LLM 驱动的闭环控制系统：

```text
状态 = 用户问题 &#43; 历史动作 &#43; 环境观察
策略 = LLM
动作 = 工具调用或结束任务
环境 = 搜索引擎、数据库、代码执行器、业务 API 等
```

---

## 3. ReAct 与几个相近概念的区别

### 3.1 ReAct vs. 直接问答

```text
直接问答：Question → LLM → Answer
ReAct：   Question → LLM → Tool → Observation → LLM → ... → Answer
```

直接问答速度快、成本低，但无法主动获取外部信息；ReAct 成本更高，却能处理实时知识和多步任务。

### 3.2 ReAct vs. Chain-of-Thought（CoT）

- CoT 重点是“推理”，通常没有外部工具反馈；
- ReAct 同时包含推理和行动，模型能用真实 Observation 修正判断；
- CoT 可能在错误前提上越想越远，ReAct 至少有机会借助外部证据纠错。

面试中可以概括为：

&gt; CoT 改善模型内部的推理过程，ReAct 则把推理过程接入环境，使 Agent 形成“决策—执行—反馈”的闭环。

### 3.3 ReAct vs. Function Calling / Tool Calling

- Function Calling 是模型输出结构化工具调用的一种接口能力；
- ReAct 是组织多轮推理、行动和观察的一种 Agent 范式；
- ReAct 可以用文本协议实现，也可以用原生 Function Calling 实现；
- 工程中通常用原生 Tool Calling 替代手写正则解析，但外层仍然可以是 ReAct 循环。

### 3.4 ReAct vs. Plan-and-Solve

- ReAct 倾向于边执行、边观察、边调整；
- Plan-and-Solve 通常先生成较完整的计划，再依次执行；
- ReAct 适合环境不确定、需要频繁反馈的任务；
- 固定长流程可采用“先规划 &#43; 分步执行”，并在每一步引入 ReAct 式纠错。

### 3.5 ReAct vs. Reflection

- ReAct 通过工具结果即时纠错；
- Reflection 强调对已经生成的答案或执行轨迹进行复盘、批评和重写；
- 二者可以组合：ReAct 完成任务后，再由 Reflection 检查证据完整性和答案质量。

---

## 4. Notebook 的整体架构

本仓库示例包含三个主要组件：

```mermaid
flowchart LR
    U[&#34;用户问题&#34;] --&gt; A[&#34;ReActAgent.run&#34;]
    A --&gt; P[&#34;构造 ReAct Prompt&#34;]
    P --&gt; L[&#34;LLMProxy.think&#34;]
    L --&gt; R[&#34;解析 Thought / Action&#34;]
    R --&gt;|&#34;Search[input]&#34;| T[&#34;ToolExecutor&#34;]
    T --&gt; S[&#34;SerpApi Search&#34;]
    S --&gt; O[&#34;Observation&#34;]
    O --&gt; H[&#34;写入 history&#34;]
    H --&gt; P
    R --&gt;|&#34;Finish[answer]&#34;| F[&#34;返回最终答案&#34;]
```

### 4.1 `LLMProxy`：模型访问层

`llm_client.py` 封装了兼容 OpenAI Chat Completions 接口的模型服务：

- 从参数或 `.env` 读取 `MODEL_ID`、`API_KEY`、`BASE_URL` 和超时时间；
- 默认 `temperature=0`，减少格式漂移和随机性；
- 使用流式输出；
- 异常时返回 `None`。

它只负责“调用模型并返回文本”，并不知道什么是 Agent 或工具。

### 4.2 `ToolExecutor`：工具注册表

`tools.py` 中的 `ToolExecutor` 维护如下结构：

```python
self.tools = {
    &#34;Search&#34;: {
        &#34;description&#34;: &#34;工具用途说明&#34;,
        &#34;func&#34;: search,
    }
}
```

三个关键方法：

- `registerTool(name, description, func)`：注册工具；
- `getAvailableTools()`：把工具名称和描述提供给模型；
- `getTool(name)`：根据模型选择的名称取得真正的 Python 函数。

这里体现了 Agent 工程中的一个重要原则：

&gt; LLM 只负责选择“调用哪个工具、传什么参数”，宿主程序负责权限校验和真正执行，不能让模型直接执行任意代码。

### 4.3 `search`：外部环境

示例工具通过 SerpApi 搜索网页，按以下优先级提取结果：

1. `answer_box_list`；
2. `answer_box.answer`；
3. `knowledge_graph.description`；
4. 前三条自然搜索结果的标题和摘要。

搜索工具让模型获得参数之外的实时信息，但“搜到了内容”不等于“内容真实可信”。Agent 仍需处理来源质量、时效性、冲突信息和引用问题。

---

## 5. Prompt 设计拆解

Notebook 中的 Prompt 主要包含四部分：

```text
角色：你是能够调用外部工具的智能助手
工具：{tools}
协议：Thought &#43; Action，Action 只能是工具调用或 Finish
上下文：Question &#43; History
```

核心输出协议：

```text
Thought: 对当前问题和下一步的分析
Action: Search[查询词]
```

或者：

```text
Thought: 已获得足够证据
Action: Finish[最终答案]
```

### Prompt 为什么必须明确规定格式

宿主程序需要把自然语言输出转换成可执行动作。如果格式不稳定，就可能出现：

- 找不到 `Action:`；
- 工具名称拼错；
- 参数括号不完整；
- 模型直接给答案，却没有输出 `Finish[...]`；
- 一次输出多个 Action，解析器不知道执行哪个；
- 工具返回文本被模型误当成新指令。

因此，ReAct Prompt 不只是“提示模型思考”，它还是 LLM 与 Agent Runtime 之间的一份通信协议。

### 本例 Prompt 的优点

- 提供了可用工具列表；
- 限定了工具调用语法；
- 明确给出终止指令；
- 将历史 Observation 反馈给模型；
- 使用 `temperature=0`，有利于稳定遵循格式。

### 本例 Prompt 可改进之处

- 没有给出完整 few-shot 示例；
- 没有规定每轮只能调用一个工具；
- 没有明确要求答案必须被 Observation 支撑；
- 没有处理搜索结果中的 Prompt Injection；
- 没有说明工具失败、结果为空、信息冲突时怎么办；
- 没有要求引用来源、发布日期或置信度；
- 工具参数只是自由文本，没有 JSON Schema；
- 把整个 Prompt 每轮作为单条 `user` 消息发送，角色边界不够清晰。

一个更稳健的策略是：固定规则放在 `system` 消息，原始问题放在 `user` 消息，动作与 Observation 作为结构化历史消息保存。

---

## 6. `ReActAgent.run()` 执行流程

### 6.1 初始化状态

```python
self.history = []
current_step = 0
```

每次运行前清空历史，防止不同用户问题之间发生上下文污染。

### 6.2 进入有界循环

```python
while current_step &lt; self.max_steps:
```

`max_steps` 是重要的安全阀，防止模型无限搜索或在错误动作间循环。示例默认最多执行 5 轮。

### 6.3 构造当前 Prompt

```python
tools_desc = self.tool_executor.getAvailableTools()
history_str = &#34;\n&#34;.join(self.history)
prompt = REACT_PROMPT_TEMPLATE.format(
    tools=tools_desc,
    question=question,
    history=history_str,
)
```

当前决策依赖三类上下文：

- 原始问题；
- 当前可用工具；
- 之前的 Action 和 Observation。

注意：本实现没有把 `Thought` 写入历史，只保留动作和观察。这能减少部分 Token，但也使后续模型看不到自己先前显式表达的计划。

### 6.4 请求模型决策

```python
response_text = self.llm_client.think(messages=messages)
```

模型的角色相当于策略函数：输入当前状态，输出下一步动作。

### 6.5 解析模型输出

```python
thought, action = self._parse_output(response_text)
```

`_parse_output()` 使用正则分别提取：

- `Thought:` 到 `Action:` 之前的文本；
- `Action:` 到响应末尾的文本。

这是一种教学友好的实现，但正则解析对格式非常敏感。

### 6.6 判断是否结束

```python
if action.startswith(&#34;Finish&#34;):
    final_answer = self._parse_action_input(action)
    return final_answer
```

模型主动决定何时信息已经足够。`Finish` 是终止动作，而不是工具。

### 6.7 执行工具并记录 Observation

```python
tool_name, tool_input = self._parse_action(action)
tool_function = self.tool_executor.getTool(tool_name)
observation = tool_function(tool_input)

self.history.append(f&#34;Action: {action}&#34;)
self.history.append(f&#34;Observation: {observation}&#34;)
```

Observation 会进入下一轮 Prompt，从而形成闭环：

```text
第 t 轮 Observation
        ↓
第 t&#43;1 轮 Thought / Action
```

这正是 ReAct 与普通单轮工具调用的核心差异。

---

## 7. 运行案例复盘

示例问题：

```text
华为最新的手机是哪一款？它的主要卖点是什么？
```

### 第 1 轮

模型认识到“最新”具有时效性，因此没有只依赖参数知识，而是调用：

```text
Search[华为 最新 手机 型号 主要 卖点]
```

这是合理的工具选择。

### 第 2 轮

模型发现第一次结果混有不同年份和不同产品线的信息，于是再次搜索确认：

```text
Search[华为 最新发布 手机 型号 及其 主要 特性]
```

这体现了图片中所说的“动态规划和纠错能力”：下一步行动不是预先写死的，而是由上一轮 Observation 决定。

### 第 3 轮

模型根据两次搜索结果执行 `Finish[...]`。

不过，这次最终答案仍存在明显质量风险：

- 搜索摘要包含未来年份或可疑信息，模型没有核验发布日期；
- 将“官网页面中出现”近似推断为“最新发布”，证据链不充分；
- 同时列出两个候选型号，没有严格回答“哪一款”；
- “影像、设计美学”等卖点缺少对应来源；
- 没有提供链接和发布日期；
- 使用了“可能包括”等模糊措辞，说明 Agent 并未真正解决歧义。

这个案例说明：

&gt; ReAct 能降低闭门造车式幻觉，但不会自动保证事实正确。工具质量、检索策略、证据校验和停止条件同样决定最终效果。

更合理的后续动作应包括：

1. 限定搜索华为官方网站；
2. 查询官网新闻或发布会页面，而非只看商品聚合页；
3. 比较产品发布日期；
4. 至少用第二个可信来源交叉验证；
5. 在答案中说明“最新发布”采用的判断口径。

---

## 8. 实践经验总结

### 8.1 优势

#### 高可解释性 / 高可观测性

开发者能够看到：

- Agent 选择了哪个工具；
- 给工具传入了什么参数；
- 工具返回了什么结果；
- Agent 在哪一步结束；
- 失败发生在模型、解析器还是工具层。

工程面试中最好补充一句：生产系统的可解释性应主要依靠**结构化决策摘要、Action、Observation、引用和状态日志**，而不是依赖向用户展示模型的完整内部思维过程。

#### 动态规划和纠错能力

ReAct 不要求开始时就生成完美计划。它可以根据环境反馈：

- 修改搜索关键词；
- 更换工具；
- 补充缺失证据；
- 发现错误后重试；
- 在达到目标后提前终止。

这使它适合开放环境和信息不完整的任务。

### 8.2 局限

#### 强依赖 LLM 能力

模型需要同时完成：

- 理解问题；
- 选择正确工具；
- 生成合法参数；
- 理解 Observation；
- 判断信息是否充足；
- 组织最终答案。

小模型可能在格式遵循、长上下文、工具选择和停止判断上表现不稳定。

#### 执行效率不高

每一轮通常都需要一次模型推理，可能还伴随一次外部工具请求，因此会增加：

- 端到端延迟；
- 输入和输出 Token；
- 模型调用费用；
- 外部 API 成本；
- 超时和失败概率。

粗略估算：若执行 `n` 轮，每轮都重新携带历史，则总输入 Token 往往不是线性常数开销，而会随着历史增长而持续增加。

#### 可能陷入局部最优

典型表现包括：

- 反复使用同一个近似搜索词；
- 过早相信第一个看似合理的结果；
- 一直修补当前方案，不尝试替代路径；
- 已经无法取得新信息，却继续调用工具；
- 过早 `Finish`，输出证据不足的答案。

#### 其他工程风险

- 工具返回恶意指令导致间接 Prompt Injection；
- 工具权限过大导致误操作；
- Observation 太长造成上下文膨胀；
- 外部 API 限流、超时或返回脏数据；
- 多工具重名或描述不清，导致路由错误；
- 没有幂等设计，重试可能重复下单、发消息或写数据。

### 8.3 调试技巧

#### 检查完整 Prompt

确认每轮实际送入模型的内容，包括：

- system 规则是否完整；
- 工具名称、描述、参数约束是否准确；
- Question 是否被错误改写；
- History 是否丢失、重复或顺序错乱；
- Observation 是否过长；
- 是否混入了不可信网页指令。

#### 分析原始模型输出和工具 I/O

至少记录：

```text
trace_id
step_id
model_name
prompt/version
raw_model_output
parsed_action
tool_name
tool_input
tool_output
latency
token_usage
error_type
```

调试时要区分：

- 模型没有选择正确工具；
- 模型选对了工具，但参数错误；
- 解析器没有正确解析模型输出；
- 工具本身失败；
- 工具返回正常，但模型误读结果；
- 停止条件设计不合理。

#### 调整 Prompt 中的示例

Few-shot 示例最好覆盖：

- 正常调用工具；
- 工具无结果后的重试；
- 工具报错后的降级；
- 多来源冲突时的核验；
- 信息充足后及时 `Finish`；
- 不需要工具时直接结束。

#### 尝试不同模型或参数

- `temperature`：工具选择型任务通常设低；
- 最大输出长度：必须容纳完整结构化动作；
- 模型能力：关注指令遵循、工具调用和长上下文能力；
- 超时和重试：区分模型超时与工具超时；
- 并发与缓存：降低重复查询的延迟和费用。

---

## 9. 对当前代码的 Code Review

### 9.1 做得好的地方

- Agent、模型客户端、工具管理器职责分离；
- 使用 `max_steps` 限制循环；
- 工具由宿主注册，不允许模型任意执行函数；
- `temperature=0` 有利于稳定输出；
- 每次 `run()` 清空历史；
- 搜索异常转换为文本 Observation，Agent 有机会继续处理；
- 教学代码短小，清晰展示 ReAct 的最小闭环。

### 9.2 解析协议不够健壮

当前正则：

```python
re.match(r&#34;(\w&#43;)\[(.*)\]&#34;, action_text, re.DOTALL)
```

存在的问题：

- `\w&#43;` 不支持带连字符或命名空间的工具名；
- 贪婪匹配可能吞入多余文本；
- 没有要求匹配到字符串末尾；
- 参数包含复杂括号时容易歧义；
- 无法表达多个具名参数和参数类型；
- `action.startswith(&#34;Finish&#34;)` 会把 `FinishWrong[...]` 也当作结束；
- 空字符串参数会被当成非法，但某些工具可能不需要参数。

工程上应优先使用模型原生 Tool Calling 和 JSON Schema，例如：

```json
{
  &#34;name&#34;: &#34;search&#34;,
  &#34;arguments&#34;: {
    &#34;query&#34;: &#34;华为官网 最新手机 发布日期&#34;
  }
}
```

并使用数据模型对参数进行类型校验。

### 9.3 缺少工具执行安全边界

当前代码直接执行：

```python
observation = tool_function(tool_input)
```

生产环境至少需要：

- 参数 Schema 校验；
- 工具白名单与用户权限校验；
- 单工具超时；
- 异常捕获与标准错误码；
- 重试和指数退避；
- 限流与熔断；
- 高风险操作二次确认；
- 幂等键；
- 沙箱隔离；
- 输出长度限制和敏感信息脱敏。

### 9.4 History 管理过于简单

当前 History 是字符串列表，会带来：

- 角色信息丢失；
- 长任务中上下文无限增长；
- 不能方便地统计单步状态；
- 难以支持并行工具调用；
- 不利于持久化、恢复和回放。

更合理的状态结构：

```python
{
    &#34;question&#34;: &#34;...&#34;,
    &#34;steps&#34;: [
        {
            &#34;step_id&#34;: 1,
            &#34;action&#34;: {&#34;tool&#34;: &#34;search&#34;, &#34;args&#34;: {&#34;query&#34;: &#34;...&#34;}},
            &#34;observation&#34;: &#34;...&#34;,
            &#34;status&#34;: &#34;success&#34;,
            &#34;latency_ms&#34;: 320,
        }
    ],
    &#34;budget&#34;: {&#34;remaining_steps&#34;: 4},
}
```

长上下文下还需要 Observation 截断、摘要、检索式记忆或状态压缩。

### 9.5 终止机制不足

当前只有两种终止方式：

- 模型输出 `Finish[...]`；
- 达到 `max_steps` 后返回 `None`。

生产系统还应支持：

- 总 Token 预算；
- 总费用预算；
- 总时间预算；
- 单工具调用次数限制；
- 重复动作检测；
- 无新增信息检测；
- 用户取消；
- 达到最大步数时生成一份“基于已有证据的受限答案”，而非直接 `None`。

### 9.6 搜索质量和答案可信度不足

搜索工具只返回前三条摘要，没有：

- URL；
- 发布时间；
- 来源类型；
- 网页正文；
- 来源可信度；
- 多来源一致性检查。

对于“最新”“第一”“最高”等问题，应把时间和排序口径作为显式约束，并优先检索官方一手来源。

---

## 10. 生产级 ReAct 的推荐设计

### 10.1 控制面与数据面分离

```text
控制面：LLM 决策、状态机、预算、权限、重试、终止条件
数据面：Search、SQL、RAG、代码执行、业务 API 等具体工具
```

不要把所有逻辑都写进 Prompt。确定性的约束应由代码执行，例如权限、预算、参数类型和幂等性。

### 10.2 使用结构化输出

推荐动作协议：

```json
{
  &#34;type&#34;: &#34;tool_call&#34;,
  &#34;tool&#34;: &#34;search&#34;,
  &#34;arguments&#34;: {&#34;query&#34;: &#34;...&#34;},
  &#34;decision_summary&#34;: &#34;需要核验官方发布日期&#34;
}
```

最终响应协议：

```json
{
  &#34;type&#34;: &#34;final&#34;,
  &#34;answer&#34;: &#34;...&#34;,
  &#34;citations&#34;: [&#34;...&#34;],
  &#34;confidence&#34;: &#34;medium&#34;
}
```

结构化输出能减少正则解析错误，也便于监控和评测。

### 10.3 建立明确的状态机

```mermaid
stateDiagram-v2
    [*] --&gt; Decide
    Decide --&gt; Validate: tool_call
    Validate --&gt; Execute: 参数及权限合法
    Validate --&gt; Repair: 参数或权限非法
    Repair --&gt; Decide
    Execute --&gt; Observe
    Observe --&gt; Decide: 尚未完成且预算充足
    Decide --&gt; Finalize: final
    Observe --&gt; Finalize: 预算耗尽/无法继续
    Finalize --&gt; [*]
```

### 10.4 工具返回值标准化

不要只返回一段自然语言，建议统一为：

```json
{
  &#34;ok&#34;: true,
  &#34;data&#34;: {},
  &#34;error&#34;: null,
  &#34;metadata&#34;: {
    &#34;source&#34;: &#34;...&#34;,
    &#34;timestamp&#34;: &#34;...&#34;,
    &#34;latency_ms&#34;: 123
  }
}
```

这样模型和 Runtime 都更容易判断工具成功、失败或部分成功。

### 10.5 安全设计

- 将工具输出视为不可信数据，而不是高优先级指令；
- 网页中的“忽略之前要求”等文本不能改变系统规则；
- 数据库工具默认只读，限制表、字段和返回行数；
- 写操作采用审批、人机确认和幂等机制；
- 对外发消息、支付、删除等动作设置更高权限等级；
- 对日志中的密钥、手机号、身份证等敏感信息脱敏。

### 10.6 可观测性与评测

线上指标至少包括：

| 维度 | 指标示例 |
|---|---|
| 任务效果 | task success rate、答案正确率、引用准确率 |
| 工具调用 | 工具选择准确率、参数正确率、调用成功率 |
| 效率 | 平均步骤数、Token、费用、P50/P95 延迟 |
| 稳定性 | 格式错误率、超时率、重试率、死循环率 |
| 安全 | 越权调用率、危险动作拦截率、注入攻击成功率 |

评测集不能只看最终答案，还应检查整条轨迹：是否选择了正确工具、是否使用了必要证据、是否有冗余调用、是否在合理时间停止。

---

## 11. 降低延迟和 Token 成本的方法

- 简单问题先做路由，不需要工具时直接回答；
- 用小模型完成工具路由，大模型负责复杂综合；
- 缓存相同或相近查询；
- 对 Observation 去噪、截断和摘要；
- 避免每轮重复注入不变的大段说明；
- 对互不依赖的只读工具进行并行调用；
- 设置重复动作检测和早停；
- 限制工具结果数量，只保留与任务相关的字段；
- 对高频固定流程采用工作流，对不确定节点局部使用 ReAct；
- 通过离线评测选择满足效果要求的最小模型。

面试中的一个成熟判断是：

&gt; 不是所有问题都应该交给自由循环 Agent。能用确定性工作流解决的部分，应优先使用工作流；只在需要动态决策和环境反馈的节点使用 ReAct。

---

## 12. 常见失败模式及解决方案

| 失败模式 | 可能原因 | 解决方案 |
|---|---|---|
| 不调用应该调用的工具 | 工具描述不清、模型能力不足 | 优化描述和示例，增加路由器或强制规则 |
| 调错工具 | 工具功能重叠、名称含糊 | 减少重叠，说明适用/不适用场景 |
| 参数格式错误 | 文本协议脆弱 | 原生 Tool Calling &#43; JSON Schema &#43; 校验/修复 |
| 重复调用同一工具 | Observation 无增量、无重复检测 | 动作去重，记录信息增益，设置调用上限 |
| 过早结束 | 停止条件模糊 | 定义证据充分性，要求引用和交叉验证 |
| 一直不结束 | 模型停止判断差 | 步数、时间、费用预算与强制 Finalize |
| 工具结果被错误理解 | 返回文本冗长或结构不清 | 标准化结构、字段裁剪、结果摘要 |
| 搜索答案仍然幻觉 | 来源不可信或证据不足 | 官方源优先、多源校验、发布日期过滤 |
| 工具被网页提示词劫持 | 间接 Prompt Injection | 数据/指令隔离、内容清洗、权限控制 |
| 重试造成重复副作用 | 工具非幂等 | 幂等键、事务、执行前检查、人工确认 |

---

## 13. 高频面试题与参考回答

### Q1：什么是 ReAct？

ReAct 是一种把语言模型推理和外部行动交替组织起来的 Agent 范式。模型根据问题与历史 Observation 决定下一步工具调用，环境执行后把结果反馈给模型，循环直到生成最终答案。它的核心是通过真实环境反馈形成闭环，而不是只依赖模型参数知识一次性作答。

### Q2：ReAct 为什么能减少幻觉？能彻底消除吗？

它能通过搜索、数据库或业务 API 获取外部证据，降低仅凭参数知识编造答案的概率；同时模型可以根据 Observation 修正计划。但它不能彻底消除幻觉，因为工具结果可能错误，模型也可能选错工具、误读结果、虚构未被证据支持的结论或过早停止。

### Q3：ReAct 的核心组件有哪些？

至少包括：LLM 决策器、Prompt/工具描述、工具注册与执行层、状态或轨迹存储、输出解析/结构化调用、终止条件，以及生产环境中的权限、预算、重试、日志和评测模块。

### Q4：为什么需要 `max_steps`？

它用于限制死循环、异常重试、Token 和费用失控。但只设置最大步数不够，还应增加时间、费用、单工具次数和重复动作等预算，并在预算耗尽时提供可解释的降级结果。

### Q5：文本版 `Action: Tool[input]` 有什么问题？

它依赖模型严格遵循格式，正则解析对换行、括号、额外文本和复杂参数都很敏感。生产中更适合使用原生 Tool Calling、JSON Schema 和严格参数校验。

### Q6：如何避免 Agent 死循环？

设置步数/时间/费用预算；检测完全相同或语义相近的重复动作；判断 Observation 是否带来新信息；限制单工具调用次数；在多次失败后切换策略或降级；必要时由状态机强制进入 Finalize。

### Q7：如何评测 ReAct Agent？

不能只看最终答案。应同时评测任务成功率、工具选择、参数正确性、证据引用、轨迹效率、步骤数、延迟、费用、格式错误、恢复能力和安全性。测试集还应覆盖工具超时、空结果、冲突结果、恶意工具输出等异常场景。

### Q8：如何设计工具描述？

需要写清工具用途、适用和不适用场景、参数含义与类型、返回值、限制、成本及副作用。功能重叠的工具应明确选择边界；高风险工具还要说明权限和确认要求。

### Q9：ReAct 如何处理 Prompt Injection？

工具内容全部视为不可信数据；系统指令与工具结果分层；明确禁止把网页文字当作系统命令；执行前由 Runtime 做权限和参数校验；高风险动作必须人工确认；使用攻击样本持续做红队评测。

### Q10：什么时候不适合使用 ReAct？

流程完全确定、追求极低延迟、操作风险很高且必须严格可控、或者简单问答不需要工具时，不应使用自由循环 ReAct。可以改用确定性工作流、规则引擎、单次工具调用或人工审批流程。

### Q11：如何把当前 Demo 改造成生产版本？

用原生 Tool Calling 替代正则；用结构化状态保存轨迹；给工具增加 Schema、权限、超时、重试和幂等；增加多维预算与重复检测；对搜索结果保留 URL、日期和来源；加入日志、Tracing、离线评测与线上指标；对危险操作增加审批；最后再根据效果和成本优化模型路由、缓存与并行调用。

### Q12：ReAct 的“可解释性”应该如何理解？

它主要提供过程可观测性：能看到工具选择、参数、返回值、引用和状态变化，这有助于调试和审计。但自然语言 Thought 不一定忠实反映模型真实计算过程，也不应默认向终端用户暴露。生产系统应优先记录结构化的决策摘要和执行证据。

---

## 14. 手写一个最小 ReAct 的伪代码

```python
def run(question, max_steps=5):
    state = {&#34;question&#34;: question, &#34;steps&#34;: []}

    for _ in range(max_steps):
        decision = llm.decide(
            question=state[&#34;question&#34;],
            tools=tool_schemas,
            history=state[&#34;steps&#34;],
        )

        if decision.type == &#34;final&#34;:
            return decision.answer

        validate_tool_call(decision)
        observation = execute_with_timeout_and_permission(decision)

        state[&#34;steps&#34;].append({
            &#34;action&#34;: decision,
            &#34;observation&#34;: observation,
        })

        if is_repeated_without_new_information(state):
            break

    return finalize_from_existing_evidence(state)
```

面试手写时要主动说明：真实系统还需要异常处理、权限控制、预算、可观测性、持久化和安全防护。

---

## 15. 一页速记

### 核心公式

```text
ReAct = Reasoning &#43; Acting &#43; Environment Feedback
```

### 基本循环

```text
Question → Thought → Action → Observation → ... → Finish
```

### 两个核心优点

- 动作和环境反馈可观测，便于调试、审计和定位故障；
- 能根据 Observation 动态规划与纠错。

### 三个主要局限

- 依赖模型的工具选择、参数生成和停止判断能力；
- 多轮 LLM &#43; 工具调用造成延迟、Token 和费用增加；
- 可能重复探索、过早结束或陷入局部最优。

### 四个调试抓手

- 检查完整 Prompt；
- 查看原始模型输出、解析结果和工具 I/O；
- 补充覆盖异常路径的 few-shot 示例；
- 对比模型、温度、预算和工具描述。

### 五个生产化关键词

```text
结构化 Tool Calling
状态机与预算
权限、幂等和沙箱
Tracing 与轨迹评测
Prompt Injection 防护
```

### 面试收尾金句

&gt; ReAct 的本质不是让模型无限自主，而是在受控 Runtime 中，让模型根据环境反馈做有限、可观测、可评测的动态决策。生产落地的关键不只是 Prompt，而是工具协议、状态管理、安全边界、停止条件和评测体系。


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlreact/  

