# Deep Research


## 0. 先用一句话理解 Deep Research

Deep Research 不是“让模型多搜索几次”，而是把复杂调研组织成一个可控制的 Agent 工作流：

```text
理解目标 → 制定计划 → 拆分并委派任务 → 多轮检索与反思
        → 汇总证据 → 统一引用 → 写入报告 → 回查需求
```

它的核心价值是让调研过程具备四种能力：

- **规划能力**：先形成 To-do，而不是直接搜索。
- **上下文管理能力**：将请求、中间资料和最终报告写入文件，避免所有内容都堆在消息历史中。
- **任务委派能力**：主 Agent 负责编排，子 Agent 负责具体资料研究。
- **过程控制能力**：通过搜索预算、反思工具、并发限制和报告规范约束 Agent。

可以将系统近似理解为：

```text
Deep Research = Agent Loop
              &#43; Planning
              &#43; Tool Use
              &#43; Sub-Agent Delegation
              &#43; Context Engineering
              &#43; Evidence Synthesis
```

---

## 1. Deep Research、普通搜索和 RAG 的区别

| 方案 | 核心动作 | 是否主动规划 | 是否多步执行 | 典型输出 |
|---|---|---:|---:|---|
| 普通搜索 | 根据关键词返回网页 | 否 | 否 | 搜索结果列表 |
| RAG | 检索相关片段后生成回答 | 通常较弱 | 通常较少 | 基于知识库的回答 |
| Tool Calling Agent | 模型自行选择和调用工具 | 有限 | 是 | 完成某项操作或回答 |
| Deep Research | 规划、委派、检索、反思、综合、验证 | 是 | 是，通常较长 | 带来源的结构化研究报告 |

Deep Research 可以使用搜索、RAG、数据库或 MCP 工具，但它们只是研究过程中的能力来源。真正使其“Deep”的，是围绕目标进行持续规划、证据收集、状态管理和结果校验的闭环。

---

## 2. 当前项目结构

```text
deep-research/
├── research_agent.ipynb       # 完整示例：创建模型、子 Agent、主 Agent 并执行调研
├── utils.py                    # Notebook 消息与 Prompt 的 Rich 格式化展示
└── research_agent/
    ├── __init__.py             # 对外导出 Prompt 和工具
    ├── prompts.py              # 主流程、研究子 Agent、委派策略三组 Prompt
    └── tools.py                # Tavily 搜索、网页抓取和反思工具
```

各文件的职责如下：

| 文件 | 关键内容 | 在架构中的角色 |
|---|---|---|
| [`research_agent.ipynb`](./research_agent.ipynb) | `create_deep_agent`、模型、子 Agent、文件后端 | 系统装配与运行入口 |
| [`research_agent/prompts.py`](./research_agent/prompts.py) | 工作流、搜索启发式、委派限制、报告规范 | Guiding Context，指导上下文 |
| [`research_agent/tools.py`](./research_agent/tools.py) | `tavily_search`、`think_tool` | Actionable Context，可执行能力 |
| [`utils.py`](./utils.py) | 格式化消息、工具调用和 Prompt | 调试与可观测性辅助 |

当前 Notebook 使用的模型配置为：

```python
model = ChatOpenAI(
    api_key=os.getenv(&#34;API_KEY&#34;),
    base_url=os.getenv(&#34;BASE_URL&#34;),
    model=&#34;qwen3-coder-plus&#34;,
    temperature=0.7,
)
```

`ChatOpenAI` 在这里是兼容 OpenAI 接口的客户端，实际模型由 `base_url` 和 `model` 决定，并不表示底层一定是 OpenAI 模型。

---

## 3. Deep Agent 核心架构

课件中的 Deep Agent 可以拆成四个核心模块：

1. **详细系统提示**：规定角色、工作流、停止条件和报告格式。
2. **规划工具**：通过 To-do 保存当前任务与进度。
3. **子 Agent**：把具体任务放进隔离上下文中执行。
4. **文件系统**：保存请求、笔记、中间结果和最终报告。

对应到本项目：

```mermaid
flowchart TD
    U[用户研究请求] --&gt; M[主 Deep Agent]

    P[主流程 Prompt&lt;br/&gt;RESEARCH_WORKFLOW_INSTRUCTIONS] --&gt; M
    D[委派 Prompt&lt;br/&gt;SUBAGENT_DELEGATION_INSTRUCTIONS] --&gt; M

    M --&gt; TODO[write_todos&lt;br/&gt;规划与状态跟踪]
    M --&gt; REQ[write_file&lt;br/&gt;research_request.md]
    M --&gt; TASK[task&lt;br/&gt;委派研究任务]

    TASK --&gt; S[research-agent 子 Agent]
    R[RESEARCHER_INSTRUCTIONS] --&gt; S
    S --&gt; SEARCH[tavily_search]
    S --&gt; THINK[think_tool]
    SEARCH --&gt; THINK
    THINK -.信息不足.-&gt; SEARCH

    S --&gt; FINDINGS[带来源的研究结果]
    FINDINGS --&gt; M
    M --&gt; SYN[综合、去重引用、组织结构]
    SYN --&gt; REPORT[write_file&lt;br/&gt;final_report.md]
    REPORT --&gt; VERIFY[回查用户请求并验证]
```

### 3.1 主 Agent 的职责

主 Agent 是 Planner、Coordinator 和 Writer，负责：

- 理解用户目标；
- 创建并更新 To-do；
- 保存原始研究请求；
- 判断应该使用几个子 Agent；
- 为子 Agent 编写清晰、独立的任务描述；
- 接收并综合不同子 Agent 的结果；
- 对引用进行统一编号和去重；
- 将最终报告写入文件；
- 回查原始需求，确认没有漏项。

它不应陷入每一条资料的搜索细节，否则主上下文会被大量网页内容和工具日志污染。

### 3.2 子 Agent 的职责

研究子 Agent 是 Researcher，负责：

- 只处理一个明确的研究主题；
- 从宽泛查询逐步收窄；
- 调用 `tavily_search` 获取资料；
- 每次搜索后调用 `think_tool` 评估证据和缺口；
- 达到停止条件后返回带来源的结构化结果。

Notebook 中的定义为：

```python
research_sub_agent = {
    &#34;name&#34;: &#34;research-agent&#34;,
    &#34;description&#34;: &#34;Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.。&#34;,
    &#34;system_prompt&#34;: RESEARCHER_INSTRUCTIONS.format(date=current_date),
    &#34;tools&#34;: [tavily_search, think_tool],
}
```

这里的 `description` 主要供主 Agent 选择子 Agent，`system_prompt` 决定子 Agent 如何执行任务，`tools` 决定它能够采取哪些动作。

### 3.3 文件系统的职责

Notebook 使用：

```python
backend = FilesystemBackend(
    root_dir=&#34;/tmp/deepagents/&#34;,
    virtual_mode=True,
)
```

主 Prompt 要求至少维护两个文件：

```text
/research_request.md   # 用户的原始研究问题
/final_report.md       # 最终调研报告
```

文件系统在这里不仅是输入输出设备，还是一种外部化工作记忆：

- 消息上下文适合保存近期对话和当前决策；
- To-do 适合保存结构化任务状态；
- 文件适合保存较长、可反复读取的资料和报告；
- 子 Agent 上下文适合保存某个局部任务的临时推理与工具结果。

这正是上下文工程中的“写入”和“隔离”。

---

## 4. 完整执行流程

`RESEARCH_WORKFLOW_INSTRUCTIONS` 定义了六步研究流程。

### 第一步：Plan

主 Agent 使用 `write_todos` 创建计划，例如：

```text
1. 规划调研任务
2. 保存用户请求
3. 委托子 Agent 研究
4. 整合结果并撰写报告
```

To-do 的价值不是展示一张清单，而是为长任务提供显式状态，使 Agent 知道：

- 当前在执行哪一步；
- 哪些步骤已完成；
- 失败后应该从哪里继续；
- 是否已经满足结束条件。

### 第二步：Save the request

将用户的原始问题写入 `/research_request.md`。

这样做可以防止长流程中的“目标漂移”：经过多轮搜索、委派和摘要后，Agent 仍能重新读取用户最初到底要求了什么。

### 第三步：Research

主 Agent 通过 `task()` 将具体研究任务交给 `research-agent`。

当前 Prompt 的默认策略是：

- 一般问题优先使用 **1 个综合型子 Agent**；
- 明确比较不同对象时，可以每个对象使用 1 个子 Agent；
- 只有主题确实相互独立时才并行拆分；
- 每轮最多并行 `3` 个研究单元；
- 最多进行 `3` 轮委派。

Notebook 中的参数：

```python
max_concurrent_research_units = 3
max_researcher_iterations = 3
```

### 第四步：Synthesize

主 Agent 汇总子 Agent 返回的发现：

- 删除重复事实；
- 解决不同来源之间的冲突；
- 按用户问题重新组织内容；
- 给每个唯一 URL 分配唯一编号；
- 避免不同子 Agent 各自使用的 `[1]`、`[2]` 直接发生冲突。

这一步不能简单拼接子 Agent 的回答。Deep Research 的最终质量很大程度上取决于综合与证据治理，而不是搜索次数。

### 第五步：Write Report

将正式报告写入 `/final_report.md`。Prompt 为三类问题提供了结构模板：

- 比较类：介绍 A、介绍 B、详细对比、结论；
- 列表/排名类：直接列举条目及细节；
- 概览类：主题概览、关键概念、结论。

报告还要求：

- 使用清晰标题；
- 默认以完整段落写作；
- 避免“我搜索了”“我发现了”等元叙述；
- 使用 `[1]`、`[2]` 形式的行内引用；
- 末尾集中列出 Sources。

### 第六步：Verify

重新读取 `/research_request.md`，检查：

- 是否回答了用户要求的每一个方面；
- 报告结构是否适合问题类型；
- 关键事实是否有来源；
- 引用编号是否连续、唯一；
- 最终文件是否成功生成。

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as 主 Agent
    participant F as 文件系统
    participant S as 研究子 Agent
    participant W as Web/Tavily

    U-&gt;&gt;M: 提交研究问题
    M-&gt;&gt;M: write_todos 制定计划
    M-&gt;&gt;F: 写入 research_request.md
    M-&gt;&gt;S: task 委派单一主题
    loop 搜索预算内迭代
        S-&gt;&gt;W: tavily_search
        W--&gt;&gt;S: 网页及正文
        S-&gt;&gt;S: think_tool 评估证据与缺口
    end
    S--&gt;&gt;M: 返回研究发现与 Sources
    M-&gt;&gt;M: 合并、去重、统一引用
    M-&gt;&gt;F: 写入 final_report.md
    M-&gt;&gt;F: 回读原始请求/最终报告
    M--&gt;&gt;U: 返回完成状态或报告摘要
```

---

## 5. 搜索子 Agent 的设计

### 5.1 搜索策略

`RESEARCHER_INSTRUCTIONS` 要求子 Agent 像人类研究员一样工作：

```text
读懂问题
  ↓
先做宽泛搜索
  ↓
评估已有信息和证据缺口
  ↓
执行更具体的搜索
  ↓
信息充分后立即停止
```

这种策略解决了两个常见问题：

- 一上来使用过窄关键词，错过关键概念或正确术语；
- 已经有足够信息后仍不断搜索，导致成本上升和上下文污染。

### 5.2 搜索预算

Prompt 中规定：

- 简单问题最多搜索 2～3 次；
- 复杂问题最多搜索 5 次；
- 找到 3 个以上相关示例/来源时可以停止；
- 最近两次搜索内容高度相似时停止；
- 5 次搜索后仍无合适来源也必须停止。

预算控制的本质是让 Agent 优化“单位 Token 的信息增益”：

```text
下一次搜索价值 ≈ 新增有效证据 - Token 成本 - 时间成本 - 噪声风险
```

### 5.3 `tavily_search`

当前工具的调用链为：

```mermaid
flowchart LR
    Q[query] --&gt; T[Tavily 搜索 URL]
    T --&gt; R[遍历搜索结果]
    R --&gt; H[httpx 获取网页 HTML]
    H --&gt; M[markdownify 转 Markdown]
    M --&gt; O[拼接标题、URL、正文]
```

关键实现：

```python
search_results = tavily_client.search(
    query,
    max_results=max_results,
    topic=topic,
)

for result in search_results.get(&#34;results&#34;, []):
    content = fetch_webpage_content(result[&#34;url&#34;])
```

这个工具不是只返回搜索摘要，而是先用 Tavily 发现 URL，再自行抓取网页全文并转成 Markdown。优点是信息更完整，代价是网页正文可能非常长、噪声较多，也更容易受到网页中的提示注入内容影响。

### 5.4 `think_tool`

`think_tool` 要求子 Agent 在每次搜索后回答四个问题：

1. 已经获得了哪些具体信息？
2. 还缺少哪些关键内容？
3. 当前证据是否足以形成可靠答案？
4. 下一步应该继续搜索还是结束？

工具本身非常轻量：

```python
return f&#34;Reflection recorded: {reflection}&#34;
```

它不会真正调用额外的推理服务，也没有把反思写入独立数据库。它的主要作用是通过一次显式工具调用打断“搜索—搜索—搜索”的惯性，让模型在循环中产生一个可见的决策检查点。

---

## 6. 为什么要使用子 Agent

课件中的协作模式是：主 Agent 负责规划和协调，不同子 Agent 分别负责研究、编码、审查等任务，并通过共享文件系统交换成果。

### 6.1 上下文隔离

假设一次调研需要分析三个相互独立的主题。如果全部放进主 Agent：

```text
主上下文 = 用户请求
         &#43; 主题 A 的网页和工具记录
         &#43; 主题 B 的网页和工具记录
         &#43; 主题 C 的网页和工具记录
         &#43; 综合报告
```

使用子 Agent 后：

```text
主上下文 = 用户请求 &#43; 任务计划 &#43; A/B/C 的压缩结论 &#43; 最终报告

子上下文 A = 主题 A &#43; A 的搜索过程
子上下文 B = 主题 B &#43; B 的搜索过程
子上下文 C = 主题 C &#43; C 的搜索过程
```

主 Agent 只接收每个子任务的高密度结果，而不必承载所有检索过程。

### 6.2 专业化

不同子 Agent 可以拥有不同的：

- 系统提示；
- 工具集合；
- Token 与搜索预算；
- 权限；
- 输出格式；
- 评估标准。

例如，研究 Agent 只需要搜索和反思工具；代码 Agent 需要读写代码和运行测试；审查 Agent 最好只读，避免既实现又自行批准。

### 6.3 并行执行

当问题明确包含独立维度时，可以并行：

```text
主 Agent
├── 子 Agent A：研究产品能力
├── 子 Agent B：研究价格与市场
└── 子 Agent C：研究风险与竞品
```

并行可以降低墙上时钟时间，但不一定降低总 Token 成本。拆分过细还会增加任务描述、结果重复、引用合并和通信开销。

### 6.4 什么时候不应拆分

以下情况通常使用一个子 Agent 更合适：

- 问题简单且信息强相关；
- 各子主题需要共享大量背景；
- 搜索范围很小；
- 子任务之间存在强顺序依赖；
- 多 Agent 的协调成本高于研究本身。

当前 Prompt 采用“默认一个子 Agent，明确比较时才并行”的策略，本质上是在平衡上下文隔离与通信成本。

---

## 7. Agent Skills：按需加载能力

&gt; 本节来自课件架构图，是对当前 Deep Research 项目的扩展理解；当前目录中没有 `SKILL.md`、技能匹配器或按需资源加载代码。

### 7.1 三层技能结构

课件将 Skill 分为三层：

| 层级 | 内容 | 加载时机 | 作用 |
|---|---|---|---|
| Layer 1：Metadata | 名称、描述、标签、适用场景 | Agent 启动时 | 让模型知道“有什么技能” |
| Layer 2：Instructions | `SKILL.md` 中的详细步骤与约束 | 任务匹配后 | 告诉模型“具体怎样做” |
| Layer 3：Scripts &amp; References | `.py`、`.md`、`.json` 等资源 | 执行到需要时 | 提供脚本、模板、知识与数据 |

```mermaid
flowchart TD
    A[Agent 启动] --&gt; M[只加载所有 Skill 元数据]
    M --&gt; U[接收用户请求]
    U --&gt; MATCH[匹配相关 Skill]
    MATCH --&gt; I[加载相关 SKILL.md 指令]
    I --&gt; NEED{执行时需要资源吗}
    NEED -- 是 --&gt; R[加载特定脚本/参考文件]
    NEED -- 否 --&gt; E[直接执行]
    R --&gt; E
```

这种设计也叫 Progressive Disclosure 或 Lazy Loading：先暴露低成本概览，匹配后再逐层展开细节。

### 7.2 MCP 急切加载与 Skill 惰性加载

课件对比了两种能力加载方式：

#### MCP 式急切加载（Eager Loading）

```text
建立连接
  → tools/list
  → 返回全部工具定义和 JSON Schema
  → 立即放入模型上下文
```

优点：

- 机制直接；
- 工具立即可用；
- 不需要额外的技能匹配过程。

问题：

- 工具很多时，工具定义占用大量 Token；
- 无关工具会干扰模型选择；
- 上下文中同时存在相似 Schema，可能降低参数生成准确率；
- 工具列表越大，安全审查和权限选择越复杂。

#### Skill 式惰性加载（Lazy Loading）

```text
启动时只加载技能元数据
  → 根据用户请求匹配少数技能
  → 加载详细说明
  → 执行时再读取脚本和参考资源
```

优点：

- 减少无关上下文；
- 能力说明可以比单个工具 Schema 更丰富；
- 便于沉淀完整工作流、脚本和模板；
- 提升工具选择的聚焦程度。

代价：

- 需要可靠的技能匹配；
- 可能发生漏召回；
- 加载链路更复杂；
- 需要管理 Skill 版本、依赖和权限。

### 7.3 如何理解课件中的 Token 数字

课件使用“100 个工具约 30,000 tokens”和“Skill 加载后降低约 70%”作为示意。这个方向是合理的，但数字不是通用定律：

```text
实际节省比例 = 工具数量
             × 每个 Schema 的平均长度
             × 当前任务所需工具比例
             × Host 的工具注入策略
             × 模型 Tokenizer 差异
```

生产系统应该直接测量：

- 启动时工具定义 Token；
- 每轮模型输入 Token；
- 被选中工具数；
- 技能召回率；
- 工具误选率；
- 最终任务成功率和总成本。

### 7.4 Skills 如何扩展当前项目

如果为本项目增加 Skills，可以设计为：

```text
skills/
├── company-research/
│   ├── SKILL.md
│   ├── source-quality.md
│   └── scripts/
│       └── normalize_citations.py
├── product-comparison/
│   ├── SKILL.md
│   └── comparison-template.md
└── paper-review/
    ├── SKILL.md
    └── evidence-levels.json
```

主 Agent 启动时只看技能名称和描述；收到“比较三家公司的 AI 战略”后，再加载 `company-research` 和 `product-comparison` 的详细指令及模板。

---

## 8. 从上下文工程理解本项目

Deep Research 的关键不是单个 Prompt，而是对上下文进行写入、选择、压缩和隔离。

| 上下文策略 | 当前项目实现 | 解决的问题 |
|---|---|---|
| 写入 Write | `write_todos`、`write_file` | 保存计划、请求和报告，减少目标漂移 |
| 选择 Select | 主 Agent 选择子 Agent；子 Agent 选择搜索词 | 只获取当前阶段需要的信息 |
| 压缩 Compress | 子 Agent 向主 Agent 返回总结而非完整搜索轨迹 | 降低主上下文长度 |
| 隔离 Isolate | `research-agent` 独立上下文 | 避免网页正文和研究细节污染主 Agent |

三类上下文也能一一对应：

| 类型 | 项目中的例子 |
|---|---|
| Guiding Context | 三组 Prompt、报告结构、搜索停止条件 |
| Information Context | 用户请求、To-do、网页正文、研究结果、报告文件 |
| Actionable Context | `task`、`write_todos`、`write_file`、`tavily_search`、`think_tool` |

---

## 9. Notebook 实际运行轨迹

Notebook 使用的示例问题是：

&gt; 给我做一个英伟达最新 GPU 型号的调研报告。

已保存的输出显示主 Agent 实际执行了：

1. 调用 `write_todos`，创建四项计划；
2. 调用 `write_file`，将问题保存到 `/research_request.md`；
3. 更新 To-do，将“委托子代理研究”标记为进行中；
4. 调用 `task`，选择 `research-agent` 并传入具体研究范围；
5. 接收子 Agent 返回的英文研究报告和三个来源；
6. 将研究任务标记为完成；
7. 主 Agent 整合并翻译内容，写入 `/final_report.md`；
8. 将所有 To-do 标记为完成；
9. 向用户返回简短完成说明。

这个运行记录证明了 To-do、文件写入和子 Agent 委派链路已经工作。

同时也暴露了一个重要现象：示例输出中没有看到主 Agent 在结束前显式调用 `read_file` 回读 `/research_request.md`。也就是说，Prompt 虽然规定了 Verify，但本次轨迹并未完整执行该步骤。**Prompt 中写了流程，不等于运行时一定严格遵守。**

另外，Notebook 中的 GPU 报告只是当时的一次 Agent 输出，不能当作长期有效的“最新 GPU”资料；涉及“最新”时必须基于执行当天重新搜索并优先核对官方来源。

---

## 10. 当前实现值得注意的问题

### 10.1 所谓“硬限制”目前主要是 Prompt 约束

搜索 2～5 次、最多 3 轮委派、最多并发 3 个研究单元，都被写进了 Prompt，但当前目录没有看到独立的运行时计数器或强制中断逻辑。

模型可能遵守，也可能遗漏。真正的硬限制应该由代码保证，例如：

```python
if state.search_count &gt;= MAX_SEARCH_CALLS:
    raise SearchBudgetExceeded()
```

或使用中间件在工具执行前检查预算。

### 10.2 主 Agent 仍然拥有搜索工具

创建主 Agent 时传入了：

```python
agent = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt=INSTRUCTIONS,
    subagents=[research_sub_agent],
    ...
)
```

这里的 `tools` 包含 `tavily_search` 和 `think_tool`。虽然主 Prompt 要求“研究必须委派给子 Agent”，但主 Agent 在权限上仍然可以直接搜索。

更严格的最小权限设计应考虑：

- 主 Agent 只拥有规划、文件和委派工具；
- 搜索工具只授予研究子 Agent；
- 用工具权限保证职责隔离，而不是只靠自然语言约束。

### 10.3 网页全文直接进入上下文

`fetch_webpage_content()` 会抓取完整 HTML 并转换为 Markdown，没有看到以下治理：

- 正文抽取；
- 长度上限；
- 按 Token 截断；
- 内容类型校验；
- 重复内容去除；
- 广告和导航过滤；
- Prompt Injection 检测；
- URL/网络访问范围限制。

网页内容属于不可信输入。Agent 应把网页中的“忽略之前指令”“执行某个工具”等文字只视为资料，不能视为系统指令。

### 10.4 搜索结果质量控制不足

当前 `tavily_search` 默认 `max_results=1`，且没有来源分级。一次搜索只抓一个结果时，容易：

- 被单一来源误导；
- 无法交叉验证；
- 混用官方资料、媒体和销售页面；
- 对时间敏感事实使用过期资料。

可引入来源优先级：

```text
官方文档/论文/监管文件
  &gt; 权威数据库与主流媒体
  &gt; 专业分析
  &gt; 博客、论坛、聚合或销售页面
```

关键事实至少应尽量由两个独立来源交叉验证。

### 10.5 引用合并只靠模型

“每个唯一 URL 只使用一个编号”的规则写在 Prompt 中，但没有确定性的 URL 规范化和去重程序。

以下地址可能实际指向同一来源：

```text
https://example.com/report
https://example.com/report/
https://example.com/report?utm_source=xxx
```

更可靠的方式是用代码完成：

- URL 规范化；
- 去除追踪参数；
- 按 canonical URL 去重；
- 生成连续编号；
- 校验正文引用与 Sources 是否一致。

### 10.6 错误被包装成普通文本

网页抓取失败时返回：

```python
return f&#34;Error fetching content from {url}: {str(e)}&#34;
```

这会让工具整体表现为“成功返回了一段文本”，模型需要自行判断它其实是错误。生产实现更适合返回结构化状态：

```json
{
  &#34;ok&#34;: false,
  &#34;url&#34;: &#34;...&#34;,
  &#34;error_type&#34;: &#34;timeout&#34;,
  &#34;retryable&#34;: true
}
```

### 10.7 日期在创建子 Agent 时固化

`current_date` 在 Notebook 创建 `research_sub_agent` 时计算。如果 Agent 服务长期运行，日期会停留在启动日，而不是每次请求的当天。

时间敏感研究应在每次运行时注入当前日期，并记录资料发布时间、检索时间和事实截止时间。

### 10.8 可复现性和测试不足

当前目录没有依赖锁定文件和自动化测试，复现实验需要自行确定：

- `deepagents` 版本；
- LangChain/LangGraph 版本；
- `tavily-python`、`httpx`、`markdownify` 版本；
- 模型服务兼容性；
- 环境变量名称和含义。

至少应补充：

- `requirements.txt` 或 `pyproject.toml`；
- `.env.example`，只写变量名，不写密钥；
- 工具单元测试；
- 模拟搜索结果的端到端测试；
- 报告引用完整性测试；
- 搜索预算与委派预算测试。

---

## 11. 更可靠的生产级设计

```mermaid
flowchart TD
    U[用户请求] --&gt; C[请求分类与风险判断]
    C --&gt; P[结构化研究计划]
    P --&gt; B[预算管理器&lt;br/&gt;时间/Token/搜索/并发]
    B --&gt; D[任务调度器]

    D --&gt; S1[研究子 Agent 1]
    D --&gt; S2[研究子 Agent 2]
    D --&gt; SN[研究子 Agent N]

    S1 --&gt; G[受控搜索网关]
    S2 --&gt; G
    SN --&gt; G

    G --&gt; Q[来源质量与安全过滤]
    Q --&gt; E[结构化证据库]
    E --&gt; Y[事实冲突检测与综合]
    Y --&gt; R[报告生成]
    R --&gt; V[自动验证器]
    V --&gt;|通过| O[最终报告]
    V --&gt;|缺少证据/漏项| D
```

建议补充以下组件：

### 11.1 结构化研究计划

每个研究单元至少包含：

```json
{
  &#34;question&#34;: &#34;需要回答的具体问题&#34;,
  &#34;scope&#34;: [&#34;包含内容&#34;],
  &#34;exclusions&#34;: [&#34;不包含内容&#34;],
  &#34;preferred_sources&#34;: [&#34;官方文档&#34;, &#34;论文&#34;],
  &#34;freshness_requirement&#34;: &#34;2026-08-23 前最新&#34;,
  &#34;output_schema&#34;: &#34;findings &#43; evidence &#43; sources&#34;
}
```

### 11.2 结构化证据

不要只让子 Agent 返回长文章，可返回：

```json
{
  &#34;claim&#34;: &#34;待写入报告的事实&#34;,
  &#34;evidence&#34;: &#34;来源中的关键内容&#34;,
  &#34;source_url&#34;: &#34;https://...&#34;,
  &#34;source_title&#34;: &#34;...&#34;,
  &#34;published_at&#34;: &#34;...&#34;,
  &#34;retrieved_at&#34;: &#34;...&#34;,
  &#34;source_type&#34;: &#34;official&#34;,
  &#34;confidence&#34;: 0.9
}
```

主 Agent 再基于证据对象生成报告，引用会更稳定，也更容易自动审计。

### 11.3 独立验证器

最终报告生成后，使用确定性程序或只读审查 Agent 检查：

- 用户问题中的每个子问题是否有对应章节；
- 每个重要事实是否带引用；
- 所有引用编号是否存在；
- Sources 中是否存在未被引用的来源；
- 是否出现相互矛盾的数字；
- 是否把旧资料写成“最新”；
- 是否包含网页提示注入造成的异常指令或内容。

### 11.4 可观测性

至少记录：

- 每次模型和工具调用耗时；
- 输入/输出 Token；
- 搜索次数和命中来源；
- 子 Agent 数量和重试次数；
- 报告验证失败原因；
- 最终成本；
- 用户或评审对报告质量的评分。

---

## 12. 面试常见问题

### 12.1 Deep Research 为什么需要主 Agent 和子 Agent？

主 Agent 负责全局目标、规划、任务调度和证据综合；子 Agent 负责局部研究。这样既能隔离网页和工具产生的大量上下文，又能为不同任务配置不同 Prompt、工具、预算与权限。代价是增加通信、去重、同步和调试成本，因此简单任务通常只使用一个研究子 Agent。

### 12.2 `think_tool` 的作用是什么？

它是研究循环中的显式反思检查点。每次搜索后，Agent 必须评估已有证据、信息缺口和继续搜索的价值，从而减少无目的工具调用。当前实现只是回显反思文本，不是真正的独立验证器或持久化记忆。

### 12.3 为什么要把用户请求写入文件？

长任务经过多轮委派和摘要后容易目标漂移。将原始请求外部化保存，可以在最终验证阶段重新读取并逐项核对；同时，文件比持续堆积在消息上下文中更适合保存长文本和稳定状态。

### 12.4 多子 Agent 一定比单 Agent 好吗？

不一定。多 Agent 的优势是上下文隔离、专业化和并行；缺点是额外 Token、重复研究、引用冲突和协调复杂度。只有问题存在明确独立维度或比较对象时才值得拆分。

### 12.5 Skills 与普通工具有什么区别？

普通工具主要描述一个可调用动作及参数 Schema；Skill 通常封装更完整的任务知识，包括适用场景、详细流程、约束、脚本、模板和参考资料。Skill 可以先加载元数据，任务匹配后再加载详细指令和资源，从而减少无关上下文。

### 12.6 Skills 能替代 MCP 吗？

不能简单替代。MCP 解决外部能力的标准连接、发现和调用；Skills 解决任务知识、操作流程和资源的组织与按需披露。两者可以组合：Skill 描述“何时以及怎样完成某类任务”，底层动作由 MCP 工具提供。

### 12.7 如何保证报告引用可靠？

不能只要求模型“带引用”。还要做来源分级、时间校验、交叉验证、URL 规范化、引用去重、正文与 Sources 一致性检查，并将关键 claim 与具体证据绑定。

### 12.8 如何防止 Deep Research 无限搜索？

同时使用两层控制：Prompt 中给出停止启发式，运行时用代码强制限制搜索次数、总 Token、总耗时、子 Agent 数量和委派轮数。只有 Prompt 约束不能算真正的硬限制。

---

## 13. 核心总结

当前项目展示了一个最小但完整的 Deep Research 原型：

```text
主 Agent
  ├── 详细工作流 Prompt
  ├── To-do 规划与状态跟踪
  ├── 文件系统工作记忆
  ├── task 子 Agent 委派
  └── 研究结果综合与报告生成

研究子 Agent
  ├── 独立系统 Prompt
  ├── Tavily 搜索与网页抓取
  ├── 搜索后的显式反思
  └── 带来源的研究结果
```

三张课件图可以统一成一个上下文工程观点：

1. **用 To-do 和文件把长任务状态写到模型上下文之外。**
2. **用子 Agent 隔离不同任务的搜索过程，只向主 Agent 返回高密度结论。**
3. **用 Skills 的分层加载避免一次性把所有能力和资源塞进上下文。**
4. **让主 Agent 专注规划与协调，让专业子 Agent 执行研究、编码或审查。**
5. **Prompt 负责指导行为，运行时代码负责真正执行预算、权限和安全边界。**

最终要记住：Deep Research 的效果不只由模型决定，而由**任务分解、上下文管理、工具质量、来源治理、运行时约束和结果验证**共同决定。

---


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mldeep-research/  

