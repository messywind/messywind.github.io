# MCP


## 0. 面试时先这样回答

### 30 秒版本

MCP（Model Context Protocol）是连接 AI 应用与外部工具、数据和提示模板的开放协议。它采用 Host、Client、Server 架构，协议消息通常基于 JSON-RPC 2.0。Server 可以暴露 Tools、Resources、Prompts，Client 负责能力发现和调用，Host 负责模型编排、权限确认、上下文管理与安全边界。它解决的是不同 Agent 框架重复适配外部系统的 `M × N` 集成问题，但不会自动解决模型规划、上下文压缩、权限控制和业务可靠性。

### 2 分钟版本

MCP 可以类比 AI 时代的“外设协议”：上层 Agent 不需要为数据库、文件系统、搜索、企业 API 分别编写私有适配，而是通过统一协议完成初始化、能力协商、工具发现和工具调用。

一次典型流程是：Host 创建 MCP Client，Client 与 Server 建立 stdio 或 Streamable HTTP 连接，双方执行 `initialize` 能力协商；Client 用 `tools/list` 获取工具及 JSON Schema；Host 把合适的工具描述提供给模型；模型选择工具后，Client 发出 `tools/call`；Server 执行业务逻辑并返回结构化结果；Host 再决定是否将结果写入模型上下文。

生产落地时我会重点关注四点：第一，工具接口要小而清晰，Schema、错误码和幂等语义稳定；第二，远程服务必须补齐认证、授权、TLS、审计和租户隔离；第三，对超时、重试、限流、取消和长任务进行治理；第四，MCP 返回值仍然占上下文窗口，要做工具筛选、结果裁剪、摘要和状态隔离。

---

## 1. MCP 解决什么问题

没有统一协议时，`M` 个 Agent/AI 应用接入 `N` 个外部系统，可能形成大量定制适配：

```text
AI 应用 A ── GitHub 私有适配
         ├── 数据库私有适配
         └── 企业搜索私有适配

AI 应用 B ── GitHub 私有适配
         ├── 数据库私有适配
         └── 企业搜索私有适配
```

MCP 把集成边界标准化：

```text
多个 AI Host / Agent  ←── MCP ──→  多个工具与数据服务
```

核心价值：

- 统一能力描述：工具名、说明、输入 Schema、输出内容有统一表达方式。
- 动态发现：Client 可以在运行时查询 Server 有哪些能力。
- 解耦模型与业务系统：切换模型或 Agent 框架时，Server 侧业务能力可以复用。
- 本地与远程统一：既能通过 stdio 拉起本地进程，也能连接远程 HTTP 服务。
- 生态复用：同一个 MCP Server 可以被多个兼容 Host 使用。

需要主动说明的边界：

- MCP 是“连接与交互协议”，不是 Agent 推理框架。
- MCP 不负责决定何时调用工具，这通常由 Host、模型或工作流负责。
- MCP 不等于 RAG；RAG 是检索增强生成方案，MCP 可以把检索能力暴露出来。
- MCP 不天然保证安全，协议之上仍需认证、授权、审计和沙箱。
- 接入 MCP 不代表上下文免费，工具定义与工具结果仍会消耗 Token。

---

## 2. 核心架构：Host、Client、Server

```mermaid
flowchart LR
    U[用户] --&gt; H[Host / AI 应用]
    H --&gt; L[LLM 与 Agent 编排]
    H --&gt; C1[MCP Client A]
    H --&gt; C2[MCP Client B]
    C1 &lt;--&gt;|stdio / Streamable HTTP| S1[MCP Server：天气]
    C2 &lt;--&gt;|stdio / Streamable HTTP| S2[MCP Server：数学]
    S1 --&gt; API[天气 API]
    S2 --&gt; CALC[计算服务]
```

### Host

Host 是用户真正使用的 AI 应用，例如 IDE、桌面助手、企业 Agent 平台。

典型职责：

- 管理模型会话和 Agent 循环。
- 创建、管理一个或多个 MCP Client。
- 决定把哪些工具描述放入当前模型上下文。
- 执行用户授权、敏感操作确认和结果展示。
- 管理不同 Server 之间的安全隔离。

### Client

Client 是 Host 内连接某个 MCP Server 的协议客户端。常见实现中，一个 Client 会维护到一个 Server 的会话。

典型职责：

- 建立连接并完成初始化、能力协商。
- 调用 `tools/list`、`tools/call` 等协议方法。
- 接收 Server 的响应和通知。
- 把协议对象转换为 Host/模型能够使用的工具描述。

### Server

Server 是对外提供上下文和操作能力的程序，可以是本地子进程，也可以是远程服务。

典型职责：

- 声明自身能力。
- 暴露 Tools、Resources、Prompts。
- 校验请求并调用数据库、文件系统或企业 API。
- 返回结构化结果与明确错误。

面试易错点：MCP Server 不等于传统机器上的“服务器”。通过 stdio 运行的本地 Python 子进程也是 MCP Server。

---

## 3. 协议层：JSON-RPC 与生命周期

MCP 的协议消息通常建立在 JSON-RPC 2.0 之上，主要包含：

- Request：有 `id`，期待 Response。
- Response：通过相同 `id` 关联请求，返回 `result` 或 `error`。
- Notification：没有 `id`，不要求响应。

### 典型生命周期

```mermaid
sequenceDiagram
    participant H as Host/LLM
    participant C as MCP Client
    participant S as MCP Server

    C-&gt;&gt;S: initialize（协议版本、Client 能力）
    S--&gt;&gt;C: Server 信息与能力
    C-&gt;&gt;S: initialized 通知
    C-&gt;&gt;S: tools/list
    S--&gt;&gt;C: 工具名、描述、inputSchema
    H-&gt;&gt;H: 模型选择工具并生成参数
    C-&gt;&gt;S: tools/call
    S--&gt;&gt;C: content / structured result / error
    C--&gt;&gt;H: 将必要结果交给模型或界面
```

为什么必须初始化：

- 协商双方支持的协议版本。
- 声明 Client 与 Server 各自支持的能力。
- 避免一方调用另一方不支持的特性。
- 为后续会话建立清晰的状态边界。

能力发生变化时，可以通过通知告诉对端重新拉取列表。面试中应强调“能力发现不是只靠写死配置”。

---

## 4. 三类 Server 原语：Tools、Resources、Prompts

| 原语 | 解决的问题 | 谁通常决定使用 | 例子 |
|---|---|---|---|
| Tools | 执行动作或计算 | 模型/Agent | 查天气、发工单、执行 SQL |
| Resources | 提供可读取的上下文数据 | 应用/Host | 文件、数据库记录、日志、知识文档 |
| Prompts | 提供可复用的提示模板或工作流入口 | 用户/应用 | 代码审查模板、周报模板 |

一种便于记忆的说法是：Tools 偏“做事”，Resources 偏“读数据”，Prompts 偏“复用交互模板”。这只是控制语义上的概括，不代表三者绝对互斥。

### Tools

工具通常包含：

- `name`：稳定、语义明确、尽量体现业务域。
- `description`：说明适用场景、限制和副作用。
- `inputSchema`：一般使用 JSON Schema 描述参数。
- 执行结果：文本、图片、结构化内容或错误信息。

工具描述会影响模型选工具和填参数的准确率，因此 Docstring 和类型标注不仅是文档，也是 Agent 可用性设计的一部分。

### Resources

Resource 适合将数据作为上下文提供给应用，常通过 URI 标识。它与工具的关键差别是：Resource 强调“可寻址、可读取的数据”，Tool 强调“执行一个操作”。

例如：

- `file:///project/README.md`
- `db://orders/20260823`
- `logs://service-a/latest`

若读取过程需要复杂查询、权限计算或明显副作用，也可以封装成 Tool，最终要按语义和治理要求选型。

### Prompts

Prompt 是 Server 提供的可复用提示模板，可以接受参数。它适合沉淀组织级最佳实践，但不应在模板中偷偷执行用户不可见的高风险动作。

### Client 侧能力

面试深入追问时，可以补充 MCP 不只允许 Server 提供能力，Client 也可以声明能力，例如：

- Roots：向 Server 声明允许访问的根目录或资源边界。
- Sampling：Server 请求 Client 借助其模型生成内容，最终控制权仍应在 Host。
- Elicitation：Server 请求 Host 向用户补充必要信息。

这些能力是否可用取决于协议版本和双方实现，不能假设所有 Host 都支持。

---

## 5. Transport：stdio 与 Streamable HTTP

| 维度 | stdio | Streamable HTTP |
|---|---|---|
| 部署位置 | 通常为本机子进程 | 本机或远程服务 |
| 生命周期 | Host 拉起并管理进程 | 服务独立部署 |
| 通信方式 | 标准输入/标准输出 | HTTP，请求可流式返回 |
| 认证 | 常依赖本机进程边界 | 需要显式认证与授权 |
| 适用场景 | IDE、本地文件/命令工具 | 企业共享服务、云端 API |
| 主要风险 | 子进程权限过大、stdout 污染协议 | 网络攻击面、越权、会话劫持 |

补充：早期实现常见 HTTP &#43; SSE。现代实现更常使用 Streamable HTTP；面试时不要把“SSE”与“Streamable HTTP”完全画等号，也不要假设所有 SDK 版本支持完全相同的 Transport 名称。

### stdio 的工程注意点

- stdout 用于协议帧，业务日志应写 stderr 或文件。
- Host 应管理子进程退出、重启和超时。
- 启动命令和工作目录必须确定，避免环境不一致。
- 不要无条件继承全部敏感环境变量。

### 远程 HTTP 的工程注意点

- 使用 TLS。
- 鉴权后还要做工具级、资源级、租户级授权。
- 校验 `Origin`、Host 等请求信息，降低 DNS rebinding 等风险。
- 绑定 `127.0.0.1` 只适合本机访问；绑定 `0.0.0.0` 前必须补齐安全措施。
- 会话标识不可替代用户身份，且要防伪造、泄露和固定会话攻击。

---

## 6. 当前目录代码逐文件拆解

目录结构：

```text
mcp_server/
├── README.md
├── mcp_supervisor.conf
├── get_weather_mcp/
│   ├── __init__.py
│   ├── __main__.py
│   ├── README.md
│   └── server.py
└── math_mcp/
    ├── __init__.py
    ├── __main__.py
    ├── README.md
    └── server.py
```

### 6.1 天气 Server

核心代码位于 `get_weather_mcp/server.py`：

```python
from fastmcp import FastMCP

mcp = FastMCP(&#34;get_weather_mcp&#34;)

@mcp.tool
def get_weather(city: str) -&gt; str:
    &#34;&#34;&#34;Get weather for a given city.&#34;&#34;&#34;
    return f&#34;{city}天气晴朗，万里碧空飘着朵朵白云!&#34;
```

从代码到协议的映射：

| Python 元素 | MCP 中的含义 |
|---|---|
| `FastMCP(&#34;get_weather_mcp&#34;)` | 创建并命名 Server |
| `@mcp.tool` | 将函数注册为 Tool |
| 函数名 `get_weather` | Tool 名称 |
| Docstring | Tool 描述的重要来源 |
| `city: str` | 生成输入 JSON Schema 的依据 |
| `-&gt; str` | 输出类型提示 |

这个示例只返回固定文案，适合演示协议，不是真实天气服务。生产版需要接天气 API，并加入城市标准化、超时、重试、缓存、错误分类和监控。

### 6.2 数学 Server

`math_mcp/server.py` 做了三件事：

1. 将 `×`、`÷` 转换成 Python 运算符。
2. 过滤出数字、括号和部分算术符号。
3. 使用 `eval(expr)` 执行表达式。

优点：

- 有输入长度限制。
- 有字符白名单和空表达式检查。
- 返回值为整数时会转换成 `int`。
- 类型标注和 Docstring 能帮助生成 Tool Schema。

面试时应主动指出的问题：

- 字符白名单不等于计算安全，`eval` 仍不应作为生产计算器的首选。
- `**` 可构造超大整数，造成 CPU/内存型拒绝服务。
- 只限制字符串长度，不能限制运算复杂度、括号深度和结果大小。
- `%`、连续运算符、除零等行为缺少明确业务约束。
- 捕获所有异常后只返回一个通用错误，不利于客户端分类处理。

更可靠的实现应使用 `ast.parse(..., mode=&#34;eval&#34;)`，只允许 `Expression`、数值常量和指定的 `BinOp`/`UnaryOp` 节点，并限制 AST 节点数量、深度、指数、数值范围和执行时间。要求更高时，应调用隔离的计算服务，而不是在 MCP 进程内解释表达式。

### 6.3 `__main__.py` 与双 Transport

两个服务都提供：

```python
def stdio():
    asyncio.run(server.mcp.run(transport=&#34;stdio&#34;))

def http():
    asyncio.run(server.mcp.run(
        transport=&#34;http&#34;,
        host=host,
        port=port,
        path=&#34;/mcp&#34;,
    ))
```

当前模块直接运行时默认执行 `http()`：

- 天气服务默认监听 `127.0.0.1:8000/mcp`。
- 数学服务默认监听 `127.0.0.1:8001/mcp`。
- 可通过 `HOST`、`PORT` 环境变量覆盖。

这里的 `transport=&#34;http&#34;` 是该 FastMCP 版本对 Streamable HTTP 的接口命名，不能据此推断所有 MCP SDK 都使用相同参数。

### 6.4 Supervisor 托管

`mcp_supervisor.conf` 将两个 Server 组成 `mcp_servers` 进程组，并设置：

- `autostart=true`：Supervisor 启动时自动拉起。
- `autorestart=true`：异常退出后自动重启。
- stdout/stderr 分文件记录。
- 启动等待和停止等待时间。

启动方式：

```bash
cd LLM08/agent-advanced/mcp_server
pip install fastmcp supervisor
supervisord -c ./mcp_supervisor.conf
lsof -i :8000
lsof -i :8001
```

停止当前配置启动的 Supervisor，生产中更推荐：

```bash
supervisorctl -c ./mcp_supervisor.conf shutdown
```

相比 `pkill -f supervisord`，它的目标更精确，不容易误伤其他 Supervisor 实例。

本目录当前没有锁定依赖版本；面试中可以建议增加 `pyproject.toml`/锁文件，确保 FastMCP API 与协议行为可复现。

---

## 7. Client 如何把 MCP Tool 交给模型

仓库相邻目录 `deep-research-from-scrach/client.py` 展示了 stdio Client 的核心过程：

```python
server_params = StdioServerParameters(
    command=&#34;python&#34;,
    args=[server_script_path],
)

stdio_transport = await exit_stack.enter_async_context(
    stdio_client(server_params)
)
read, write = stdio_transport

session = await exit_stack.enter_async_context(
    ClientSession(read, write)
)

await session.initialize()
tools = (await session.list_tools()).tools
result = await session.call_tool(tool_name, tool_args)
```

完整的 Agent Tool Loop 通常是：

```text
1. Client 发现工具
2. Host 按当前任务筛选工具
3. Host 将工具 Schema 提供给模型
4. 模型输出工具名与参数
5. Host 做参数校验、权限检查和高风险确认
6. Client 调用 MCP Server
7. Host 检查错误、裁剪结果并回填模型
8. 模型决定继续调用还是生成最终答案
```

关键点：模型不能直接访问 MCP Server。真正发起网络或进程调用的是 Host/Client，Host 应保留最终控制权。

---

## 8. MCP 与上下文工程

第一张图将上下文工程分为“写入、选择、压缩、隔离”四类，这与 MCP 的关系如下。

### 8.1 写入上下文

可写入的内容包括：

- 对话内临时状态，例如计划、任务清单、反思结果。
- 跨会话长期记忆。
- 文件或外部状态。

MCP 可以通过 Tool/Resource 读写外部系统，但“写什么、保存多久、谁能读取”是 Host 和业务系统的策略，不由 MCP 自动决定。

### 8.2 选择上下文

Host 不应把所有 Server 的所有工具都塞给模型，而应先做：

- 按用户、租户、角色过滤无权限工具。
- 按任务路由到相关 Server。
- 按语义检索少量相关 Tool/Resource。
- 从长期记忆或知识库只召回必要片段。

工具越多不一定越好：工具名相似、描述重叠时，模型的选择准确率可能下降，输入 Token 和延迟也会增加。

### 8.3 压缩上下文

MCP Tool 可能返回网页、日志、表格等超长结果。Host 应：

- 优先让 Server 支持过滤、分页、字段选择和聚合。
- 对超长结果截断，但保留“已截断”和分页信息。
- 用确定性程序先抽取关键字段，再考虑 LLM 摘要。
- 保存原始结果的引用，而不是反复把全文放入对话。

### 8.4 隔离上下文

适用于多 Agent 或复杂任务：

- 不同子 Agent 只连接完成任务所需的 Server。
- 敏感工具放入独立进程、容器或权限域。
- 子任务只返回摘要、证据引用和结构化产物。
- 避免一个 Agent 的提示注入内容污染其他 Agent。

一句面试总结：MCP 扩展了 Agent 可获得的上下文和动作空间；上下文工程负责控制哪些能力和结果在何时进入模型上下文。

---

## 9. MCP、Function Calling、RAG、A2A、ANP 的区别

### 9.1 MCP vs Function Calling

| 对比项 | Function Calling | MCP |
|---|---|---|
| 层级 | 模型 API 的能力 | Host 与外部能力之间的协议 |
| 工具来源 | 常由应用代码静态传入 | 可从 Server 动态发现 |
| 生命周期 | 通常跟随一次模型请求 | 有连接、初始化、能力协商和会话 |
| 可移植性 | 受模型供应商 API 影响 | 目标是跨 Host、Server 复用 |
| 二者关系 | 模型决定调用什么 | MCP Client 真正连接和调用能力 |

二者不是替代关系。常见架构是：MCP 提供工具，Host 把 MCP Tool 转成模型的 Function Calling Schema，模型返回调用意图，Host 再通过 MCP 执行。

### 9.2 MCP vs RAG

- RAG 是“检索相关知识并增强生成”的应用模式。
- MCP 是“如何发现、读取或调用外部能力”的协议。
- 可以用 MCP Resource 或 Tool 暴露向量检索服务，从而实现 RAG。

### 9.3 MCP vs A2A vs ANP

结合后两张图，可以用“纵向接工具、横向找 Agent、网络化发现”理解：

| 协议/概念 | 主要连接对象 | 核心问题 | 典型场景 |
|---|---|---|---|
| MCP | Host/Agent 与工具、数据、API | 能力发现与上下文/工具调用 | Agent 查库、读文件、调用企业服务 |
| A2A | 独立 Agent 与独立 Agent | 跨框架委派任务、同步状态与产物 | 采购 Agent 委派风控 Agent 审核 |
| ANP | Agent 与开放 Agent 网络 | 身份、注册、发现、协作网络 | 不同主体的 Agent 在网络中发现服务 |

MCP 与 A2A 可以组合：每个 Agent 内部通过 MCP 使用本地工具，Agent 之间通过 A2A 协作。课件中的 A2A 图正体现了“组织或技术边界上方用 A2A，边界内部向下用 MCP 接企业服务”。

对于 ANP，应采用谨慎表述：在该课件语境中，它强调 Agent 的注册、发现和点对点协作网络；相关方案仍在演进，成熟度、治理方式和行业统一程度不能与已广泛实现的 MCP 简单等同。面试时先确认面试官所说的 ANP 具体指哪套规范。

---

## 10. 生产级 MCP Server 设计清单

### 10.1 工具设计

- 一个工具完成一个清晰任务，避免“万能 execute”。
- 工具名稳定，使用领域前缀减少冲突，如 `order_get_detail`。
- Description 写清适用场景、限制、单位、副作用和失败条件。
- 输入 Schema 尽量具体：枚举、范围、格式、必填项、最大长度。
- 输出优先结构化，避免让 Client 从大段自然语言中二次解析。
- 查询与变更工具分开，高风险操作明确标记。
- 写操作支持幂等键，避免超时重试造成重复扣款、重复发信。

### 10.2 错误与可靠性

- 区分参数错误、无权限、资源不存在、限流、上游超时和内部错误。
- 只有幂等操作才适合自动重试，并加入指数退避和抖动。
- 设置连接、读取、执行总超时。
- 长任务返回任务 ID，支持查询状态、取消和进度通知。
- 对上游依赖使用限流、熔断、隔离舱和降级。
- 避免把内部堆栈、密钥或数据库细节返回给模型。

### 10.3 安全

- 参数 Schema 校验只是第一层，业务层还要做语义校验。
- 认证解决“你是谁”，授权解决“你能调用什么、操作哪些数据”。
- 高风险 Tool 采用最小权限、显式确认、审批或双人复核。
- 将用户身份和授权上下文安全传到下游，防止 confused deputy（混淆代理）问题。
- 防 Prompt Injection：外部 Tool/Resource 返回内容只能视为不可信数据，不能当系统指令执行。
- 密钥由密钥管理系统注入，不写入 Tool 描述、日志或模型上下文。
- 本地 Server 也要做文件路径规范化、根目录限制和命令白名单。

### 10.4 可观测性

建议每次调用记录：

- `trace_id`、`request_id`、会话 ID。
- Server、Tool、版本、租户和调用方。
- 参数摘要或脱敏后的参数指纹。
- 总耗时、排队耗时、上游耗时。
- 成功/失败、错误分类、重试次数。
- 输入输出大小、模型 Token 影响。

不要记录明文密码、Token、个人隐私和完整敏感业务数据。

### 10.5 性能

- 工具列表按任务筛选，避免每轮重复注入全部 Schema。
- 查询型工具支持缓存，但缓存键必须包含租户和权限维度。
- 大结果使用分页、游标、字段投影和服务端聚合。
- I/O 型工具使用异步调用；CPU 密集型任务交给进程池或独立服务。
- 明确并发上限和背压，防止模型并行调用击穿下游。

---

## 11. 系统设计题：企业智能助手如何接 MCP

题目示例：设计一个能查订单、申请退款、检索内部文档的企业 Agent。

### 架构

```text
用户
  ↓
Agent Gateway：登录、租户、限流、审计
  ↓
Host：会话、模型路由、计划、工具筛选、用户确认
  ├── MCP Client → Order MCP Server → 订单服务
  ├── MCP Client → Refund MCP Server → 支付/审批系统
  └── MCP Client → Knowledge MCP Server → 搜索/向量库
```

### 回答要点

1. 登录身份进入 Host 后生成短期授权上下文，不能把长期主密钥交给模型。
2. Host 按租户和角色过滤工具，只给当前任务最相关的 Tool Schema。
3. 查订单是只读工具；申请退款是写工具，必须带幂等键和订单版本。
4. 模型生成退款参数后，Host 先做确定性校验，再让用户确认金额和原因。
5. Refund Server 再做一次服务端授权，不能信任 Client 已经检查过。
6. 长审批返回任务 ID，Agent 后续查询状态，而不是保持无限长 HTTP 请求。
7. 文档检索结果先做权限过滤、去重和裁剪，再进入模型上下文。
8. 全链路记录审计，但日志必须脱敏。
9. 对超时和失败区分“可重试”与“不可重试”，写操作依赖幂等保障。
10. Server 故障时允许只读能力降级，但不能绕过退款审批。

---

## 12. 高频面试题与参考答案

### Q1：为什么有 REST API 还需要 MCP？

REST 解决服务间 HTTP API 设计，MCP 解决 AI Host 如何统一发现和调用上下文能力。MCP Server 内部完全可以再调用 REST API。MCP 增加了 AI 场景需要的能力描述、Schema、生命周期和发现机制，不是要替代所有 REST 服务。

### Q2：MCP 是否减少了所有集成成本？

它减少协议适配和重复胶水代码，但业务字段映射、权限模型、错误语义、SLA 和数据治理仍然存在。复杂度被标准化和分层，而不是消失。

### Q3：Tool、Resource 怎么选？

可寻址、偏读取的数据优先考虑 Resource；有参数化计算、动作、副作用或复杂业务逻辑时使用 Tool。最终还要看 Host 支持程度和权限治理要求。

### Q4：模型如何知道 Server 有什么工具？

Client 初始化后调用工具列表接口，得到名称、描述和输入 Schema；Host 再选择合适的工具提供给模型。模型不应自己扫描网络寻找 Server。

### Q5：工具调用失败怎么办？

Server 返回可分类错误；Host 根据错误类型决定修正参数、提示用户、降级或重试。只有幂等且符合重试条件的操作才能自动重试。

### Q6：stdio 与 HTTP 怎么选？

本地、单用户、需要访问本机文件或命令时优先 stdio；需要多人共享、跨机器部署和独立扩缩容时使用 Streamable HTTP。远程方案必须额外处理 TLS、认证、授权、租户隔离和网络治理。

### Q7：MCP 最大的安全风险是什么？

不存在单一最大风险，常见组合是过度授权、恶意或被攻陷的 Server、Prompt Injection、敏感数据泄露和高风险工具误调用。核心控制是 Host 保留执行权、最小权限、服务端二次授权、用户确认、输入输出校验和完整审计。

### Q8：如何防止模型调用危险工具？

不能只靠 Prompt。需要在模型之外做工具白名单、基于身份的授权、参数策略校验、风险分级、用户确认、幂等控制和服务端二次鉴权。

### Q9：MCP 如何支持多 Agent？

每个 Agent 可以通过自己的 MCP Client 使用工具，但多 Agent 的任务分解、消息传递和协作协议不属于 MCP 的核心职责。跨 Agent 协作可由工作流框架或 A2A 类协议负责。

### Q10：工具很多时怎么办？

按业务域、用户权限和当前意图先路由，再只向模型提供少量候选工具；必要时建立工具检索索引。还要改善名称和描述，减少功能重叠。

### Q11：MCP 如何与上下文窗口配合？

工具描述和结果都会进入上下文预算。Host 应选择性注入 Tool Schema，对结果分页、字段裁剪、聚合或摘要，并把大对象保存为外部引用。

### Q12：如何做版本升级？

协议层通过初始化协商版本；业务 Tool 层要保持向后兼容，新增可选字段优于直接修改语义。破坏性变化使用新 Tool 名或独立 Server 版本，并监控旧版本调用量后逐步下线。

### Q13：为什么 Tool Schema 很重要？

它既用于参数校验，也直接影响模型能否选对工具、填对参数。过于宽泛的字符串参数会把错误推迟到运行期，降低稳定性和可观测性。

### Q14：如何保证写操作不会重复执行？

让 Host 生成幂等键，Server 持久化幂等记录，并在重试时返回第一次执行结果；同时使用业务唯一约束或版本号处理并发。不能仅靠模型承诺“不重复调用”。

### Q15：如何测试 MCP Server？

分四层：纯函数单测；Tool Schema 和参数边界契约测试；真实 Client 的初始化、发现和调用集成测试；超时、断连、并发、权限和恶意输入的故障/安全测试。

---

## 13. 针对本目录的改进题

如果面试官让你把当前 Demo 改成生产版本，可以按下面回答。

### 天气服务

- 接入真实天气提供方并封装 Provider 接口，避免绑定单一供应商。
- 将城市转换为标准 location ID，处理重名城市。
- 输入增加单位、日期、语言等枚举和范围约束。
- 对上游设置超时、有限重试、熔断和缓存。
- 输出结构化字段：温度、单位、天气代码、更新时间、数据来源。
- 区分城市不存在、上游限流、上游超时等错误。

### 数学服务

- 移除 `eval`，使用 AST 白名单解释器或隔离计算服务。
- 限制节点数、括号深度、指数、数值大小和总执行时间。
- 明确支持的运算符和小数精度。
- 返回结构化结果，包括规范化表达式、结果和错误类型。
- 增加属性测试、模糊测试和 DoS 用例。

### 部署

- 锁定 Python、FastMCP 及依赖版本。
- 增加健康检查、优雅退出、结构化日志和指标。
- 使用非 root 用户、只读文件系统和最小网络权限。
- 远程访问增加 TLS、认证、授权和租户隔离。
- Supervisor 可用于简单环境；容器平台中由编排系统管理重启、探针和扩缩容。

---

## 14. 易错表述纠正

| 易错说法 | 更准确的说法 |
|---|---|
| MCP 是一个 Agent 框架 | MCP 是连接 AI Host 与上下文能力的协议 |
| MCP Server 一定是远程服务器 | 本地 stdio 子进程也可以是 Server |
| MCP 就是 Function Calling | Function Calling 是模型接口；MCP 是外部能力协议，可组合使用 |
| 接了 MCP 就能自动选对工具 | 工具选择仍取决于 Host 路由、Schema 质量和模型能力 |
| Schema 校验后工具就是安全的 | 仍需业务授权、语义校验、沙箱、确认和审计 |
| 所有工具都放进上下文效果最好 | 工具过多会增加 Token、延迟和误选率 |
| A2A 会替代 MCP | A2A 偏 Agent 间协作，MCP 偏 Agent/Host 接工具与数据 |
| ANP 已经只有一种统一实现 | 先确认具体规范，相关生态仍在演进 |
| SSE 就是 Streamable HTTP | 二者相关但不是完全相同的 Transport 设计 |

---

## 15. 最后一分钟背诵清单

1. MCP = AI Host 与外部工具、数据、提示模板之间的开放协议。
2. 三角色 = Host、Client、Server。
3. 三类 Server 原语 = Tools、Resources、Prompts。
4. 协议基础 = JSON-RPC 2.0、初始化、能力协商、发现、调用、通知。
5. 两类常见 Transport = stdio、Streamable HTTP。
6. Function Calling 决定模型如何表达调用；MCP 负责发现并连接外部能力。
7. MCP 不负责 Agent 规划，也不会自动解决上下文和安全问题。
8. 上下文工程四动作 = 写入、选择、压缩、隔离。
9. MCP 接工具，A2A 连 Agent；两者可以上下组合。
10. 生产关键字 = 最小权限、Schema、幂等、超时、限流、审计、可观测、结果裁剪。
11. 当前天气 Demo 在 8000，数学 Demo 在 8001，路径均为 `/mcp`。
12. 当前数学 Demo 的 `eval` 有资源消耗风险，生产版应换 AST 白名单或隔离计算。

---

## 16. 一句有区分度的结尾

&gt; 我认为 MCP 真正的工程价值不只是“让模型会调工具”，而是把 AI 应用与外部能力的连接边界协议化；但要达到生产可用，仍必须在 Host 和 Server 两侧补齐上下文治理、确定性校验、权限控制与可靠性工程。


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlmcp/  

