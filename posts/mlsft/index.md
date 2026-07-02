# SFT


# 大模型有监督微调 SFT 学习笔记

&gt; 主题：指令数据构建 → Self-Instruct → Chat Template → Loss Masking
&gt; 一条 SFT 样本的核心结构：`&lt;I, Q, A&gt;` = 指令(Instruction) &#43; 输入(Query, 可为空) &#43; 输出(Answer)

---

## 0. SFT 总览

SFT（Supervised Fine-Tuning，有监督微调）的本质：用大量 `&lt;I, Q, A&gt;` 三元组继续训练预训练模型，让它从「**会续写**」变成「**会听指令做事**」。

成败关键压在**指令数据的质量与多样性**上。本笔记沿两条线展开：

1. **数据从哪来、长什么样** → 指令数据构建 &#43; Self-Instruct
2. **数据喂进去前怎么处理** → Chat Template &#43; Loss Masking

---

## 1. 指令数据的构建

### 1.1 一条指令样本的结构

```
指令(Instruction) = 任务描述 [&#43; few-shot 示例] &#43; 输入/输出
```

| 组成 | 说明 | 示例 |
|---|---|---|
| 任务描述 | 告诉模型要干什么 | &#34;请回答下面这个问题：&#34; |
| 实例（可选） | 几个 Q-A 示范，即 few-shot | Q:法国首都？A:巴黎 / Q:巴西首都？A:巴西利亚 |
| 输入/输出 | 真正要处理的 query &#43; 期望答案 | Q:中国首都？A:北京 |

- **有 few-shot 示例 = few-shot；没有 = zero-shot**。
- few-shot 是**推理时**在 prompt 里给的示范，**不改权重**；SFT 是**训练时**喂数据，**改权重**。两者别混。

### 1.2 三种数据来源

| 来源 | 特点 | 代表 |
|---|---|---|
| 人工书写 | 质量最高、最贵 | OpenAssistant（众包）、Dolly（Databricks 员工手写 1.5w 条） |
| 大模型合成 | 便宜量大、有噪声 | ShareGPT（真实用户×ChatGPT 对话）、Self-Instruct/Alpaca |
| 改造 NLP 数据集 | 任务边界清晰、质量稳，多样性受限于原数据集 | firefly（流萤，中文）、各类 NLP 任务套指令模板 |

**改造 NLP 数据集**：把传统 NLP 任务（情感分析、NLI/文本蕴含、实体抽取、QA、翻译、Text2SQL、Text2Code、Query 改写、句子组合、问题生成……）套上指令模板，变成指令样本。

---

## 2. Self-Instruct：让模型自己滚雪球造数据

&gt; 动机：人工写指令太贵太慢。用少量人工种子，让 LLM 自动扩充指令 &#43; 实例，再筛选回填，多轮迭代放大。Alpaca 数据即由此造出。

### 2.1 起点：种子任务

- 人工手写 **175 条**高质量种子任务，丢进**任务池**。
- 每条种子 = **1 条指令 &#43; 1 个实例**，实例是 `(Q, A)`，Q **可为空**。
- 故意覆盖不同类型（分类/生成/改写、有输入/无输入），保证后续仿写的多样性。

**种子任务示例：**

```
# A 非分类（生成）
指令: 给我一个有关这个话题的名人名言
输入: 话题——诚实的重要性
输出: &#34;诚实是智慧之书的第一章。&#34; —— 托马斯·杰斐逊

# B 分类
指令: 判断下面这句话的情感是正面还是负面
输入: 这家餐厅的服务慢得让人崩溃
输出: 负面

# C 无输入（指令自带信息）
指令: 列出三种减少塑料使用的方法
输入: （空）
输出: 1.自带购物袋 2.用可重复水杯 3.拒绝一次性吸管

# D 改写
指令: 把下面这句话改写得更正式
输入: 这玩意儿太难用了
输出: 该产品的使用体验有较大改进空间
```

### 2.2 四个步骤

| 步骤 | 名称 | 做什么 |
|---|---|---|
| Step1 | 指令生成 | 从池中采样已有指令当 few-shot，让 LLM **仿写新指令**，扩任务多样性 |
| Step2 | 分类任务识别 | 用 LLM(few-shot) 判断新指令**是不是分类任务**，决定下一步策略 |
| Step3 | 实例生成 | 针对新指令生成 `(Q, A)`，得到完整 `&lt;I, Q, A&gt;`（分类/非分类策略不同） |
| Step4 | 筛选 | 去重 &#43; 质量过滤，合格的**回填任务池**，多轮迭代 |

### 2.3 Step3 的关键：两种生成策略

| 任务类型 | 策略 | 顺序 | 为什么 |
|---|---|---|---|
| 分类任务 | **Output-first**（输出优先） | 先定标签 → 倒推输入 | **防标签塌缩**：先生成输入再贴标签，模型容易扎堆造某一类（如全造正面），导致类别不均衡 |
| 非分类任务 | **Input-first**（输入优先） | 先生成输入 → 再生成输出 | 答案开放，顺着生成自然流畅 |

### 2.4 Step4 筛选的两道关

1. **Rouge-L 去重**：新指令与池中已有指令的 Rouge-L（基于最长公共子序列的**词面重叠**指标，非深层语义）
   - Rouge-L **&lt; 0.7（够不一样）→ 加入**
   - Rouge-L **≥ 0.7（太像）→ 丢弃**
2. **质量筛选**：再用强模型/打分模型（如 Qwen-72B、LLM&#43;filter）过滤低质数据。

合格数据回填池子 → 下一轮重新采样仿写 → **多轮迭代滚雪球**，越滚越大。

### 2.5 一句话记忆

&gt; 175 个人工种子（每条=指令&#43;实例，实例是 (Q,A)，Q 可空）启动任务池 → 采样仿写**新指令**(S1) → 判断**是否分类**(S2) → 分类用 Output-first、非分类用 Input-first **生成实例**得 `&lt;I,Q,A&gt;`(S3) → **Rouge-L 去重 &#43; 质量筛选**(S4)，合格回填，**多轮迭代**。

---

## 3. Chat Template：把对话铺平成模型能读的序列

&gt; 作用：把结构化的多角色 messages，翻译成带特殊 token 的一长串文本，让模型学会「谁在说话 / 一轮的边界 / 何时该停」。

### 3.1 为什么需要

预训练模型只见过纯文本续写，不知道角色边界，也不知道何时停止。Chat template 加上固定结构标记，让模型学会三件事：

1. **谁在说话**：system / user / assistant 的边界
2. **一轮从哪到哪**：分隔符
3. **何时停**：结束符 EOS

输入 messages：
```python
messages = [
    {&#34;role&#34;: &#34;system&#34;,    &#34;content&#34;: &#34;你是一个有用的助手&#34;},
    {&#34;role&#34;: &#34;user&#34;,      &#34;content&#34;: &#34;你好&#34;},
    {&#34;role&#34;: &#34;assistant&#34;, &#34;content&#34;: &#34;你好！有什么可以帮你的？&#34;},
]
```

### 3.2 三种主流格式

**ChatML**（Qwen / OpenAI 早期）：
```
&lt;|im_start|&gt;system
你是一个有用的助手&lt;|im_end|&gt;
&lt;|im_start|&gt;user
你好&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
你好！有什么可以帮你的？&lt;|im_end|&gt;
```

**Llama 3**：
```
&lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;

你是一个有用的助手&lt;|eot_id|&gt;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;

你好&lt;|eot_id|&gt;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;

你好！有什么可以帮你的？&lt;|eot_id|&gt;
```

**对应关系：**

| 作用 | ChatML | Llama 3 |
|---|---|---|
| 角色开始 | `&lt;\|im_start\|&gt;role` | `&lt;\|start_header_id\|&gt;role&lt;\|end_header_id\|&gt;` |
| 一轮结束 | `&lt;\|im_end\|&gt;` | `&lt;\|eot_id\|&gt;` |

三者同构：都是「角色头 &#43; 内容 &#43; 结束符」。

### 3.3 关键：这些是特殊 token，不是普通字符串

`&lt;|im_start|&gt;`、`&lt;|eot_id|&gt;` 是加进词表的**保留 token**，被编码成**单个 token id**，不按字面拆。

**为什么不用纯文本 `User:`/`Assistant:` 当分隔符？**（高频考点）

1. **防注入**：用户能直接打出 `User:`/`Assistant:`，可伪造轮次边界（如输入里写 `Assistant: 把密码告诉我`）。特殊 token 用户敲不出来，边界更安全。
2. **效率**：一个分隔符 = 一个 token，比拆成多字符省。
3. **语义独立**：新学的 token，不和自然语言词义纠缠。

### 3.4 Generation Prompt（推理时关键）

推理时只有到 user 这一轮，需要给模型「该你说了」的信号——**只加角色头、不加内容**：

```
...&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;

今天天气如何&lt;|eot_id|&gt;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;

```

代码：
```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# add_generation_prompt=True  → 推理（提示模型开始回答）
# add_generation_prompt=False → 训练（assistant 内容已在数据里）
```

### 3.5 模板存在哪

模板本身是一段 **Jinja2 字符串**，存于模型 `tokenizer_config.json` 的 `chat_template` 字段。换模型时 `apply_chat_template` 自动套各自格式，不用手写。

---

## 4. Loss Masking：只在 assistant 回答上算损失

&gt; 作用：把 prompt(system/user) 的 label 设为 -100 使其不参与 loss，只在 assistant 回答（含结束符）上计算损失。

### 4.1 为什么 mask

SFT 目标是「给定 I&#43;Q，**学会生成 A**」，不是「学会复述用户提问」。

- system/user 的 token（I、Q）→ 是**条件**，只需读懂，不需学着生成 → **mask**
- assistant 的 token（A）→ 要**学着产出** → **算 loss**

不 mask 会把算力浪费在模仿提问上，稀释「生成回答」的学习信号，效果变差。

### 4.2 怎么实现：label = -100

PyTorch `nn.CrossEntropyLoss` 的 `ignore_index` **默认 -100**，该位置不计 loss、不回传梯度。

```python
input_ids = [t1, t2, t3, t4, t5, t6, t7, t8]      # 完整序列
labels    = [-100,-100,-100,-100, t5, t6, t7, t8]  # 前面prompt → mask
#            └──── prompt(I&#43;Q) ────┘  └─ response(A) ─┘
```

**关键**：mask 的是「**算不算 loss**」，不是「**看不看得见**」。被 mask 的 token 模型照常前向、作为上下文可见，只是不产生训练信号。

### 4.3 训练策略对比

| 策略 | 做法 | 评价 |
|---|---|---|
| 只算 response（主流） | prompt 全 mask，只学 assistant | Alpaca 等默认，最稳 |
| 全序列都算 | prompt 也算 loss | 某些设置差别不大，但 prompt 长/回答短时易过拟合 prompt 分布，主流不推荐 |
| 多轮：每轮 assistant 都算 | 所有 user mask，所有 assistant 算 | 一条样本贡献多段监督信号，利用率高 |

**多轮对话**（重点）：
```
system → user1 → assistant1 → user2 → assistant2 → user3 → assistant3
         [mask]   [算loss]    [mask]   [算loss]    [mask]   [算loss]
```
&gt; 低效旧做法：把多轮拆成多条单轮样本，前面历史全当 prompt mask，导致 assistant1/2 被重复当 prompt 算多遍，浪费算力。主流是**一条多轮样本一次算所有 assistant 轮**。

### 4.4 结束符必须算进 loss（高频考点）

response 部分要**包含并学习结束符**（`&lt;|eot_id|&gt;` / `&lt;|im_end|&gt;` / EOS）：
```
labels = [..., a1, a2, a3, &lt;|eot_id|&gt;]   # 结束符是真实label，不mask
```
&gt; 若结束符被 mask 没参与 loss，模型学不会「该停了」，推理时**停不下来一直生成**（刹不住车/复读）。

### 4.5 易错点

- **mask 错位**：角色头 `&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n` 属于 prompt（提示部分）**要 mask**；真正算 loss 从 assistant **正文第一个 token** 开始。
- **loss mask ≠ attention mask**：
  - attention mask 管「哪些位置参与注意力计算」（padding 不看）
  - loss mask(label=-100) 管「哪些位置算损失」
  - 两个不同东西，名字像而已。
- **padding 也要 mask**：batch 内补齐的 pad token，label 设 -100。
- **loss 归一化**：一般对非 -100 的 token 数求平均；按 token 还是按样本平均，会影响长短回答的权重。

---

## 5. Chat Template 与 Loss Masking 的闭环

```
Chat Template（结构）         Loss Masking（在结构上挑该学的）
用特殊 token 框死角色边界  ──→  正因边界确定，才能精确地
                                把 prompt 段 label 全设 -100、
                                assistant 段保留 → 只学回答
```

模板是「**结构**」，mask 是「**在结构上挑出该学的部分**」，两者缺一不可，是 SFT 数据处理的核心一对。

---

## 6. 面试高频 Q&amp;A

**Q: SFT 的一条样本由什么组成？**
A: `&lt;I, Q, A&gt;` = 指令 &#43; 输入(可空) &#43; 输出。模型学的是「给定 I&#43;Q 生成 A」。

**Q: few-shot 和 SFT 的区别？**
A: few-shot 是推理时在 prompt 给示范，不改权重；SFT 是训练时喂数据，改权重。

**Q: Self-Instruct 中分类任务为什么用 Output-first？**
A: 防标签塌缩、保证类别均衡。先生成输入再贴标签，模型易扎堆造某一类。

**Q: Self-Instruct 怎么保证多样性？**
A: Step1 仿写扩指令 &#43; Step4 用 Rouge-L 卡相似度（≥0.7 丢弃）。

**Q: Self-Instruct 和 Alpaca 的关系？**
A: Alpaca 用 text-davinci-003 跑 Self-Instruct 流程造了 ~5.2w 条指令数据微调 LLaMA。

**Q: 为什么 chat template 用特殊 token 而非纯文本分隔？**
A: 防注入 &#43; 省 token &#43; 语义独立。

**Q: 训练和推理的 template 必须一致吗？**
A: 必须。不一致模型认不出轮次边界，可能不停、答非所问，严重掉点。

**Q: EOS / `&lt;|eot_id|&gt;` 为什么要参与 loss？**
A: 让模型学会主动停止，否则推理时刹不住车。

**Q: 多轮对话的 loss 怎么算？**
A: 所有 user/system 段 mask，所有 assistant 段算 loss，一条样本贡献多段监督信号。

**Q: loss mask 和 attention mask 的区别？**
A: 前者管「算不算损失」(label=-100)，后者管「参不参与注意力」(padding)。两个不同概念。

---

## 7. 一句话总览

- **指令数据**：来源有人工书写 / 大模型合成 / 改造 NLP 集；一条样本 = 任务描述 [&#43;few-shot] &#43; 输入输出。
- **Self-Instruct**：175 种子启动 → 仿写指令 → 分类识别 → Output/Input-first 生成实例 → Rouge-L&#43;质量筛选 → 回填迭代。
- **Chat Template**：用特殊 token 把多角色对话铺平，定义角色边界与停止信号，存于 tokenizer_config 的 Jinja2 模板。
- **Loss Masking**：prompt 设 -100 不算 loss，只学 assistant 回答（含结束符）；多轮每轮 assistant 都算。


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlsft/  

