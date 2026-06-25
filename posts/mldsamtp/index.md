# DSA &amp; MTP


# DSA &amp; MTP 详细学习笔记

&gt; DeepSeek 系列两大关键创新:
&gt; - **DSA (DeepSeek Sparse Attention)** —— DeepSeek-V3.2-Exp / V3.2 引入的细粒度稀疏注意力机制
&gt; - **MTP (Multi-Token Prediction)** —— DeepSeek-V3 引入的多 token 预测训练目标
&gt;
&gt; 二者解决的是不同维度的问题:DSA 优化**长上下文推理/训练的计算复杂度**,MTP 优化**训练信号密度 &#43; 推理加速(投机解码)**。

---

# Part 1 · DSA(DeepSeek Sparse Attention)

## 1.1 背景与动机

标准 attention 是 **O(L²)** 复杂度(L 为序列长度)。在长上下文(如 128K)场景下,attention 计算成为主要瓶颈,训练和推理成本都急剧上升。

DSA 的核心思想:**从&#34;attend to everything&#34;转向&#34;attend to what matters&#34;**。
- 实证观察:每个 head、每个样本的 attention 分布高度稀疏(&gt;90% 的权重接近 0),但&#34;哪些 token 重要&#34;这件事是**随输入和 head 动态变化的**。
- 因此固定模式的稀疏(滑动窗口 / block / 随机 / 全局 token)效果受限,需要**内容自适应(content-adaptive)、动态(input-dependent)** 的稀疏。

DSA 是 DeepSeek-V3.2 相比 V3.1-Terminus **唯一的架构改动**,通过 continued training(在已有 checkpoint 上继续训练)引入。

&gt; 关键定位:DSA 的目标**不是**超过 V3.1-Terminus 的性能,而是**在引入稀疏后尽量减少性能退化的同时,获得效率收益**(性能持平 &#43; 成本大降)。

---

## 1.2 整体架构

DSA = **两个组件**:

```
                  ┌─────────────────────────┐
   query token h_t│                         │
   ───────────────▶  Lightning Indexer       │  计算 index score I_{t,s}
                  │  (轻量、FP8、少量 head)   │  对每个历史 token 打分
                  └───────────┬─────────────┘
                              │ top-k 选择
                              ▼
                  ┌─────────────────────────┐
                  │ Fine-grained Token       │  选出 top-k 个最相关的
                  │ Selection (top-k)        │  KV entry(如 2048 个)
                  └───────────┬─────────────┘
                              │ 只在被选中的 token 上
                              ▼
                  ┌─────────────────────────┐
                  │  主注意力(基于 MLA)     │  仅对 top-k 个 KV 做 attention
                  └─────────────────────────┘
```

DSA 是**实例化在 MLA 之上**的(instantiated under MLA),不是替换 MLA。

---

## 1.3 组件一:Lightning Indexer(闪电索引器)

作用:对于第 t 个 query token `h_t`,快速计算它与每个前序 token `h_s` 的**相关性分数(index score)** `I_{t,s}`,决定哪些 token 值得被选中。

索引分数公式(直观形式):

$$
I_{t,s} = \sum_{j=1}^{H^{I}} w^{I}\_{t,j} \cdot \mathrm{ReLU}\!\left(q^{I}\_{t,j} \cdot k^{I}_{s}\right)
$$

- $H^I$:indexer 的 head 数(很少,远小于主注意力的 head 数)
- $q^I_{t,j}, k^I_s$:indexer 专用的 query / key 投影(低维)
- $w^I_{t,j}$:可学习的标量权重
- **激活函数选 ReLU**:出于吞吐量(throughput)考虑,而非 softmax

为什么它&#34;轻量(lightning)&#34;:
1. **head 数少** —— 只需要少量 head 做粗粒度打分。
2. **可用 FP8 执行** —— indexer 的乘加运算用 FP8,进一步降本。
3. 虽然 indexer 本身名义上仍是 $O(L²)$,但因为它极其轻量(少 head &#43; 低维 &#43; FP8),它的成本被&#34;不再对整个 context 做完整 MLA&#34;省下的成本**完全淹没**。

---

## 1.4 组件二:Fine-grained Token Selection(细粒度 token 选择)

- 基于 indexer 算出的分数 `I_{t,s}`,对每个 query token 选出 **top-k** 个最相关的 KV entry。
- DeepSeek-V3.2 中 **k = 2048**(从 128K 的上下文里选 2048 个),这是一个超参。
- 选择是 **per-query、细粒度** 的:不同 query token 各自选自己的 top-k,而不是固定窗口或固定 block。
- indexer 和 selector **都是可学习的**,二者共同构成一个动态的 attention mask,把不相关 token 过滤掉。

工程实现要点(面试加分):
- top-k 用 GPU 的 partial sort 算法实现,可融合成单个 kernel(参考 DeepSeek 开源的 TileLang kernel)。
- gather 操作 &#43; 主注意力用定制 CUDA kernel,处理**每个 query 稀疏度可变**的情况。

---

## 1.5 在 MLA 下的实例化:为什么用 MQA mode

DeepSeek-V3.2 从 V3.1-Terminus continued training,而后者用的是 **MLA**。DSA 必须兼容 MLA。

关键约束:**kernel 层面,每个 KV entry 必须被多个 query 共享**才能高效。

因此 DSA 实现在 **MLA 的 MQA(Multi-Query Attention)模式**下:
- MLA 的每个 latent 向量(即 MLA 的 KV entry)被**所有 query head 共享**。
- 这样在做 gather(根据 top-k 索引去取 KV)时,一份 KV 服务所有 head,kernel 效率高。

&gt; 一句话总结:**DSA 套在 MLA 上,且用 MQA 模式跑,让一份被选中的 latent KV 被所有 query head 共享,从而在稀疏 gather 时保持 kernel 级效率。**

---

## 1.6 训练流程(两阶段 continued pre-training)

DSA 不是从头训练,而是在 V3.1-Terminus checkpoint 上**继续训练**引入。数据分布与 V3.1-Terminus 的 128K 长上下文扩展数据对齐。

### 阶段 A:Dense Warm-up Stage(稠密热身,初始化 indexer)

目标:让新加入的 indexer 学会模仿主注意力的分布。

- **冻结除 indexer 外的所有参数**,保持稠密注意力。
- 构造对齐目标分布 $p_{t,:}$:
  - 对第 t 个 query,把主注意力分数**沿所有 head 求和**;
  - 再沿序列维度做 **L1 归一化**,得到目标分布 $p_{t,:} \in \mathbb{R}^{t}$。
- 训练目标 = **KL 散度损失**,让 indexer 输出对齐 $p_{t,:}$。
- 超参:LR = $10^{-3}$,训 **1000 步**,每步 16 条 128K 序列 → 共 **2.1B tokens**。

### 阶段 B:Sparse Training Stage(稀疏训练,适配稀疏模式)

- 引入 top-k token selection,**放开所有参数一起训**,让主模型适配稀疏 attention 模式。
- 仍然做 indexer 对齐,但 **只在被选中的 token 集合 $S_t$ 上**对齐。
- 超参:LR = $7.3 \times 10^{-6}$,每个 query 选 **2048** 个 KV token,训 **15000 步**,每步 480 条 128K 序列 → 共 **943.7B tokens**。

### 一个重要的结构选择:detach indexer 输入

- 把 indexer 的输入**从计算图中 detach(切断梯度)**。
- 效果:**indexer 只被 indexer loss(KL)优化,主模型只被语言建模 loss 优化**,两路梯度互不干扰。

---

## 1.7 复杂度与推理收益

- 核心注意力复杂度:**O(L²) → O(Lk)**,其中 $k \ll L$。当 L=128K、k=2048 时收益巨大。
- 实测(H800 集群):长序列场景下,端到端 GPU 成本约**减半(~2×)**;成本节省随 token 在序列中的位置越靠后越明显(因为前面要 attend 的历史越长)。
- 长上下文任务(如 Fiction.liveBench)上性能**不退化**,验证了 DSA 训练的稳定性。

&gt; 深层含义(面试可引申):DSA 本质是**把花在&#34;参数&#34;上的 FLOPs 换成花在&#34;token&#34;上的 FLOPs**。它让长链推理(test-time compute / 大量 reasoning token)变得便宜,从而能在**线性成本包络**内堆更多推理 token —— 这也是 V3.2 能逼近 GPT-5 级推理的关键之一。

---

## 1.8 与其他稀疏注意力对比

| 方法 | 稀疏模式 | 是否内容自适应 | 缺点 |
|---|---|---|---|
| 滑动窗口(Sliding Window,如 Gemma 3) | 固定局部窗口 | 否 | 丢失远距离依赖 |
| Block / Strided | 固定块 | 否 | 需手工调 token 布局 |
| Random / Global token | 部分固定 | 部分 | 不够灵活 |
| **DSA** | **动态 top-k** | **是(per-query 学习)** | indexer 需额外训练对齐 |

DSA 的优势:**每个 token 学会去关注它认为最相关的少数历史 token**,而不是固定窗口或固定模式 —— 在保持质量的前提下做到内容自适应稀疏。

---

## 1.9 DSA 面试问答

**Q1:DSA 解决什么问题?和 MLA 是什么关系?**
A:DSA 解决长上下文下 attention 的 O(L²) 计算瓶颈。它不替换 MLA,而是**实例化在 MLA 之上**:用一个轻量 indexer 对历史 token 打分,选 top-k(如 2048)个最相关的 latent KV,只在这些 KV 上跑 MLA,把复杂度降到 O(Lk)。

**Q2:Lightning Indexer 为什么&#34;快&#34;?**
A:三点 —— ① head 数很少;② 低维投影;③ 用 FP8 计算,且激活用 ReLU(不用 softmax)以提吞吐。虽然 indexer 名义上仍 O(L²),但它极轻量,成本被&#34;不对全 context 做完整 MLA&#34;省下的部分淹没。

**Q3:为什么 DSA 要在 MLA 的 MQA 模式下实现?**
A:kernel 层面需要每个 KV entry 被多个 query 共享才高效。MQA 模式让 MLA 的一个 latent KV 被所有 query head 共享,这样按 top-k 索引做 gather 时一份 KV 服务全部 head,kernel 效率最高。

**Q4:indexer 是怎么训练出来的?为什么要 detach?**
A:两阶段。先 dense warm-up:冻结其他参数,用主注意力分布(跨 head 求和 &#43; L1 归一)作为目标,用 KL 散度训 indexer 模仿它。再 sparse training:引入 top-k,放开全部参数,只在选中集合上对齐。**detach indexer 输入**是为了让 indexer 只被 KL loss 优化、主模型只被 LM loss 优化,两路梯度解耦,训练更稳。

**Q5:DSA 和滑动窗口稀疏有什么本质区别?**
A:滑动窗口是**固定**的局部模式,丢远距离依赖;DSA 是**学习得到的、内容自适应**的 top-k,每个 query 动态选自己最相关的历史 token,既稀疏又不丢关键长程依赖。

**Q6:DSA 带来的收益?**
A:复杂度 O(L²)→O(Lk);长序列端到端成本约减半;长上下文性能基本不退化。更深一层,它让 test-time compute 变便宜,使大规模 RL / 长推理在线性成本内可行。

---
---

# Part 2 · MTP(Multi-Token Prediction)

## 2.1 背景与动机

标准语言模型每个位置只预测**下一个 token**(next-token prediction)。MTP 把预测范围扩展到**每个位置预测多个未来 token**。

两个动机:
1. **稠密化训练信号(densify training signals)**:每个 token 不只贡献一个预测 loss,而是贡献多个,提升数据效率。
2. **让模型预先规划表示(pre-plan representations)**:为了预测更远的 token,模型被迫学习更有前瞻性的内部表示。

&gt; 灵感来自 Gloeckle et al. (2024),但有关键区别 —— 见 2.2。

---

## 2.2 核心区别:Sequential vs Parallel

| | Gloeckle et al. (2024) | **DeepSeek-V3 MTP** |
|---|---|---|
| 预测方式 | **并行**预测 D 个 token | **顺序(sequential)** 预测 D 个 token |
| 输出头 | D 个**独立**输出头 | 模块串联,**共享**输出头 |
| 因果链 | 不保持完整因果链 | **保持完整因果链(complete causal chain)** |

DeepSeek 的关键改进:**用串联的多个 MTP 模块顺序预测,并在每个预测深度都保持完整的因果链**,从而保留自回归特性。这一点和 EAGLE 的&#34;维持因果链&#34;思想类似,但 EAGLE 主要目标是投机解码,DeepSeek 主要目标是**提升训练**。

---

## 2.3 MTP 模块组成

用 **D 个串联 MTP 模块**预测 D 个额外 token。第 k 个 MTP 模块包含 4 个部件:

| 部件 | 是否共享 | 说明 |
|---|---|---|
| Embedding Layer $\mathrm{Emb}(\cdot)$ | **与主模型共享** | 复用主模型学到的 token 表示 |
| Output Head $\mathrm{OutHead}(\cdot)$ | **与主模型共享** | 复用主模型 hidden→词表概率的映射 |
| Transformer Block $\mathrm{TRM}_k(\cdot)$ | **独立(每个深度一个)** | 处理该深度的组合表示 |
| Projection Matrix $M_k$ | **独立(每个深度一个)** | 把拼接后的表示投影回模型维度 |

&gt; 记忆口诀:**Embedding 和 Output Head 共享,Transformer Block 和 Projection 各深度独立。**

---

## 2.4 前向计算流程(核心公式)

设主模型已对输入 $t_1, \dots, t_T$ 算出 hidden states。对第 i 个 token、预测深度 k:

**Step 1 — 拼接 &#43; 投影**:把上一深度的 hidden $h_i^{k-1}$ 与&#34;未来第 k 个真实 token&#34;的 embedding 拼接,经 $M_k$ 投影:

$$
h_i^{\prime k} = M_k \left[\, \mathrm{RMSNorm}(h_i^{k-1}) \,;\, \mathrm{RMSNorm}(\mathrm{Emb}(t_{i&#43;k})) \,\right]
$$

**Step 2 — 过 Transformer Block**:

$$
h_{1:T-k}^{k} = \mathrm{TRM}\_k\!\left(h_{1:T-k}^{\prime k}\right)
$$

**Step 3 — 共享输出头预测**:

$$
p_{i&#43;k&#43;1}^{k} = \mathrm{OutHead}\!\left(h_i^{k}\right)
$$

直观例子(D=2,输入 t1~t4,主模型预测 t5):
- 主模型:用 h1~h5 预测 t5(下一个 token)
- MTP(k=1):用 h1~h4 预测 **t6**
- MTP(k=2):用 h1~h3 预测 **t7**

注意:**深度越深,可用的输入序列越短**(因为每深一层就要多看一个未来 token 作为输入)。

---

## 2.5 训练目标(Loss)

每个深度 k 计算一个交叉熵损失 $\mathcal{L}_{\mathrm{MTP}}^{k}$,然后对所有深度取平均,再乘一个权重因子 $\lambda$:

$$
\mathcal{L}\_{\mathrm{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\mathrm{MTP}}^{k}
$$

- MTP loss 作为**主训练目标(next-token)的辅助损失**加入。
- 核心目的是**提升主模型本身的性能**(让主模型表示更有前瞻性)。

---

## 2.6 推理阶段:两种用法

MTP 模块在推理时有两条路:

**用法 1 —— 直接丢弃(默认)**
- MTP 的主要目的是改善训练。推理时**可直接丢弃所有 MTP 模块**,主模型独立正常工作,不增加任何推理开销。

**用法 2 —— 复用为投机解码(Speculative Decoding)**
- 把 MTP 模块当作&#34;draft model&#34;,一次前向草拟多个未来 token;
- 主模型并行验证这些草稿 token,匹配则接受,跳过若干计算步;
- DeepSeek 报告:**第二个预测 token(MTP1)接受率约 85–90%(&gt;80%)**,带来约 **1.8× 的生成吞吐(TPS)提升**;在 SGLang 等框架上实测端到端延迟/吞吐提升约 1.2–2.1×。

&gt; MTP 模块&#34;轻量&#34;(单层 transformer head &#43; 共享 embedding/head),所以做草稿模型成本很低。

---

## 2.7 MTP 面试问答

**Q1:MTP 和普通的 next-token prediction 区别?**
A:普通 LM 每个位置只预测下一个 token;MTP 每个位置预测多个未来 token。好处:① 稠密化训练信号、提升数据效率;② 逼模型学更有前瞻性的表示。

**Q2:DeepSeek 的 MTP 和 Gloeckle(Meta)的 MTP 有什么不同?**
A:Meta 版用 D 个独立输出头**并行**预测、不保持因果链;DeepSeek 用串联模块**顺序**预测,**保持完整因果链**,且共享 embedding 和 output head。保持因果链让它天然兼容投机解码,也更符合自回归本质。

**Q3:一个 MTP 模块由哪些部件组成?哪些共享哪些独立?**
A:四个部件 —— 共享的 Embedding 和 Output Head(复用主模型),独立的 Transformer Block 和 Projection Matrix(每个深度各一个)。

**Q4:MTP 第 k 个模块的输入是什么?**
A:上一深度的 hidden $h_i^{k-1}$ 与&#34;未来第 k 个真实 token&#34;的 embedding $\mathrm{Emb}(t_{i&#43;k})$,各自 RMSNorm 后拼接,再经 $M_k$ 投影到模型维度,送入该深度的 Transformer Block。

**Q5:MTP 的 loss 怎么算?**
A:每个深度一个交叉熵,跨 D 个深度求平均,再乘权重 $\lambda$,作为辅助损失加到主 next-token 目标上。

**Q6:训练用了 MTP,推理时怎么办?**
A:两种 —— ① 直接丢弃 MTP 模块,主模型独立工作,零额外开销;② 复用为投机解码的草稿模型。后者中 MTP1 接受率约 85–90%,带来约 1.8× 吞吐提升。

**Q7:为什么 MTP 既能&#34;提训练&#34;又能&#34;提推理&#34;?**
A:训练时它稠密化信号、强迫模型学前瞻表示,提升主模型质量;推理时这些已经训好的轻量模块恰好能当草稿模型做投机解码 —— 一份训练投入,两处收益。

---
---

# Part 3 · DSA vs MTP 对比 &amp; 综合面试题

## 3.1 一张表理清两者

| 维度 | DSA | MTP |
|---|---|---|
| 出处 | DeepSeek-V3.2-Exp / V3.2 | DeepSeek-V3 |
| 解决的问题 | 长上下文 attention 的 **计算复杂度** | 训练信号密度 &#43; 推理加速 |
| 作用阶段 | 训练 **和** 推理都受益 | 训练为主;推理可选复用 |
| 核心机制 | indexer 打分 &#43; top-k 稀疏选择 | 串联模块顺序预测多 token |
| 复杂度影响 | O(L²) → O(Lk) | 不改主模型推理复杂度 |
| 与 MLA 关系 | 实例化在 MLA 之上(MQA mode) | 正交,独立模块 |
| 推理时去留 | 始终启用(就是推理机制) | 可丢弃或复用为投机解码 |
| 是否改架构 | 是(V3.2 唯一架构改动) | 是(加 MTP 模块) |

## 3.2 综合面试题

**Q1:DeepSeek 一整条技术线里,MLA、MoE、MTP、DSA 各自解决什么?**
A:
- **MLA**:压缩 KV cache,降推理显存(低秩 latent KV)。
- **MoE**:稀疏激活,扩参数量但控激活算力。
- **MTP**:多 token 预测,稠密训练信号 &#43; 可做投机解码。
- **DSA**:稀疏 attention,把长上下文 attention 从 O(L²) 降到 O(Lk)。
四者分别打**显存、参数效率、训练信号/推理加速、长上下文计算**四个不同维度的瓶颈。

**Q2:DSA 和 MTP 都说能&#34;加速推理&#34;,加速的是同一回事吗?**
A:不是。
- DSA 加速的是**单次前向里 attention 的计算量**(长上下文下尤其明显),是算法复杂度层面的下降。
- MTP 加速的是**生成阶段的 token 吞吐**(通过投机解码一次草拟多个 token、并行验证),是解码策略层面的加速。
- 两者正交,可叠加。

**Q3:如果让你给一个长上下文 &#43; 高并发推理的 Agent 系统选型,这两个特性你怎么用?**
A:DSA 必开 —— Agent 场景上下文长(工具调用历史、检索内容),O(Lk) 能显著降本;MTP 模块复用为投机解码 —— Agent 输出往往较长,1.8× 左右的吞吐提升直接降低端到端延迟。两者一个管&#34;读得起长上下文&#34;,一个管&#34;吐得快&#34;。

---

## 参考与延伸

- DeepSeek-V3 Technical Report(MTP 原始实现,Figure 3 &#43; 公式 21–25)
- DeepSeek-V3.2 / V3.2-Exp Report(DSA:lightning indexer &#43; token selection &#43; 两阶段训练)
- 工程实现参考:vLLM、SGLang、Megatron-LM 的 MTP/DSA 支持文档


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mldsamtp/  

