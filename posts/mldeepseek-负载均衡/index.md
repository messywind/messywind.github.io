# DeepSeek 负载均衡


# DeepSeek 负载均衡策略详解

&gt; 学习笔记 · 聚焦 MoE 路由的负载均衡（Load Balancing）
&gt; 演进路线：**DeepSeekMoE / V2（辅助损失）→ V3（无辅助损失 Auxiliary-Loss-Free）**

---

## 0. 为什么 MoE 需要负载均衡？

MoE（Mixture-of-Experts）每个 token 只激活 Top-K 个专家，路由器（Router/Gate）决定 token 去哪些专家。如果不加约束，会出现两类问题：

1. **路由坍缩（Routing Collapse）**
   训练初期某些专家偶然被多选 → 它们训练得更好 → 更容易被选中 → 形成&#34;马太效应&#34;。最终只有少数专家被频繁使用，其余专家几乎&#34;饿死&#34;，相当于浪费了模型容量，MoE 退化成一个小的稠密模型。

2. **计算/通信效率低下（系统层面）**
   专家分布在不同 GPU/节点上。如果负载不均，热门专家所在设备成为瓶颈，其他设备空转等待，拖慢整体吞吐；同时 all-to-all 通信也会不均衡。

&gt; **核心矛盾**：负载均衡（让专家被均匀使用）与模型性能（让 token 去最合适的专家）天然冲突。DeepSeek 的整条技术演进，本质就是在&#34;如何更优雅地化解这对矛盾&#34;。

负载均衡需要在**两个层面**同时考虑：
- **算法层面**：每个专家被选中的频率大致均衡（避免坍缩）。
- **系统层面**：设备（device）、节点（node）之间的计算与通信负载均衡。

---

## 1. 前置：DeepSeekMoE 架构基础

DeepSeek 的 MoE 有两个标志性设计，理解负载均衡前需要先知道：

- **细粒度专家分割（Fine-Grained Expert Segmentation）**
  把每个专家的 FFN 中间维度切小，专家数量变多（比如把 N 个专家拆成 mN 个更小的专家），相应每个 token 激活的专家数也变多。好处是专家组合更灵活、知识分解更细致。

- **共享专家隔离（Shared Expert Isolation）**
  把一部分专家设为**共享专家（Shared Expert）**，对所有 token 恒定激活，用来学习通用/共性知识；其余为**路由专家（Routed Expert）**，由 Router 动态选择，学习专门化知识。

&gt; **关键点**：负载均衡只作用于**路由专家**。共享专家恒定激活，不存在&#34;被不被选&#34;的问题。

记号约定（下文统一使用）：
- $u_t$：第 $t$ 个 token 的输入隐藏向量
- $e_i$：第 $i$ 个路由专家的中心向量（centroid）
- $s_{i,t}$：token $t$ 与专家 $i$ 的亲和度（affinity）
- $N_r$：路由专家总数；$K_r$：每个 token 激活的路由专家数
- $T$：序列长度（token 数）

---

## 2. 第一代方案：辅助损失（Auxiliary Loss）—— DeepSeekMoE / V2

思路：在语言建模主损失之外，**额外加几个均衡损失项**，用梯度&#34;逼&#34;路由器把负载摊平。V2 一共用了**三个**辅助损失 &#43; 两个工程策略。

### 2.1 专家级负载均衡损失（Expert-Level Balance Loss）

防止单个专家被过度使用（防坍缩）。

$$
\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i
$$

- $f_i$：专家 $i$ 实际被选中的 **频率（fraction）**
$$
f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}\big(\text{token } t \text{ 选了专家 } i\big)
$$
- $P_i$：专家 $i$ 的**平均门控概率（soft 概率）**
$$
P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}
$$

**直觉**：$f_i$ 是&#34;硬&#34;的选中比例（不可导），$P_i$ 是&#34;软&#34;的概率（可导）。两者相乘求和，当某专家又被频繁选中（$f_i$ 大）又被赋予高概率（$P_i$ 大）时，损失变大，梯度会通过 $P_i$ 把该专家的概率压下去。最小化该损失 ⟺ 把选中分布推向均匀分布。

### 2.2 设备级负载均衡损失（Device-Level Balance Loss）

把专家分成 $D$ 组，每组放一个设备上，**让设备间的负载均衡**（比专家级更粗粒度，更贴近实际算力分配）。

$$
\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f&#39;_i P&#39;_i,
\quad
f&#39;_i = \frac{1}{|\mathcal{E}_i|}\sum_{j \in \mathcal{E}_i} f_j,
\quad
P&#39;_i = \sum_{j \in \mathcal{E}_i} P_j
$$

其中 $\mathcal{E}_i$ 是第 $i$ 个设备上的专家集合。

### 2.3 通信负载均衡损失（Communication Balance Loss）

设备级均衡保证&#34;每个设备接收的 token 大致相等&#34;，但不能保证&#34;每个设备**发送**的 token 也相等&#34;。这个损失约束**all-to-all 通信中每个设备的发送量**均衡，避免某些设备成为通信热点。

$$
\mathcal{L}_{\text{CommBal}} = \alpha_3 \sum_{i=1}^{D} f&#39;&#39;_i P&#39;&#39;_i
$$

（形式与上面类似，只是统计的是发往各设备的通信量。）

### 2.4 设备受限路由（Device-Limited Routing）

**机制约束**，不是损失：限制每个 token 最多只能被路由到 $M$ 个设备上（先按设备上专家的最高亲和度选出 Top-M 设备，再在这些设备内选 Top-K 专家）。

**作用**：直接给通信成本设上界——一个 token 的 all-to-all 通信最多只涉及 $M$ 个设备，避免 token 把专家散布到所有设备造成通信爆炸。

### 2.5 Token 丢弃策略（Token-Dropping）

即便有均衡损失，训练中仍会有瞬时不均。为控制计算浪费，V2 采用**设备级 token 丢弃**：每个设备设一个容量上限（capacity factor），超过容量的、亲和度最低的 token 被丢弃（不参与该专家计算）。注意：会保护对均衡损失影响大的 token。

---

## 3. 辅助损失的根本缺陷 ⭐（V3 改进的动机）

辅助损失虽然有效，但有个绕不开的问题：**干扰梯度（Interference Gradients）**。

- 辅助损失的梯度和语言建模主损失的梯度**方向不一定一致**，相当于在主任务上引入了&#34;噪声/拉扯&#34;。
- 这就产生一个两难的**权衡困境（trade-off dilemma）**：
  - $\alpha$（损失权重）**太大** → 均衡得很好，但损害模型性能（主任务被干扰）。
  - $\alpha$ **太小** → 不干扰性能，但均衡不住，专家又会坍缩、token 被大量丢弃。
- 调 $\alpha$ 成了玄学，且无论怎么调，都是在&#34;性能&#34;和&#34;均衡&#34;之间做妥协，不存在两全。

&gt; **一句话总结动机**：辅助损失把&#34;负载均衡&#34;硬塞进了&#34;语言建模&#34;的梯度里，两个目标互相打架。能不能让负载均衡**不通过主损失的梯度**来实现？—— 这就是 V3 的答案。

---

## 4. 第二代方案：无辅助损失负载均衡（Auxiliary-Loss-Free）—— DeepSeek-V3 ⭐⭐⭐

&gt; 这是 V3 在 MoE 训练上的**头号贡献**，面试几乎必考。

### 4.1 核心思想：给每个专家加一个可学习的偏置项（Bias Term）

为每个路由专家 $i$ 维护一个**偏置 $b_i$**。这个偏置**只在 Top-K 选择时**加到亲和度上，用来&#34;调节&#34;哪些专家更容易被选中；但它**不参与最终门控值（gating value）的计算**。

### 4.2 数学表达

V3 的门控（注意 V3 把亲和度从 V2 的 **Softmax 改成了 Sigmoid**）：

$$
s_{i,t} = \text{Sigmoid}\big(u_t^{\top} e_i\big)
$$

Top-K 选择时用的是 **加了偏置** 的分数：

$$
g&#39;_{i,t} =
\begin{cases}
s_{i,t}, &amp; s_{i,t} &#43; b_i \in \operatorname{TopK}\big(\{\, s_{j,t} &#43; b_j \mid 1 \le j \le N_r \,\},\; K_r\big) \\[4pt]
0, &amp; \text{otherwise}
\end{cases}
$$

最后归一化得到门控权重：

$$
g_{i,t} = \frac{g&#39;_{i,t}}{\sum_{j=1}^{N_r} g&#39;_{j,t}}
$$

### 4.3 偏置的动态更新（无梯度，规则式调整）

$b_i$ **不是靠损失反向传播学的**，而是每个训练步后按&#34;专家当前负载&#34;用一条简单规则更新：

- 统计本 batch 每个专家收到的 token 数（负载 $c_i$）。
- 若专家 $i$ **过载**（负载高于平均）→ $b_i \leftarrow b_i - \gamma$（降低偏置，让它后续更难被选中）。
- 若专家 $i$ **欠载**（负载低于平均）→ $b_i \leftarrow b_i &#43; \gamma$（提高偏置，让它后续更易被选中）。

其中 $\gamma$ 是**偏置更新速度**（bias update speed），是一个超参。可写成：

$$
b_i \leftarrow b_i &#43; \gamma \cdot \text{sign}\big(\bar{c} - c_i\big)
$$

这是一个**负反馈控制器**：热门专家被自动&#34;调冷&#34;，冷门专家被自动&#34;调热&#34;，整个系统自我趋向均衡。

### 4.4 关键洞察：偏置只改路由，不改门控值 ⭐

这是整个方案的精髓，**一定要理解**：

- $b_i$ **只用于 Top-K 的筛选**（决定&#34;谁被选&#34;）。
- 一旦专家被选中，乘到 token 表示上的权重 $g_{i,t}$ 仍来自**原始亲和度 $s_{i,t}$**，**不含 $b_i$**。

因此：
- 偏置实现了负载均衡，却**没有往主损失里注入任何梯度**——彻底避开了&#34;干扰梯度&#34;。
- 负载均衡和语言建模**解耦**，不再有 $\alpha$ 的权衡困境。
- 结果：V3 既保持了优秀的均衡，又拿到了比&#34;靠辅助损失&#34;更好的模型性能。

### 4.5 互补的序列级辅助损失（Complementary Sequence-Wise Auxiliary Loss）

偏置法保证的是**全局/batch 级**均衡，但无法防止**单条序列内部**出现极端不均（某条序列把 token 全堆给少数专家）。所以 V3 仍保留一个**极小权重 $\alpha$** 的序列级均衡损失作为兜底：

$$
\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i
$$

$$
f_i = \frac{N_r}{K_r T}\sum_{t=1}^{T}\mathbb{1}\big(\text{token } t \text{ 选了专家 } i\big),
\quad
P_i = \frac{1}{T}\sum_{t=1}^{T} s&#39;_{i,t},
\quad
s&#39;_{i,t} = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}
$$

&gt; 注意：它是&#34;**序列级（sequence-wise）**&#34;统计（在每条序列内部算），且 $\alpha$ **极小**——只起兜底作用，不会重新引入明显的干扰梯度。这是它和 V2 辅助损失的本质区别。

### 4.6 节点受限路由（Node-Limited Routing）

V2 的&#34;设备受限&#34;在 V3 升级为&#34;**节点受限**&#34;：每个 token 最多被路由到 $M$ 个节点（V3 中 $M=4$）。选节点的依据是&#34;该节点上专家亲和度 Top-K 之和&#34;最高的 $M$ 个节点。同样是为了**给跨节点通信成本设上界**，让计算-通信尽量重叠。

### 4.7 训练全程不丢弃 Token（No Token-Dropping）

由于偏置法把负载均衡得足够好，V3 在**训练和推理全程都不丢 token**，避免了 token-dropping 带来的信息损失和训练-推理不一致。

---

## 5. V2（辅助损失）vs V3（无辅助损失）对比

| 维度 | DeepSeek-V2 / MoE | DeepSeek-V3 |
|---|---|---|
| 均衡核心手段 | 多个**辅助损失**（专家级/设备级/通信级） | **偏置项** $b_i$（无梯度规则更新） |
| 是否引入干扰梯度 | 是（核心痛点） | **否**（均衡与主损失解耦） |
| 亲和度激活函数 | Softmax | **Sigmoid** |
| 偏置是否影响门控值 | —（无偏置） | **否**，只影响 Top-K 选择 |
| 是否还有辅助损失 | 有，且权重需谨慎调 | 仅保留**极小权重**的序列级兜底损失 |
| 路由范围约束 | 设备受限路由（Device-Limited） | 节点受限路由（Node-Limited，$M=4$） |
| Token 丢弃 | 设备级 token-dropping | **全程不丢弃** |
| 调参负担 | 需要在性能/均衡间艰难权衡 $\alpha$ | 只需调更新速度 $\gamma$，鲁棒得多 |

---

## 6. 面试直觉总结 ⭐（背这几条就够答）

1. **MoE 为什么要负载均衡？** 防路由坍缩（避免容量浪费）&#43; 系统层面设备/通信均衡（避免瓶颈）。

2. **传统辅助损失的问题？** 引入**干扰梯度**，造成&#34;性能 vs 均衡&#34;的权衡困境，$\alpha$ 难调。

3. **V3 怎么解决的？** 用**偏置项**做负载均衡：偏置**只影响 Top-K 路由选择，不进入门控值计算**，所以不产生干扰梯度，把均衡和主损失**解耦**。

4. **偏置怎么更新？** 不靠反向传播，而是规则式负反馈：过载专家降偏置、欠载专家升偏置，速度由 $\gamma$ 控制。

5. **为什么还留一个辅助损失？** 偏置管不了**单条序列内部**的极端不均，所以保留一个**极小权重的序列级损失**兜底。

6. **一句话电梯陈述**：
   &gt; &#34;V3 把负载均衡从&#39;损失项&#39;变成了&#39;路由偏置&#39;——它只在选专家时起作用、不参与权重计算，因此不污染语言建模的梯度，从而同时拿到了好的均衡和好的性能。&#34;

---

## 7. 极简代码：无辅助损失偏置路由

下面是 V3 风格 router 的最小可运行骨架，重点演示 **4.2 / 4.3 / 4.4** 三个核心点（偏置只用于选择、门控值用原始亲和度、偏置规则更新）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxFreeRouter(nn.Module):
    &#34;&#34;&#34;DeepSeek-V3 风格的无辅助损失负载均衡路由器（教学简化版）&#34;&#34;&#34;

    def __init__(self, hidden_dim: int, n_experts: int, top_k: int,
                 bias_update_speed: float = 1e-3):
        super().__init__()
        # 专家中心向量 e_i，用一个 Linear 表示 u_t^T e_i
        self.gate = nn.Linear(hidden_dim, n_experts, bias=False)
        self.top_k = top_k
        self.gamma = bias_update_speed
        # 偏置 b_i：注意是 buffer 不是 Parameter —— 它不参与反向传播！
        self.register_buffer(&#34;bias&#34;, torch.zeros(n_experts))

    def forward(self, x):
        # x: [num_tokens, hidden_dim]
        # 1) 亲和度用 Sigmoid（V3 的改动，V2 是 Softmax）
        affinity = torch.sigmoid(self.gate(x))          # s_{i,t}, [T, N]

        # 2) Top-K 选择时加上偏置 b_i（核心点之一）
        scored = affinity &#43; self.bias                    # s &#43; b
        topk_idx = scored.topk(self.top_k, dim=-1).indices  # [T, K]

        # 3) 门控值仍来自“原始亲和度” s，不含 b（核心点之二！）
        topk_affinity = affinity.gather(-1, topk_idx)    # 注意是 affinity 不是 scored
        gates = topk_affinity / topk_affinity.sum(-1, keepdim=True)  # 归一化

        # 训练时统计每个专家负载，供 update_bias 使用
        if self.training:
            load = torch.bincount(topk_idx.flatten(),
                                  minlength=self.bias.numel()).float()
            self._last_load = load

        return topk_idx, gates

    @torch.no_grad()
    def update_bias(self):
        &#34;&#34;&#34;每个训练步之后调用：规则式负反馈更新偏置（无梯度，核心点之三）&#34;&#34;&#34;
        load = self._last_load
        avg = load.mean()
        # 过载(load&gt;avg) -&gt; 降偏置；欠载 -&gt; 升偏置
        self.bias &#43;= self.gamma * torch.sign(avg - load)


# ---- 使用示意 ----
# router = AuxFreeRouter(hidden_dim=512, n_experts=64, top_k=6)
# for batch in loader:
#     idx, gates = router(batch)        # 路由
#     loss = compute_lm_loss(...)       # 只有语言建模损失（&#43;极小序列级兜底）
#     loss.backward(); optimizer.step()
#     router.update_bias()              # 关键：在优化器之外单独更新偏置
```

**三个最该记住的代码细节**：
- `self.bias` 用 `register_buffer` 而非 `nn.Parameter` —— 强调它**不走反向传播**。
- 计算 `gates` 时 gather 的是 **`affinity`（原始 $s$）** 而不是 `scored`（$s&#43;b$）—— 偏置**不进门控值**。
- `update_bias()` 在 `optimizer.step()` **之外**单独调用 —— 负载均衡与主损失优化**解耦**。

---

## 附：延伸阅读

- DeepSeekMoE 论文：细粒度专家 &#43; 共享专家隔离
- DeepSeek-V2 论文：三个辅助损失 &#43; 设备受限路由 &#43; token 丢弃
- DeepSeek-V3 论文：Auxiliary-Loss-Free Load Balancing（本笔记第 4 章核心来源）


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mldeepseek-%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1/  

