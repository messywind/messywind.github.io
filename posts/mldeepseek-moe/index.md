# DeepSeek MoE


# DeepSeek 大模型 — MoE 架构笔记

## 0. 一句话脉络

MoE（Mixture of Experts）的核心思想是 **&#34;用多个专家网络 &#43; 一个门控（路由）网络，按输入动态决定让哪些专家参与计算&#34;**。

演进路线：

```
Adaptive Mixtures of Local Experts (1991, bagging 思想)
        ↓
Dense MoE        —— 所有专家都算，门控给权重
        ↓
Sparse MoE       —— 只激活 Top-K 专家（Sparsely-Gated MoE, Google 2017）
        ↓
Mixtral MoE      —— 8 个 FFN 专家，Top-2 激活
        ↓
DeepSeek MoE     —— 细粒度路由专家 &#43; 共享专家（专家隔离）
```

---

## 1. Dense MoE（稠密 MoE）

### 结构

```
输入 X [B, D]
   ├──&gt; Expert1 ──&gt; o1 [B, M] ──┐
   ├──&gt; Expert2 ──&gt; o2 [B, M] ──┤
   └──&gt; Expert3 ──&gt; o3 [B, M] ──┤
                                 ├──&gt; Σ ──&gt; 输出 Y [B, M]
   X ──&gt; 门控 (Gating) ──&gt; [B, 3] ┘
        输出权重 w1, w2, w3 ∈ [0, 1]
```

- **门控网络**：对输入打分，输出每个专家的权重，激活函数用 **sigmoid / softmax**，把权重压到 `[0, 1]`。
- **Dense 的特点**：**每个专家都要计算**，再用门控权重做加权求和。计算量大，没有稀疏性。

### 公式

$$
y = \sum_{k=1}^{K} o_k \cdot w_k
$$

其中 $o_k$ 是第 $k$ 个专家的输出，$w_k$ 是门控给出的权重。

---

## 2. 稀疏 MoE 的三种门控公式（核心）

&gt; 这一组公式是面试常考的&#34;数学层&#34;，建议背下来。

### ① Dense MoE

$$
y = \sum_{i=1}^{n} G(x_i) \cdot E(x_i)
$$

$$
G(x) = \mathrm{softmax}(x \cdot W_g) \in [B, L, n]
$$

- 门控输出维度是 `n`（专家总数），**所有专家都有非零权重**。

### ② 稀疏 MoE（Sparse）

$$
G(x) = \mathrm{softmax}\big(\mathrm{KeepTopK}(H(x), k)\big) \in [B, L, K]
$$

$$
H(x) = x \cdot W_g
$$

- 先打分 $H(x)$，再用 **KeepTopK 只保留 Top-K**，其余置 $-\infty$，softmax 后变成 0 → 实现稀疏激活。
- 门控有效维度从 `n` 降到 `K`。

### ③ 带噪稀疏 MoE（Noisy Top-K Gating）

&gt; 即 Google 的 **Sparsely-Gated MoE**（Shazeer et al., 2017）

$$
H(x_i) = x_i \cdot W_g &#43; \mathrm{StandardNormal}(\cdot) \cdot \mathrm{Softplus}(x_i \cdot W_{noise}) \in [B, L, n]
$$

$$
\mathrm{Softplus}(x) = \log(1 &#43; e^{x})
$$

$$
\mathrm{KeepTopK}(v, k)_i =
\begin{cases}
v_i &amp; \text{if } v_i \text{ 在 } v \text{ 的 Top-}k \text{ 中} \\
-\infty &amp; \text{otherwise}
\end{cases}
$$

- **加噪的作用**：在打分上加可学习的高斯噪声（噪声幅度由 $W_{noise}$ 控制），**鼓励探索、缓解专家&#34;赢者通吃&#34;、帮助负载均衡**。
- Softplus 保证噪声幅度恒为正。

---

## 3. Mixtral MoE 架构

### 关键参数

- **Mixtral 8×7B**：8 个 FFN 专家，每个约 7B。
- **56B 总参数，14B 激活**（Top-K=2，约 2×7B 被激活）。

### 结构

```
输入 X [B, L, D]
   └─&gt; 路由网络 ──&gt; 选出 Top-2 专家
         ├──&gt; FFN1 (7b) ──&gt; o1 [B,L,D] ──┐
         ├──&gt; FFN2 (7b) ──&gt; o2  ✗(未选中) │
         ├──&gt; ......                       ├──&gt; Σ ──&gt; 输出 Y [B,L,D]
         └──&gt; FFN8 (7b) ──&gt; o8 [B,L,D] ──┘
   专家权重 [B, L, Top-K], K=2
```

- 用 **FFN** 充当专家（取代 Transformer Block 里的标准 FFN）。
- 路由网络选 **Top-2**，未选中的专家（如图中 FFN2）**不参与计算**（红叉），权重路径被切断。
- 总参数大但激活参数小 → **推理时算力 ≈ 2 个 FFN**，但模型容量 ≈ 8 个 FFN。

---

## 4. DeepSeek MoE 架构（专家隔离 &#43; 共享专家）

### 结构

```
                            ┌─&gt; 共享专家 (Shared) ──────────────┐
输入 X [B,L,D] ──────────────┤                                    │
   └─&gt; 路由网络 ─────────────┼─&gt; 路由专家1 ──&gt; o1 ──&gt; w1 ──┐      │
        ↑                    ├─&gt; 路由专家2 ──&gt; o2  ✗         ├──&gt; Σ ──&gt; 输出 Y [B,L,D]
   高斯噪声(打分加噪)         └─&gt; 路由专家n ──&gt; on ──&gt; wn ──┘      │
                              专家权重 [B, L, Top-K]                │
                                                                    ┘
```

### 两类专家

| 类型 | 作用 | 是否每次都激活 |
|------|------|----------------|
| **共享专家 (Shared Expert)** | 提供通用特征提取，所有 token 都过 | ✅ 始终激活 |
| **路由专家 (Routed Expert)** | 细粒度、各有专长，由路由网络选 Top-K | ❌ 只激活 Top-K |

&gt; 路由网络打分时会**加入高斯噪声**（对应第 2 节的带噪门控），帮助负载均衡。

### 版本参数对比

| 版本 | 路由专家 | 共享专家 | Top-K | 激活/总参数 |
|------|----------|----------|-------|-------------|
| **DeepSeek V2** | 160 routed | 2 shared | 6 | 21B / 236B |
| **DeepSeek V3** | 256 routed | 1 shared | 8 | 37B / 671B |

&gt; 趋势：**专家更细粒度（数量更多）、共享专家更少、Top-K 更多、激活率更低**。

---

## 5. Token 路由直观示例

例句：**&#34;他昨天很荣幸地收到了期待已久的 offer&#34;**

假设三个专家的专长：

- **专家 1**：擅长处理情感词汇和情感表达
- **专家 2**：擅长处理实体指代和关系理解
- **专家 3**：擅长处理语法结构和句法关系
- **共享专家**：提供通用的特征提取

| Token | 词性 | 可能路由到 |
|-------|------|-----------|
| 他 | 代词 (Pronoun) | 专家 2（实体指代） |
| 昨天 | 时间状语 (Time Adverb) | 专家 2（情境理解） |
| 很荣幸地 | 副词&#43;形容词 | 专家 1（情感词汇） |
| 收到 | 动词 (Verb) | 专家 3（动词/语法结构） |
| 了 | 助词 (Particle) | 专家 3（语法助词） |
| 期待已久的 | 形容词短语 | 专家 1（情感）&#43; 专家 2（描述性信息） |
| offer | 名词 (Noun) | 专家 2（实体） |
| **整个句子** | — | **都会经过共享专家** |

&gt; 直觉：不同 token 按&#34;语义功能&#34;被路由到不同的专长专家；共享专家兜底通用信息。
&gt; 在自回归生成中，**每一步、每一层、每个 token 激活的专家组合都不同**（图 2 的逐步生成示意）。

---

## 6. 面试题 1：为什么要有&#34;共享专家&#34;的设计？有什么好处？

**① 减少冗余，让 Routed experts 更专业**
- 通用知识由共享专家承担，路由专家不必各自重复学习通用特征，可以更专注于细分领域 → 提升专业化程度、提高参数利用率。

**② 计算更高效（与 All2All 通信相关）**
- MoE 的专家并行需要 **All2All 通信**把 token 分发到对应专家所在的设备。
- 共享专家始终在本地参与计算，可以 **让通信与计算重叠（overlap）**，把 All2All 的通信延迟&#34;隐藏&#34;在共享专家的计算里 → 整体吞吐更高。

&gt; 关键词：**减少冗余 / 专业化 / 通信-计算重叠 / 通信隐藏 / All2All**

---

## 7. 面试题 2：MoE 的门控输出必须用 softmax 吗？

**结论：不是必须。** DeepSeek V2 用 softmax，**V3 改用 sigmoid**。

为什么 V3 换成 sigmoid：

**① Softmax 的 score 区分度会随专家数 N 增大而降低**
- 专家数 `N↑` 时，softmax 归一化把概率摊薄 → 分布**变平坦**，各专家得分差异变小，难以区分。

**② sigmoid 值域更宽，不受专家数量 N 影响**
- sigmoid 对每个专家**独立**打 `[0,1]` 分，不做跨专家归一化 → 专家数再多，单个分数也不会被稀释。

**③ 从&#34;单选&#34;变&#34;多选&#34;，有助于负载均衡**
- softmax 天然是&#34;互斥竞争&#34;（此消彼长）；sigmoid 是独立判断，更像多标签 → token 不会被强行挤到少数专家，**有利于负载均衡**。

&gt; 关键词：**区分度 / N↑ 变平坦 / 值域不受 N 影响 / 单选→多选 / 负载均衡**

---

## 8. 速记小结

- **Dense MoE**：全专家计算 &#43; 加权求和，无稀疏。
- **Sparse MoE**：`softmax(KeepTopK(...))`，只激活 Top-K。
- **Noisy Top-K（Google 2017）**：打分加高斯噪声 × Softplus，促进负载均衡。
- **Mixtral**：8×7B，Top-2，56B 总 / 14B 激活。
- **DeepSeek MoE**：路由专家 &#43; 共享专家（专家隔离）；V2: 160&#43;2 / Top-6 / 21B@236B；V3: 256&#43;1 / Top-8 / 37B@671B。
- **共享专家好处**：减冗余 &#43; 专业化 &#43; 通信计算重叠（隐藏 All2All）。
- **门控激活函数**：softmax 非必须，V3 用 sigmoid（区分度、值域、负载均衡三点）。

## 代码：

```python
&#34;&#34;&#34;
DeepSeek MoE 手撕实现（学习用）
==================================

本文件把 notebook 的渐进式搭建过程整理为模块化结构，方便对照架构学习：

    ModelArgs           —— 超参数配置
    TopkRouter          —— 带噪声的 Top-K 门控路由
    Expert              —— 标准 FFN 专家（ReLU），对应基础版 MoE
    SparseMOE           —— 基础稀疏 MoE（仅路由专家）
    DeepSeekExpert      —— SwiGLU 专家，DeepSeek 用的 FFN
    DeepSeekMOE         —— 完整 DeepSeek MoE（路由专家 &#43; 共享专家）

整体数据流（一句话总结）：
    x --router--&gt; 选 Top-K 个专家并给出门控权重
      --gather--&gt; 只把被选中的 token 喂给对应专家计算
      --weighted sum--&gt; 按门控权重加权求和写回
      --shared experts--&gt; 所有 token 都过共享专家，结果叠加

运行：python deepseek_moe.py
&#34;&#34;&#34;

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 超参数配置
# ============================================================
@dataclass
class ModelArgs:
    n_dim: int              # 隐藏维度 d
    n_experts: int          # 路由专家数量
    n_shared_experts: int   # 共享专家数量（DeepSeek 的关键设计）
    top_k: int              # 每个 token 激活的路由专家数
    dropout: float
    batch: int
    seq_len: int
    add_noise: bool         # 训练时是否给门控 logits 加噪声


# ============================================================
# Router：带噪声的 Top-K 门控
# ============================================================
class TopkRouter(nn.Module):
    &#34;&#34;&#34;Top-K MoE 门控单元。

    作用：对每个 token，决定走哪 top_k 个专家，并给出归一化的门控权重。

    关键步骤：
        1. 线性层得到 logits        [b, l, n_experts]
        2. 加可学习的高斯噪声        noise = N(0,1) * softplus(W_noise · x)
        3. topk 选出每个 token 的前 k 个专家
        4. 用 scatter 把非 Top-K 位置填成 -inf 得到稀疏 logits
        5. softmax 得到门控权重（-inf 位置自动变 0，权重只在被选专家间归一化）
    &#34;&#34;&#34;

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.top_k
        self.add_noise = args.add_noise

        # 门控打分层
        self.top_k_linear = nn.Linear(args.n_dim, args.n_experts)
        # 噪声幅度也是可学习的（Noisy Top-K Gating）
        self.noise_linear = nn.Linear(args.n_dim, args.n_experts)

    def forward(self, mha_output):
        # logits: [b, l, n_experts]
        logits = self.top_k_linear(mha_output)

        # 加噪声：normal(.) * softplus(W_noise · x)
        # softplus 保证噪声幅度非负；噪声能鼓励探索、缓解专家&#34;赢者通吃&#34;
        if self.add_noise:
            noise_logits = self.noise_linear(mha_output)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits &#43; noise

        # 选出每个 token 的 top_k 专家
        # 修正：原 notebook 这里写的是 args.top_k（依赖全局变量），改用 self.top_k 让类自洽
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

        # 构造稀疏 logits：先全填 -inf，再把 Top-K 的真实分数 scatter 回去
        infs = torch.full_like(logits, float(&#34;-inf&#34;))
        sparse_logits = infs.scatter(-1, top_k_indices, top_k_logits)

        # softmax：-inf 位置 -&gt; 0，权重只在被选中的专家之间归一化
        gating_output = F.softmax(sparse_logits, dim=-1)

        # gating_output: [b, l, n_experts]（非 Top-K 位置为 0）
        # top_k_indices : [b, l, top_k]
        return gating_output, top_k_indices


# ============================================================
# 专家网络
# ============================================================
class Expert(nn.Module):
    &#34;&#34;&#34;标准 FFN 专家（基础版 MoE 用），结构：Linear -&gt; ReLU -&gt; Linear -&gt; Dropout。&#34;&#34;&#34;

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(args.n_dim, 4 * args.n_dim),
            nn.ReLU(),
            nn.Linear(4 * args.n_dim, args.n_dim),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class DeepSeekExpert(nn.Module):
    &#34;&#34;&#34;DeepSeek 风格的 SwiGLU 专家。

    SwiGLU(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )

    与标准 FFN 的区别：用门控乘法（gate * up）替代单一激活，表达能力更强，
    是 LLaMA / DeepSeek 等现代模型的标配。
    &#34;&#34;&#34;

    def __init__(self, args: ModelArgs, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(args.n_dim, 4 * args.n_dim, bias=bias)
        self.up_proj = nn.Linear(args.n_dim, 4 * args.n_dim, bias=bias)
        self.down_proj = nn.Linear(4 * args.n_dim, args.n_dim, bias=bias)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


# ============================================================
# 基础版稀疏 MoE（仅路由专家）
# ============================================================
class SparseMOE(nn.Module):
    &#34;&#34;&#34;完整的稀疏 MoE 网络（基础版）。

    核心思想：先算 router 决定每个 token 走哪些专家，
    再&#34;按专家分组&#34;批量计算——没被选中的专家完全不参与计算（这才是稀疏的省算力来源）。
    &#34;&#34;&#34;

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.router = TopkRouter(args)
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.n_experts)])
        self.top_k = args.top_k

    def forward(self, x):
        # gating_output: [b, l, n_experts]; indices: [b, l, top_k]
        gating_output, indices = self.router(x)

        # 输出初始化为 0，形状与输入相同 [b, l, d]
        final_output = torch.zeros_like(x)

        # 摊平到 token 维度，方便按专家做 batch 计算
        flat_x = x.view(-1, x.size(-1))                              # [b*l, d]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # [b*l, n_experts]

        # 逐个专家处理（同一专家选中的 token 一起算）
        for i, expert in enumerate(self.experts):
            # 该专家是否出现在某个 token 的 Top-K 里：[b, l]
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)                         # [b*l]

            if flat_mask.any():
                # 只取选中该专家的 token：[select_k, d]
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)                # [select_k, d]

                # 取这些 token 在专家 i 上的门控权重：[select_k] -&gt; [select_k, 1]
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # 写回输出（一个 token 可能被多个专家更新，所以用 &#43;=）
                final_output[expert_mask] &#43;= weighted_output.squeeze(1)

        return final_output


# ============================================================
# DeepSeek MoE（路由专家 &#43; 共享专家）
# ============================================================
class DeepSeekMOE(nn.Module):
    &#34;&#34;&#34;完整的 DeepSeek MoE。

    相比基础版的两点关键改动：
        1. 专家用 SwiGLU（DeepSeekExpert）
        2. 增加&#34;共享专家&#34;：所有 token 都会经过，提供公共/通用知识，
           让路由专家更专注于各自擅长的细分能力。
    最终输出 = 路由专家的加权和 &#43; 所有共享专家之和
    &#34;&#34;&#34;

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.router = TopkRouter(args)
        self.routed_experts = nn.ModuleList(
            [DeepSeekExpert(args) for _ in range(args.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [DeepSeekExpert(args) for _ in range(args.n_shared_experts)]
        )
        self.top_k = args.top_k

    def forward(self, x):
        gating_output, indices = self.router(x)             # [b,l,n_experts], [b,l,top_k]

        final_output = torch.zeros_like(x)                  # [b, l, d]

        flat_x = x.view(-1, x.size(-1))                     # [b*l, d]
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # ---- 路由专家：稀疏激活，仅算被选中的 token ----
        for i, expert in enumerate(self.routed_experts):
            expert_mask = (indices == i).any(dim=-1)        # [b, l]
            flat_mask = expert_mask.view(-1)                # [b*l]

            if flat_mask.any():
                expert_input = flat_x[flat_mask]            # [select_k, d]
                expert_output = expert(expert_input)        # [select_k, d]

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] &#43;= weighted_output.squeeze(1)

        # ---- 共享专家：所有 token 都经过，直接叠加（无门控权重）----
        # 修正：原 notebook 用 expert(flat_x) 得到 [b*l, d] 加到 [b,l,d] 上，
        #       靠 batch=1 的广播碰巧能跑；这里直接用 expert(x) 保持 [b,l,d]，对任意 batch 都正确。
        for expert in self.shared_experts:
            final_output = final_output &#43; expert(x)

        return final_output, indices


# ============================================================
# Demo / 测试
# ============================================================
def _demo_forward(args: ModelArgs):
    &#34;&#34;&#34;跑一遍前向，确认输入输出维度一致。&#34;&#34;&#34;
    torch.manual_seed(1)
    mha_output = torch.randn(args.batch, args.seq_len, args.n_dim)

    moe = DeepSeekMOE(args)
    final_output, indices = moe(mha_output)

    print(&#34;=== 前向测试 ===&#34;)
    print(&#34;input  shape:&#34;, tuple(mha_output.shape))
    print(&#34;output shape:&#34;, tuple(final_output.shape))
    print(&#34;选中的专家 indices:\n&#34;, indices)


def _demo_load_balance(args: ModelArgs, count: int = 100):
    &#34;&#34;&#34;统计专家负载，直观感受路由是否均衡。

    理想情况下 count*seq_len*top_k 次激活会比较均匀地分到各专家上；
    若某些专家长期被冷落，就需要负载均衡损失（aux loss）来约束——
    这也是 MoE 面试常被追问的点。
    &#34;&#34;&#34;
    import numpy as np

    torch.manual_seed(1)
    moe = DeepSeekMOE(args)

    utilization = np.zeros(args.n_experts, dtype=int)
    for _ in range(count):
        mha_output = torch.randn(args.batch, args.seq_len, args.n_dim)
        _, indices = moe(mha_output)
        for idx in indices.detach().cpu().numpy().flatten():
            utilization[idx] &#43;= 1

    print(&#34;\n=== 专家负载测试 ===&#34;)
    print(&#34;专家负载:&#34;, utilization)
    print(&#34;激活总次数:&#34;, utilization.sum(),
          f&#34;(= count {count} * seq_len {args.seq_len} * top_k {args.top_k})&#34;)


if __name__ == &#34;__main__&#34;:
    args = ModelArgs(
        n_dim=32,
        n_experts=4,
        n_shared_experts=1,
        top_k=2,
        dropout=0.1,
        batch=1,
        seq_len=5,
        add_noise=True,
    )
    print(args, &#34;\n&#34;)

    _demo_forward(args)
    _demo_load_balance(args)
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mldeepseek-moe/  

