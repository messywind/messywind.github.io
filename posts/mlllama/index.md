# LLaMA


# LLaMA 家族架构底座与代际演进（v1 - v4）

在大模型面试中，对 LLaMA 的考察通常分为两条线：**“横向看架构底座”**（考底层算子与推导）与“纵向看代际演进”（考技术视野与选型把控）。

## 一、 LLaMA 家族通用底座架构（高频硬核考点）

LLaMA 确立了现代开源大模型的“行业标准架构”，以下四大组件是重中之重：

### 1. 归一化：RMSNorm &#43; Pre-Norm

* **为什么用 RMSNorm？** 原生的 LayerNorm 需要计算均值 $\mu$ 和方差 $\sigma^2$ ，并引入学习参数 $\gamma$ 和 $\beta$。RMSNorm 认为“均值归零”和“偏置项”对模型收益不大，直接砍掉了 $\mu$ 和 $\beta$，仅保留均方根（RMS）进行缩放。这消除了跨显存的重复读写，前向计算速度飙升 10% ~ 50%。
* **为什么用 Pre-Norm？** 经典的 Post-Norm 在深层网络中容易导致梯度消失或爆炸。LLaMA 在进入 Attention 和 FFN 之前先做归一化（Pre-Norm），确保主干残差的特征体量始终大于新算出的特征，保障了深层训练的绝对稳定性。

### 2. 位置编码：RoPE (旋转位置编码)

* **核心思想：** “用绝对位置的旋转，来实现相对位置的计算”。通过在复数域上对 Query 和 Key 向量乘以一个随位置变化的旋转矩阵，点积时绝对位置自然抵消，仅保留相对距离特征。
* **长文本扩展：** 面对长文本，直接外推会导致低频维度越界（OOD）。业界通常通过对 RoPE 的基础频率 $\theta$（Base）进行动态缩放（如 NTK-aware Scaling 或 YaRN 算法），做到“高频不改，低频压缩”，实现无损的长文本上下文扩展。

### 3. 激活函数：SwiGLU（非线性与门控之美）

在原始 Transformer 中，FFN 采用带 Bias 的 ReLU。LLaMA 移除了 Bias，并引入了门控机制（Gate）。

* **底层数学公式：**
首先计算带有门控的 SwiGLU 激活输出：

$$SwiGLU(x) = SiLU(xW_1) \otimes (xW_3)$$



然后乘上降维矩阵 $W_2$ 得到最终的 FFN 输出：

$$FFN_{LLaMA}(x) = (SiLU(xW_1) \otimes xW_3)W_2$$


* **机制解析：** $xW_1$ 经过 $SiLU$ 激活后充当“门控”，控制 $xW_3$ 提取的信息流向下一层的比例。
* **参数对齐：** 由于多了一个权重矩阵 $W_3$，为了与经典 Transformer 保持相同的总参数量，LLaMA 将 FFN 的隐藏层维度从经典的 $4d$ 缩小到了 $\frac{8}{3}d$ 的倍数。


SiLU 其实就是 $\beta=1$ 时的 Swish 函数，它的数学公式是输入 $x$ 乘以 $x$ 的 Sigmoid 激活值：

$$SiLU(x) = x \cdot \sigma(x) = \frac{x}{1 &#43; e^{-x}}$$

#### 💡 面试加分项：为什么 LLaMA 偏爱 SiLU 而不是 ReLU？

如果在面试中写出这个公式，面试官大概率会追问：“**从数学图像上看，SiLU 相比 ReLU 有什么优势？**” 你可以补充以下两点：

1. **平滑的非单调性（Smooth Non-monotonicity）：**
ReLU 在 $x=0$ 处是一个不可导的尖角（拐点），而 SiLU 处处平滑可导。这种平滑的误差曲面能让优化器（如 AdamW）在训练时更容易找到全局最优解，收敛更稳定。
2. **保留微弱的负梯度（解决 Dying ReLU 问题）：**
当 $x &lt; 0$ 时，ReLU 会直接将输出暴力截断为 0，导致神经元“死亡”（梯度彻底消失）。而 SiLU 在 $x &lt; 0$ 的区域依然会保留一个微小的负值输出（大约在 $x \approx -1.28$ 时达到最小值），这使得即使输入是负数，信息和梯度依然能够微弱地流动，极大地提升了网络的表达能力。

### 4. 推理优化：GQA (分组查询注意力)

* **解决显存灾难：** 传统 MHA 为每个 Query 头保存独立 KV 矩阵，导致显存受限（Memory-bound）。
* **核心机制：** GQA 将 Query 头分组，每组共享一组 KV 头。这是 MQA 和 MHA 的折中方案，大幅降低了 KV Cache 的显存占用，提高了推理的 Batch Size 上限，同时几乎无损模型性能。

---

## 二、 LLaMA 1 到 4 的代际演进史（宏观视野）

### 💡 第一阶段：基础奠定与开源破局 (LLaMA 1 &amp; 2)

* **LLaMA 1 (2023.02)：** 确立了 RMSNorm、RoPE、SwiGLU 的基本盘，但上下文仅 2K，无官方对话对齐版。
* **LLaMA 2 (2023.07)：** 走向商用。上下文扩至 4K，在 34B/70B 中正式引入 GQA。官方发布了 SFT 和 RLHF 对齐的 LLaMA-2-Chat，并引入 Ghost Attention (GAtt) 控制多轮对话。

### 💡 第二阶段：工程暴力的巅峰 (LLaMA 3 &amp; 3.1 &amp; 3.2)

* **LLaMA 3 (2024.04)：** 原生 8K Context；词表从 32K 扩容至 128K（极大提升多语言与中文效率）；全系标配 GQA；使用惊人的 15T Token 训练，极致压榨了 8B 小模型的潜力。
* **LLaMA 3.1：** RoPE 扩展使上下文暴增至 128K，发布 405B 超大杯。
* **LLaMA 3.2：** 走向端侧与多模态。推出 1B/3B 轻量级模型（应用剪枝与蒸馏），并引入 11B/90B 视觉编码器。

### 💡 第三阶段：稀疏 MoE 与原生多模态时代 (LLaMA 4)

LLaMA 4 首次全面引入了**稀疏混合专家（Sparse MoE）架构**和早期融合（Early Fusion）技术。以其标志性的双子星模型为例：

| 比较维度 | Llama 4 Scout (17B-16E) | Llama 4 Maverick (17B-128E) |
| --- | --- | --- |
| **底层总参数量** | ~109B | ~400B |
| **单次推理激活** | 17B | 17B |
| **MoE 专家配置** | 16 个专家 | 128 个极其细粒度的专家 |
| **上下文长度** | 行业级标准配置 | 极限可达 512K - 1M |
| **核心业务取向** | 高速并发、通用提取（~99%直接提取准确率） | 复杂逻辑推理、长文档分析、代码与 Agent 驱动 |
| **面试高分总结** | 适合追求低延迟、低成本、高并发的企业通用任务。 | 证明了“专家数量越多、底层库越大”，越能涌现深度的 System 2（慢思考）推理能力。 |

---

## 三、 经典连环追问实战模拟（大厂必考）

**Q1：LLaMA 在生成时突然不断重复同一个词，从解码策略上怎么解决？**

&gt; **解答：** 这属于典型的“退化式生成”。可以通过调整解码策略（Decoding Strategy）解决：
&gt; 1. 引入 **Top-P（核采样）** 和 **Top-K** 截断长尾低概率词的干扰。
&gt; 2. 调高惩罚项中的 **Repetition Penalty**（重复惩罚系数），强行压低已生成词的输出概率。
&gt; 3. 适当调高 **Temperature（温度参数）**，软化 Softmax 分布，增加生成的随机性。
&gt; 
&gt; 

**Q2：如果要训练一个垂直领域（如医疗/金融）的 LLaMA，完整的落地流程是怎样的？**

&gt; **解答：**
&gt; 1. **词表扩充（Vocab Extension）：** 评估专有名词比例，必要时向 128K 词表中注入行业 Token，并初始化对应的 Embedding 层。
&gt; 2. **CPT (继续预训练 / Continue Pre-training)：** 喂入海量无标注的行业研报、文档，让模型注入垂直领域的“暗知识”。
&gt; 3. **SFT (监督微调)：** 使用高质量的 Q&amp;A 数据对，约束模型的回答格式与语气。为了节省算力，通常采用 **LoRA 或 QLoRA** 进行参数高效微调（PEFT）。
&gt; 4. **推理部署：** 采用 vLLM 等框架，利用 PagedAttention 技术对 KV Cache 进行分块加载，极大提升生产环境的并发吞吐量。
&gt; 
&gt;

## 手撕代码：
```python
&#34;&#34;&#34;
LLaMA3 —— 手撕版（从 notebook 整理而来的模块化实现）

核心组件：
    1. RMSNorm           —— 均方根归一化
    2. RoPE              —— 旋转位置编码（复数实现）
    3. Attention (GQA)   —— 分组查询注意力 &#43; 因果掩码
    4. FeedForward       —— SwiGLU 前馈网络
    5. TransformerBlock  —— 一个 Decoder Layer（两段残差）
    6. Transformer       —— Embedding &#43; N 层 Block &#43; 输出头

约定的张量维度记号：
    B   batch size
    T   序列长度 (seq_len)
    D   模型维度 dim          (= 4096)
    H   注意力头数 n_heads    (= 32)
    Hkv KV 头数 n_kv_heads   (= 8)
    hd  每个头的维度 head_dim (= dim // n_heads = 128)
&#34;&#34;&#34;

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# 0. 配置（对应 params.json）
# ============================================================
@dataclass
class ModelArgs:
    dim: int = 4096            # 模型隐藏维度 D
    n_layers: int = 32         # Decoder 层数
    n_heads: int = 32          # Q 的注意力头数 H
    n_kv_heads: int = 8        # K/V 的头数 Hkv（GQA：H 个 Q 共享 Hkv 组 KV）
    vocab_size: int = 128256   # 词表大小
    multiple_of: int = 1024    # FFN 隐藏维度对齐到该值的倍数
    ffn_dim_multiplier: float = 1.3
    norm_eps: float = 1e-5     # RMSNorm 的 eps
    rope_theta: float = 500000.0  # RoPE 频率基数 θ
    max_seq_len: int = 2048

    @property
    def head_dim(self) -&gt; int:
        return self.dim // self.n_heads      # hd = D / H

    @property
    def n_rep(self) -&gt; int:
        return self.n_heads // self.n_kv_heads  # 每组 KV 被复制的次数 (kv_group)


# ============================================================
# 1. RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    &#34;&#34;&#34;x / sqrt(mean(x^2) &#43; eps) * weight

    与 LayerNorm 的区别：不减均值、无偏置，只做缩放，更省算力。
    &#34;&#34;&#34;

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习缩放（shift）

    def _norm(self, x: torch.Tensor) -&gt; torch.Tensor:
        # rsqrt = 1/sqrt；在最后一维 D 上求均方
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) &#43; self.eps)

    def forward(self, x: torch.Tensor) -&gt; torch.Tensor:
        # 归一化用 float32 保证数值稳定，再转回原 dtype
        return self._norm(x.float()).type_as(x) * self.weight


# ============================================================
# 2. RoPE 旋转位置编码
# ============================================================
def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float) -&gt; torch.Tensor:
    &#34;&#34;&#34;预计算每个位置、每个频率对应的旋转复数 e^{i·m·θ_k}

    返回 shape: (seq_len, head_dim // 2) 的复数张量。
    &#34;&#34;&#34;
    # θ_k = 1 / theta^(2k/hd)，k = 0,1,...,hd/2-1
    k = torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim
    freqs = 1.0 / (theta ** k)                       # (hd/2,)

    m = torch.arange(seq_len)                        # 位置索引 (T,)
    freqs = torch.outer(m, freqs)                    # m·θ_k -&gt; (T, hd/2)

    # 用模长为 1、角度为 freqs 的极坐标，构造复数 cos&#43;isin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (T, hd/2) complex
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -&gt; tuple[torch.Tensor, torch.Tensor]:
    &#34;&#34;&#34;对 Q、K 施加旋转位置编码。

    xq: (B, T, H,   hd)
    xk: (B, T, Hkv, hd)
    把相邻两维拼成复数 -&gt; 乘以旋转复数 -&gt; 再拆回实数。
    &#34;&#34;&#34;
    # (..., hd) -&gt; (..., hd/2, 2) -&gt; 复数 (..., hd/2)
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (T, hd/2) -&gt; (1, T, 1, hd/2) 以便和 (B,T,H,hd/2) 广播
    freqs_cis = freqs_cis[None, :, None, :]

    # 复数相乘即在极坐标上旋转；再 view_as_real 拆回 (..., hd/2, 2) -&gt; 展平回 (..., hd)
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================
# 3. GQA 注意力
# ============================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -&gt; torch.Tensor:
    &#34;&#34;&#34;把 KV 头复制 n_rep 份，让 Hkv 对齐到 H。

    x: (B, T, Hkv, hd) -&gt; (B, T, Hkv*n_rep, hd)
    &#34;&#34;&#34;
    B, T, Hkv, hd = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]                  # (B, T, Hkv, 1,    hd)
        .expand(B, T, Hkv, n_rep, hd)        # (B, T, Hkv, n_rep,hd)
        .reshape(B, T, Hkv * n_rep, hd)      # (B, T, H,         hd)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = args.n_rep
        self.head_dim = args.head_dim

        # 注意：Q 输出 H*hd，K/V 输出 Hkv*hd（GQA 的关键，KV 更小）
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor | None
    ) -&gt; torch.Tensor:
        B, T, _ = x.shape

        # 线性投影并拆成多头
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # RoPE（只作用在 Q、K 上）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # GQA：把 KV 复制到与 Q 相同的头数
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # (B, T, H, hd) -&gt; (B, H, T, hd)，把头维提前便于做注意力
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力分数 QK^T / sqrt(hd)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores &#43; mask          # 因果掩码（上三角 -inf）

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        out = torch.matmul(scores, xv)      # (B, H, T, hd)

        # 合并多头 -&gt; (B, T, H*hd) -&gt; 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ============================================================
# 4. SwiGLU 前馈网络
# ============================================================
class FeedForward(nn.Module):
    &#34;&#34;&#34;FFN(x) = w2( SiLU(w1 x) * w3 x )

    w1 = gate（门控），w3 = up（升维），w2 = down（降维）。
    &#34;&#34;&#34;

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 计算隐藏维度：先 4*dim*2/3，再乘 multiplier，并向上对齐到 multiple_of
        hidden = int(2 * (4 * args.dim) / 3)
        hidden = int(args.ffn_dim_multiplier * hidden)
        hidden = args.multiple_of * ((hidden &#43; args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(args.dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, args.dim, bias=False)   # down

    def forward(self, x: torch.Tensor) -&gt; torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================
# 5. 一个 Decoder Layer
# ============================================================
class TransformerBlock(nn.Module):
    &#34;&#34;&#34;两段「Norm -&gt; 子层 -&gt; 残差」：

        h = x &#43; Attention(RMSNorm(x))
        out = h &#43; FFN(RMSNorm(h))
    （Pre-Norm 结构）
    &#34;&#34;&#34;

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: torch.Tensor | None
    ) -&gt; torch.Tensor:
        h = x &#43; self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h &#43; self.feed_forward(self.ffn_norm(h))
        return out


# ============================================================
# 6. 完整模型
# ============================================================
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(TransformerBlock(args) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)               # 最后的输出归一化
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)  # 映射回词表

        # 预计算 RoPE 频率，注册为 buffer（不参与训练，但随模型搬到 GPU）
        freqs_cis = precompute_freqs_cis(args.head_dim, args.max_seq_len, args.rope_theta)
        self.register_buffer(&#34;freqs_cis&#34;, freqs_cis, persistent=False)

    def forward(self, tokens: torch.Tensor) -&gt; torch.Tensor:
        &#34;&#34;&#34;tokens: (B, T) 的 token id -&gt; logits: (B, T, vocab_size)&#34;&#34;&#34;
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)            # (B, T, D)

        freqs_cis = self.freqs_cis[:T]             # 取前 T 个位置的旋转复数

        # 构造因果掩码：上三角（不含对角线）为 -inf，禁止看未来
        mask = None
        if T &gt; 1:
            mask = torch.full((T, T), float(&#34;-inf&#34;), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)    # (T, T)，会广播到 (B, H, T, T)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)                           # 输出前最后一次归一化
        logits = self.output(h)                    # (B, T, vocab_size)
        return logits

    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -&gt; torch.Tensor:
        &#34;&#34;&#34;最简单的贪心解码（每步取 argmax）。tokens: (B, T0)&#34;&#34;&#34;
        for _ in range(max_new_tokens):
            tokens_cond = tokens[:, -self.args.max_seq_len:]
            logits = self(tokens_cond)             # (B, T, V)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


# ============================================================
# 7. （可选）加载 Meta 官方权重 consolidated.00.pth
# ============================================================
def load_meta_weights(model: Transformer, ckpt_path: str) -&gt; Transformer:
    &#34;&#34;&#34;把官方 state_dict 的命名映射到本模块。

    官方权重命名本就和这里基本一致（tok_embeddings / layers.{i}.attention.wq ...），
    所以直接 load_state_dict 即可。&#34;&#34;&#34;
    state = torch.load(ckpt_path, map_location=&#34;cpu&#34;)
    model.load_state_dict(state, strict=True)
    return model


# ============================================================
# 8. demo：随机权重跑一遍前向，验证维度
# ============================================================
if __name__ == &#34;__main__&#34;:
    # 用一份缩小的配置，方便在 CPU 上快速验证 shape 是否对齐
    args = ModelArgs(
        dim=256, n_layers=2, n_heads=8, n_kv_heads=2,
        vocab_size=1000, multiple_of=64, ffn_dim_multiplier=1.0, max_seq_len=128,
    )
    model = Transformer(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f&#34;参数量: {n_params/1e6:.2f}M&#34;)

    tokens = torch.randint(0, args.vocab_size, (2, 16))   # (B=2, T=16)
    logits = model(tokens)
    print(&#34;logits shape:&#34;, logits.shape)                  # 期望 (2, 16, 1000)

    out = model.generate(tokens, max_new_tokens=5)
    print(&#34;generate shape:&#34;, out.shape)                   # 期望 (2, 21)
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlllama/  

