# GQA


# GQA（Grouped Query Attention）笔记

## 1. 为什么需要 GQA：KV Cache 的显存瓶颈

自回归生成时，为了避免每生成一个 token 都重新计算前面所有 token 的 K/V，会把历史的 K/V 缓存下来，这就是 **KV Cache**。它用「显存」换「计算」，是推理提速的关键，但显存开销会随序列变长而线性增长。

单个样本 KV Cache 的显存量（字节）约为：

```
2 (K和V)  ×  L (层数)  ×  H (头数)  ×  S (序列长度)  ×  D (head_dim)  ×  dtype字节数
```

可以看到，**头数 H 越多，KV Cache 越大**。在长上下文、大 batch 的推理场景下，KV Cache 经常会超过模型权重本身的显存占用，成为瓶颈。GQA 就是冲着「在尽量不掉点的前提下，缩小 KV Cache」来的。

## 2. MHA / MQA / GQA 三者对比

三者的唯一区别在于 **Key/Value 头的数量**，Query 头数始终是 H：

| 方案 | Query 头数 | KV 头数 | KV Cache 大小 | 效果 |
|------|-----------|---------|--------------|------|
| **MHA**（标准多头） | H | H | 基准（最大） | 质量最好 |
| **MQA**（Multi-Query） | H | 1 | 基准的 1/H | 省最多，但质量下降明显 |
| **GQA**（Grouped-Query） | H | G（1&lt;G&lt;H） | 基准的 G/H | 折中，质量接近 MHA |

一句话：**MQA 是 G=1 的极端，MHA 是 G=H 的极端，GQA 是中间的可调平衡点。**

```
MHA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8        每个 Q 头配一个独立 KV 头
       |  |  |  |  |  |  |  |
      K1 K2 K3 K4 K5 K6 K7 K8

GQA:  Q1 Q2  Q3 Q4  Q5 Q6  Q7 Q8     每 2 个 Q 头共享 1 个 KV 头（H=8,G=4）
       \  /   \  /   \  /   \  /
       K1     K2     K3     K4

MQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8        所有 Q 头共享同一个 KV 头
        \  \  \  | /  /  /  /
              K1
```

## 3. GQA 的核心原理

GQA 把 H 个 Query 头平均分成 **G 组**，**组内所有 Query 头共享同一份 K/V**。

实现上只改两处：

1. **投影维度不同**：Q 投影到 `H * head_dim`，而 K/V 只投影到 `G * head_dim`。
   ```python
   self.q_proj = nn.Linear(dim, n_heads  * head_dim, bias=False)  # H 头
   self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)  # G 头
   self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)  # G 头
   ```
2. **算注意力前把 G 头的 K/V 复制成 H 头**，让维度和 Q 对齐：
   ```python
   n_rep = n_heads // kv_heads
   k = k.repeat_interleave(n_rep, dim=1)  # [B, G, T, D] -&gt; [B, H, T, D]
   v = v.repeat_interleave(n_rep, dim=1)
   ```

其余流程（QKᵀ / √d → mask → softmax → 乘 V → 输出投影）和标准 MHA 完全一样。

## 4. 两个最容易踩坑的实现细节

### 4.1 复制必须用 `repeat_interleave`，不能用 `repeat`

分组要求「相邻的 Query 头对应同一个 KV 头」，所以复制方式必须是**就地重复**：

```
G=4 -&gt; H=8

repeat_interleave:  [1,2,3,4] -&gt; [1,1,2,2,3,3,4,4]   ✅ 正确，组内相邻
repeat:             [1,2,3,4] -&gt; [1,2,3,4,1,2,3,4]   ❌ 错误，分组对应关系乱了
```

### 4.2 KV Cache 缓存的是「复制前」的 G 头 K/V

这是 GQA 省显存的根本所在。一定要**先拼接 cache、保存 cache，再做复制**：

```python
# 先拼历史
if past_key is not None:
    k = torch.cat([past_key, k], dim=2)
# 保存的是 G 头版本（省显存的关键）
past_key_values = (k, v)
# 复制只是为了当前这步算注意力，复制后的 H 头版本【不入 cache】
k = k.repeat_interleave(n_rep, dim=1)
```

如果顺序搞反，把复制后的 H 头 K/V 存进 cache，那 GQA 的省显存优势就完全没了。

## 5. 收益分析

设 `n_rep = H / G`：

- **KV Cache 显存**：缩小为原来的 `G / H`。例如 H=8、G=4 时，KV Cache 减半；G=1（MQA）时缩到 1/H。
- **K/V 投影参数量**：`k_proj`、`v_proj` 的输出维度从 `H*D` 降到 `G*D`，参数也按 `G/H` 缩小（Q 和输出投影不变）。
- **注意力计算量**：因为复制后仍是 H 头在算，主体计算量基本不变，省的是**访存/带宽**和**显存**——而解码阶段恰恰是访存受限的，所以 GQA 在实际推理里也能加速。

&gt; 上面手撕代码的实测（dim=4096, H=8, batch=8, 序列110）：MHA 的 KV Cache 约 27.5 MB，GQA(G=4) 约 13.75 MB，正好减半。

## 6. 实际应用

GQA 已是当前大模型的主流配置，典型如：

- **LLaMA-2 70B / LLaMA-3 系列**：用 GQA（如 64 个 Q 头、8 个 KV 头）。
- **Mistral / Mixtral**、**Qwen2** 等：普遍采用 GQA。

小模型有时仍用 MHA（KV Cache 压力不大），而追求极致推理吞吐时也有用 MQA 的。GQA 因为「省显存 &#43; 几乎不掉点」成了最常见的折中选择。

## 7. 面试速答

- **GQA 解决什么问题？** 缩小 KV Cache 显存（以及 K/V 投影参数），缓解长上下文 / 大 batch 推理的显存与访存瓶颈。
- **和 MHA、MQA 的关系？** 三者只差 KV 头数。MHA：KV 头=Q 头；MQA：KV 头=1；GQA：介于两者之间，是可调的折中。
- **怎么实现共享？** Q 投影 H 头、K/V 只投影 G 头；算注意力前用 `repeat_interleave` 把 K/V 从 G 头扩到 H 头。
- **为什么用 `repeat_interleave` 不用 `repeat`？** 保证组内 Query 头与对应 KV 头相邻一致，分组语义才正确。
- **KV Cache 存哪个版本？** 存复制前的 G 头 K/V，否则省显存的意义就没了。
- **GQA 省了计算量吗？** 主要省显存和访存带宽；复制后注意力主体计算量与 MHA 接近，但在访存受限的解码阶段仍能加速。


## 手撕代码：

```python
&#34;&#34;&#34;
手撕代码：MHA / GQA &#43; KV Cache 的最小实现
========================================

包含三部分：
    1. KV Cache 显存占用的计算工具
    2. MultiHeadAttentionKVCache —— 标准多头注意力 &#43; KV Cache
    3. GQAWithKVCache          —— 分组查询注意力（GQA）&#43; KV Cache

约定的张量维度记号：
    B  = batch_size
    S  = seq_len（当前这一步输入的 token 数）
    T  = 历史总长度（past &#43; 当前）
    H  = n_heads      （Query 头数）
    G  = kv_heads     （Key/Value 头数，GQA 中 G &lt;= H）
    D  = head_dim     （= dim // n_heads）

阶段说明：
    Prefill  阶段：一次性喂入整段 prompt，seq_len = S（&gt; 1）
    Decoding 阶段：自回归逐 token 生成，每步 seq_len = 1
&#34;&#34;&#34;

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. KV Cache 显存占用计算
# ----------------------------------------------------------------------
def calculate_kv_cache_size(past_key_values):
    &#34;&#34;&#34;计算 past_key_values 占用的总显存（单位 MB）。

    past_key_values 是一个 (k, v) 元组，k / v 形状均为 [B, G, T, D]。
    numel()       -&gt; 元素个数
    element_size() -&gt; 每个元素的字节数（fp32=4, fp16/bf16=2）
    &#34;&#34;&#34;
    total_size = 0
    if past_key_values is None:
        return total_size

    for tensor in past_key_values:
        total_size &#43;= tensor.numel() * tensor.element_size()

    return total_size / (1024 * 1024)  # bytes -&gt; MB


# ----------------------------------------------------------------------
# 2. 多头注意力（MHA） &#43; KV Cache
# ----------------------------------------------------------------------
class MultiHeadAttentionKVCache(nn.Module):
    &#34;&#34;&#34;标准多头注意力 &#43; KV Cache 版本。

    每次 forward 把当前 step 算出的 K/V 与历史 past_key/past_value 在
    序列维度（dim=2）上拼接，并返回新的 (k, v) 供下一步复用，
    从而避免对历史 token 重复做投影和注意力计算。
    &#34;&#34;&#34;

    def __init__(self, dim=512, n_heads=8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Wq, Wk, Wv, Wo（一般 attention 内部不带 bias）
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q, k, v, past_key=None, past_value=None, mask=None):
        B, S, _ = q.shape

        # 1) 线性投影
        q = self.q_proj(q)  # [B, S, H*D]
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 2) 切分多头：[B, S, H, D] -&gt; [B, H, S, D]
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # 3) KV Cache：在序列维度拼接历史
        #    past: [B, H, T_past, D]  &#43;  new: [B, H, S, D]  -&gt;  [B, H, T_past&#43;S, D]
        if past_key is not None:
            k = torch.cat([past_key, k], dim=2)
        if past_value is not None:
            v = torch.cat([past_value, v], dim=2)

        # 保存给下一步复用
        past_key_values = (k, v)

        # 4) 注意力分数：[B, H, S, T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 5) Causal Mask（mask 中 True 表示需要被遮蔽的位置）
        #    注意：被遮蔽位置填的是 -inf（不是 1e-9），softmax 后才会变成 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float(&#34;-inf&#34;))

        # 6) softmax &#43; 加权求和
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]

        # 7) 拼回多头：[B, H, S, D] -&gt; [B, S, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.dim)

        # 8) 输出投影
        output = self.o_proj(attn_output)
        return output, past_key_values


# ----------------------------------------------------------------------
# 3. 分组查询注意力（GQA） &#43; KV Cache
# ----------------------------------------------------------------------
class GQAWithKVCache(nn.Module):
    &#34;&#34;&#34;Grouped Query Attention &#43; KV Cache 版本。

    GQA 的核心：Query 仍有 H 个头，但 Key/Value 只有 G 个头（G &lt; H），
    H 个 Query 头被分成 G 组，组内共享同一份 K/V。
    这样 KV Cache 的体积按 G/H 比例缩小（G=1 即退化为 MQA）。
    &#34;&#34;&#34;

    def __init__(self, dim=512, n_heads=8, kv_heads=4):
        super().__init__()
        assert n_heads % kv_heads == 0, &#34;n_heads 必须能被 kv_heads 整除&#34;

        self.dim = dim
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = dim // n_heads

        # Q 投影到 H 个头；K/V 只投影到 G 个头 —— 这是省显存的关键
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, q, k, v, past_key=None, past_value=None, mask=None):
        B, S, _ = q.shape

        # 1) 线性投影
        q = self.q_proj(q)  # [B, S, H*D]
        k = self.k_proj(k)  # [B, S, G*D]
        v = self.v_proj(v)  # [B, S, G*D]

        # 2) 切分多头
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)   # [B, H, S, D]
        k = k.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)  # [B, G, S, D]
        v = v.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)  # [B, G, S, D]

        # 3) KV Cache：拼接历史（注意 cache 存的是 G 头的 K/V，更省）
        if past_key is not None:
            k = torch.cat([past_key, k], dim=2)
        if past_value is not None:
            v = torch.cat([past_value, v], dim=2)

        # 保存的是「复制前」的 G 头 K/V，这正是 GQA 省显存的地方
        past_key_values = (k, v)

        # 4) 把 G 头的 K/V 复制成 H 头，以便和 Q 对齐做注意力
        #    用 repeat_interleave：G=4,H=8 时 [1,2,3,4] -&gt; [1,1,2,2,3,3,4,4]
        #    （不能用 repeat，那样会变成 [1,2,3,4,1,2,3,4]，分组对应关系就错了）
        n_rep = self.n_heads // self.kv_heads
        k = k.repeat_interleave(n_rep, dim=1)  # [B, H, T, D]
        v = v.repeat_interleave(n_rep, dim=1)  # [B, H, T, D]

        # 5) 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 6) Causal Mask（True 处填 -inf）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float(&#34;-inf&#34;))

        # 7) softmax &#43; 加权
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, S, D]

        # 8) 拼回 &#43; 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.dim)
        output = self.o_proj(attn_output)
        return output, past_key_values


# ----------------------------------------------------------------------
# 工具：构造下三角 causal mask
# ----------------------------------------------------------------------
def build_causal_mask(seq_len):
    &#34;&#34;&#34;返回 [1, 1, S, S] 的 bool mask，上三角（不含对角线）为 True，表示遮蔽未来。&#34;&#34;&#34;
    mask = torch.full((1, 1, seq_len, seq_len), True)
    return torch.triu(mask, diagonal=1)


# ----------------------------------------------------------------------
# 演示：Prefill &#43; Decoding 自回归循环
# ----------------------------------------------------------------------
def run_demo(attn, x, mask, n_decode=100, tag=&#34;&#34;):
    # ---- Prefill：一次性处理整段 prompt ----
    output, (past_k, past_v) = attn(x, x, x, mask=mask)
    kv_mem = calculate_kv_cache_size((past_k, past_v))
    print(f&#34;[{tag}] prefill -&gt; output: {tuple(output.shape)}, &#34;
          f&#34;KV: {tuple(past_k.shape)}, kv_mem: {kv_mem:.2f} MB&#34;)

    # ---- Decoding：逐 token 生成，每步只送 1 个新 token ----
    begin = time.time()
    for _ in range(n_decode):
        new_x = output[:, [-1], :]  # [B, 1, dim]
        output, (past_k, past_v) = attn(
            new_x, new_x, new_x, past_key=past_k, past_value=past_v
        )
    cost = time.time() - begin

    kv_mem = calculate_kv_cache_size((past_k, past_v))
    print(f&#34;[{tag}] after {n_decode} steps -&gt; KV: {tuple(past_k.shape)}, &#34;
          f&#34;kv_mem: {kv_mem:.2f} MB, cost: {cost:.3f}s\n&#34;)


if __name__ == &#34;__main__&#34;:
    print(&#34;torch version:&#34;, torch.__version__, &#34;\n&#34;)

    # 公共超参（dim 较大是为了让 KV Cache 显存差异更明显）
    batch, seq_len, dim, heads = 8, 10, 4096, 8
    x = torch.randn(batch, seq_len, dim)
    mask = build_causal_mask(seq_len)

    # ---- Test Case 1：MHA &#43; KV Cache ----
    mha = MultiHeadAttentionKVCache(dim=dim, n_heads=heads)
    run_demo(mha, x, mask, n_decode=100, tag=&#34;MHA&#34;)

    # ---- Test Case 2：GQA &#43; KV Cache（kv_heads=4，KV Cache 减半）----
    gqa = GQAWithKVCache(dim=dim, n_heads=heads, kv_heads=4)
    run_demo(gqa, x, mask, n_decode=100, tag=&#34;GQA&#34;)
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlgqa/  

