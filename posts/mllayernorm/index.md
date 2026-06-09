# LayerNorm


## 深度学习核心组件：LayerNorm (层归一化) 学习笔记

### 一、 核心概念：什么是 LayerNorm？

在深度学习中，尤其是处理文本和序列数据（如 Transformer 模型）时，数据分布的剧烈波动会导致模型难以训练。**LayerNorm 的核心作用是“抹平样本内部的特征偏差”，把各种千奇百怪的输入数据，强制拉回到一个稳定、统一的分布标准上。**

**💡 直观比喻：**
假设有 32 个班级考 256 门科目。LayerNorm 不管其他班级考得如何，它只盯着**当前这一个学生**的 256 门成绩。它计算这名学生自己的平均分和成绩波动程度，然后把该学生的成绩标准化，消除偏科带来的绝对数值差异，只保留相对特征。

---

### 二、 底层数学逻辑：Z-score 标准化

LayerNorm 的前置核心操作是标准化，其底层公式为：

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

#### 1. 平移操作：减去均值 ($x - \mu$)

* **含义**：寻找数据的“重心”。
* **作用**：把整个坐标系的原点移动到该组数据的平均值位置。消除数据原本自带的绝对数值大小，转化为“高于平均”或“低于平均”的相对关系。

#### 2. 缩放操作：除以标准差 ($\dots \div \sigma$)

* **什么是标准差 ($\sigma$)**：衡量数据的“离散程度”或“波动大小”。它定义了在这组数据中，偏离均值多少才算“正常”。
* **为什么要除以它**：为了**消除量纲（统一评价标准）**。如果仅仅减去均值，波动大的数据依然会产生极大的值。除以标准差后，所有特征的单位都被抹掉，变成了一个全新的通用单位——**“几个标准差”**。
* **最终效果**：无论原始特征的数值是几千还是零点几，处理后都会变成**均值为 0，标准差为 1** 的标准分布，防止大数值特征在梯度下降时“喧宾夺主”。

---

### 三、 LayerNorm 的完整公式与代码映射

标准化的数据过于理想，可能会破坏网络本该学习到的非线性特征表达。因此，LayerNorm 引入了可学习参数。

**完整公式：**


$$y = \frac{x - \mu}{\sigma &#43; \epsilon} \odot \gamma &#43; \beta$$

结合你的手写 PyTorch 代码，它们是一一对应的：

* **计算均值 $\mu$ 与标准差 $\sigma$**：
```python
mean = x.mean(-1, keepdim=True) 
std = x.std(-1, keepdim=True) 

```


&gt; **注**：`-1` 代表在最后一个维度（即特征维度 `dim=256`）上进行计算，这是 LayerNorm 的灵魂所在。


* **防止分母为零的微小常数 $\epsilon$**：
```python
self.eps = eps # 默认 1e-6

```


&gt; **注**：纯粹的工程技巧，防止程序因 $\sigma=0$ 而崩溃（Division by zero）。


* **可学习的缩放平移参数 $\gamma$ 与 $\beta$**：
```python
self.gamma = nn.Parameter(torch.ones(dim))
self.beta = nn.Parameter(torch.zeros(dim))

```


&gt; **注**：这是模型的“后悔药”。$\gamma$ 和 $\beta$ 在训练中不断更新，如果模型发现完全归一化会损失重要特征，它可以自行学习，将分布缩放或偏移回更利于分类/生成的最佳状态。


* **最终计算**：
```python
return self.gamma * (x - mean) / (std &#43; self.eps) &#43; self.beta

```



---

### 四、 进阶考点：为什么 NLP (Transformer) 偏爱 LayerNorm 而非 BatchNorm？

| 归一化方式 | 计算维度 | 核心优劣 | 适用场景 |
| --- | --- | --- | --- |
| **BatchNorm (BN)** | 跨样本 (Batch) 计算同一个特征的均值/方差。 | 依赖 Batch Size 大小。在 NLP 中，句子长度不一（需 Padding 补零），大量的 0 会严重干扰均值和方差的统计，导致失真。 | 计算机视觉 (CV)、CNN。 |
| **LayerNorm (LN)** | 独立样本内计算所有特征的均值/方差。 | **不受 Batch Size 和序列长度 (Seq Len) 的影响**。针对每个词向量内部进行归一化，在处理变长序列时极其稳定。 | 自然语言处理 (NLP)、RNN、Transformer。 |

---

### 五、 核心总结

1. **目的**：通过统一特征尺度，加速网络收敛，防止梯度消失或爆炸。
2. **本质**：减去均值定中心，除以标准差统尺度，最后乘加 $\gamma$ 和 $\beta$ 保留网络表达能力。
3. **定位**：大模型（如 GPT、BERT）中不可或缺的基石组件，通常放置在多头注意力机制（Multi-Head Attention）和前馈神经网络（FFN）的前后。

```python
import torch
import torch.nn as nn

class CustomLayerNorm(nn.Module):
    &#34;&#34;&#34;
    手写 LayerNorm：针对输入张量的最后一个维度（特征维度）进行标准化
    底层公式: y = [ (x - mean) / (std &#43; eps) ] * gamma &#43; beta
    &#34;&#34;&#34;
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        
        # dim: 特征维度的大小 (例如词向量的维度 256)
        self.dim = dim
        self.eps = eps
        
        # 1. 定义可学习参数 gamma (缩放比例) 和 beta (平移偏移量)
        # 初始状态下，gamma 全为 1，beta 全为 0，即初始不改变标准化后的分布
        # 使用 nn.Parameter 包装，使其能够被优化器捕捉并随梯度更新
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 假设输入 x 的 shape 为: [batch_size, seq_len, dim]
        # 例如: [32, 10, 256] -&gt; 32个句子，每个句子10个词，每个词是256维的向量

        # 2. 计算均值 (mu)
        # 在最后一个维度 (dim=-1) 上求均值
        # keepdim=True 保持维度为 [32, 10, 1]，利用广播机制方便后续与原始张量相减
        mean = x.mean(dim=-1, keepdim=True) 

        # 3. 计算标准差 (sigma)
        # 注意：这里使用无偏估计计算标准差 (PyTorch 默认，或者可以通过 ((x-mean)**2).mean计算方差后开根号)
        std = x.std(dim=-1, keepdim=True) 

        # 4. 执行标准化与参数缩放平移
        # (x - mean): 平移，寻找数据重心
        # / (std &#43; eps): 缩放，消除量纲，eps防止除以0崩溃
        # * gamma &#43; beta: 赋予模型“后悔药”，保留学习非线性特征的能力
        norm_x = (x - mean) / (std &#43; self.eps)
        output = self.gamma * norm_x &#43; self.beta
        
        return output

# ================= 验证代码 =================
if __name__ == &#34;__main__&#34;:
    # 模拟超参数
    batch_size = 32
    seq_len = 10
    hidden_dim = 256

    # 构造随机正态分布的输入数据
    features = torch.randn(batch_size, seq_len, hidden_dim)

    # 实例化我们手写的 LayerNorm
    ln = CustomLayerNorm(dim=hidden_dim)
    
    # 前向传播
    features_output = ln(features)

    print(&#34;=== Shape 检查 ===&#34;)
    print(f&#34;输入 Shape:  {features.shape}&#34;)
    print(f&#34;输出 Shape:  {features_output.shape}\n&#34;)

    print(&#34;=== 分布检查 (查看第一个 batch 的第一个 token) ===&#34;)
    # 原始数据的均值和方差通常是随机波动的
    print(f&#34;标准化前 -&gt; 均值: {features.mean(-1)[0, 0].item():.6f}, 标准差: {features.std(-1)[0, 0].item():.6f}&#34;)
    
    # 经过 LayerNorm 后，均值应极度接近 0，标准差应极度接近 1
    print(f&#34;标准化后 -&gt; 均值: {features_output.mean(-1)[0, 0].item():.6f}, 标准差: {features_output.std(-1)[0, 0].item():.6f}&#34;)
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mllayernorm/  

