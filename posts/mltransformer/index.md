# Transformer


## 核心论文：Attention Is All You Need

&lt;center&gt;
	&lt;embed src=&#34;/pdf/Attention Is All You Need.pdf&#34; width=&#34;1050&#34; height=&#34;1000&#34;&gt;
&lt;/center&gt;

## 结构

&lt;center&gt;
  &lt;img width=&#34;500&#34; alt=&#34;image&#34; src=&#34;/image/ML/transformer1.png&#34;&gt;
&lt;/center&gt;

## 输入

### embedding

输入是由一句话包含若干个单词组成，我们会将单词进行 embedding，也就是转成向量的形式，更方便处理。

每一个单词的向量表示会有很多维度，论文里的维度为 $d_m = 512$

若某个序列长度小于最长的序列长度，则用 Padding Mask 填充，也就是使用 `&lt;pad&gt;` 标记填充，便于序列对齐。

在训练时，输入包含原句子和翻译后的句子，分别输入到 Inputs 和 Outputs，其中 Outputs 需要右移一位，在句子开始加一个标志 `&lt;begin&gt;`，用来后续方便处理掩码。

![](/image/ML/word2vec.png)

![](/image/ML/embedding.png)

### Positional Encoding

在 embedding 之后，需要对句子中每个词进行位置编码，因为 Transformer 不像 RNN 那样可以知道单词的时序信息，输入时是一整个句子作为输入。

位置编码公式如下：

$$
\begin{aligned}
\text{PE}(\text{pos}, 2i) &amp;= \sin\left(\text{pos}\times{10000^{-\frac{2i}{d_m}}}\right) \\\\
\text{PE}(\text{pos}, 2i &#43; 1) &amp;= \cos\left(\text{pos}\times{10000^{-\frac{2i}{d_m}}}\right)
\end{aligned}
$$

其中，pos 表示单词在句子中的位置，$d_m$ 表示 embedding 的维度，$2i$ 表示该位置单词的偶数维度，$2i &#43; 1$ 表示该位置单词的奇数维度。

得到 PE 之后**直接与词向量进行相加**，就完成了输入处理部分。

## 注意力机制

### Self-Attention

#### 结构：

![](/image/ML/Attention.png)

#### 公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

可以理解为 $Q$ 是查询，$K$ 是键值，$V$ 是值。

首先，需要将输入的每个词向量分别乘上 $W_Q, W_K, W_V$ 矩阵才变成 $Q, K, V$，其中 $W_Q, W_K, W_V$ 是学习参数。

![](/image/ML/wq.png)

然后将 $Q$ 和 $K^\top$ 做矩阵乘法，得到一个分数矩阵，每一列相当于该单词与各个单词分别做内积，值越大，相关度越高。

![](/image/ML/qk.png)

随后再将每个值除以 $\sqrt{d_k}$，这个是矩阵 $Q$ 的维度，因为避免数值过大影响梯度。

之后对每一列进行 softmax，这样做的理由是使得值归一化，

![](/image/ML/softmaxqk.png)

最终，再乘上 $V$，得到 attention

### Multi-Head Attention

#### 结构：

![](/image/ML/Multi-HeadAttention.jpg)

Multi-Head Attention 是由多个 Self-Attention 组合而成，这也叫多头注意力机制。

#### 公式：

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &amp;= \text{Concat}(\text{head}_1, \cdots, \text{head}_h) W ^ O \\\\
\text{where } \text{head}_i &amp;= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

其中参数 $W_i^Q, W_i^K \in \mathbb{R}^{d_m \times d_k}, W_i^V \in \mathbb{R}^{d_m \times d_v}, W ^ O \in \mathbb{R}^{h d_v \times d_m}$，$h$ 代表头的个数。

论文里取 $h = 8$，将输入分到 $8$ 个头中，由原来的 $512$ 维变成 $512 / 8 = 64$ 维（注意要保证维度被头数整除）并行执行，最后再拼接起来，进行线性变换变回 $512$ 维。

## Encoder

### 结构：

![](/image/ML/encoder.png)

这一部分是 Encoder 的结构，可以看到是由 Multi-Head Attention, Add &amp; Norm, Feed Forward, Add &amp; Norm 组成的。其中 $N \times$ 的意思是有 $N$ 个重复的 Encoder 块堆叠起来。

### FeedForward

这是一个前馈神经网络，其中包含两层全连接层，第一层的激活函数是 Relu，第二层不使用激活函数。

公式：

$$
\text{FFN}(x) = \max(0,xW_1 &#43; b_1)W_2 &#43; b_2
$$

### Add &amp; Norm

Add 是指残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到，用于缓解梯度消失。

![](/image/ML/res.png)

Norm 指 Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛。

这两层公式如下：

$$
\begin{aligned}
\text{LayerNorm}&amp;(X &#43; \text{MultiHead}(X)) \\\\
\text{LayerNorm}&amp;(X &#43; \text{FFN}(X)) \\\\
\end{aligned}
$$

最后，encoder 层的输出会当作 $V, K$ 输入到 Cross-Attention

## Decoder

### 结构：

![](/image/ML/decoder.png)

这里的 $N \times$ 同样是 $N$ 个重复的 Decoder 层堆叠起来。

### Masked Multi-Head Attention

加入 masked 的意义是当生成一个序列时，进行到一个位置时，不能让模型看到未来的单词，这样会影响训练，所以要在注意力机制里引入 masked，用来屏蔽未来信息。

为了让每一步后面的词不影响前面的词，我们在 softmax 之前，引入一个 masked 矩阵，将上三角矩阵全部变为 $-\infty$

例如这是一个 $4 \times 4$ 的矩阵:

$$
\text{masked} = \begin{bmatrix}
0&amp;-\infty&amp;-\infty&amp;-\infty\\\\
0&amp;0&amp;-\infty&amp;-\infty\\\\
0&amp;0&amp;0&amp;-\infty\\\\
0&amp;0&amp;0&amp;0
\end{bmatrix}
$$

于是我们将 masked 矩阵直接与 $QK^\top$ 相加，由于 $-\infty$ 的部分加任何数都是本身，所以 softmax 会使它趋于 $0$

### Cross-Attention

在第二个 Multi-Head Attention 中，将 Masked Multi-Head Attention 的输入当作此时的 $Q$，$K, V$ 则是 encoder 部分的输出。

这样做的意义是因为这是两个不同的序列，它可以让 $Q$ 的每一个元素都能注意到 $K, V$ 序列的所有元素，从而学习到两个序列之间的关系。

交叉注意力就像是一个学生 ($Q$) 在听老师讲课 ($K, V$ 序列)，学生的问题和理解 ($Q$) 去匹配老师的知识点 ($K$)，并吸收老师的具体讲解内容 ($V$)，从而丰富和更新学生的知识。

### Softmax

最后，使用 softmax 输出下一个单词，然后进行自回归。

## Transformer 常见问题

### 1. 为什么自注意力机制在计算时需要对点积结果进行 $\sqrt{d_k}$ 的缩放？

进行 $\sqrt{d_k}$ 的缩放是为了防止点积结果在维度 $d_k$ 较大时过大，这会导致 softmax 函数处于饱和区，使得梯度变得非常小，难以通过反向传播有效地训练。缩放有助于维持点积的稳定性，确保梯度在一个合适的范围内。

### 2. Transformer 的前馈网络 (FFN) 在模型中承担什么角色？

FFN为模型提供了非线性处理能力，它在每个位置上独立地作用于其输入，有助于增加模型的复杂度和表达能力。

### 3. 为何 Transformer 模型中采用 Layer Normalization 而非 Batch Normalization？

Layer Normalization 对每个样本独立进行归一化，适用于序列化数据和变长输入，而 Batch Normalization 在批处理时对特征进行归一化，不适用于序列长度变化的情况。 

### 4. Transformer 模型中的“残差连接”如何有助于缓解梯度消失问题？

残差连接通过直接将输入加到子层的输出上，使得深层网络中的信号能够直接传递到较浅层，有助于缓解梯度消失问题。

### 5. 在 Transformer 模型中，为什么要使用 Dropout？

Dropout 是一种正则化技术，通过随机丢弃一部分网络连接，可以有效减少模型的过拟合，增强模型的泛化能力。

### 6. 如何理解 Transformer 中的自回归属性？

在 Transformer 的解码器中，自回归属性指模型在生成每个输出时，只能依赖于先前生成的输出，确保在生成序列时的顺序性和一致性。

### 7. 为什么 Transformer 模型能够有效处理长距离依赖问题？ 

由于自注意力机制能够直接计算序列中任意两点之间的依赖关系，Transformer 模型能够有效捕捉长距离依赖，避免了传统序列模型（如 RNN）中信息传递路径长导致的信息丢失。

### 8. 位置编码为什么使用正弦和余弦函数？

正弦和余弦函数被用于位置编码因为它们具有周期性，这使得模型能够更容易地学习和推理关于序列长度和元素位置的信息。此外，它们可以帮助模型捕捉到相对位置信息，因为正弦和余弦函数的值可以通过叠加和差分运算来编码元素间的相对距离。

### 9. Transformer 中为什么需要线性变换？

- 线性变换的好处：在 $QK^\top$ 部分，线性变换矩阵将 $Q, K$ 投影到了不同的空间，增加了表达能力（这一原理可以同理 SVM 中的核函数-将向量映射到高维空间以解决非线性问题），这样计算得到的注意力矩阵的泛化能力更高。

- 不用线性变换的坏处：在 $QK^\top$ 部分，如果不做线性变换，即 $X=Q=K$，则会导致注意力矩阵是对称的，即这样的效果明显是差的，比如“我是一个女孩”这句话，女孩对修饰我的重要性应该要高于我修饰女孩的重要性。

### 10. Transformer 为何使用多头注意力机制？

- 并行计算：多头注意力机制允许模型同时关注输入序列的不同部分，每个注意力头可以独立计算，从而实现更高效的并行计算。这样能够加快模型的训练速度。

- 提升表征能力：通过引入多个注意力头，模型可以学习到不同类型的注意力权重，从而捕捉输入序列中不同层次、不同方面的语义信息。这有助于提升模型对输入序列的表征能力。

- 降低过拟合风险：多头注意力机制使得模型可以综合不同角度的信息，从而提高泛化能力，降低过拟合的风险。

- 增强模型解释性：每个注意力头可以关注输入序列的不同部分，因此可以更好地理解模型对于不同输入信息的关注程度，使得模型的决策更具解释性。

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mltransformer/  

