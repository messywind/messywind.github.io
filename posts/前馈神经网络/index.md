# 前馈神经网络


## 模型
模型由多层感知机构成，每层有若干个神经元，第一层是输入层，最后一层是输出层。如下图所示

![1](/image/ML/5.jpg)



其中输入层表示输入数据的若干个特征 记为 $x_1, x_2, \cdots, x_n$，输出层表示分类的概率。

### 神经元

![1](/image/ML/6.jpg)

每个节点就是一个神经元，从图中不难发现上一层的每个神经元都有一条边连向该神经元，其中边有边权 $w$

定义 $w ^ l_{jk}$ 为第 $l-1$ 层的第 $k$ 个神经元和第 $l$ 层的第 $j$ 个神经元连接对应的权重 ($l - 1$ 层表示他左边的层)

定义 $z^l_j$ 为第 $l$ 层第 $j$ 个神经元的未激活值(加权计算值)。

定义 $a^l_j$ 为第 $l$ 层第 $j$ 个神经元的激活值。

定义 $b^l_j$ 为第 $l$ 层第 $j$ 个神经元带有的偏置。

其中

$$
z_j^l = \sum\limits_k w_{jk} ^ l a_k ^ {l - 1} &#43; b_j ^ l \\\\
a_j^l=\sigma(z_j^l)
$$

$\sigma(x)$ 代表 sigmoid 函数，$k$ 下标求和范围为 $l$ 层神经元个数。

直观来讲就是该神经元的值是上一层每个神经元的值乘以到他的边权，累加起来再加上一个该神经元带的偏置，最后套一个 sigmoid 函数。

那么为什么要套 sigmoid？因为线性组合如果叠加在一起，那么始终可以用一个式子来表达，就用不着多层网络了，所以要引入一个非线性的复合，并且，sigmoid 可以让数据归一化，始终在 $(0, 1)$ 之间。当然 ReLU 也是可以的。

### 损失函数

定义 $y_j$ 表示输出层第 $j$ 个输出的真实值。

考虑用 MSE (均方误差) 来作为损失函数 $C$
$$
C = \frac{1}{2}\sum_{k}(y_k - a_k ^ l) ^ 2
$$

### 反向传播

现在考虑最小化损失函数，一般来说都是梯度下降，但是我们需要调整整个网络中的 $w, b$ 来使损失函数最小化，这样不好计算。

我们知道，训练过程就是要让模型的预测 $a^l_j$ 越接近真实值 $y_j$ 越好。那么 $a^l_j$ 跟什么有关呢？根据样本的真实值 $y_j$，可以计算 $a^l_j$ 和 $y_j$ 的误差，计算出误差之后，我们肯定知道要增大还是减小 $a^l_j$ 才能让模型的预测更好。而我们能够改变的量只有：第 $l - 1$ 层和第 $l$ 层之间的每个权重 $w^l_{jk}$，偏置项 $b_j^l$，或者是第 $l-1$ 层的激活函数的输出值 $a^{l-1}_k$，但是 $a_k^{l-1}$ 并无法直接改变，它是由更前面的权重和偏置项的值决定的。当我们从后往前根据预测的误差，考虑要如何修改每一层的权重和偏置项的时候，就是在做反向传播。

上述过程解释了 $a_j^l$ 想要如何调整模型的权重和偏置项。当然，我们还需要考虑输出层中除了$a_j^l$ 以外的神经元的“意见”。他们各自对如何改变模型的权重和偏置项的“意见”并不一定相同。最后，我们需要考虑输出层中所有神经元的意见来更新模型的权重和偏差。

#### 四个基本公式

我们希望能得到任意的 $\dfrac{\partial C}{\partial w_{jk}}, \dfrac{\partial C}{\partial b^l_j}$

在此之前，我们先定义几个东西。

##### 公式 1

定义 $\delta^l_j \equiv \dfrac{\partial C}{\partial z^l_j}$
那么有

$$
\begin{aligned}
\delta_j^l &amp;= \frac{\partial C}{\partial z_j^l} \\\\
&amp;= \frac{\partial C}{\partial a_j^l} \frac{\partial a_j^l}{\partial z_j^l} \\\\
&amp;= \frac{\partial C}{\partial a_j^l} \frac{\partial \sigma(z_j^l)}{\partial z_j^l}  \\\\
&amp;= \frac{\partial C}{\partial a_j^l}\sigma \&#39; (z_j^l)
\end{aligned}
$$

然后 $\dfrac{\partial C}{\partial a^l_j} = (y_j - a^l_j)$，所以 $\delta_j^l = (y_j - a^l_j)\sigma \&#39; (z_j^l)$

那么 $l$ 层的 $\delta$ 用矩阵表示：

$$
\delta ^ l = \begin{bmatrix}
\frac{\partial C}{\partial a_1^l}\sigma \&#39; (z_1^l) \\\\
\frac{\partial C}{\partial a_2^l}\sigma \&#39; (z_2^l) \\\\
\vdots \\\\
\frac{\partial C}{\partial a_k^l}\sigma \&#39; (z_k^l) \\\\
\end{bmatrix} = \begin{bmatrix}
\frac{\partial C}{\partial a_1^l} \\\\
\frac{\partial C}{\partial a_2^l} \\\\
\vdots \\\\
\frac{\partial C}{\partial a_k^l} \\\\
\end{bmatrix} \odot \begin{bmatrix}
\sigma \&#39; (z_1^l) \\\\
\sigma \&#39; (z_2^l) \\\\
\vdots \\\\
\sigma \&#39; (z_k^l) \\\\
\end{bmatrix} = \nabla C_{a^l} \odot \sigma \&#39; (z^l)
$$

其中 $\odot$ 表示按元素 (element-wise) 乘。

##### 公式 2

继续对于 $l - 1$ 层进行推导。

$$
\begin{aligned}
\delta_j^{l - 1} &amp;= \frac{\partial C}{\partial z_j^{l - 1}} \\\\
&amp;= \frac{\partial C}{\partial a_j^{l - 1}} \frac{\partial a_j^{l - 1}}{\partial z_j^{l - 1}} \\\\
&amp;= \frac{\partial C}{\partial a_j^{l - 1}} \frac{\partial \sigma(z_j^{l - 1})}{\partial z_j^{l - 1}}  \\\\
&amp;= \frac{\partial C}{\partial a_j^{l - 1}} \sigma \&#39; (z_j^{l - 1})
\end{aligned}
$$

发现此时 $\dfrac{\partial C}{\partial a_j^{l - 1}}$ 不好求了，但我们知道 $a_j^{l - 1}$ 和 $z_j^l$ 都有关。

拿 $a^{l - 1}_1$ 举例子，如下图

![1](/image/ML/7.jpg)

所以我们可以先把 $C$ 看成 $z^l_1, z^l_2, \cdots, z^l_k$ 的复合函数。

$$
\begin{aligned}
\frac{\partial C}{\partial a_1^{l - 1}} &amp;= \frac{\partial C(z^l_1, z^l_2, \cdots, z^l_k)}{\partial a_1^{l - 1}} \\\\
&amp;= \sum_k \frac{\partial C}{\partial z^l_k} \frac{\partial z^l_k}{\partial a_1^{l - 1}}
\end{aligned}
$$

由于 $z_j^l = \sum\limits_k w_{jk} ^ l a_k ^ {l - 1} &#43; b_j ^ l$，所以 $\dfrac{\partial z^l_k}{\partial a_1^{l - 1}} = w_{k1} ^ l$

又根据公式 1 的定义 $\delta^l_j \equiv \dfrac{\partial C}{\partial z^l_j}$，将两个式子带入上述求和公式得：

$$
\sum_k \frac{\partial C}{\partial z^l_k} \frac{\partial z^l_k}{\partial a_1^{l - 1}} = \sum_k w^l_{k1}\delta_k^l
$$

同理，对于其他的 $a^{l - 1}_{j}$，都有：

$$
\frac{\partial C}{\partial a_j^{l - 1}} = \sum_k w^l_{kj}\delta_k^l
$$

所以说，$\delta_j^{l - 1} = \dfrac{\partial C}{\partial a_j^{l - 1}} \sigma \&#39; (z_j^{l - 1}) = \left(\sum\limits_k w^l_{kj}\delta_k^l \right) \times \sigma \&#39; (z_j^{l - 1})$

推广到矩阵形式，即对 $l$ 层所有 $\delta ^ l$ (层数下标先平移一层)：

$$
\delta ^ l = 
\begin{bmatrix}
\left(\sum\limits_k w^{l &#43; 1}\_{k1} \delta^{l &#43; 1}_k \right) \times \sigma \&#39; (z_1^l) \\\\
\left(\sum\limits_k w^{l &#43; 1}\_{k2} \delta^{l &#43; 1}_k \right) \times \sigma \&#39; (z_2^l) \\\\
\vdots \\\\
\left(\sum\limits_k w^{l &#43; 1}\_{kn} \delta^{l &#43; 1}_k \right) \times \sigma \&#39; (z_n^l) \\\\
\end{bmatrix} 
= ((w^{l&#43;1})^\top \delta^{l&#43;1})\odot \sigma \&#39; (z^l)
$$

##### 公式 3

好了现在终于可以解决 $w, b$ 的偏导问题了

$$
\begin{aligned}
\frac{\partial C}{\partial b^l_j} &amp;= \frac{\partial C}{\partial z^l_j} \frac{\partial z^l_j}{\partial b^l_j} \\\\
&amp;= \frac{\partial C}{\partial z^l_j} \\\\
&amp;= \delta^l_j
\end{aligned}
$$

由于 $z_j^l = \sum\limits_k w_{jk} ^ l a_k ^ {l - 1} &#43; b_j ^ l$，所以 $\dfrac{\partial z^l_j}{\partial b^l_j} = 1$

##### 公式 4

$$
\begin{aligned}
\frac{\partial C}{\partial w_{jk}^l}&amp;=\frac{\partial C}{\partial z^l_j}\frac{\partial z^l_j}{\partial w_{jk}^l} \\\\
&amp;=\frac{\partial C}{\partial z^l_j} a_k^{l-1} \\\\
&amp;=\delta_j^l a_k^{l-1}
\end{aligned}
$$

由于 $z_j^l = \sum\limits_k w_{jk} ^ l a_k ^ {l - 1} &#43; b_j ^ l$，所以 $\dfrac{\partial z^l_j}{\partial w^l_{jk}} = a^{l - 1}_k$

#### 过程

至此我们基于 4 个公式，给出反向传播的流程：

首先前向传播，计算每个 $z_j^l$，$a_j^l$，根据公式 1 计算输出层的 $\delta^l$

然后从后往前：

1. 根据公式 2 计算每一层的梯度向量 $\delta^l$，注意我们总是可以根据 $\delta^{l&#43;1}$ 计算出 $\delta^l$
2. 根据公式 3 可以计算出每个偏置项 $b^l_j$ 的梯度 $\delta^l_j$
3. 根据公式 4 可以计算出每个权重 $w^l_{jk}$ 的梯度 $\delta_j^la_k^{l-1}$

上面这个过程也回答了为什么反向传播是一个高效的算法这个问题:

1. 根据公式 2，计算第 $l$ 层的梯度向量 $\delta^l$ 的时候 $\delta^{l&#43;1}$ 已经算好了，不用从头从输出层开始推导。
2. 根据公式 3 和公式 4，直接算出了损失函数对当前层权重和偏置项的梯度，而不是其他什么中间的梯度结果。

## 参考

https://martinlwx.github.io/zh-cn/backpropagation-tutorial/

https://zhuanlan.zhihu.com/p/683499770

https://playground.tensorflow.org/

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/  

