# 感知机


## 感知机

### 模型

设输入的特征向量为 $X \subseteq \textbf{R}^n$，每个向量对应输出 $Y = \{1, -1\}$，也就是两种分类。

设函数 $f(x) = \text{sign}(w \cdot x &#43; b)$ 称为感知机，其中 $w \in \textbf{R}^n$ 表示一组权值，$b \in \textbf{R}$ 叫做偏置，$\cdot$ 表示向量点乘 (内积)。$\text{sign}$ 是符号函数如下：
$$
\text{sign}(x) = \begin{cases}
1,&amp;x \ge 0 \\\\
-1, &amp;x &lt; 0
\end{cases}
$$
通过感知机的学习，得到方程 $w \cdot x &#43; b = 0$，表示一个 $\textbf{R}^n$ 空间的超平面，将特征向量划分为两类。

### 学习策略

#### 数据集的线性可分性

对所有正类的数据集都有 $w \cdot x &#43; b &gt; 0$，负类数据集 $w \cdot x &#43; b &lt; 0$

![1](/image/ML/4.png)

如上图，考虑两个特征 $x_1,x_2$，类型用 x 和 o 来表示，x 类型都在直线的下方，带入直线方程会发现均小于 $0$，那么 $\text{sign}$ 值就为 $-1$

#### 损失函数

首先特征向量 $x_0$ 到超平面的距离为 $\dfrac{|wx_0 &#43; b|}{||w||}$，其中 $||w||$ 是 $L_2$ 范数，即 $\sqrt{\sum\limits_{i = 1} ^ {n}w_i^2}$

我们只考虑错误分类的点到超平面的距离，对于误分类的点 $(x_i,y_i)$，$-y_i(w \cdot x &#43; b) &gt; 0$ 成立，因为假设该点为正类 ($y_i = 1$)，由于误分类那么他会在直线下方，导致 $w \cdot x &#43; b  &lt; 0$，所以 $-1 \times (w \cdot x &#43; b) &gt; 0$，负类则相同。

此时误分类点到超平面的距离就为  $-\dfrac{y_i(wx_i &#43; b)}{||w||}$，设误分类点的集合为 $M$，那么总距离就为
$$
-\frac{1}{||w||}\sum_{x_i \in M} y_i (w \cdot x_i &#43; b)
$$
不考虑 $\dfrac{1}{||w||}$，就得到了损失函数 $L(w, b) = -\sum\limits_{x_i \in M} y_i (w \cdot x_i &#43; b)$，此函数连续可导。

我们考虑最小化损失函数。首先任选一个超平面 $w_0,b_0$，然后随机选择**一个**误分类点梯度下降。
$$
\frac{\partial L(w, b)}{\partial w} = -\sum_{x_i \in M}y_ix_i \\
\frac{\partial L(w, b)}{\partial b} = -\sum_{x_i \in M}y_i
$$
随机选择误分类点 $(x_i,y_i)$ 对 $w, b$ 进行更新：$w &#43; \alpha y_ix_i, b &#43; \alpha y_i$，最后直到损失函数为 $0$ 为止。

#### 对偶形式

由于每次对随机一个点进行梯度下降，那么我们从结果考虑，假设第 $i$ 个点更新的次数为 $k_i$ 次，那么最终的 $w, b$ 就为 $w = \sum\limits_{i = 1} ^ {n}\alpha k_i y_ix_i, b = \sum\limits_{i = 1} ^ {n}\alpha k_i y_i$，那么感知机模型就为 $\text{sign}\left(\sum\limits_{j = 1} ^ {n}\alpha k_j y_j x_j \cdot x &#43;b \right)$

训练的时候对于某个点 $(x_i,y_i)$ 如果 $y_i \left(\sum\limits_{j = 1} ^ {n}\alpha k_j y_j x_j \cdot x_i &#43;b \right) \le 0$，就：$k_i &#43; 1, b &#43; \alpha y_i$，直到没有误分类数据。

由于数据大量出现 $x_i \cdot x_j$，为了方便可以先算出 Gram 矩阵 $\textbf{G} = [x_i\cdot x_j]_{n \times n}$，即 $x$ 向量组自己和自己做 $n \times n$ 的矩阵乘法。

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E6%84%9F%E7%9F%A5%E6%9C%BA/  

