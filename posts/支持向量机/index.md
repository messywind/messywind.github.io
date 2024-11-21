# 支持向量机 (SVM)

## 线性可分二分类模型

在二维空间上，两类点被一条直线完全分开叫做线性可分。

还是拿之前的数据集举例，$(x_1, y_1), (x_2,y_2), \cdots (x_n, y_n)$，$x_i \in \textbf{R} ^ d$，$d$ 为特征向量维度。

模型的目标是找到一条直线 $w x &#43; b = 0$ 对于每个 $x_i$ 满足

$$
\text{sign}(w^\top x_i &#43; b) = \begin{cases}
1, y_i = 1 \\\\
-1, y_i = -1
\end{cases}
$$

也就是说

$$
\begin{cases}
w^\top x_i &#43; b &gt; 0, y_i = 1 \\\\
w^\top x_i &#43; b &lt; 0, y_i = -1
\end{cases}
$$

我们可以再简化一下式子，注意到 $y_i$ 乘以 $w^\top x_i &#43; b$ 始终大于 $0$，所以式子转化为 $y_i(w^\top x_i &#43; b) &gt; 0$

### 最大化间隔

我们需要找到这条直线将正负样本划分开来，然后样本到直线的距离必须都要最远。

### 支持向量

正负样本中距离超平面最近的一些点，这些点叫做支持向量。

![1](/image/ML/8.png)

### 最优化

我们的目标是支持向量到划分超平面的距离最大。

考虑点到超平面的距离 $\dfrac{|w^\top x &#43; b|}{||w||}$

根据支持向量到超平面的距离为 $d$，其他点到超平面的距离大于 $d$ 我们有：

$$
\begin{cases}
\dfrac{1}{||w||} (w^\top x_i &#43; b) \ge d, y_i = 1 \\\\
\dfrac{1}{||w||} (w^\top x_i &#43; b) \le -d, y_i = -1
\end{cases}
$$

把 $d$ 除过去：

$$
\begin{cases}
\dfrac{1}{||w||d} (w^\top x_i &#43; b) \ge 1, y_i = 1 \\\\
\dfrac{1}{||w||d} (w^\top x_i &#43; b) \le -1, y_i = -1
\end{cases}
$$

令这一坨 $\dfrac{1}{||w||d}$ 为 $1$ (方便推导，且对目标函数无影响) 得到：

$$
\begin{cases}
w^\top x_i &#43; b \ge 1, y_i = 1 \\\\
w^\top x_i &#43; b \le -1, y_i = -1
\end{cases}
$$

式子简化为 $y_i (w^\top x_i &#43; b) \ge 1$

含义如下图所示：

![1](/image/ML/9.png)

回顾我们要最大化的目标 $\dfrac{|w^\top x_i &#43; b|}{||w||}$，由于 $y_i (w^\top x_i &#43; b) \ge 1$，那么 $y_i (w^\top x_i &#43; b) = |w^\top x_i &#43; b|$，然后只考虑支持向量，那么 $y_i (w^\top x_i &#43; b) = 1$，再为了推导方便，我们将式子整体乘 $2$ (无影响)，原式变为 $\max \dfrac{2}{||w||}$，相当于 $\min \dfrac{||w||}{2}$，然后我们把 $||w||$ 加一个平方，因为 $x ^ 2$ 单调 (指大于 $0$ 时)不影响最小值点，式子变为

$$
\min \frac{1}{2} ||w|| ^ 2 \\\\
\text{s.t.} \ y_i (w^\top x_i &#43; b) \ge 1
$$

这就是硬间隔线性 SVM

## 对偶型
接下来考虑对偶型。
### KKT 条件

要先引入一个方法就是 KKT 条件。

首先看一个同时含有等式和不等式约束的多元函数极值。

$$
\begin{aligned}
\min_u\ &amp;f(u) \\\\
\text{s.t.} \ &amp; g_i(u) \le 0, &amp; i = 1,2,\cdots,m \\\\
&amp; h_j(u) = 0, &amp; j = 1,2,\cdots, n
\end{aligned}
$$

对于等式直接拉格朗日乘数法没什么好说的，重点关注一下不等式。

我们假设目标函数为 $L(u, \alpha, \beta) = f(u) &#43; \sum\limits_{i = 1} ^ {m} \alpha_i g_i(u) &#43; \sum\limits_{j = 1} ^ {n} \beta_jh_j(u)$

对于不等式来说，有 $g(u) &lt; 0$ 和 $g(u) = 0$ 两种情况。

- 当 $g(u) &lt; 0$ 时，相当于此条件作废，也就是没有限制条件了，约束函数不起作用，那么此时相当于 $\alpha = 0$

- 当 $g(u) = 0$ 时，相当于是等式的情况了，可行解 $u ^ *$ 满足 $\nabla f(u ^ *) = -\alpha \nabla g(u ^ *)$，且梯度方向相反且平行，所以 $\alpha &gt; 0$

如下图所示 ($u$ 替换为 $x$)：

![1](/image/ML/10.jpg)

综上所述，满足 $\alpha_i^* g(u*) = 0$，称作互补松弛

### 约束问题转换为 min max 拉格朗日函数

约束问题等价于

$$
\begin{aligned}
\min_u \max_{\alpha, \beta}\ &amp; L(u, \alpha, \beta) \\\\
\text{s.t.} \ &amp; \alpha_i \ge 0, &amp; i = 1,2,\cdots,m
\end{aligned}
$$

证明：

原式为

$$
\min_u \max_{\alpha, \beta} \left(f(u) &#43; \sum\_{i = 1} ^ {m} \alpha_i g_i(u) &#43; \sum_{j = 1} ^ {n} \beta_jh_j(u) \right)
$$

$f(u)$ 无关，将 $\max_{\alpha, \beta}$ 拿进去

$$
\min_u  \left(f(u) &#43; \max_{\alpha, \beta} \left(\sum\_{i = 1} ^ {m} \alpha_i g_i(u) &#43; \sum_{j = 1} ^ {n} \beta_jh_j(u) \right) \right)
$$

- 如果 $u$ 不满足约束，那么 $g_i(u) &gt; 0$，由于约束 $\alpha_i \ge 0$，可以取 $\alpha_i = \infty$，使得式子为 $\infty$

- 如果 $u$ 不满足等式约束，即 $h_j(u) \ne 0$，由于 $\beta_j$ 没有正负限制，可以取 $\beta_j = \text{sign}(h_j(u)) = \infty$

- 如果 $u$ 满足约束，那么 $\alpha_i \ge 0, \alpha_i g(u) \le 0$，且 $\beta_jh_j(u) = 0$，式子结果为 $0$

所以转化为

$$
\min_u  \left(f(u) &#43; \begin{cases}0, \forall i,j \ g_i(u) \le 0, h_j(u) = 0\\\\ \infty, \text{else}\end{cases} \right)
$$

分配 $\min$

$$
\min_u  f(u) &#43; \min_u\begin{cases}0, \forall i,j \ g_i(u) \le 0, h_j(u) = 0\\\\ \infty, \text{else}\end{cases} 
$$

等价于 $\min_u f(u)$，且 $u$ 满足约束。

#### 交换 min max

由于硬间隔线性 SVM 满足 Slater 条件 (https://www.cnblogs.com/guanyang/p/16287060.html)，所以可以交换 min max 等价：

$$
\min_u \max_{\alpha, \beta}\  L(u, \alpha, \beta) = \max_{\alpha, \beta} \min_u \  L(u, \alpha, \beta)
$$

### 对偶问题

回顾一下原始问题

$$
\min \frac{1}{2} ||w|| ^ 2 \\\\
\text{s.t.} \ 1 - y_i (w^\top x_i &#43; b) \le 0
$$

该问题没有等式约束，那么定义拉格朗日函数为 $L(w, b, \alpha) = \dfrac{1}{2} ||w||^2 &#43; \sum\limits_{i = 1} ^ {m} \alpha_i(1 - y_i (w^\top x_i &#43; b))$

根据对偶性转换为

$$
\max_\alpha \min_{w, b} \ L(w, b, \alpha) \\\\
\text{s.t.} \ \alpha_i \ge 0, i = 1, 2, \cdots, m
$$

先考虑里面的 $\min\limits_{w, b} \ L(w, b, \alpha)$，我们直接对 $w, b$ 求偏导令成 $0$：

$$
\begin{aligned}
\frac{\partial L}{\partial w} &amp;= 0 \\\\
w - \sum_{i = 1} ^ m \alpha_i x_i y_i &amp;= 0 \\\\
\sum_{i = 1} ^ m \alpha_i x_i y_i &amp;= w
\end{aligned}
$$

解得最优值 $w ^ * = \sum\limits_{i = 1} ^ m\alpha_i x_i y_i$，接下来对 $b$ 偏导：

$$
\begin{aligned}
\frac{\partial L}{\partial b} &amp;= 0 \\\\
\sum_{i = 1} ^ m\alpha_i y_i &amp;= 0
\end{aligned}
$$

得到了一个等式 $\sum\limits_{i = 1} ^ m\alpha_i y_i = 0$

我们将两个偏导结果带入到原式：

$$
\begin{aligned}
L(w ^ *, b ^ *, \alpha) &amp;= \frac{1}{2} ||w ^ *||^2 &#43; \sum_{i = 1} ^ {m} \alpha_i(1 - y_i(w^\top x_i &#43; b)) \\\\
&amp;= \frac{1}{2} ||w ^ *||^2 &#43; \sum_{i = 1} ^ {m} \alpha_i - \sum_{i = 1} ^ {m} \alpha_i y_i w^\top x_i - \sum_{i = 1} ^ {m} \alpha_i y_i b \\\\
&amp;= \frac{1}{2} ||w ^ *||^2 &#43; \sum_{i = 1} ^ {m} \alpha_i - ||w ^ *|| ^ 2 - b \times 0 \\\\
&amp;= -\frac{1}{2} ||w ^ *||^2 &#43; \sum_{i = 1} ^ {m} \alpha_i \\\\
&amp;= -\frac{1}{2} \left(\sum\limits_{i = 1} ^ m\alpha_i x_i y_i\right) \left(\sum\limits_{j = 1} ^ m\alpha_j x_j y_j\right) &#43; \sum_{i = 1} ^ {m} \alpha_i \\\\
&amp;= -\frac{1}{2} \sum\limits_{i = 1} ^ m \sum\limits_{j = 1} ^ m \alpha_i \alpha_j x_i x_j y_i y_j &#43; \sum_{i = 1} ^ {m} \alpha_i
\end{aligned}
$$

然后问题变为：

$$
\max_\alpha \left(-\frac{1}{2} \sum\limits_{i = 1} ^ m \sum\limits_{j = 1} ^ m \alpha_i \alpha_j x_i x_j y_i y_j &#43; \sum_{i = 1} ^ {m} \alpha_i\right) \\\\
\text{s.t.} \ \alpha_i \ge 0, i = 1, 2, \cdots, m \\\\
\sum_{i = 1} ^ m\alpha_i y_i = 0
$$

加个负号变成 $\min$：

$$
\min_\alpha \left(\frac{1}{2} \sum\limits_{i = 1} ^ m \sum\limits_{j = 1} ^ m \alpha_i \alpha_j x_i x_j y_i y_j - \sum_{i = 1} ^ {m} \alpha_i\right) \\\\
\text{s.t.} \ \alpha_i \ge 0, i = 1, 2, \cdots, m \\\\
\sum_{i = 1} ^ m\alpha_i y_i = 0
$$

这就是硬间隔线性 SVM 的对偶型。

### SMO 算法

这是一个二次规划问题，可以用 SMO(Sequential Minimal Optimization) 即序列最小优化算法求解。

SMO 核心思想是每次只优化一个参数，其他固定。但是我们这里有 $\sum\limits_{i = 1} ^ m\alpha_i y_i = 0$，如果我们优化其中一个 $\alpha_i$，其他固定，那么 $\alpha_i$ 也成定值了。

所以我们考虑固定住两个参数 $\alpha_i, \alpha_j$，约束变为

$$
\alpha_i y_i &#43; \alpha_j y_j = C \\\\
C = -\sum_{k \ne i,j} \alpha_k y_k
$$

由此得出 $\alpha_j = \dfrac{C - \alpha_i y_i}{y_j}$，我们带入原式，就得到了只含有一个 $\alpha_i$ 变量的式子，直接对 $\alpha_i$ 梯度下降。之后再选一个新的变量即可。

完事之后根据式子 $w = \sum\limits_{i = 1} ^ m\alpha_i x_i y_i$ 可以求出 $w$，再根据 $y_i(w^\top x_i &#43; b) = 1$ 求出 $b$，可以把每个支持向量带进去求一个 $b$ 的平均值。

## 核函数

假设现在是一个线性不可分的样本。

![1](/image/ML/11.jpg)

我们可以把特征映射到更高维度。这样在高维就线性可分了。

![1](/image/ML/12.jpg)

处理方式是将 $x_i, x_j$ 变为 $\Phi(x_i), \Phi(x_j)$，我们把核函数写作 $K(x_i, x_j)$

$$
\min_\alpha \left(\frac{1}{2} \sum\limits_{i = 1} ^ m \sum\limits_{j = 1} ^ m \alpha_i \alpha_j \Phi(x_i) \Phi(x_j) y_i y_j - \sum_{i = 1} ^ {m} \alpha_i\right) \\\\
\text{s.t.} \ \alpha_i \ge 0, i = 1, 2, \cdots, m \\\\
\sum_{i = 1} ^ m\alpha_i y_i = 0
$$

### 常见的核函数

#### 线性核函数

$$
K(x_i, x_j) = x_i ^ \top x_j
$$

#### 多项式核函数

$$
K(x_i, x_j) = (\gamma x_i ^ \top x_j &#43; b) ^ d
$$

#### 高斯核函数

$$
K(x_i, x_j) = \exp(-\gamma||x_i - x_j|| ^ 2)
$$

#### sigmoid 核函数

$$
K(x_i, x_j) = \tanh(\gamma x ^ \top x_j &#43; b)
$$

## 参考
统计学习方法(第2版)李航

https://zhuanlan.zhihu.com/p/77750026

https://zhuanlan.zhihu.com/p/38163970

https://zhuanlan.zhihu.com/p/55532322

https://zhuanlan.zhihu.com/p/261061617

https://zhuanlan.zhihu.com/p/480302399

https://blog.csdn.net/qq_25018077/article/details/139541976

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/  

