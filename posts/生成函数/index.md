# 生成函数

## 形式幂级数

设数列 $a_0,a_1, a_2,\cdots,a_n,\cdots$ 那么他的形式幂级数就为

$$\sum_{i = 0}^{\infty}a_ix^i$$  

#### 运算：

设 $f(x) = \sum_{i = 0}^{\infty}a_ix^i,g(x) = \sum_{i = 0}^{\infty}b_ix^i$

加法：

$$f(x) &#43; g(x) = \sum_{i = 0}^{\infty}(a_i &#43; b_i)x^i$$

减法：

$$f(x) - g(x) = \sum_{i = 0}^{\infty}(a_i - b_i)x^i$$

乘法：

$$f(x) * g(x) = \sum_{i = 0}^{\infty}\sum_{k = 0}^{i}a_k * b_{i - k}x^i$$

记号：

记形式幂级数 $f(x)$ 的 $x^n$ 项系数为 $[x^n] f(x)$

## 常生成函数

### 定理：

设 $S = {a_1,a_2,\cdots,a_k}$ ，且 $a_i$ 可以取的次数的集合为 $M_i$ ，记 $F_i(x) = \sum_{u \in M_i}x^u$ ，则从 $S$ 中取 $n$ 个元素组成集合的方案数 $g(n)$ 的常生成函数 $G(x) = \sum_{i =0}^{\infty}g(i)x^i$ 满足：

$$G(x) = F_1(x)F_2(x) \cdots F_k(x)$$

### 形式幂级数的逆元：

$A(x)B(x) = 1$

逆元存在的条件：$[x^0]A(x) \ne 0$

暴力计算的方法：递推

#### 常见的逆：

$$\sum_{i = 0}^{\infty}x^i = \frac{1}{1 - x}$$ 

$$\sum_{i = 0}^{\infty} a^ix^i = \frac{1}{1 - ax}$$

$$\sum_{ i = 0}^{\infty}\binom{i &#43; k - 1}{i}x^i=\frac{1}{(1 - x)^k}$$

### 例题：

#### 食物

&gt;在一个自助水果店，有苹果、香蕉、草莓三种水果，你可以取 $n$ 个水果，但是水果店要求，取的苹果数必须是偶数，取的香蕉数必须是 $3$ 的倍数，取的草莓数不能超过 $5$ 个。求有多少种取 $n$ 个水果的方案。

苹果：$A(x) = 1&#43;x^2&#43;x^4&#43;\cdots$

香蕉：$B(x) = 1 &#43; x^3 &#43; x^6&#43;\cdots$

草莓：$C(x) = 1 &#43; x &#43; x ^2 &#43; x ^3 &#43; x ^ 4 &#43; x ^ 5$

那么答案就是 $[x^n]A(x)B(x)C(x)$

#### Devu and Flowers

&gt;$n$ 种花，分别有 $f_1,f_2,\cdots,f_n$ 个，求取 $s$ 朵花的方案数
&gt;
&gt;$(1 \le n \le 20, 0 \le f_i \le 10^{12}, 0 \le s \le 10^{14})$

每一朵花的生成函数：

$$F_i(x) = 1 &#43; x &#43; x ^2 &#43; \cdots &#43; x^{f_i} = \frac{1 - x^{f_i &#43; 1}}{1 - x}$$

方案数：

$$F(x) = F_1(x)F_2(x)\cdots F_n(x) = \frac{\prod_{i = 1}^{n}(1-x^{f_i &#43; 1})}{(1-x)^n}$$

设 $A(x) = \prod_{i = 1}^{n}(1-x^{f_i &#43; 1})$

那么答案就为

$$[x^s]F(x) = \sum_{i = 0}^{s}[x^i]A(x) [x^{ s- i}] \frac{1}{(1 - x)^n}$$

等价于

$$[x^s]F(x) = \sum_{i = 0}^{s}[x^i]A(x) C_{s - i &#43;  n - 1}^{n - 1} $$

#### [CEOI2004] Sweets

&gt;$n$ 种糖果，别分有 $m_1,m_2,\cdots,m_n$ 个，求取不少于 $a$ 不多于 $b$ 颗糖果的方案数。
&gt;
&gt;$(1 \le n \le 10, 0 \le a \le b \le 10^7,0 \le m_i \le 10^6)$

每一个糖果的生成函数：

$$F_i(x) = 1 &#43; x &#43; x ^2 &#43; \cdots &#43; x^{m_i} = \frac{1 - x^{m_i &#43; 1}}{1 - x}$$

方案数：

$$F(x) = F_1(x)F_2(x)\cdots F_n(x) = \frac{\prod_{i = 1}^{n}(1-x^{m_i &#43; 1})}{(1-x)^n}$$

那么答案就为

$$\sum_{s = a}^{b}[x^s]F(x) = \sum_{s = a}^{b}\sum_{i = 0}^{s}[x^i]A(x) C_{s - i &#43;  n - 1}^{n - 1}$$

根据组合数公式

$$C_{a}^{j}&#43;C_{a &#43; 1}^{j} &#43; \cdots&#43;C_{b}^{j} = C_{b &#43; 1}^{j &#43; 1} - C_{a}^{j &#43; 1}$$

得到

$$\sum_{i = 0}^{s}[x^i]A(x) (C_{a - i &#43; n}^{n} - C_{b - i&#43; n - 1}^{n})$$

## 指数生成函数

一个数列 $\{a_n\}$ 对应的指数生成函数为 $f(x) = \sum_{i = 0}^{\infty}a_i\dfrac{x_i}{i!}$

### 定理：

设 $S = {a_1,a_2,\cdots,a_k}$ ，且 $a_i$ 可以取的次数的集合为 $M_i$ ，记 $F_i(x) = \sum_{u \in M_i}\dfrac{x^u}{u!}$ ，则从 $S$ 中取 $n$ 个元素排成一列的方案数 $g(n)$ 的指数生成函数 $G(x) = \sum_{i =0}^{\infty}g(i)\dfrac{x^i}{i!}$ 满足：

$$G(x) = F_1(x)F_2(x) \cdots F_k(x)$$

#### 常见公式：

$$\sum_{i = 0}^{\infty}\frac{x^i}{i!} = e^x$$

$$\sum_{i = 0}^{\infty}a^i\frac{x^i}{i!} = e^{ax}$$

### 例题：

#### blocks

&gt;一段长度为 $n$ 的序列，你有红黄蓝绿 $4$ 种颜色的砖块，一块砖长度为 $1$，问你铺砖的方案数，其中红色砖和绿色砖的数量必须为偶数
&gt;
&gt;答案可能很大，请输出 $\bmod 10007$ 后的结果

设

$$F(x) = 1 &#43; \dfrac{x^2}{2!} &#43; \dfrac{x^4}{4!} &#43;\cdots = \dfrac{e^{x}&#43;e^{-x}}{2}$$

$$G(x) = 1 &#43; x&#43;\dfrac{x^2}{2!} &#43; \dfrac{x^3}{3!} &#43;\cdots = e^x$$

则答案为

$$n!\times[x^n]F^2(x)G^2(x) = n!\times[x^n]\frac{e^{4x}&#43;2e^{2x} &#43; 1}{4}=\frac{4^n&#43;2\times2^n}{4} = 4^{n - 1} &#43; 2^{n - 1}$$

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E7%94%9F%E6%88%90%E5%87%BD%E6%95%B0/  

