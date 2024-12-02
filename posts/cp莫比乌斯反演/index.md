# 莫比乌斯反演


## 前置知识1:整除分块
## $①:\lfloor \dfrac{a}{bc} \rfloor = \lfloor \dfrac{\lfloor \dfrac{a}{b} \rfloor}{c} \rfloor$
**证明：**

$$\dfrac{a}{b}=\lfloor \dfrac{a}{b}\rfloor &#43; r,r\in[0,1)$$

$$\lfloor \dfrac{a}{bc} \rfloor = \lfloor \dfrac{a}{b} ·\dfrac{1}{c} \rfloor$$

$$=\lfloor \dfrac{1}{c}·(\lfloor \dfrac{a}{b}\rfloor &#43; r)\rfloor$$

$$=\lfloor \dfrac{\lfloor \dfrac{a}{b} \rfloor}{c}&#43;\dfrac{r}{c} \rfloor$$

$$\because r&lt;c$$

$$\therefore \lfloor \dfrac{\lfloor \dfrac{a}{b} \rfloor}{c}&#43;\dfrac{r}{c} \rfloor=\lfloor \dfrac{\lfloor \dfrac{a}{b} \rfloor}{c} \rfloor$$
## $②:i \in [1,n],|\lfloor \dfrac{n}{i}\rfloor| \le 2 \sqrt{n}$

**证明：**

当 $i \in[1,\lfloor \sqrt n\rfloor]$ 时，$\lfloor \dfrac{n}{i} \rfloor$有$\lfloor \sqrt n\rfloor$种取值

当 $i\in(\lfloor \sqrt n\rfloor, n]$ 时，$\lfloor \dfrac{n}{i} \rfloor\le \lfloor \sqrt n\rfloor$，有$\lfloor \sqrt n\rfloor$种取值

## $③:$ 对 $\lfloor\dfrac{n}{i}\rfloor$ 求和， $\forall i\in[1,n]$ 只需要找到最大的一个 $j(i \le j \le n)$，使得 $\lfloor\dfrac{n}{i}\rfloor=\lfloor\dfrac{n}{j}\rfloor$，此时 $j=\lfloor\dfrac{n}{\lfloor\dfrac{n}{j}\rfloor}\rfloor$

**证明：** 
先证明 $j \ge i$：
$$\lfloor \dfrac{n}{i}\rfloor \le \dfrac{n}{i} $$

$$\Leftrightarrow \lfloor\dfrac{n}{\lfloor\dfrac{n}{i}\rfloor}\rfloor \ge \lfloor\dfrac{n}{\dfrac{n}{i}}\rfloor = \lfloor i \rfloor=i$$

$$\Leftrightarrow i \le \lfloor\dfrac{n}{\lfloor\dfrac{n}{i}\rfloor}\rfloor =j$$

再证明最大值
设 $k=\lfloor\dfrac{n}{i}\rfloor$，则

$$k \le \dfrac{n}{j} &lt;k &#43; 1$$

$$\dfrac{1}{k &#43; 1} &lt; \dfrac{j}{n} \le \dfrac{1}{k}$$

$$\dfrac{n}{k &#43; 1} &lt; j \le \dfrac{n}{k}$$

因为 $j$ 是整数，所以最大值为 $\lfloor \dfrac{n}{k}\rfloor$

所以每次将 $[i,j]$ 分为一块求解累加到答案上

## 代码：
```cpp
for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res &#43;= (r - l &#43; 1) * (n / l);
}
```

## 前置知识2：莫比乌斯函数
**定义：**

设 $n=p_1^{c_1}\cdots p_k^{c_k}$
$$\mu(n)=\begin{cases}
0,&amp;\exists i \in[1,k],c_i &gt;1 \\\\
1,&amp;k \equiv 0\pmod2,\forall i \in[1,k],c_i=1\\\\
-1,&amp;k\equiv1\pmod2,\forall i\in [1,k],c_i=1
\end{cases}$$

**性质：** 

$$\sum_{d|n}\mu(d) = \begin{cases}
1,&amp;n=1\\\\
0,&amp;n&gt;1
\end{cases}$$

**证明：**

$$取n=\prod_{i=1}^{k}p_i$$

$$\sum_{d|n}\mu(d)=\sum_{i=0}^{k}C_{k}^{i}·(-1)^i$$

$$=(1&#43;(-1))^k=0(二项式定理)$$

## 线性筛莫比乌斯函数：
```cpp
void get_mobius(int n) {
    mobius[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            mobius[i] = -1;
        }
        for (int j = 0; primes[j] * i &lt;= n; j &#43;&#43;) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                mobius[t] = 0;
                break;
            }
            mobius[t] = -mobius[i];
        }
    }
}
```

## 莫比乌斯反演：
设 $f(n),g(n)$ 为两个数论函数

**形式一：**

如果

$$f(n)=\sum_{d|n}g(d)$$

则有

$$g(n)=\sum_{d|n}\mu(d)f(\dfrac{n}{d})$$

**证明：**

将

$$f(n)=\sum_{d|n}g(d)$$

带入

$$\sum_{d|n}\mu(d)f(\dfrac{n}{d})$$

得

$$\sum_{d|n}\mu(d)\sum_{k \mid \frac{n}{d}}g(k)$$

交换求和次序：
因为 $k \mid \dfrac{n}{d}$ ，那么 $k$ 也是 $n$ 的因子，所以枚举 $n$ 的所有因子 $d$ 等价于枚举 $k$，$k \mid \dfrac{n}{d}=d \mid \dfrac{n}{k}$

所以

$$\sum_{k|n}g(k)\sum_{d|\frac{n}{k}}\mu(d)$$

根据 $\sum_{d \mid n}\mu(d)=[n=1]$

所以 $n=k$，所以 

$$\sum_{k|n}g(k)\sum_{d \mid \frac{n}{k}}\mu(d)=g(n)$$

证毕。

**形式二(常用)：**

如果

$$f(n)=\sum_{n|d}g(d)$$

则有

$$g(n)=\sum_{n|d}\mu(\dfrac{d}{n})f(d)$$

**证明：**

将

$$f(n)=\sum_{n|d}g(d)$$

带入

$$\sum_{n|d}\mu(\dfrac{d}{n})f(d)$$

得

$$\sum_{n|d}\mu(\dfrac{d}{n})\sum_{d|k}g(k)$$

交换求和次序：

$$\sum_{n|k}g(k)\sum_{\frac{d}{n}|\frac{k}{n}}\mu(\dfrac{d}{n})$$

根据 $\sum_{d|n}\mu(d)=[n=1]$

所以 $n=k$，所以 

$$\sum_{n|k}g(k)\sum_{\frac{d}{n} \mid \frac{k}{n}}\mu(\dfrac{d}{n})=g(n)$$

证毕。
## 反演技巧

$$\sum_{i=1}^{n} f(i) \sum_{j=1}^{m}g(j)=\sum_{j=1}^{m}g(j)\sum_{i=1}^{n}f(i)$$

$$\sum_{i=1}^{n} f(i) \sum_{d 
\mid i}g(d)=\sum_{d=1}^{n}g(d)\sum_{i=1}^{\lfloor \frac{n}{d} \rfloor}f(id)$$

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E8%8E%AB%E6%AF%94%E4%B9%8C%E6%96%AF%E5%8F%8D%E6%BC%94/  

