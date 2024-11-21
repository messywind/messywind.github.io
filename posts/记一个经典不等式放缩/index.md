# 记一个经典不等式放缩


今晚闲来无事看了看 $2022$ 年的济南一模最后一题第三问，发现真的是两年没碰高考数学手生疏了，不过还好做出来了。
**题意**
证明
$$
\sum_{x = 2}^{n} \frac{1}{\ln x} &gt; 1 - \frac{1}{n}
$$
**分析：**
$1 - \dfrac{1}{n} = \dfrac{n - 1}{n}$，左边是 $n$ 个数求和，所以考虑裂项右边

$$
\frac{n - 1}{n} = \frac{1}{1 \times 2} &#43; \frac{1}{2 \times 3} &#43; \cdots &#43; \frac{1}{(n - 1) \times n}
$$

接下来只需证 $x ^ 2 - x - \ln x &gt; 0 (x \ge 2)$

设 $f(x) = x ^ 2 - x - \ln x$
则 $f&#39;(x) = \dfrac{(2x&#43; 1)(x-1)}{x} &gt; 0$
由于 $f(2) = 2 - \ln2 &gt; 0$
所以 $f(x)&gt; 0$

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E8%AE%B0%E4%B8%80%E4%B8%AA%E7%BB%8F%E5%85%B8%E4%B8%8D%E7%AD%89%E5%BC%8F%E6%94%BE%E7%BC%A9/  

