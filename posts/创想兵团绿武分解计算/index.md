# 创想兵团绿武分解计算


假设分解服从二项分布，设分解的绿武为 $n$ 个，分解到材料的概率为 $p$，那么分解到金币的概率就为 $1 - p$，则获得材料的期望为 $n \times p$，获得金币的期望为 $1000 \times n \times (1 - p)$，那么消耗的总金币为
$2400 \times n - 1000 \times n \times (1 - p)$

假设一万金币价值为 $x$ 兑换券，一个材料价值为 $y$ 兑换券，那么成本价为
$\dfrac{2400 \times n - 1000 \times n \times (1 - p)}{10000} x$

收益价为 $n \times p \times y$

解不等式

$$\frac{2400 \times n - 1000 \times n \times (1 - p)}{10000} \times x \le n \times p \times y \Leftrightarrow \frac{0.14 x}{y - 0.1x} \le p $$

假设 $x = 3, y = 2$，可知概率大约 $p$ 在 $24.7\%$ 概率可以盈利。


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E5%88%9B%E6%83%B3%E5%85%B5%E5%9B%A2%E7%BB%BF%E6%AD%A6%E5%88%86%E8%A7%A3%E8%AE%A1%E7%AE%97/  

