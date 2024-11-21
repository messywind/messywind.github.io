# 2020年普通高等学校招生全国统一考试（新高考全国Ⅰ卷）数学21题

21.（12分）
已知函数 $f(x)=ae^{x-1}-\ln x&#43;\ln a$

(1) 当 $a = e$ 时，求曲线 $y=f(x)$ 在点 $(1,f(1))$ 处的切线与两个坐标轴围成的三角形的面积。

(2) 若 $f(x)\geq1$ ，求 $a$ 的取值范围。

这里重点解一下第二问。

同构法：$ae^{x-1}-\ln x&#43;\ln a \geq 1 \Leftrightarrow ae^{x-1} \geq  \ln\dfrac{ex}{a}$

观察到左边有 $x-1$ 次方，右边为 $ex$，所以两边同乘 $ex$

$axe^{x}\geq ex \ln\dfrac{ex}{a}$

此时形势已经很明了，是 $xe^x$ 形式的同构：

$f(x)e^{f(x)} \ge g(x)e^{g(x)} $

当 $a&gt;0$ 时， $xe^{x}\geq \dfrac{ex}{a}\ln\dfrac{ex}{a} \Leftrightarrow xe^{x}\geq \ln\dfrac{ex}{a}e^{\ln\frac{ex}{a}}$

 ∴只需证 $x\geq \ln\dfrac{ex}{a}$ ，整理得 $ae^{x}\geq ex$，显然是 $e^{x}\geq ex$ 的放缩。故而 $a≥1 $

当 $a&lt;0$ 时，$xe^{x}\leq \dfrac{ex}{a}\ln\dfrac{ex}{a} \Leftrightarrow xe^{x}\leq \ln\dfrac{ex}{a}e^{\ln\frac{ex}{a}}$

所以只需证 $x\leq \ln\dfrac{ex}{a}$，整理得 $ae^{x}\geq ex$，与 $a&gt;0$ 情况相同。

综上 $a\in[1,\infty)$


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2020%E5%B9%B4%E9%AB%98%E8%80%83%E6%95%B0%E5%AD%A621%E9%A2%98/  

