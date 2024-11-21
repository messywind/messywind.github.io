# 狄利克雷卷积


## 定义：
对于两个数论函数 $f(x),g(x)$ 那么它们的卷积 $h(x)$ 记作 $f(x) * g(x)$，式子如下：

$$f(x) * g(x) = h(x) = \sum_{d \mid n} f(d)g(\frac{n}{d})$$

简记为 $h = f * g$

## 性质：
**交换律：** $f * g = g * f$

**结合律：** $(f * g) * h = f * (g * h)$

**分配律：** $(f &#43; g) * h = f * h &#43; g * h$

**单位元：** 对 $\forall f(x), f * \varepsilon = f$

**逆元：** 对一个非零的数论函数 $f(x)$，和另一个数论函数 $g(x)$ 满足 $f*g=\varepsilon$，则称 $g(x)$ 为 $f(x)$ 的逆元。

$g(x) = \dfrac{\varepsilon - \sum \limits_{d \mid x,d \ne 1} f(d)g(\dfrac{x}{d})}{f(1)}$

**积性函数的逆元还是积性函数**

**两个积性函数的狄利克雷卷积还是积性函数**

## 常见积性函数：

 **1. 莫比乌斯函数：$\mu(x)$**

 &gt;设 $n=p_1^{c_1}\cdots p_k^{c_k}$
 &gt;$$\mu(n)=\begin{cases}
 &gt;0,&amp;\exists i \in[1,k],c_i &gt;1 \\\\
 &gt;1,&amp;k \equiv 0\pmod2,\forall i \in[1,k],c_i=1\\\\
 &gt;-1,&amp;k\equiv1\pmod2,\forall i\in [1,k],c_i=1
 &gt;\end{cases}$$

 **2. 欧拉函数：$\varphi(x)$**
 &gt;$\varphi(n) = \sum \limits_{i=1} ^{n}[\gcd(i,n) = 1]$

 **3. 单位函数：$\varepsilon(x)$**
&gt;$\varepsilon(n) = [n = 1]$

 **4. 恒等函数：$Id(x)$**
 &gt;$Id(n) = n$

 **5. 常数函数：$I(x)$**
 &gt;$I(n)=1$

**6. 约数个数函数：$d(x)$**
&gt;$d(n)=\sum \limits_{i \mid n}1$

**7. 约数和函数：$\sigma(x)$**
&gt;$\sigma(n)=\sum \limits_{d \mid n} d$

## 常见卷积：
**1. $\varepsilon = \mu * 1$**
&gt;$\varepsilon = [n=1]=\sum \limits _{d \mid n} \mu (d)$

**2. $d = 1 * 1$**
&gt;$d(n)=\sum \limits_{i \mid n}1$

**3. $Id * 1 = \sigma$**
&gt;$\sigma(n)=\sum \limits_{d \mid n} d$

**4. $\mu * Id = \varphi$**
&gt;$\varphi(n)=\sum \limits _{d \mid n} d \cdot \mu(\dfrac{n}{d})$

**5. $\varphi * 1 = Id$**
&gt;$Id(n)=\sum \limits _{d \mid n} \varphi(d)$

## 狄利克雷卷积证莫比乌斯反演：
已知 $f = g * 1$ 其中 $1$ 的逆元为 $\mu$，所以有 $f * \mu = g$

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E7%8B%84%E5%88%A9%E5%85%8B%E9%9B%B7%E5%8D%B7%E7%A7%AF/  

