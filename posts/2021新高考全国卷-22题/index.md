# 2021新高考全国Ⅰ卷 22题


## 题面:
$已知函数f(x)=x(1-\ln{x})$

$(1)讨论f(x)的单调性$

$(2)设a,b&gt;0,a\ne b,且b\ln{a}-a\ln{b}=a-b,证明:2&lt;\frac{1}{a}&#43;\frac{1}{b}&lt;e$

## 题解:
$(1)f&#39;(x)=-\ln{x},则x\in(0,1]时,f(x)单调递增,x\in[1,&#43;\infty)时,f(x)单调递减$

$(2)$ $$\because b\ln{a}-a\ln{b}=a-b$$

$$两边同时除ab$$

$$\therefore \frac{\ln{a}}{a}-\frac{\ln{b}}{b}=\frac{1}{b}-\frac{1}{a}$$

$$\therefore \frac{\ln{a}}{a}&#43;\frac{1}{a}=\frac{\ln{b}}{b}&#43;\frac{1}{b}$$

$$\therefore \frac{1}{a} ( 1&#43;\ln{a})=\frac{1}{b} (1&#43; \ln{b})$$

$$\therefore \frac{1}{a} ( 1-\ln{\frac{1}{a}})=\frac{1}{b} ( 1-\ln{\frac{1}{b}})$$


$$由(1)知:0&lt;\frac{1}{a}&lt;1&lt;\frac{1}{b}&lt;e$$

$$令x_1=\frac{1}{a},x_2=\frac{1}{b}$$

$$左边:x_1&#43;x_2&gt;2$$

$$\Leftrightarrow  x_2&gt;2-x_1$$

$$\Leftrightarrow f(x_2)=f(x_1)&lt;f(2-x_1)$$

$$\Leftrightarrow  f(2-x_1)-f(x_1)&gt;0$$

$$令g(x)=f(2-x)-f(x),x &gt;1$$

$$g&#39;(x)=\ln{(2-x)}&#43;\ln{x}=\ln{(2x-x^2)}&gt;0,g(x)单调递增$$

$$\because g(1)=0\therefore g(x)&gt;0,原式成立$$

$$右边:x_1&#43;x_2&lt;e$$

$$f(x)在(e,0)点的切线为y=e-x$$

$$切线放缩:x(1-\ln{x})\le e-x$$

$$\because x_1(1-\ln{x_1})=x_2(1-\ln{x_2})$$

$$\therefore x_1(1-\ln{x_1})&lt;e-x_2$$

$$\Leftrightarrow  x_1&#43;x_2&lt;e&#43;x_1\ln{x_1}$$

$$\because x_1 \in (0,1)$$

$$\therefore x_1\ln{x_1}&lt;0$$

$$\therefore x_1&#43;x_2&lt;e&#43;x_1\ln{x_1}\ &lt;e$$

$$\therefore 2&lt;x_1&#43;x_2&lt;e \Leftrightarrow 2&lt;\frac{1}{a}&#43;\frac{1}{b}&lt;e$$

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2021%E6%96%B0%E9%AB%98%E8%80%83%E5%85%A8%E5%9B%BD%E5%8D%B7-22%E9%A2%98/  

