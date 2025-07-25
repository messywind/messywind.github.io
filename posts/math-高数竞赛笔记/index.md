# 高数竞赛笔记

## $1.已知函数f(x)连续且f(x&#43;2)-f(x)=\cos x,\int_{0}^{2}f(x)dx=1,求\int^{1}_{-1}f(x)dx$

**分析：** 构造。题目中给出了两点差的函数式，想到构造变动上下限积分，题眼$\int^{2}_{0}f(x)dx=1$，提示构造$F(x)$并解出任意常数$C$

$$令F(x)=\int^{x&#43;2}_{x}f(t)dt,则F&#39;(x)=f(x&#43;2)-f(x)=\cos x$$

$$\therefore F(x)=\sin x&#43;C$$

$$\because \int^{2}_{0}f(x)dx=1,\therefore F(0)=1$$

$$带入 F(x)=\sin x&#43;C,得C=1$$

$$\therefore F(x)=\sin x&#43;1$$

$$则\int^{1}_{-1}f(x)dx=F(-1)=1-\sin 1$$

## $2.计算I=\int_{L}\frac{(x-1)\text{d}y-y\text{d}x}{(x-1)^2&#43;y^2},L是从(-2,0)到(2,0)的上半椭圆\frac{x^2}{4}&#43;y^2=1$

**分析：** 格林公式&#43;第二类曲线积分的路径无关性。

看到这种形式应该想到路径无关，同时分母是个圆的方程，要注意考虑分母为$0$的情况，经典套路用一个很小的圆计算无意义的点，再次用格林公式。

$$I=\int_{L}\frac{-y\text{d}x&#43;(x-1)\text{d}y}{(x-1)^2&#43;y^2}$$

$$P=\frac{-y}{(x-1)^2&#43;y^2},Q=\frac{x-1}{(x-1)^2&#43;y^2}$$

$$则\frac{\partial P}{\partial y}=\frac{\partial Q}{\partial x}=\frac{y^2-(x-1)^2}{((x-1)^2&#43;y^2)^2}$$

$$\because L为单连通区域\therefore I与路径无关$$

$$\because (x-1)^2&#43;y^2\ne 0,\therefore 不包含(1,0)$$

$$\therefore 取L_{\varepsilon}=(x-1)^2&#43;y^2=\varepsilon ^2(\varepsilon &gt;0且足够小),y\ge 0$$

$$\int_{L_\varepsilon}\frac{(x-1)\text{d}y-y\text{d}x}{(x-1)^2&#43;y^2}=-\frac{1}{\varepsilon^2}\iint_{L_\varepsilon}\text{d}x\text{d}y=-\pi$$

$$\therefore I=0&#43;\int_{L_\varepsilon}=-\pi$$

## $3.设f(x)在[0,1]上二阶可导,f(0)=0,f(1)=1,\int^{1}_{0}f(x)\text{d}x=1$

## $(1)证明 \exists \xi \in(0,1),使得f&#39;(\xi)=0$

## $(2)证明：\exists \eta \in (0,1),使得f&#39;\&#39;(\eta)&lt;-2$

$(1)$ **分析：** 看见导数先想到罗尔定理，两个点一个用1，一个就要去题干找点，发现积分式$\int_{0}^{1}f(x)\text{d}x=1$，发现恰好符合积分中值定理$\frac{\int^{1}_{0}f(x)\text{d}x}{1-0}=f(\xi)$，于是找到第二个点$\xi_1$。
再用这两个点找到$\xi$

![QQ图片20210616161403.png](https://cdn.acwing.com/media/article/image/2021/06/16/63738_0393588bce-QQ图片20210616161403.png) 


$$\because \int^{1}_{0}f(x)\text{d}x=1$$

$$\therefore 由积分中值定理得:\exists \xi_1\in(0,1),使得f(\xi_1)=1$$

$$\because f(\xi_1)=1,f(1)=1$$

$$\therefore 由罗尔中值定理得:\exists \xi\in(\xi_1,1),使得f&#39;(\xi)=0$$

$(2)$ **分析：** 构造&#43;反证。看到$f&#39;\&#39;(\eta)&lt;-2$应该想到构造函数，二阶导$&gt;0$可以判断函数的凹凸性，再根据题目的已知条件有积分，所以利用凹函数的积分性质来证明不等式。

$$令g(x)=f(x)&#43;x^2$$

$$假设f&#39;\&#39;(x)\ge-2,即f&#39;\&#39;(x)&#43;2\ge0$$

$$\because g&#39;\&#39;(x)=f&#39;\&#39;(x)&#43;2$$

$$\therefore g&#39;\&#39;(x)\ge0$$

$$\therefore g&#39;\&#39;(x)为凹函数$$

$$则有:\int_{0}^{1}g(x) \text{d}x \le \int_{0}^{1}2x\text{d}x=1$$


$$\int_{0}^{1}g(x) \text{d}x=\int_{0}^{1}(f(x)&#43;x^2) \text{d}x=\frac{4}{3} &gt; 1,矛盾$$

$$\therefore \exists \eta \in (0,1),使得g&#39;\&#39;(x)&lt;0$$

$$即:\exists \eta \in (0,1),使得f&#39;\&#39;(\eta)&lt;-2$$

## $4.设正值函数f(x)在[a,b]上连续,\int_{a}^{b}f(x)\text{d}x=A,证明:\int_{a}^{b}f(x)e^{f(x)}\text{d}x\int_{a}^{b}\frac{1}{f(x)}\text{d}x \ge (b-a)(b-a&#43;A)$

**法一：化为二重积分**

**分析：** 题目给了两个积分，想到把另一个积分变元为$y$，转为二重积分，而区间$[a,b]$变为矩形区域，可以利用其性质变为二重积分。根据对称性，二重积分可以互换变量，构造$\frac{1}{2}$，再用重要不等式$e^x\ge x&#43;1$进行放缩，最后证出答案。

$$I=\int_{a}^{b}f(x)e^{f(x)}\text{d}x\int_{a}^{b}\frac{1}{f(x)}\text{d}x=\int_{a}^{b}f(x)e^{f(x)}\text{d}x\int_{a}^{b}\frac{1}{f(y)}\text{d}y$$

$$\because D: [a,b] ×[a,b]为矩形区域$$

$$\therefore I=\iint\limits_{D}^{}\frac{f(x)}{f(y)}e^{f(x)}\text{d}x\text{d}y =\iint\limits_{D}^{}\frac{f(y)}{f(x)}e^{f(y)}\text{d}x\text{d}y$$

$$\therefore I=\frac{1}{2}\iint\limits_{D}^{}\frac{f(x)}{f(y)}e^{f(x)}\text{d}x\text{d}y &#43;\iint\limits_{D}^{}\frac{f(y)}{f(x)}e^{f(y)}\text{d}x\text{d}y$$

$$I=\frac{1}{2}\iint\limits_{D}\frac{f^2(x)e^{f(x)}&#43;f^2(y)e^{f(y)}}{f(x)f(y)}\text{d}x\text{d}y \ge \iint\limits_{D}\sqrt{e^{f(x)&#43;f(y)}}\text{d}x\text{d}y$$

$$I=\iint\limits_{D}e^{\frac{f(x)&#43;f(y)}{2}}\text{d}x\text{d}y \ge \iint\limits_{D}(1&#43;\frac{f(x)&#43;f(y)}{2})\text{d}x\text{d}y$$

$$I\ge \iint\limits_{D}\text{d}x\text{d}y&#43;\iint\limits_{D} f(x) \text{d}x\text{d}y=(b-a)^2&#43;\int_{a}^{b}f(x)\text{d}x\int_{a}^{b}\text{d}y$$

$$I\ge(b-a)^2&#43;(b-a)A=(b-a)(b-aA)$$

**法二：柯西不等式**

**分析：** 看到两个积分可以凑柯西不等式，再放缩证出答案。

$$\because  I= \int_{a}^{b}f^2(x)\text{d}x\int_{a}^{b}g^2(x)\text{d}x \ge (\int_{a}^{b}f(x)g(x)\text{d}x)^2$$

$$\therefore \int_{a}^{b}f(x)e^{f(x)}\text{d}x\int_{a}^{b}\frac{1}{f(x)}\text{d}x \ge (\int_{a}^{b}\sqrt{e^{f(x)}}\text{d}x)^2$$

$$(\int_{a}^{b}\sqrt{e^{f(x)}}\text{d}x)^2 = (\int_{a}^{b} e^{\frac{f(x)}{2}} \text{d}x)^2 \ge [\int_{a}^{b}(1&#43;\frac{f(x)}{2})\text{d}x]^2$$

$$\int_{a}^{b}(1&#43;\frac{f(x)}{2})\text{d}x=b-a&#43;\frac{A}{2}$$

$$\therefore I\ge (b-a&#43;\frac{A}{2})^2$$

$$先证: \forall x,(\frac{x}{2}&#43;k)^2 \ge k(k&#43;x),k为常数$$

$$\Leftrightarrow \frac{x^2}{4}&#43;k^2&#43;kx\ge k^2&#43; kx$$

$$\Leftrightarrow  x^2\ge0恒成立$$

$$\therefore I \ge (b-a)(b-a&#43;A)$$

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/math-%E9%AB%98%E6%95%B0%E7%AB%9E%E8%B5%9B%E7%AC%94%E8%AE%B0/  

