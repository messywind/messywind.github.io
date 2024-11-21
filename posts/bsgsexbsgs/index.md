# BSGS、exBSGS


## $\text{BSGS (Baby Step Giant Step)}:$

给定正整数 $a,b,p,a \perp p$ ，求满足

$$a^{x} \equiv b(\text{ mod } p)$$

的**最小非负整数** $x$

首先由欧拉定理的推论有：

$$a^x\equiv a^{x\text{ mod }\varphi (p)}(\text{ mod }p)$$

所以 $a^x$ 在模 $p$ 意义下的最小循环节为 $\varphi(p)$ ，那么只需要考虑 $x \in [0,\varphi(p)-1]$ 即可，为了简便避免算欧拉函数，我们对欧拉函数进行放缩 $\varphi(p)  \le p &#43; 1$，那就是枚举 $x \in [0,p]$。那么我们对暴力枚举的算法做一个优化：令 $x =kt-y$ $,k=\lfloor \sqrt{p} \rfloor &#43;1$，则原式为 

$$a^{kt}\equiv ba^{y} (\text{ mod }p)$$

$y$ 的范围是 $[0,k-1]$ ，所以可以枚举每个 $y$ ，预处理右边的值，插入到一个哈希表中，再枚举左边的 $t\in[1,k]$ 如果从哈希表找到值，那么就是答案，特判 $t=0$ 的情况，时间复杂度 $O(\sqrt{p})$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
int a, b, p;
int bsgs(int a, int b, int p) {
    if (1 % p == b % p) return 0;
    int k = sqrt(p) &#43; 1;
    unordered_map&lt;int, int&gt; mp;
    for (int i = 0, j = b % p; i &lt; k; i &#43;&#43;) {
        mp[j] = i;
        j = j * a % p;
    }
    int ak = 1;
    for (int i = 0; i &lt; k; i &#43;&#43;) ak = ak * a % p;
    for (int i = 1, j = ak; i &lt;= k; i &#43;&#43;) {
        if (mp.count(j)) return i * k - mp[j];
        j = j * ak % p;
    }
    return -1;
}
signed main() {
    while (cin &gt;&gt; a &gt;&gt; p &gt;&gt; b, a || b || p) {
        int res = bsgs(a, b, p);
        if (res == -1) {
            cout &lt;&lt; &#34;No Solution&#34; &lt;&lt; endl;
        } else {
            cout &lt;&lt; res &lt;&lt; endl;
        }
    }
}
```

## $\text{exBSGS}:$
给定正整数 $a,b,p$ ，$a,p$ 不一定互质，求满足

$$a^{x} \equiv b(\text{ mod } p)$$

的**最小非负整数** $x$

分情况来看，如果 $x=0$ 时，满足 $1 \equiv b(\text{ mod } p)$，答案就是 $0$

如果 $\gcd(a,p)=1$，那么就直接用朴素 $\text{BSGS}$ 算法

如果 $\gcd(a,p) &gt; 1$，由裴蜀定理得

$$a^x&#43;kp = b$$

设 $\gcd(a,p)=d$，如果$d\nmid p$ 那么无解，否则，等式两边同除 $d$ 得

$$\frac{a}{d}a^{x-1}&#43;k\frac{p}{d} = \frac{b}{d}$$

等价于同余方程：

$$\frac{a}{d}a^{x-1} \equiv\frac{b}{d} (\text{ mod }\frac{p}{d})$$

由于 $\gcd(\frac{a}{d},\frac{p}{d})=1$ ，所以把 $\frac{a}{d}$ 移到等式右边，就是乘 $\frac{a}{d}$ 的逆元

$$a^{x-1} \equiv\frac{b}{d}  (\frac{a}{d})^{-1}(\text{ mod }\frac{p}{d})$$

用新变量替换：

$$
(a\&#39;)^{x} \equiv b\&#39; \pmod  {p\&#39;}
$$


由于 $x \ge 1$，这样就可以递归地用 $\text{BSGS}$ 求解了，新的解就为 $x&#43;1$，逆元可以用扩展欧几里得算法求解。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
int a, b, p;
int exgcd(int a, int b, int&amp; x, int&amp; y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}
int bsgs(int a, int b, int p) {
    if (1 % p == b % p) return 0;
    int k = sqrt(p) &#43; 1;
    unordered_map&lt;int, int&gt; mp;
    for (int i = 0, j = b % p; i &lt; k; i &#43;&#43;) {
        mp[j] = i;
        j = j * a % p;
    }
    int ak = 1;
    for (int i = 0; i &lt; k; i &#43;&#43;) ak = ak * a % p;
    for (int i = 1, j = ak; i &lt;= k; i &#43;&#43;) {
        if (mp.count(j)) return i * k - mp[j];
        j = j * ak % p;
    }
    return -1;
}
int exbsgs(int a, int b, int p) {
    b = (b % p &#43; p) % p;
    if (1 % p == b % p) return 0;
    int x, y;
    int d = exgcd(a, p, x, y);
    if (d &gt; 1) {
        if (b % d) return -2e9;
        exgcd(a / d, p / d, x, y);
        return exbsgs(a, b / d * x % (p / d), p / d) &#43; 1;
    }
    return bsgs(a, b, p);
}
signed main() {
    while (cin &gt;&gt; a &gt;&gt; p &gt;&gt; b, a || b || p) {
        int res = exbsgs(a, b, p);
        if (res &lt; 0) {
            cout &lt;&lt; &#34;No Solution&#34; &lt;&lt; endl;
        } else {
            cout &lt;&lt; res &lt;&lt; endl;
        }
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/bsgsexbsgs/  

