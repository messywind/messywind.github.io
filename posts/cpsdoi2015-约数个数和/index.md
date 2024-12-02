# [SDOI2015] 约数个数和


[题目链接](https://www.luogu.com.cn/problem/P3327)

**题意：**

设 $d(x)$ 为 $x$ 的约数个数，求

$$\sum_{i=1}^{N}\sum_{j=1}^{M} d(ij)$$

**分析：**

首先 $i=p_1 ^{\alpha_1}\cdots p_k^{\alpha_k},j=p_1 ^{\beta_1}\cdots p_k^{\beta_k},i×j=p_1 ^{\alpha_1&#43;\beta_1}\cdots p_k^{\alpha_k&#43;\beta_k}$

约数个数和为

$$d(ij)=(\alpha_1&#43;\beta_1&#43;1)\cdots(\alpha_k&#43;\beta_k&#43;1)=\alpha_1\beta_2&#43;\cdots &#43; 1$$

所以$$d(ij)=\sum_{x|i}\sum_{y|j}[\gcd(x,y)=1]$$

所以原式为

$$\sum_{i=1}^{N}\sum_{j=1}^{M}\sum_{x|i}\sum_{y|j}[\gcd(x,y)=1]$$

设 $g(n)$ 为：

$$g(n)=\sum_{i=1}^{N}\sum_{j=1}^{M}\sum_{x|i}\sum_{y|j}[n\mid\gcd(x,y)]$$

设 $f(n)$ 为：

$$\sum_{i=1}^{N}\sum_{j=1}^{M}\sum_{x|i}\sum_{y|j}[\gcd(x,y)=n]$$

则有

$$g(n)=\sum_{n\mid d} f(d)$$

那么就可以莫比乌斯反演了

$$f(n)=\sum_{n\mid d}\mu(\frac{d}{n})g(d)$$

交换 $g(n)$ 的求和次序

由于 $x,y$ 分别是 $i,j$ 的约数，所以可以先枚举 $x,y$

那么后面的 $[n \mid \gcd(x,y)]$ 与 $i,j$ 无关不需要考虑，所以只需要计算 $i$ 中有多少 $x$ 的倍数， $j$ 中有多少 $y$ 的倍数，所以是 $\lfloor \frac{N}{x} \rfloor \lfloor \frac{M}{y} \rfloor$

所以式子就变为了

$$\sum_{x=1}^{N}\sum_{y=1}^{M}\lfloor \frac{N}{x} \rfloor \lfloor \frac{M}{y} \rfloor[n \mid \gcd(x,y)]$$

$[n \mid \gcd(x,y)]$ 只需要关心 $n$ 的倍数即可，那么就是 

$$x&#39;=\frac{x}{n},y&#39;=\frac{y}{n}$$

替换得

$$\sum_{x&#39;=1}^{\frac{N}{n}}\sum_{y&#39;=1}^{\frac{M}{n}} \lfloor\frac{N}{nx&#39;} \rfloor \lfloor\frac{M}{ny&#39;} \rfloor$$

设

$$N&#39;=\frac{N}{n},M&#39;=\frac{M}{n}$$

则

$$\sum_{x&#39;=1}^{N&#39;}\sum_{y&#39;=1}^{M&#39;} \lfloor\frac{N&#39;}{x&#39;} \rfloor \lfloor\frac{M&#39;}{y&#39;} \rfloor$$

假设有二重积分

$$\iint_{D}f(x,y)\text{d}x\text{d}y$$

当区域 $D$ 为矩形区域时，可以转为二次积分

$$\int_{D_1}f(x,y)\text{d}x\int_{D_2}f(x,y)\text{d}y$$

那么原式可以变为

$$\sum_{x&#39;=1}^{N&#39;}\lfloor\frac{N&#39;}{x&#39;} \rfloor \sum_{y&#39;=1}^{M&#39;}\lfloor\frac{M&#39;}{y&#39;} \rfloor$$

设

$$h(x)=\sum_{i=1}^{x}\lfloor\frac{x}{i} \rfloor$$

那么

$$f(n)=\sum_{n\mid d}\mu(\frac{d}{n})g(d)$$

答案为

$$f(1)=\sum_{d=1}^{N}\mu(d)g(d)$$

带入 $g(d)$

$$\sum_{i=1}^{\min(N,M)}\mu(i)h(\lfloor \frac{N}{i} \rfloor)h(\lfloor \frac{M}{i} \rfloor)$$

可以用整除分块计算，$h(x)$ 也同样可以用整除分块预处理。

## 代码：
```cpp
#include &lt;stdio.h&gt;
#include &lt;algorithm&gt;
#define int long long
using namespace std;
const int N = 5e4 &#43; 5;
int T, n, m, primes[N], mobius[N], cnt, sum[N], h[N];
bool st[N];
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
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = sum[i - 1] &#43; mobius[i];
}
void get_h(int n) {
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int l = 1, r; l &lt;= i; l = r &#43; 1) {
            r = min(i, i / (i / l));
            h[i] &#43;= (r - l &#43; 1) * (i / l);
        }
    }
}
signed main() {
    get_mobius(N - 1), get_h(N - 1);
    scanf(&#34;%lld&#34;, &amp;T);
    while (T --) {
        scanf(&#34;%lld%lld&#34;, &amp;n, &amp;m);
        int res = 0, k = min(n, m);
        for (int l = 1, r; l &lt;= k; l = r &#43; 1) {
            r = min(k, min(n / (n / l), m / (m / l)));
            res &#43;= (sum[r] - sum[l - 1]) * h[n / l] * h[m / l];
        }
        printf(&#34;%lld\n&#34;, res);
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cpsdoi2015-%E7%BA%A6%E6%95%B0%E4%B8%AA%E6%95%B0%E5%92%8C/  

