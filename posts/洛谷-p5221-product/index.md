# [洛谷 P5221] Product


[原题链接](https://www.luogu.com.cn/problem/P5221)

**题意**

求

$$\prod_{i=1}^{n}\prod^{n}_{j=1}\frac{\text{lcm}(i,j)}{\gcd(i,j)}$$

对 $104857601$ 取模

$1 \le n  \le 10^6$

**分析：**

将 $\text{lcm}(i,j)=\dfrac{i \cdot j}{\gcd(i,j)}$ 替换原式得

$$\prod_{i=1}^{n}\prod^{n}_{j=1}\frac{i \cdot j}{\gcd(i,j)^2}$$

对 $\gcd(i,j)^2$ 求逆元

$$\prod_{i=1}^{n}\prod^{n}_{j=1}i \cdot j \cdot \gcd(i,j)^{-2}$$

前后分开来看

$$\prod_{i=1}^{n}\prod_{j=1} ^ {n} i \cdot j\prod_{i=1}^{n}\prod^{n}_{j=1}\gcd(i,j)^{-2}$$

前面 $\prod\limits_{i=1}^{n}\prod\limits_{j=1} ^{n}i \cdot j=\prod\limits_{i=1}^{n}i^n\prod\limits_{j=1} ^{n}j=\prod\limits_{i=1}^{n}i^n\cdot n!=(n!)^{n}\prod\limits_{i=1}^{n}i^n=(n!)^{2n}$

后半部分 先看原来的值再求逆元

$$\prod_{i=1}^{n}\prod^{n}_{j=1}\gcd(i,j)^{2}$$

先把平方拿出去看里面

$$\prod_{i=1}^{n}\prod^{n}_{j=1}\gcd(i,j)$$

枚举 $\gcd(i,j)$

$$\prod_{d=1}^{n}d^{\sum\limits_{i=1}^{n}\sum\limits^{n}_{j=1}[\gcd(i,j)=d]}$$

指数部分 因为 $\sum\limits_{i=1}^{n}\sum\limits_{i=1}^{n}[\gcd(i,j)=1]=2\sum\limits_{i=1}^{n}\varphi(i)-1$ 所以

$$\prod_{d=1}^{n}d^{2\sum\limits_{i=1}^{\lfloor\frac{n}{d}
\rfloor}\varphi(i)-1}$$

最后答案为

$$(n!)^{2n}\cdot (\prod_{d=1}^{n}d^{2\sum\limits_{i=1}^{\lfloor\frac{n}{d}
\rfloor}\varphi(i)-1})^{-2}$$

## 代码(毒瘤出题人卡空间)：
```cpp
#include &lt;cstdio&gt;
using namespace std;
const int N = 1e6 &#43; 5, mod = 104857601;
int n, fact = 1, euler[N], primes[N], cnt, res = 1, inv = 1;
bool st[N];
int qmi(int a, int b) {
    int res = 1;
    while (b) {
        if (b &amp; 1) res = 1ll * res * a % mod;
        a = 1ll * a * a % mod;
        b &gt;&gt;= 1;
    }
    return res;
}
void get_eulers(int n) {
    euler[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; primes[j] * i &lt;= n; j &#43;&#43;) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                euler[t] = euler[i] * primes[j];
                break;
            }
            euler[t] = euler[i] * (primes[j] - 1);
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) euler[i] = (euler[i - 1] &#43; 2 * euler[i]) % (mod - 1);
}
signed main() {
    scanf(&#34;%d&#34;, &amp;n);
    get_eulers(n);
    for (int i = 1; i &lt;= n; i &#43;&#43;) fact = 1ll * fact * i % mod;
    res = qmi(fact, 2 * n);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        inv = (1ll * inv * qmi(i, euler[n / i] - 1)) % mod;
    }
    printf(&#34;%d\n&#34;, 1ll * res * qmi(1ll * inv * inv % mod, mod - 2) % mod);
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E6%B4%9B%E8%B0%B7-p5221-product/  

