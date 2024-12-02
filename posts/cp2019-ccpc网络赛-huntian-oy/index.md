# [2019 CCPC网络赛] Huntian Oy


[原题链接](http://acm.hdu.edu.cn/showproblem.php?pid=6706)

**题意**

$T$ 组输入，给定 $n,a,b$  求

$$f(n,a,b)=\sum_{i=1} ^{n}\sum_{j=1}^{i} \gcd(i^a-j^a, i ^ b - j ^ b)[\gcd(i,j)=1]$$

对 $10^9&#43;7$ 取模，$1 \le n,a,b \le10^9$ $a,b$ 互质

**分析：**

$\gcd$ 的性质：

$$\gcd(a ^ m - b ^ m, a ^ n - b ^ n)=a^{\gcd(n,m)}-b^{\gcd(n,m)}$$

条件：$\gcd(a,b)=1$

套用结论带入原式

$$\sum_{i=1} ^{n}\sum_{j=1}^{i}( i -j)[\gcd(i,j)=1]$$

也就是

$$\sum_{i=1} ^{n}\sum_{j=1}^{i}i[\gcd(i,j)=1]-\sum_{i=1} ^{n}\sum_{j=1}^{i}j[\gcd(i,j)=1]$$

前半部分$\sum\limits_{i=1} ^{n}i\sum\limits_{j=1}^{i}[\gcd(i,j)=1]$是欧拉函数的定义 $\varphi(i)$

$$\sum_{i=1} ^{n}i \varphi(i)-\sum_{i=1} ^{n}\sum_{j=1}^{i}j[\gcd(i,j)=1]$$

看后半部分，这不就是疯狂LCM那个题吗，所以是 $\dfrac{i\varphi(i)&#43;1}{2}$

$$\sum_{i=1} ^{n}i \varphi(i)-\sum_{i=1} ^{n}\frac{i\varphi(i)&#43;1}{2}$$

整理得

$$\sum_{i=1} ^{n}\frac{i\varphi(i)-1}{2}$$

用杜教筛，令 $f(x)=x\varphi(x)$，令 $S(n)=\sum\limits_{i=1} ^{n}f(i)$

$$g(1)S(n) = \sum_{i=1} ^{n}f*g-\sum_{i=2}^{n} g(i)S(\lfloor\frac{n}{i} \rfloor)$$

$$f*g=\sum_{d \mid n} d\varphi(d)g(\frac{n}{d})$$

要消掉一个 $d$，所以令 $g(x)=x$

$$f*g=n\sum_{d \mid n} \varphi(d)=n \cdot \varphi * I$$

因为 $\varphi*I=Id$，所以 $f * g=n^2$，杜教筛原式为

$$S(n) = \sum_{i=1} ^{n}i^2-\sum_{i=2}^{n} iS(\lfloor\frac{n}{i} \rfloor)$$

$\sum\limits_{i=1}^{n}i^2=\dfrac{n(n&#43;1)(2n&#43;1)}{6}$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5, mod = 1e9 &#43; 7;
int T, a, b, n, primes[N], euler[N], cnt, sum[N], inv2, inv6;
bool st[N];
unordered_map&lt;int, int&gt; mp;
int qmi(int a, int b) {
    int res = 1;
    while (b) {
        if (b &amp; 1) res = res * a % mod;
        a = a * a % mod;
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
        for (int j = 0; primes[j] &lt;= n / i; j &#43;&#43;) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                euler[t] = primes[j] * euler[i];
                break;
            }
            euler[t] = (primes[j] - 1) * euler[i];
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = (sum[i - 1] &#43; i * euler[i] % mod) % mod;
}
int s2(int n) {
    return n * (n &#43; 1) % mod * (2 * n &#43; 1) % mod * inv6 % mod;
}
int Sum(int n) {
    if (n &lt; N) return sum[n];
    if (mp[n]) return mp[n];
    int res = s2(n);
    for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res = (res - (r - l &#43; 1) * (l &#43; r) % mod * inv2 % mod * Sum(n / l) % mod &#43; mod) % mod;
    }
    return mp[n] = res;
}
signed main() {
    inv2 = qmi(2, mod - 2);
    inv6 = qmi(6, mod - 2);
    get_eulers(N - 1);
    cin &gt;&gt; T;
    while (T --) {
        cin &gt;&gt; n &gt;&gt; a &gt;&gt; b;
        cout &lt;&lt; (Sum(n) - 1 &#43; mod) % mod * inv2 % mod &lt;&lt; endl;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2019-ccpc%E7%BD%91%E7%BB%9C%E8%B5%9B-huntian-oy/  

