# [2023杭电多校5 1002] GCD Magic


[原题链接](https://acm.hdu.edu.cn/showproblem.php?pid=7325)

**题意**
求
$$
\sum_{i = 1} ^ {n} \sum_{j = 1} ^ {n}[\gcd(2 ^ i - 1, 2 ^ j - 1)] ^ k
$$

对 $998\,244\,353$ 取模。

$1 \le n \le 10 ^ 9, 0 \le k \le 10$

**分析：**

易证 $\gcd(2 ^ i - 1, 2 ^ j - 1) = 2 ^ {\gcd(i, j)} - 1$，代入得

$$
\sum_{i = 1} ^ {n} \sum_{j = 1} ^ {n}(2^{\gcd(i, j)} - 1) ^ k
$$

常规枚举 $d$

$$
\sum_{d = 1} ^ {n} \sum_{i = 1} ^ {n} \sum_{j = 1} ^ {n}(2^d - 1) ^ k [\gcd(i,j) = d]
$$

把 $d$ 拿到求和上界

$$
\sum_{d = 1} ^ {n} \sum_{i = 1} ^ { \lfloor \frac{n}{d} \rfloor } \sum_{j = 1} ^ { \lfloor \frac{n}{d} \rfloor}\left(2^d - 1\right) ^ k[\gcd(i,j) = 1]
$$

我们知道 $\sum\limits_{i = 1} ^ {n}\sum\limits_{j = 1} ^ {n}[\gcd(i,j) = 1] = 2\sum\limits_{i = 1} ^ {n}\varphi(i) - 1$，($-1$ 在 $\sum$ 外面)，代入得

$$
\sum_{d = 1} ^ {n}\left(2^d - 1\right) ^ k \left(2\sum_{i = 1} ^ { \lfloor \frac{n}{d} \rfloor}\varphi(i) - 1\right)
$$

考虑整除分块，后面欧拉函数前缀和可以用杜教筛，那么考虑如何快速求 $\left(2^d - 1\right) ^ k$ 的前缀和。记

$$
S(n) = \sum_{i = 1} ^ {n}(2 ^ i - 1) ^ k 
$$

将 $\left(2^i - 1\right) ^ k$ 二项式展开

$$
S(n) = \sum_{i = 1} ^ {n}\sum_{j = 0} ^ {k} \binom{k}{j} \times 2 ^ {i\times j} \times (-1) ^ {k - j}
$$

交换求和顺序

$$
\sum_{j = 0} ^ {k} \binom{k}{j}\times (-1) ^ {k - j} \sum_{i = 1} ^ {n}  (2 ^ {j}) ^ {i}
$$

其中 $\sum\limits_{i = 1} ^ {n}  (2 ^ {j}) ^ {i}$ 用等比数列求和公式

$$
\sum_{j = 0} ^ {k} \binom{k}{j}\times (-1) ^ {k - j} \times 2 ^ j \times \frac{2 ^ {j \times n} - 1}{2 ^ j - 1}
$$

这样求 $S(n)$ 就变为 $O(k \log n)$ 了，注意特判 $j = 0$ 和欧拉降幂
时间复杂度 $O(n ^ {\frac{2}{3}} &#43; k \sqrt n \log n)$
## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5, mod = 998244353;
int n, k, primes[N], euler[N], cnt, sum[N], fact[N], infact[N];
bool st[N];
unordered_map&lt;int, int&gt; mp;
void get_eulers(int n) {
    euler[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; i * primes[j] &lt;= n; j &#43;&#43;) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                euler[t] = primes[j] * euler[i];
                break;
            }
            euler[t] = (primes[j] - 1) * euler[i];
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        sum[i] = (sum[i - 1] &#43; euler[i]) % mod;
    }
}
int qmi(int a, int b) {
    int res = 1;
    while (b) {
        if (b &amp; 1) res = res * a % mod;
        a = a * a % mod;
        b &gt;&gt;= 1;
    }
    return res;
}
int C(int m, int n) {
    return fact[m] * infact[m - n] % mod * infact[n] % mod;
}
int Sum_euler(int n) {
    if (n &lt; N) return sum[n];
    if (mp[n]) return mp[n];
    int res = n * (n &#43; 1) / 2 % mod;
    for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res = (res - Sum_euler(n / l) * (r - l &#43; 1) % mod &#43; mod) % mod;
    }
    return mp[n] = res;
}
int Sum(int n) {
    int res = 0;
    for (int j = 0; j &lt;= k; j &#43;&#43;) {
        int f = (k - j) % 2 == 1 ? mod - 1 : 1;
        if (!j) {
            res = (res &#43; n * f % mod) % mod;
        } else {
            int omod = mod - 1;
            int t = (qmi(2, j * n % omod) - 1 &#43; mod) % mod;
            int S = C(k, j) * f % mod % mod * qmi(2, j) % mod * t % mod;
            int inv = (qmi(2, j) - 1 &#43; mod) % mod;
            S = S * qmi(inv, mod - 2) % mod;
            res = (res &#43; S) % mod;
        }
    }
    return res;
}
signed main() {
    get_eulers(N - 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i &lt; N; i &#43;&#43;) fact[i] = fact[i - 1] * i % mod;
    infact[N - 1] = qmi(fact[N - 1], mod - 2);
    for (int i = N - 2; i; i --) infact[i] = infact[i &#43; 1] * (i &#43; 1) % mod;
    int T;
    cin &gt;&gt; T;
    while (T --) {
        int res = 0;
        cin &gt;&gt; n &gt;&gt; k;
        for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
            r = n / (n / l);
            res = (res &#43; (2 * Sum_euler(n / l) % mod - 1 &#43; mod) % mod * (Sum(r) - Sum(l - 1) &#43; mod) % mod) % mod;
        }
        cout &lt;&lt; res &lt;&lt; &#34;\n&#34;;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2023%E6%9D%AD%E7%94%B5%E5%A4%9A%E6%A0%A15-1002-gcd-magic/  

