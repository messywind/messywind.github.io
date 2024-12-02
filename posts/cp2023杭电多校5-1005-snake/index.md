# [2023杭电多校5 1005] Snake


[原题链接](https://acm.hdu.edu.cn/showproblem.php?pid=7328)

**题意**

有 $n$ 个标号为 $1,2,\cdots,n$ 的球，放到 $m$ 个无标号盒子 (盒内顺序有标号)，且每个盒子球数不超过 $k$，求方案数对 $998\,244\,353$ 取模。

$1 \le m,k \le n \le 10 ^ 6$

**分析：**

考虑每个盒子内球的生成函数 $\sum\limits_{i = 1} ^ {k}x ^ i$，那么 $m$ 个盒子的生成函数就为 $\left( \sum\limits_{i = 1} ^ {k}x ^ i\right) ^ m$，那么方案数就为第 $n$ 项系数。

由于球带标号，所以需要对答案全排列，也就是乘 $n!$，又由于盒子不带标号，所以要对答案除 $m!$，那么答案为 

$$
\frac{n!}{m!} \times [x ^ n]\left( \sum\limits_{i = 1} ^ {k}x ^ i\right) ^ m
$$

$10 ^ 6$ 用多项式快速幂会超时，考虑

$$
\left( \sum\limits_{i = 1} ^ {k}x ^ i\right) ^ m= x ^ m \left( \sum\limits_{i = 0} ^ {k - 1}x ^ i\right) ^ m = x ^ m \frac{(1 -x ^ k)^m}{(1 - x) ^ m}
$$

转为求 $[x^{n - m}] \dfrac{(1 -x ^ k)^m}{(1 - x) ^ m}$ 其中

$$
(1 - x ^ k) ^ m = \sum_{i = 0} ^ {m}\binom{m}{i} \times (-1) ^ i \times x ^ {i \times k}
$$

$$
\frac{1}{(1 - x) ^ m} = \sum_{i = 0} ^ {\infty} \binom{m - 1 &#43; i}{m - 1} \times x ^ i
$$

于是枚举第一个式子的 $i$，那么只需要求第二个式子的 $n - m - i \times k$ 项系数即可。
## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
using i64 = long long;
constexpr int mod = 998244353;
template&lt;class T&gt;
T power(T a, int b) {
    T res = 1;
    for (; b; b /= 2, a *= a) {
        if (b % 2) {
            res *= a;
        }
    }
    return res;
}
template&lt;int mod&gt;
struct ModInt {
    int x;
    ModInt() : x(0) {}
    ModInt(i64 y) : x(y &gt;= 0 ? y % mod : (mod - (-y) % mod) % mod) {}
    ModInt &amp;operator&#43;=(const ModInt &amp;p) {
        if ((x &#43;= p.x) &gt;= mod) x -= mod;
        return *this;
    }
    ModInt &amp;operator-=(const ModInt &amp;p) {
        if ((x &#43;= mod - p.x) &gt;= mod) x -= mod;
        return *this;
    }
    ModInt &amp;operator*=(const ModInt &amp;p) {
        x = (int)(1LL * x * p.x % mod);
        return *this;
    }
    ModInt &amp;operator/=(const ModInt &amp;p) {
        *this *= p.inv();
        return *this;
    }
    ModInt operator-() const {
        return ModInt(-x);
    }
    ModInt operator&#43;(const ModInt &amp;p) const {
        return ModInt(*this) &#43;= p;
    }
    ModInt operator-(const ModInt &amp;p) const {
        return ModInt(*this) -= p;
    }
    ModInt operator*(const ModInt &amp;p) const {
        return ModInt(*this) *= p;
    }
    ModInt operator/(const ModInt &amp;p) const {
        return ModInt(*this) /= p;
    }
    bool operator==(const ModInt &amp;p) const {
        return x == p.x;
    }
    bool operator!=(const ModInt &amp;p) const {
        return x != p.x;
    }
    ModInt inv() const {
        int a = x, b = mod, u = 1, v = 0, t;
        while (b &gt; 0) {
            t = a / b;
            swap(a -= t * b, b);
            swap(u -= t * v, v);
        }
        return ModInt(u);
    }
    ModInt pow(i64 n) const {
        ModInt res(1), mul(x);
        while (n &gt; 0) {
            if (n &amp; 1) res *= mul;
            mul *= mul;
            n &gt;&gt;= 1;
        }
        return res;
    }
    friend ostream &amp;operator&lt;&lt;(ostream &amp;os, const ModInt &amp;p) {
        return os &lt;&lt; p.x;
    }
    friend istream &amp;operator&gt;&gt;(istream &amp;is, ModInt &amp;a) {
        i64 t;
        is &gt;&gt; t;
        a = ModInt&lt;mod&gt;(t);
        return (is);
    }
    int val() const {
        return x;
    }
    static constexpr int val_mod() {
        return mod;
    }
};
using Z = ModInt&lt;mod&gt;;
vector&lt;Z&gt; fact, infact;
void init(int n) {
    fact.resize(n &#43; 1), infact.resize(n &#43; 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        fact[i] = fact[i - 1] * i;
    }
    infact[n] = fact[n].inv();
    for (int i = n; i; i --) {
        infact[i - 1] = infact[i] * i;
    }
}
Z C(int n, int m) {
    if (n &lt; 0 || m &lt; 0 || n &lt; m) return Z(0);
    return fact[n] * infact[n - m] * infact[m];
}
void solve() {
    int n, m, k;
    cin &gt;&gt; n &gt;&gt; m &gt;&gt; k;
    Z ans;
    for (int i = 0; i &lt;= m; i &#43;&#43;) {
        Z f = i &amp; 1 ? Z(-1) : Z(1);
        ans &#43;= f * C(m, i) * C(n - k * i - 1, m - 1);
    }
    cout &lt;&lt; ans * fact[n] / fact[m] &lt;&lt; &#34;\n&#34;;
}
signed main() {
    init(1e6);
    cin.tie(0) -&gt; sync_with_stdio(0);
    int T;
    cin &gt;&gt; T;
    while (T --) {
        solve();
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2023%E6%9D%AD%E7%94%B5%E5%A4%9A%E6%A0%A15-1005-snake/  

