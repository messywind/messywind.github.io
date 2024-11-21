# [2022 牛客多校4 C] Easy Counting Problem


[原题链接](https://ac.nowcoder.com/acm/contest/33189/C)

**题意**

给定一个正整数 $w$ 及 $w$ 个数 $c_0, c_1, \cdots,c_{w - 1}$

$q$ 组询问，每次询问给定一个正整数 $n$，计算有多少个长度为 $n$ 的字符串满足：

- 每个字符只能取数字 $0 \sim w - 1$
- 数字 $i$ 至少出现 $c_i$ 次

对 $998 \, 244 \, 353$ 取模。

$2 \le w \le 10, 1 \le c_i \le 5 \times 10 ^ 4, \sum\limits_{i = 0} ^ {w - 1}c_i \le 5 \times 10 ^ 4$

$1 \le q \le 300, 1 \le n \le 10 ^ 7$

**分析：**

首先我们可以写出每个数字 $i$ 的 $\textbf{EGF}$
$$
\sum_{j = c_i} ^ {\infty} \frac{x ^ j}{j!}
$$
那么每个数字的 $\textbf{EGF}$ 做乘积表示满足条件的所有长度的字符串的方案数
$$
\prod_{i = 0} ^ {w - 1}\sum_{j = c_i} ^ {\infty} \frac{x ^ j}{j!}
$$
可以把和式用前缀和相减拆一下 $\sum\limits_{j = c_i} ^ {\infty} \dfrac{x ^ j}{j!} = \sum\limits_{j = 0} ^ {\infty} \dfrac{x ^ j}{j!} - \sum\limits_{j = 0} ^ {c_i - 1} \dfrac{x ^ j}{j!}$，发现第一项为 $e ^ x$，故答案为
$$
\prod_{i = 0} ^ {w - 1}(e ^ x - \sum_{j = 0} ^ {c_i - 1} \frac{x ^ j}{j!})
$$
由于 $w \le 10$，所以考虑暴力展开式子，做换元 $e ^ x \rightarrow y$

在展开式子的过程中，假设当前的多项式为 $f = A_0 &#43; A_1 y &#43; A_2y ^ 2 &#43; A_3y^3 &#43; \cdots$，那么新遇到一个多项式 $(y &#43; g_i)$ 其中 $g_i = -\sum\limits_{j = 0} ^ {c_i - 1} \dfrac{x ^ j}{j!}$， 则结果变为 $f * y &#43; f * g_i$ ($\*$ 表示多项式卷积)，前一项为 $A_0y &#43; A_1 y ^ 2 &#43; A_2y ^ 3 &#43; A_3y^4 &#43; \cdots$，那么后一项是 $f$ 的每一项系数与 $g_i$ 的多项式卷积，为 $A_0 * g_i &#43; (A_1 * g_i) y &#43; (A_2 * g_i)y ^ 2 &#43; (A_3 * g_i)y^3 &#43; \cdots$，那么答案就为

$$
A_0 \* g_i &#43; (A_0 &#43; A_1 \* g_i)y &#43; (A_1 &#43; A_2 \* g_i)y ^ 2 &#43; (A_2 &#43; A_3 \* g_i)y ^ 3 &#43; \cdots
$$

这样就预处理好了总答案，现考虑回答每组询问，我们知道最后的答案是形如 $\sum\limits_{i = 0} ^ {w - 1} e ^ {ix} F_i(x)$ 的多项式，我们需要知道每一项的第 $n$ 项系数，由于 $\sum\limits_{i = 0} ^ {w - 1}c_i \le 5 \times 10 ^ 4$，我们可以在询问里对于每个 $i$ 直接枚举 $F_i(x)$ 的项数，设当前枚举到了第 $j$ 项，那么需要在 $e ^ {ix}$ 中取出第 $n - j$ 项，也就是 $e ^ {ix} = 1 &#43; \dfrac{(ix) ^ 1}{1!} &#43; \dfrac{(ix) ^ 2}{2!} &#43; \dfrac{(ix) ^ 3}{3!} &#43; \cdots$ 的第 $n - j$ 项，为 $\dfrac{i ^ {n - j}}{(n - j)!}$

那么答案就为
$$
n! \times \sum_{i = 0} ^ {w} \sum_{j = 0} ^ {\min(n, |F_i(x)|)} [x ^ j] F_i(x) \times \frac{i ^ {n - j}}{(n - j)!}
$$

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
using i64 = long long;
constexpr int mod = 998244353;
int norm(int x) {
    if (x &lt; 0) {
        x &#43;= mod;
    }
    if (x &gt;= mod) {
        x -= mod;
    }
    return x;
}
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
struct Z {
    int x;
    Z(int x = 0) : x(norm(x)) {}
    int val() const {
        return x;
    }
    Z operator-() const {
        return Z(norm(mod - x));
    }
    Z inv() const {
        assert(x != 0);
        return power(*this, mod - 2);
    }
    Z &amp;operator*=(const Z &amp;rhs) {
        x = i64(x) * rhs.x % mod;
        return *this;
    }
    Z &amp;operator&#43;=(const Z &amp;rhs) {
        x = norm(x &#43; rhs.x);
        return *this;
    }
    Z &amp;operator-=(const Z &amp;rhs) {
        x = norm(x - rhs.x);
        return *this;
    }
    Z &amp;operator/=(const Z &amp;rhs) {
        return *this *= rhs.inv();
    }
    friend Z operator*(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res *= rhs;
        return res;
    }
    friend Z operator&#43;(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res &#43;= rhs;
        return res;
    }
    friend Z operator-(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res -= rhs;
        return res;
    }
    friend Z operator/(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res /= rhs;
        return res;
    }
    friend istream &amp;operator&gt;&gt;(istream &amp;is, Z &amp;a) {
        i64 v;
        is &gt;&gt; v;
        a = Z(v);
        return is;
    }
    friend ostream &amp;operator&lt;&lt;(ostream &amp;os, const Z &amp;a) {
        return os &lt;&lt; a.val();
    }
};
vector&lt;int&gt; rev;
vector&lt;Z&gt; roots{0, 1};
void dft(vector&lt;Z&gt; &amp;a) {
    int n = a.size();
    if (int(rev.size()) != n) {
        int k = __builtin_ctz(n) - 1;
        rev.resize(n);
        for (int i = 0; i &lt; n; i &#43;&#43;) {
            rev[i] = rev[i &gt;&gt; 1] &gt;&gt; 1 | (i &amp; 1) &lt;&lt; k;
        }
    }
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        if (rev[i] &lt; i) {
            swap(a[i], a[rev[i]]);
        }
    }
    if (int(roots.size()) &lt; n) {
        int k = __builtin_ctz(roots.size());
        roots.resize(n);
        while ((1 &lt;&lt; k) &lt; n) {
            Z e = power(Z(3), (mod - 1) &gt;&gt; (k &#43; 1));
            for (int i = 1 &lt;&lt; (k - 1); i &lt; (1 &lt;&lt; k); i &#43;&#43;) {
                roots[i &lt;&lt; 1] = roots[i];
                roots[i &lt;&lt; 1 | 1] = roots[i] * e;
            }
            k &#43;&#43;;
        }
    }
    for (int k = 1; k &lt; n; k *= 2) {
        for (int i = 0; i &lt; n; i &#43;= 2 * k) {
            for (int j = 0; j &lt; k; j &#43;&#43;) {
                Z u = a[i &#43; j], v = a[i &#43; j &#43; k] * roots[k &#43; j];
                a[i &#43; j] = u &#43; v, a[i &#43; j &#43; k] = u - v;
            }
        }
    }
}
void idft(vector&lt;Z&gt; &amp;a) {
    int n = a.size();
    reverse(a.begin() &#43; 1, a.end());
    dft(a);
    Z inv = (1 - mod) / n;
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        a[i] *= inv;
    }
}
struct Poly {
    vector&lt;Z&gt; a;
    Poly() {}
    Poly(const vector&lt;Z&gt; &amp;a) : a(a) {}
    Poly(const initializer_list&lt;Z&gt; &amp;a) : a(a) {}
    int size() const {
        return a.size();
    }
    void resize(int n) {
        a.resize(n);
    }
    Z operator[](int idx) const {
        if (idx &lt; size()) {
            return a[idx];
        } else {
            return 0;
        }
    }
    Z &amp;operator[](int idx) {
        return a[idx];
    }
    Poly mulxk(int k) const {
        auto b = a;
        b.insert(b.begin(), k, 0);
        return Poly(b);
    }
    Poly modxk(int k) const {
        k = min(k, size());
        return Poly(vector&lt;Z&gt;(a.begin(), a.begin() &#43; k));
    }
    Poly divxk(int k) const {
        if (size() &lt;= k) {
            return Poly();
        }
        return Poly(vector&lt;Z&gt;(a.begin() &#43; k, a.end()));
    }
    friend Poly operator&#43;(const Poly &amp;a, const Poly &amp;b) {
        vector&lt;Z&gt; res(max(a.size(), b.size()));
        for (int i = 0; i &lt; int(res.size()); i &#43;&#43;) {
            res[i] = a[i] &#43; b[i];
        }
        return Poly(res);
    }
    friend Poly operator-(const Poly &amp;a, const Poly &amp;b) {
        vector&lt;Z&gt; res(max(a.size(), b.size()));
        for (int i = 0; i &lt; int(res.size()); i &#43;&#43;) {
            res[i] = a[i] - b[i];
        }
        return Poly(res);
    }
    friend Poly operator*(Poly a, Poly b) {
        if (a.size() == 0 || b.size() == 0) {
            return Poly();
        }
        int sz = 1, tot = a.size() &#43; b.size() - 1;
        while (sz &lt; tot) {
            sz *= 2;
        }
        a.a.resize(sz);
        b.a.resize(sz);
        dft(a.a);
        dft(b.a);
        for (int i = 0; i &lt; sz; i &#43;&#43;) {
            a.a[i] = a[i] * b[i];
        }
        idft(a.a);
        a.resize(tot);
        return a;
    }
    friend Poly operator*(Z a, Poly b) {
        for (int i = 0; i &lt; int(b.size()); i &#43;&#43;) {
            b[i] *= a;
        }
        return b;
    }
    friend Poly operator*(Poly a, Z b) {
        for (int i = 0; i &lt; int(a.size()); i &#43;&#43;) {
            a[i] *= b;
        }
        return a;
    }
    Poly &amp;operator&#43;=(Poly b) {
        return (*this) = (*this) &#43; b;
    }
    Poly &amp;operator-=(Poly b) {
        return (*this) = (*this) - b;
    }
    Poly &amp;operator*=(Poly b) {
        return (*this) = (*this) * b;
    }
    Poly deriv() const {
        if (a.empty()) {
            return Poly();
        }
        vector&lt;Z&gt; res(size() - 1);
        for (int i = 0; i &lt; size() - 1; i &#43;&#43;) {
            res[i] = (i &#43; 1) * a[i &#43; 1];
        }
        return Poly(res);
    }
    Poly integr() const {
        vector&lt;Z&gt; res(size() &#43; 1);
        for (int i = 0; i &lt; size(); i &#43;&#43;) {
            res[i &#43; 1] = a[i] / (i &#43; 1);
        }
        return Poly(res);
    }
    Poly inv(int m) const {
        Poly x{a[0].inv()};
        int k = 1;
        while (k &lt; m) {
            k *= 2;
            x = (x * (Poly{2} - modxk(k) * x)).modxk(k);
        }
        return x.modxk(m);
    }
    Poly log(int m) const {
        return (deriv() * inv(m)).integr().modxk(m);
    }
    Poly exp(int m) const {
        Poly x{1};
        int k = 1;
        while (k &lt; m) {
            k *= 2;
            x = (x * (Poly{1} - x.log(k) &#43; modxk(k))).modxk(k);
        }
        return x.modxk(m);
    }
    Poly pow(int k, int m) const {
        int i = 0;
        while (i &lt; size() &amp;&amp; a[i].val() == 0) {
            i &#43;&#43;;
        }
        if (i == size() || 1LL * i * k &gt;= m) {
            return Poly(vector&lt;Z&gt;(m));
        }
        Z v = a[i];
        auto f = divxk(i) * v.inv();
        return (f.log(m - i * k) * k).exp(m - i * k).mulxk(i * k) * power(v, k);
    }
    Poly sqrt(int m) const {
        Poly x{1};
        int k = 1;
        while (k &lt; m) {
            k *= 2;
            x = (x &#43; (modxk(k) * x.inv(k)).modxk(k)) * ((mod &#43; 1) / 2);
        }
        return x.modxk(m);
    }
    Poly mulT(Poly b) const {
        if (b.size() == 0) {
            return Poly();
        }
        int n = b.size();
        reverse(b.a.begin(), b.a.end());
        return ((*this) * b).divxk(n - 1);
    }
};
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
signed main() {
    init(1e7);
    cin.tie(0) -&gt; sync_with_stdio(0);
    int w;
    cin &gt;&gt; w;
    vector&lt;Poly&gt; ans(w &#43; 1);
    ans[0] = {1};
    for (int i = 1; i &lt;= w; i &#43;&#43;) {
        int c;
        cin &gt;&gt; c;
        vector&lt;Z&gt; g(c);
        for (int j = 0; j &lt; c; j &#43;&#43;) {
            g[j] = -infact[j];
        }
        for (int j = i; j; j --) {
            ans[j] = ans[j] * Poly(g) &#43; ans[j - 1];
        }
        ans[0] = ans[0] * Poly(g);
    }
    int m;
    cin &gt;&gt; m;
    while (m --) {
        int n;
        cin &gt;&gt; n;
        Z res;
        for (int i = 0; i &lt;= w; i &#43;&#43;) {
            int v = min(ans[i].size() - 1, n);
            Z Pow = power(Z(i), n - v);
            for (int j = v; ~j; j --) {
                res &#43;= ans[i][j] * Pow * infact[n - j];
                Pow *= i;
            }
        }
        cout &lt;&lt; res * fact[n] &lt;&lt; &#34;\n&#34;;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2022-%E7%89%9B%E5%AE%A2%E5%A4%9A%E6%A0%A14-c-easy-counting-problem/  

