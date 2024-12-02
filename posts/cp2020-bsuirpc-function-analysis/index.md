# [2020 BSUIRPC] Function Analysis


[原题链接](https://codeforces.com/gym/103637/problem/F)

**题意**

给定三个正整数 $n, d, k$ ，现有排列 $p = (1,2,3,\cdots,n)$ ，有 $n - d &#43; 1$ 个询问，对于每个询问有正整数 $m (d \le m \le n)$ ，现从 $p$ 中**随机可重复地**选取 $m$ 个数构成序列 $q$ ，求 $q$ 中第 $d$ 小数大于 $k$ 的概率，

对 $998244353$ 取模。

**分析：**

对于满足第 $d$ 小数大于 $k$ 的序列 $q = (a_1, a_2, \cdots,a_i,\cdots,a_m)$ ，假设最后一个小于等于 $k$ 的数为 $a_i$，那么一定有 $i \le d - 1$，所以前 $i$ 个数一定要在 $1 \sim k$ 中任取，方案记为 $F(x)$，那么后 $m - i$ 个数一定要在 $k &#43;1 \sim n$ 中取，方案记为 $G(x)$，可以考虑写出 $F(x)$ 与 $G(x)$ 的生成函数，因为数是可排列的，所以要用 $\textbf{EGF}$

在 $1 \sim k$ 中取 $i$ 个数的方案数为 $k ^ i$，作为 $F(x)$ 取 $i$ 个数生成函数的系数，$1 \sim k$ 中至多取 $d - 1$ 个数，所以 $F(x)$ 的项数为 $d$
$$
F(x) = 1&#43;\frac{k^1}{1!}x^1 &#43; \frac{k^2}{2!}x^2&#43;\cdots&#43;\frac{k^{d - 1}}{(d-1)!}x^{d - 1}
$$
在 $k &#43; 1 \sim n$ 中取 $i$ 个数的方案数为 $(n - k) ^ {i}$，作为 $G(x)$ 取 $i$ 个数生成函数的系数，$k &#43; 1 \sim n$ 中可以取至多 $n$ 个数，所以 $G(x)$ 项数为 $n &#43; 1$
$$
G(x) = 1 &#43; \frac{(n - k) ^1}{1!}x^1 &#43; \frac{(n - k)^2}{2!}x^2 &#43; \cdots &#43; \frac{(n - k) ^ n}{n!}x^n
$$
	那么选 $m$ 个数满足第 $d$ 小数大于 $k$ 的总方案数就为 $m!\times[x^m] F(x)G(x)$，所有取数的情况为 $n ^ m$，所以概率就为
$$
\frac{m!\times[x^m] F(x)G(x)}{n ^m}
$$
可以用 $\text{NTT}$ 算出 $F(x) * G(x)$

时间复杂度 $O(n\log n)$

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
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n, d, k;
    cin &gt;&gt; n &gt;&gt; d &gt;&gt; k;
    vector&lt;Z&gt; fact(n &#43; 1), infact(n &#43; 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        fact[i] = fact[i - 1] * i;
    }
    infact[n] = fact[n].inv();
    for (int i = n - 1; i; i --) {
        infact[i] = infact[i &#43; 1] * (i &#43; 1);
    }
    vector&lt;Z&gt; f(d);
    for (int i = 0; i &lt;= d - 1; i &#43;&#43;) {
        f[i] = power(Z(k), i) * infact[i];
    }
    vector&lt;Z&gt; g(n &#43; 1);
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        g[i] = power(Z(n - k), i) * infact[i];
    }
    Poly ans = Poly(f) * Poly(g);
    for (int m = d; m &lt;= n; m &#43;&#43;) {
        Z Inv = power(Z(n), m);
        cout &lt;&lt; fact[m] * ans[m] * Inv.inv() &lt;&lt; &#34;\n&#34;;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2020-bsuirpc-function-analysis/  

