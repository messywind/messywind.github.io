# [2022 牛客多校2 E] Falfa With Substring


[原题链接](https://ac.nowcoder.com/acm/contest/33187/E)

**题意**

定义 $F_{n, k}$ 为所有长度为 $n$ 的字符串 $S$ 中恰好出现了 $k$ 次 `bit` 的个数。

求 $F_{n, 0},F_{n, 1}, \cdots,F_{n,n}$ 对 $998\,244\,353$ 取模。

$(1 \le n \le 10 ^ 6)$

**分析：**

看到求恰好出现 $k$ 次，首先想到求出大于等于 $k$ 次再进行容斥。

考虑钦定出现 $k$ 次 `bit` 的字符串，将 $k$ 个 `bit` 进行捆绑，那么有 $n - 3k &#43; k = n - 2k$ 个位置，并且剩下 $n - 3 k$ 个字母任意取值，从 $n - 2k$ 个位置选出 $k$ 个放 `bit`，方案数为 $\dbinom{n - 2k}{k} \times 26 ^ {n - 3k}$，记为 $f(k)$

那么 $f(k)$ 由所有恰好出现 $k, k &#43; 1, \cdots, n$ 次的方案数加起来，还要乘上对应次数选出 $k$ 个的方案数，记恰好出现 $k$ 次的方案数为 $g(k)$
$$
f(k) = \sum_{i = k} ^ {n} \binom{i}{k} g(i)
$$
根据二项式反演公式 $f(n)= \sum\limits_{i = n} ^ {m} \binom{i}{n} g(i) \Leftrightarrow g(n) = \sum\limits_{i = n} ^ {m}(-1) ^ {i - n}\binom{i}{n}g(i)$
$$
g(k) = \sum_{i = k} ^ {n} (-1) ^ {i - k} \binom{i}{k}f(i)
$$
展开组合数 $\dbinom{i}{k}= \dfrac{i!}{k! \times (i - k)!}$
$$
g(k) = \sum_{i = k} ^ {n} i!f(i) \dfrac{(-1) ^ {i - k}}{k! \times (i - k)!} \\
\Leftrightarrow k!g(k) = \sum_{i = k} ^ {n} i!f(i) \dfrac{(-1) ^ {i - k}}{(i - k)!}
$$
设 $P(k) = k!g(k), F(i) = i!f(i), G(i) = \dfrac{(-1) ^ {i}}{i!}$，则
$$
P(k) = \sum_{i = k} ^ {n} F(i) \times G(i - k)
$$
令 $(i - k) \rightarrow i$
$$
P(k) = \sum_{i = 0} ^ {n - k} G(i) \times F(i &#43; k)
$$
考虑多项式加速。我们知道常规的多项式卷积是 $F(n) = \sum\limits_{i = 0} ^ {n} f(i) \times g(n - i)$，所以上式中 $F$ 对应的下标应为 $n - k - i$，发现 $n - k - i &#43; i &#43; k = n$ 恰好是对称关系，所以可以对 $F$ 函数做翻转进行多项式卷积。

答案为 $\dfrac{P(i)}{i!}$ $(0 \le i \le n)$ 

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
    int n;
    cin &gt;&gt; n;
    vector&lt;Z&gt; fact(n &#43; 1), infact(n &#43; 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        fact[i] = fact[i - 1] * i;
    }
    infact[n] = fact[n].inv();
    for (int i = n; i; i --) {
        infact[i - 1] = infact[i] * i;
    }
    auto C = [&amp;](int n, int m) {
	    if (n &lt; 0 || m &lt; 0 || n &lt; m) return Z(0);
	    return fact[n] * infact[n - m] * infact[m];
	};
	vector&lt;Z&gt; f(n / 3 &#43; 1), g(n / 3 &#43; 1);
	for (int i = 0; i &lt;= n / 3; i &#43;&#43;) {
		f[i] = fact[i] * C(n - 2 * i, i) * power(Z(26), n - 3 * i);
		g[i] = infact[i] * (i % 2 == 0 ? 1 : -1);
	}
	reverse(f.begin(), f.end());
	auto ans = Poly(f) * Poly(g);
	for (int i = 0; i &lt;= n; i &#43;&#43;) {
		cout &lt;&lt; (i &lt;= n / 3 ? infact[i] * ans[n / 3 - i] : 0) &lt;&lt; &#34; \n&#34;[i == n];
	}
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2022-%E7%89%9B%E5%AE%A2%E5%A4%9A%E6%A0%A12-e-falfa-with-substring/  

