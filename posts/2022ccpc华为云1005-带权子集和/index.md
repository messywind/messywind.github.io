# [2022CCPC华为云1005] 带权子集和


[原题链接](https://acm.hdu.edu.cn/showproblem.php?pid=7260)

**题意**

给定一个大小为 $n$ 的多重集 $A = \{a_1, a_2, \cdots,a_n\}$ 和两个非负整数 $k, t$，求
$$
\sum_{S \subseteq A,S \ne \varnothing } t ^ {|S|} \left ( \sum_{i \in S} i \right ) ^ k
$$
保证 $k=0$ 时没有子集的和是 $0$，对 $998 \, 244 \,353$ 取模。

$(1 \le n \le 10 ^ 2, 0 \le k \le 10 ^4, 0 \le x,a_i &lt; 988244352)$

**分析：**

若直接枚举子集复杂度为 $O(n \times 2 ^ n)$，无法接受，考虑化简，右边和式的 $k$ 次方可以做一下展开，那么

$$
\left ( \sum\limits_{i \in S} i \right ) ^ k = \underbrace{(i_1 &#43; i_2&#43; \cdots&#43;i_{|S|}) \times \cdots \times (i_1 &#43; i_2&#43; \cdots &#43; i_{|S|})}_{k项}
$$

也就是在 $i_1,i_2,\cdots,i_{|S|}$ 中任选 $k$ 个可重复的数的所有乘积和，考虑 $A$ 中每个数 $a_i$ 的贡献，对于每个 $a_i$ 都有选与不选两种状态，组成了集合 $S$，那么假设不考虑后面的 $k$ 次方和式，我们可以写出生成函数 $(1 &#43; tx) ^ n$，但是现在多乘了 $k$ 次方和式，我们再只考虑这个和式，也就是说每个 $a_i$ 都可以被选 $0 \sim k$ 次，而且随意排列，所以 $\textbf{EGF}$ 为
$$
1 &#43; \frac{a_i}{1!}x &#43; \frac{a_i ^ 2}{2!}x ^ 2 &#43; \cdots &#43; \frac{a_i ^ k}{k!}x ^ k
$$
那么对于某个子集的贡献就为
$$
[x ^ k] \prod _ {i = 1} ^ n (1 &#43; \frac{a_i}{1!}x &#43; \frac{a_i ^ 2}{2!}x ^ 2 &#43; \cdots &#43; \frac{a_i ^ k}{k!}x ^ k)
$$
 所以只需要将这两个生成函数结合一下，也就是把这个 $\textbf{EGF}$ 带入到每个 $1 &#43; tx$ 中

$$
\prod_{i = 1} ^ {n} \left (1 &#43; t \times (1 &#43; \frac{a_i}{1!}x &#43; \frac{a_i ^ 2}{2!}x ^ 2 &#43; \cdots &#43; \frac{a_i ^ k}{k!}x ^ k) \right) \\\\
= \prod _{i = 1} ^ {n}\left (1 &#43; t &#43; \frac{t \times a_i}{1!}x &#43; \frac{t \times a_i ^ 2}{2!}x ^ 2 &#43; \cdots &#43; \frac{t \times a_i ^ k}{k!}x ^ k\right)
$$
由于题目保证了 $k=0$ 时没有子集的和是 $0$，所以如果 $k = 0$ 答案需要减去 $1$，也就是空集的情况。

那么最后的答案就为 $[x ^ k] \prod \limits_{i = 1} ^ {n} (1 &#43; t &#43; \dfrac{t \times a_i}{1!}x &#43; \dfrac{t \times a_i ^ 2}{2!}x ^ 2 &#43; \cdots &#43; \dfrac{t \times a_i ^ k}{k!}x ^ k) - [k=0]$

注意每次 $\texttt{NTT}$ 卷积需要将大小设为 $k$，不然会超时。

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
void solve() {
    int n, k, x;
    cin &gt;&gt; n &gt;&gt; k &gt;&gt; x;
    vector&lt;int&gt; a(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        cin &gt;&gt; a[i];
    }
    vector&lt;vector&lt;Z&gt;&gt; f(n &#43; 1, vector&lt;Z&gt;(k &#43; 1));
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        f[i][0] = 1 &#43; x;
        for (int j = 1; j &lt;= k; j &#43;&#43;) {
            f[i][j] = x * power(Z(a[i]), j) * infact[j];
        }
    }
    function&lt;Poly(int, int)&gt; dc = [&amp;](int l, int r) {
        if (l == r) return Poly(f[l]);
        int mid = l &#43; r &gt;&gt; 1;
        auto ans = dc(l, mid) * dc(mid &#43; 1, r);
        ans.resize(k &#43; 1);
        return ans;
    };
    cout &lt;&lt; fact[k] * dc(1, n)[k] - !k &lt;&lt; &#34;\n&#34;;
}
signed main() {
    init(1e4);
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
> URL: https://blog.messywind.top/posts/2022ccpc%E5%8D%8E%E4%B8%BA%E4%BA%911005-%E5%B8%A6%E6%9D%83%E5%AD%90%E9%9B%86%E5%92%8C/  

