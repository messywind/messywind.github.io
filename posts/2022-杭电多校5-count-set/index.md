# [2022 杭电多校5] Count Set


[原题链接](https://acm.hdu.edu.cn/showproblem.php?pid=7191)

**题意**

给定一个长度为 $n$ 的排列 $p$ $\{1, 2, \cdots, n\}$ 和一个非负整数 $k$，计算排列 $p$ 中的子集 $T$，满足集合大小为 $k$ 且 $T$ 与 $P(T)$ 没有交集，$P(T) = \{y \mid y= p_x,x \in T\}$

**分析：**

考虑将排列 $p$ 看成图，$i$ 向 $p_i$ 连边，会形成若干个环，那么原问题等价于从图中选出 $k$ 个点且每个环中不能有相邻被选择的点的方案数。考虑构造每个环 $i$ 的生成函数
$$
1 &#43; f_{S_i,1}x &#43; f_{S_i,2} x ^ 2 &#43; \cdots &#43; f_{S_i,\lfloor \frac{S_i}{2} \rfloor} x ^ {\lfloor \frac{S_i}{2} \rfloor}
$$
其中 $S_i$ 表示环 $i$ 的大小，$f_{S_i,j}$ 表示大小为 $S_i$ 的环中选出 $j$ 个互不相邻的点的方案数，根据鸽巢原理，若 $j &gt; \lfloor \dfrac{S_i}{2} \rfloor$，一定有两个点相邻，所以生成函数只需要取到 $\lfloor \dfrac{S_i}{2} \rfloor$ 项即可。那么答案就为
$$
[x ^ k]\prod_{i = 1} ^ {\text{cnt}} \sum_{j = 0} ^ {\lfloor \frac{S_i}{2} \rfloor} f_{S_i,j}x^j
$$
$\text{cnt}$ 为图中环的数量。那么现在考虑求出 $f(n, m)$，即大小为 $n$ 的环选出 $m$ 个互不相邻的点的方案数。

我们先考虑不是环的情况，也就是链式不相邻问题，那么可以先放 $m$ 个被选择的球，考虑把中间 $m - 1$ 个空放上一个不被选择的球，那么剩下 $n - 2\times m - 1$ 个球就可以随便放，问题就相当于有 $m &#43; 1$ 个盒子，每个盒子可空的方案数，那么就是经典隔板法，方案数为 $g(n, m) =\dbinom{n - 2\times m - 1 &#43; m &#43; 1 - 1}{m &#43; 1 - 1}=\dbinom{n - m &#43; 1}{m}$

现在考虑是环的情况，假设对于环上一个点，有两种情况，若这个点被选择，则这个点的相邻点不能被选择，那么其他 $n - 3$ 个点就是 $g(n - 3, m - 1)$，若这个点不被选择，那么剩下 $n - 1$ 个点就是 $g(n - 1, m)$，所以 $f(n, m) = g(n - 3, m - 1) &#43; g(n - 1, m) = \dbinom{n - m - 1}{m - 1} &#43; \dbinom{n - m}{m}$

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
        int sz = 1, tot = min(5000000, a.size() &#43; b.size() - 1);
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
Z C(int n, int m) {
    if (n &lt; 0 || m &lt; 0 || n &lt; m) return Z(0);
    return fact[n] * infact[n - m] * infact[m];
}
struct DSU {
    vector&lt;int&gt; p, Size;
    DSU(int n) : p(n), Size(n, 1) {
        iota(p.begin(), p.end(), 0);
    }
    int find(int x) {
        return p[x] == x ? p[x] : p[x] = find(p[x]);
    }
    bool same(int u, int v) {
        return find(u) == find(v);
    }
    void merge(int u, int v) {
        u = find(u), v = find(v);
        if (u != v) {
            Size[v] &#43;= Size[u];
            p[u] = v;
        }
    }
};
void solve() {
    int n, k;
    cin &gt;&gt; n &gt;&gt; k;
    DSU p(n &#43; 1);
    vector&lt;int&gt; a(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        cin &gt;&gt; a[i];
        p.merge(i, a[i]);
    }
    vector&lt;vector&lt;Z&gt;&gt; f(n &#43; 1);
    int cnt = 0;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        if (p.find(i) == i) {
            cnt &#43;&#43;;
            f[cnt].resize(p.Size[i] / 2 &#43; 1);
            for (int j = 0; j &lt;= p.Size[i] / 2; j &#43;&#43;) {
                f[cnt][j] = C(p.Size[i] - j - 1, j - 1) &#43; C(p.Size[i] - j, j);
            }
        }
    }
    function&lt;Poly(int, int)&gt; dc = [&amp;](int l, int r) {
        if (l == r) return Poly(f[l]);
        int mid = l &#43; r &gt;&gt; 1;
        return dc(l, mid) * dc(mid &#43; 1, r);
    };
    Poly ans = dc(1, cnt);
    ans.resize(k &#43; 1);
    cout &lt;&lt; ans[k] &lt;&lt; &#34;\n&#34;;
}
signed main() {
    init(1e7);
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
> URL: https://blog.messywind.top/posts/2022-%E6%9D%AD%E7%94%B5%E5%A4%9A%E6%A0%A15-count-set/  

