# [2021ICPC上海 B] Strange Permutations


[原题链接](https://codeforces.com/gym/103446/problem/B)

**题意**

给定一个长度为 $n$ 的 $1 \sim n$ 排列 $P$，找到有多少个 $1 \sim n$ 的排列 $Q$ 使得 $\forall i \in[1, n - 1], Q_{i &#43; 1} \ne P_{Q_i}$ 

对 $998244353$ 取模

$(1 \le n \le 10 ^ 5, 1 \le P_i \le n)$

**分析：**

如果只观察式子可能看不出什么规律，我们可以把条件转化为 $n$ 个点的图，考虑把排列 $Q$ 表示为边集 $\{(Q_1, Q_2),(Q_2, Q_3),\cdots,(Q_{n-1},Q_n)\}$ ，那么排列 $P$ 的意思就是图中不能存在边集 $\{(1,P_1),(2,P_2),\cdots,(n,P_n)\}$ ，那么就等价于在一张图中选一条哈密顿路径的方案数，所以考虑对每条不存在的边集进行容斥。考虑计算选了 $i$ 个不存在的边的方案数，发现排列 $P$ 一定会成环，所以对于每个 $k$ 元环可以选择 $0 \sim k - 1$ 个不存在的边(哈密顿路径无环所以不能包含 $k$ 个不存在的边)，那么可以用生成函数 $f(k)$ 来表示
$$
f(k) = 1 &#43; \binom{k}{1}x &#43; \binom{k}{2}x ^ 2 &#43; \binom{k}{3}x ^ 3 &#43; \cdots &#43; \binom{k}{k - 1}x ^ {k - 1}
$$
$x$ 项的系数 $m$ 表示 $k$ 元环中选了 $m$ 条不存在的边，那么系数显然是 $\dbinom{k}{m}$

所以只需找出排列 $P$ 的所有环及其环的大小，假设有 $t$ 个 $a_1,a_2,\cdots,a_t$ ，$a_i$ 表示第 $i$ 个环的大小。

那么方案就是
$$
\prod_{i = 1} ^ {t} f(a_i)
$$
做一次分治 $\text{NTT}$ 或启发式合并得到多项式 $F(x)$

最后容斥计算答案，钦定选了 $i$ 条不存在的边其他边的数量就是 $(n - i)!$ 那么最后的答案就为
$$
\sum_{i = 0} ^ {n}(-1) ^ i(n - i)![x^i]F(x)
$$
时间复杂度 $O(n\log ^2n)$

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
    for (int i = n - 1; i; i --) {
        infact[i] = infact[i &#43; 1] * (i &#43; 1);
    }
    auto C = [&amp;](int m, int n) {
        if (n &lt; 0 || m &lt; 0 || m &lt; n) return Z(0);
        return fact[m] * infact[m - n] * infact[n];
    };
    vector&lt;int&gt; p(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        cin &gt;&gt; p[i];
    }
    vector&lt;bool&gt; st(n &#43; 1);
    vector&lt;int&gt; cnt;
    int circle = 0;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        if (st[i]) {
            continue;
        }
        cnt.push_back(0);
        for (int j = i; !st[j]; j = p[j]) {
            st[j] = true, cnt[circle] &#43;&#43;;
        }
        circle &#43;&#43;;
    }
    int idx = 0;
    vector&lt;Poly&gt; f(n &#43; 1);
    for (int i = 0; i &lt; circle; i &#43;&#43;) {
        if (!cnt[i]) {
            continue;
        }
        idx &#43;&#43;;
        f[idx].resize(cnt[i]);
        for (int j = 0; j &lt; cnt[i]; j &#43;&#43;) {
            f[idx][j] = C(cnt[i], j);
        }
    }
    function&lt;Poly(int, int)&gt; dc = [&amp;](int l, int r) {
        if (l == r) return f[l];
        int mid = l &#43; r &gt;&gt; 1;
        return dc(l, mid) * dc(mid &#43; 1, r);
    };
    Poly ans = dc(1, idx);
    ans.resize(n &#43; 1);
    Z res;
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        if (i &amp; 1) {
            res -= fact[n - i] * ans[i];
        } else {
            res &#43;= fact[n - i] * ans[i];
        }
    }
    cout &lt;&lt; res &lt;&lt; &#34;\n&#34;;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2021icpc%E4%B8%8A%E6%B5%B7-b-strange-permutations/  

