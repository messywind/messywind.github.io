# [2023牛客多校7 F] Counting Sequences


[原题链接](https://ac.nowcoder.com/acm/contest/57361/F)

**题意**

计算满足下列条件长度为 $n$ 的序列 $(a_1,a_2,\cdots,a_n)$ 的个数。

- 对于 $1 \le i \le n$ 满足 $0 \le a_i \le 2 ^ m$
- $\sum\limits_{i = 1} ^ {n} \text{cnt}_1(a_i \oplus b_i) = k$

其中序列 $b$ 为 $(a_2,a_3,\cdots,a_n,a_1)$，$\text{cnt}_1(x)$ 代表 $x$ 二进制中 $1$ 的个数。

对 $998\,244\,353$ 取模。

$2 \le n &lt; 998\,244\,353, 1 \le m \le 10 ^ 8,1 \le k \le 5\times 10 ^ 4$

**分析：**

首先观察第二个条件，由于每一位是独立的，考虑拆位。

统计 $m$ 位中每一位对答案的贡献，记 $a_i = \sum\limits_{j = 0} ^ {m - 1} c_{ij} \times 2 ^ j$，其中 $c_{ij}$ 表示 $a_i$ 二进制的第 $j$ 位。

那么对第 $j$ 位来说，只需要知道序列 $(c_{1j} \oplus c_{2j},c_{2j} \oplus c_{3j},\cdots, c_{(n - 1)j} \oplus c_{nj}, c_{nj} \oplus c_{1j})$ 中产生若干个 $1$ 的方案数。

下面简单证一下序列中只可能包含偶数个 $1$：

$c_{pj} \oplus c_{qj} = 1$ 等价于 $c_{pj} \ne c_{qj}$ 

$c_{pj} \oplus c_{qj} = 0$ 等价于 $c_{pj} = c_{qj}$ 

所以问题等价于在 $01$ 环选择 $u$ 条边使相邻点的值不相等，那么剩下 $n - u$ 条边使得相邻点的值相等。

由于相等边不改变值，所以我们可以将不相等边构成的连通块缩成一个连通块，也就是将相等边构成连通块的两边节点合并到一起。

此时问题变为二分图染色问题，我们知道二分图染色奇数环是不可行的，所以环大小必定为偶数，由此推出 $1$ 必须为偶数个。

那么得到 $2u$ 个 $1$ 的方案数就为 $2\dbinom{n}{2u}$，考虑每一位的生成函数

$$
F(x) = \sum_{u = 0} ^ {\lfloor \frac{n}{2} \rfloor } 2\binom{n}{2u}x ^ {2u}
$$

由于每一位独立，所以 $m$ 个生成函数都相等，所以答案为

$$
[x ^ k] F^ m(x)
$$

由于 $n  &lt; 998\,244\,353$ ，但 $k \le 5\times 10 ^ 4$，所以 $F(x)$ 至多算到第 $k$ 项，那么 $\dbinom{n}{2u}$ 经典维护下降幂。最后使用多项式快速幂求解即可。
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
            res[i] = a[i &#43; 1] * (i &#43; 1);
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
    vector&lt;Z&gt; eval(vector&lt;Z&gt; x) const {
        if (size() == 0) {
            return vector&lt;Z&gt;(x.size(), 0);
        }
        const int n = max(int(x.size()), size());
        vector&lt;Poly&gt; q(n &lt;&lt; 2);
        vector&lt;Z&gt; ans(x.size());
        x.resize(n);
        function&lt;void(int, int, int)&gt; build = [&amp;](int p, int l, int r) {
            if (r - l == 1) {
                q[p] = Poly{1, -x[l]};
            } else {
                int m = l &#43; r &gt;&gt; 1;
                build(p &lt;&lt; 1, l, m);
                build(p &lt;&lt; 1 | 1, m, r);
                q[p] = q[p &lt;&lt; 1] * q[p &lt;&lt; 1 | 1];
            }
        };
        build(1, 0, n);
        function&lt;void(int, int, int, const Poly &amp;)&gt; work = [&amp;](int p, int l, int r, const Poly &amp;num) {
            if (r - l == 1) {
                if (l &lt; int(ans.size())) {
                    ans[l] = num[0];
                }
            } else {
                int m = (l &#43; r) / 2;
                work(p &lt;&lt; 1, l, m, num.mulT(q[p &lt;&lt; 1 | 1]).modxk(m - l));
                work(p &lt;&lt; 1 | 1, m, r, num.mulT(q[p &lt;&lt; 1]).modxk(r - m));
            }
        };
        work(1, 0, n, mulT(q[1].inv(n)));
        return ans;
    }
    Poly inter(const Poly &amp;y) const {
        vector&lt;Poly&gt; Q(a.size() &lt;&lt; 2), P(a.size() &lt;&lt; 2);
        function&lt;void(int, int, int)&gt; dfs1 = [&amp;](int p, int l, int r) {
            int m = l &#43; r &gt;&gt; 1;
            if (l == r) {
                Q[p].a.push_back(-a[m]);
                Q[p].a.push_back(Z(1));
                return;
            }
            dfs1(p &lt;&lt; 1, l, m), dfs1(p &lt;&lt; 1 | 1, m &#43; 1, r);
            Q[p] = Q[p &lt;&lt; 1] * Q[p &lt;&lt; 1 | 1];
        };
        dfs1(1, 0, a.size() - 1);
        Poly f;
        f.a.resize((int)(Q[1].size()) - 1);
        for (int i = 0; i &#43; 1 &lt; Q[1].size(); i &#43;&#43;) {
            f[i] = Q[1][i &#43; 1] * (i &#43; 1);
        }
        Poly g = f.eval(a);
        function&lt;void(int, int, int)&gt; dfs2 = [&amp;](int p, int l, int r) {
            int m = l &#43; r &gt;&gt; 1;
            if (l == r) {
                P[p].a.push_back(y[m] * power(g[m], mod - 2));
                return;
            }
            dfs2(p &lt;&lt; 1, l, m), dfs2(p &lt;&lt; 1 | 1, m &#43; 1, r);
            P[p].a.resize(r - l &#43; 1);
            Poly A = P[p &lt;&lt; 1] * Q[p &lt;&lt; 1 | 1];
            Poly B = P[p &lt;&lt; 1 | 1] * Q[p &lt;&lt; 1];
            for (int i = 0; i &lt;= r - l; i &#43;&#43;) {
                P[p][i] = A[i] &#43; B[i];
            }
        };
        dfs2(1, 0, a.size() - 1);
        return P[1];
    }
};
Poly toFPP(vector&lt;Z&gt; &amp;a) {
    int n = a.size();
    vector&lt;Z&gt; b(n);
    iota(b.begin(), b.end(), 0);
    auto F = Poly(a).eval(b);
    vector&lt;Z&gt; f(n), g(n);
    for (int i = 0, sign = 1; i &lt; n; i &#43;&#43;, sign *= -1) {
        f[i] = F[i] * infact[i];
        g[i] = Z(sign) * infact[i];
    }
    return Poly(f) * Poly(g);
}
Poly toOP(vector&lt;Z&gt; &amp;a) {
    int n = a.size();
    vector&lt;Z&gt; g(n);
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        g[i] = infact[i];
    }
    auto F = Poly(a) * Poly(g);
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        F[i] *= fact[i];
    }
    vector&lt;Z&gt; p(n);
    iota(p.begin(), p.end(), 0);
    return Poly(p).inter(F);
}
Poly FPPMul(Poly a, Poly b) {
    int n = a.size() &#43; b.size() - 1;
    Poly p;
    p.resize(n);
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        p[i] = infact[i];
    }
    a *= p, b *= p;
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        a[i] *= b[i] * fact[i];
    }
    for (int i = 1; i &lt; n; i &#43;= 2) {
        p[i] = -p[i];
    }
    a *= p;
    a.resize(n);
    return a;
}
Poly Lagrange2(vector&lt;Z&gt; &amp;f, int m, int k) {
    int n = f.size() - 1;
    vector&lt;Z&gt; a(n &#43; 1), b(n &#43; 1 &#43; k);
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        a[i] = f[i] * ((n - i) &amp; 1 ? -1 : 1) * infact[n - i] * infact[i];
    }
    for (int i = 0; i &lt;= n &#43; k; i &#43;&#43;) {
        b[i] = Z(1) / (m - n &#43; i);
    }
    Poly ans = Poly(a) * Poly(b);
    for (int i = 0; i &lt;= k; i &#43;&#43;) {
        ans[i] = ans[i &#43; n];
    }
    ans.resize(k &#43; 1);
    Z sum = 1;
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        sum *= m - i;
    }
    for (int i = 0; i &lt;= k; i &#43;&#43;) {
        ans[i] *= sum;
        sum *= Z(m &#43; i &#43; 1) / (m - n &#43; i);
    }
    return ans;
}
Poly Chirp_Z_Transform(vector&lt;Z&gt; &amp;a, int c, int m) {
    int n = a.size();
    a.resize(n &#43; m - 1);
    Poly f, g;
    f.resize(n &#43; m - 1), g.resize(n &#43; m - 1);
    for (int i = 0; i &lt; n &#43; m - 1; i &#43;&#43;) {
        f[n - 1 &#43; m - 1 - i] = power(Z(c), i * (i - 1LL) / 2 % (mod - 1));
        g[i] = power(Z(c), mod - 1 - i * (i - 1LL) / 2 % (mod - 1)) * a[i];
    }
    Poly res = f * g, ans;
    ans.resize(m);
    for (int i = 0; i &lt; m; i &#43;&#43;) {
        ans[i] = res[n - 1 &#43; m - 1 - i] * power(Z(c), mod - 1 - i * (i - 1LL) / 2 % (mod - 1));
    }
    return ans;
}
Poly S2_row;
void S2_row_init(int n) {
    vector&lt;Z&gt; f(n &#43; 1), g(n &#43; 1);
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        f[i] = power(Z(i), n) * infact[i];
        g[i] = Z(i &amp; 1 ? -1 : 1) * infact[i];
    }
    S2_row = Poly(f) * Poly(g);
}
Poly S2_col;
void S2_col_init(int n, int k) {
    n &#43;&#43;;
    vector&lt;Z&gt; f(n);
    for (int i = 1; i &lt; n; i &#43;&#43;) {
        f[i] = infact[i];
    }
    auto ans = Poly(f).pow(k, n);
    S2_col.resize(n &#43; 1);
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        S2_col[i] = ans[i] * fact[i] * infact[k];
    }
}
Poly Bell;
void Bell_init(int n) {
    vector&lt;Z&gt; f(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        f[i] = infact[i];
    }
    auto ans = Poly(f).exp(n &#43; 1);
    Bell.resize(n &#43; 1);
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        Bell[i] = ans[i] * fact[i];
    }
}
vector&lt;Z&gt; p;
void p_init(int n) {
    vector&lt;int&gt; f(n &#43; 1);
    p.resize(n &#43; 1);
    p[0] = 1;
    f[0] = 1, f[1] = 2, f[2] = 5, f[3] = 7;
    for (int i = 4; f[i - 1] &lt;= n; i &#43;&#43;) {
        f[i] = 3 &#43; 2 * f[i - 2] - f[i - 4];
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = 0; f[j] &lt;= i; j &#43;&#43;) {
            p[i] &#43;= Z(j &amp; 2 ? -1 : 1) * p[i - f[j]];
        }
    }
}
Poly P;
void p_init(int n, int m) {
    vector&lt;Z&gt; a(n &#43; 1);
    for (int i = 1; i &lt;= m; i &#43;&#43;) {
        for (int j = i; j &lt;= n; j &#43;= i) {
            a[j] &#43;= Z(j / i).inv();
        }
    }
    P = Poly(a).exp(n &#43; 1);
}
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n, m, k;
    cin &gt;&gt; n &gt;&gt; m &gt;&gt; k;
    int sz = min(n, k);
    Poly f;
    f.resize(sz &#43; 1);
    f[0] = 2;
    Z fsum = 1, fn = n;
    for (int i = 1; i &lt;= sz; i &#43;&#43;) {
        fsum = fsum * fn / i;
        f[i] = i &amp; 1 ? Z(0) : fsum * 2;
        fn -= 1;
    }
    cout &lt;&lt; f.pow(m, k &#43; 1)[k] &lt;&lt; &#34;\n&#34;;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2023%E7%89%9B%E5%AE%A2%E5%A4%9A%E6%A0%A17-f-counting-sequences/  

