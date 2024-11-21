# [2021 CCPC 广州 A] Math Ball


[原题链接](https://codeforces.com/gym/103415/problem/A)

**题意**

给定 $n$ 个权值为 $c_1,c_2,\cdots,c_n$ 的物品，总共最多取 $W$ 个，求
$$
\sum_{k_1&#43;k_2&#43;\cdots&#43;k_n \le W} \prod_{i = 1} ^ {n} k_i ^ {c_i}
$$
对 $998\,244\,353$ 取模，其中 $k_i$ 代表第 $i$ 个物品取的次数。

$1 \le n \le 10 ^ 5, \sum\limits_{i = 1} ^ {n} c_i \le 10 ^ 5, W \le 10 ^ {18}$

**分析：**

首先观察题目要我们求的式子，可以看出是一个多项式卷积形式，不难写出每个物品的生成函数，设第 $i$ 个物品的生成函数为
$$
f_i(x) = \sum_{j = 0} ^ {\infty} j ^ {c_i}x ^ j
$$
那么答案就是
$$
\sum_{i = 0} ^ {W} [x ^ i]\prod_{j = 1} ^ {n}f_j(x)
$$
但 $W$ 是 $10 ^ {18}$ 的，我们肯定不能这么求，所以考虑将答案求一次前缀和，计算第 $W$ 项的系数。

我们知道给一个多项式乘以 $\sum\limits_{i = 0} ^ {\infty} x ^ i = \dfrac{1}{1 -x}$ 就相当于求一次前缀和，故答案为
$$
[x ^ W]\frac{1}{1 - x} \times \prod_{j = 1} ^ {n}f_j(x)
$$
但这样还是不能解决问题。

所以考虑化简每个物品的生成函数 $f_i(x)$，我们知道有自然数幂展开 $i ^ k = \sum\limits_{j = 0} ^ {k} {k \brace j} \times j! \times \dbinom{i}{j}$，所以 $f_i(x)$ 就为
$$
f_i(x) = \sum_{j = 0} ^ {\infty} \sum_{k = 0} ^ {c_i}{c_i \brace k} \times k! \times \binom{j}{k} \times x ^ j
$$
交换求和次序
$$
f_i(x) = \sum_{k = 0} ^ {c_i} {c_i \brace k} \times k! \sum_{j = 0} ^ {\infty}\binom{j}{k} x ^ j
$$
现在考虑化简 $\sum\limits_{j = 0} ^ {\infty}\dbinom{j}{k} x ^ j$，我们根据广义二项式定理知道 $\sum\limits_{i = 0} ^ {\infty} \dbinom{i &#43; k - 1}{i}x ^ i$ 的封闭形式为 $\dfrac{1}{(1 - x) ^ k}$，那么 $\sum\limits_{i = 0} ^ {\infty} \dbinom{i &#43; k}{i}x ^ i = \dfrac{1}{(1 - x) ^ {k &#43; 1}}$

由组合数性质有 $\dbinom{i &#43; k}{i}=\dbinom{i &#43; k}{k}$，所以 $\sum\limits_{i = 0} ^ {\infty} \dbinom{i &#43; k}{i}x ^ i = \sum\limits_{i = 0} ^ {\infty} \dbinom{i &#43; k}{k}x ^ i = \sum\limits_{i = k} ^ {\infty} \dbinom{i}{k}x ^ {i - k} = \dfrac{1}{(1 - x) ^ {k &#43; 1}}$ 再等式两边同乘 $x ^ k$，得出结论
$$
\sum_{i = 0} ^ {\infty} \binom{i}{k}x ^ i = \frac{x ^ k}{(1 - x) ^ {k &#43; 1}}
$$
(由于 $\dbinom{n}{m}$ 在 $n &lt; m$ 时为 $0$，所以 $i$ 从 $0$ 或从 $k$ 开始都一样)

所以进一步化简了 $f_i(x)$，为
$$
f_i(x) = \sum_{k = 0} ^ {c_i} {c_i \brace k} \times k! \times \frac{x ^ k}{(1 - x) ^ {k &#43; 1}}
$$
尽管如此，$f_i(x)$ 仍然不好算，我们注意到题目条件 $\sum\limits_{i = 1} ^ {n} c_i \le 10 ^ 5$，一般会往分治 $\texttt{NTT}$ 上考虑，我们不妨将 $f_i(x)$ 的形式变成
$$
f_i(x) = \sum\limits_{k = 0} ^ {c_i} {c_i \brace k} \times k! \times (\frac{x}{1-x}) ^ {k} \times \frac{1}{1 - x}
$$
此时如果令 $y = \dfrac{x}{1 - x},f_i(x) = g_i(y) \times \dfrac{1}{1 - x}$，其中 $g_i(y) = \sum\limits_{k = 0} ^ {c_i} {c_i \brace k} \times k! \times y ^ {k}$ 那么答案就为
$$
[x ^ W] \frac{\prod_{j = 1} ^ {n} g_j(y)}{(1 - x) ^ {n &#43; 1}}
$$
这样的话 $\prod\limits_{j = 1} ^ {n} g_j(y)$ 是可以用分治 $\texttt{NTT}$ 求解的，其中需要用到快速求解第二类斯特林数的每一行，求出之后考虑计算第 $W$ 项的系数。

令 $F(x) = \prod\limits_{j = 1} ^ {n} g_j(y)$，那么 $F(x)$ 的第 $k$ 项就形如 $a_k \times (\dfrac{x}{1 - x}) ^ k$，其中 $a_k$ 为 $F(x)$ 的第 $k$ 项系数，那么 $a_k \times (\dfrac{x}{1 - x}) ^ k = a_k \times x ^ k \times \dfrac{1}{(1 - x) ^ k}$，把 $\dfrac{1}{(1 - x) ^ k}$ 拿到下面，变为 $\dfrac{1}{(1 - x) ^ {n &#43; k &#43; 1}}$

由于 $F(x)$ 项数较少，所以考虑枚举 $F(x)$ 的每一项，即答案为 $a_k$ 与 $\dfrac{1}{(1 - x) ^ {n &#43; k &#43; 1}}$ 的第 $W - k$ 项的乘积之和，考虑展开 $\dfrac{1}{(1 - x) ^ {n &#43; k &#43; 1}} = \sum\limits_{i = 0} ^ {\infty} \dbinom{n &#43; k &#43; i}{i}x ^ i$，所以第 $W - k$ 项为 $\dbinom{W &#43; n}{W - k}$，虽然 $W$ 较大，但 $n$ 很小，经典维护 $W$ 的下降幂即可。

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
    init(1e5);
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n;
    Z w;
    cin &gt;&gt; n &gt;&gt; w;
    vector&lt;int&gt; c(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        cin &gt;&gt; c[i];
    }
    function&lt;Poly(int, int)&gt; dc = [&amp;](int l, int r) {
        if (l == r) {
            S2_row_init(c[l]);
            vector&lt;Z&gt; f(c[l] &#43; 1);
            for (int i = 0; i &lt;= c[l]; i &#43;&#43;) {
                f[i] = fact[i] * S2_row[i];
            }
            return Poly(f);
        }
        int mid = l &#43; r &gt;&gt; 1;
        return dc(l, mid) * dc(mid &#43; 1, r);
    };
    auto ans = dc(1, n);
    Z sum = infact[n];
    for (i64 i = w.val() &#43; n; i &gt; w.val(); i --) {
        sum *= Z(i);
    }
    Z res;
    for (int i = 0; i &lt; ans.size() &amp;&amp; i &lt;= w.val(); i &#43;&#43;) {
        res &#43;= ans[i] * sum;
        sum *= (w - i) / (n &#43; i &#43; 1);
    }
    cout &lt;&lt; res &lt;&lt; &#34;\n&#34;;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2021-ccpc-%E5%B9%BF%E5%B7%9E-a-math-ball/  

