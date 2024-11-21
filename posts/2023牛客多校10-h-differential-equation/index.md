# [2023牛客多校10 H] Differential Equation


[原题链接](https://ac.nowcoder.com/acm/contest/57364/H)

**题意**

定义函数 $f_k(x) = (x ^ 2 - 1)f\&#39; _{k - 1}(x)$，其中 $f_0(x) = x$，求 $f_n(x_0)$ 对 $998\,244\,353$ 取模。

$0\le n \le 2 \times 10 ^ 5, 0 \le x_0 \le 998\,244\,352$

**分析：**

首先看到递推式含有导数，我们可以考虑序列 $f_0(x),f_1(x),f_2(x),\cdots$ 的 EGF。由于 $x$ 变量已存在，我们用二元函数 $F(x, y)$ 来刻画，即：

$$
F(x,y) = \sum_{i = 0} ^ {\infty}\frac{f_i(x)}{i!} y ^ i
$$

我们希望在 $F(x, y)$ 中看到 $f&#39;_i(x)$，于是对 $F(x, y)$ 求一次 $x$ 的偏导。

$$
\frac{\partial F(x,y)}{\partial x} = \sum_{i = 0} ^ {\infty}\frac{f&#39;_i(x)}{i!} y ^ i
$$

这样我们就可以把 $f\&#39; _ {i}(x)$ 替换为 $\dfrac{f_{i &#43; 1}(x)}{x ^ 2 - 1}$ (此时分母必然不为 $0$，否则不满足样例解释)

$$
\frac{\partial F(x,y)}{\partial x} = \sum_{i = 0} ^ {\infty}\frac{f_{i &#43; 1}(x)}{(x ^ 2 - 1) \times i!} y ^ i
$$

我们希望右边也凑成 $\sum\limits_{i = 0} ^ {\infty}\dfrac{f_i(x)}{i!} y ^ i$，所以把 $(x ^ 2 - 1)$ 乘到左边。

$$
(x ^ 2 - 1)\frac{\partial F(x,y)}{\partial x} = \sum_{i = 0} ^ {\infty}\frac{f_{i &#43; 1}(x)}{i!} y ^ i
$$

此时差别为 $f_{i &#43; 1}(x)$ 产生的错位，所以我们可以对 $F(x, y)$ 求一次 $y$ 的偏导。

$$
\begin{array}{c}
&amp;&amp;\dfrac{\partial F(x,y)}{\partial y} &amp;=&amp; \sum\limits_{i = 1} ^ {\infty}\dfrac{f_i(x)}{(i - 1)!} y ^ {i - 1}
\\\\
&amp;\Leftrightarrow&amp; \dfrac{\partial F(x,y)}{\partial y} &amp;=&amp; \sum\limits_{i = 0} ^ {\infty}\dfrac{f_{i &#43; 1}(x)}{i!} y ^ i
\end{array}
$$

这样两个等式右边就一样了，于是直接联立。

$$
\frac{\partial F(x,y)}{\partial y} = (x ^ 2 - 1)\frac{\partial F(x,y)}{\partial x}
$$

问题变为求解这个一阶线性偏微分方程，考虑用[特征线法](https://zhuanlan.zhihu.com/p/340161952)。方程改写为：

$$
a(x,y,F)F_x&#43;b(x,y,F)F_y=c(x,y,F)
$$

其中 $a(x,y,F) = 1 - x ^ 2, b(x,y,F) = 1,c(x,y,F) = 0$，故有特征系统：

$$
\begin{array}{c}
\dfrac{\mathrm{d}x}{\mathrm{d}t}  &amp;=&amp; 1 - x ^ 2 &amp;(1)\\\\
\dfrac{\mathrm{d}y}{\mathrm{d}t}  &amp;=&amp; 1 &amp; (2) \\\\
\dfrac{\mathrm{d}F}{\mathrm{d}t}  &amp;=&amp; 0 &amp; (3)
\end{array}
$$

将 $(1), (2)$ 方程联立消掉 $\mathrm{d}t$ 得 $\mathrm{d}y = \dfrac{1}{1 - x ^ 2} \mathrm{d}x$，两边积分得 $2y &#43; C_1 = \ln\left|\dfrac{1 &#43; x}{1 - x}\right|$

再由 $(3)$ 得 $F = C_2$，与上式结合得：

$$
F(x, y) = h\left(\ln\left|\dfrac{1 &#43; x}{1 - x}\right| - 2y\right)
$$

其中 $h$ 为任意函数，此时由边界条件 $F(x, 0) = x$ 代入得： 

$$
x = h\left(\ln\left|\dfrac{1 &#43; x}{1 - x}\right|\right)
$$

于是令 $u = \ln\left|\dfrac{1 &#43; x}{1 - x}\right|$，反解 $x$ 得 $\dfrac{e ^ u - 1}{e ^ u &#43; 1} = x$，所以 $h(u) = \dfrac{e ^ u - 1}{e ^ u &#43; 1}$，即 $h(x) = \dfrac{e ^ x - 1}{e ^ x &#43; 1}$

将 $\ln\left|\dfrac{1 &#43; x}{1 - x}\right| - 2y$ 代入到 $h(x)$，整理出 $F(x, y)$：

$$
F(x, y) = \dfrac{1 &#43; x - (1 - x)e ^ {2y}}{1 &#43; x &#43; (1 - x)e ^ {2y}}
$$

故

$$
\dfrac{1 &#43; x - (1 - x)e ^ {2y}}{1 &#43; x &#43; (1 - x)e ^ {2y}} = \sum_{i = 0} ^ {\infty}\frac{f_i(x)}{i!} y ^ i
$$

直接代入 $x_0$ 并求 $f_n(x_0)$，那么答案就为

$$
f_n(x_0) = n!\times [y ^ n] \left(\dfrac{1 &#43; x_0 - (1 - x_0) e ^ {2y}}{1 &#43; x_0 &#43; (1 - x_0)e ^ {2y}}\right)
$$

展开 $e ^ {2y}$ 到 $n$ 项，进行多项式求逆再卷积即可。

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
    int n, x0;
    cin &gt;&gt; n &gt;&gt; x0;
    init(n);
    vector&lt;Z&gt; f(n &#43; 1), g(n &#43; 1);
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        f[i] = g[i] = power(Z(2), i) * infact[i];
        f[i] *= x0 - 1, g[i] *= 1 - x0;
    }
    f[0] &#43;= 1 &#43; x0, g[0] &#43;= 1 &#43; x0;
    Poly ans = Poly(f) * Poly(g).inv(n &#43; 1);
    cout &lt;&lt; fact[n] * ans[n] &lt;&lt; &#34;\n&#34;;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2023%E7%89%9B%E5%AE%A2%E5%A4%9A%E6%A0%A110-h-differential-equation/  

