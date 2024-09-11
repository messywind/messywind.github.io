---
title: "[2023牛客多校10 H] Differential Equation"
date: 2023-08-22 16:34:01
tags:
- 生成函数
- 多项式
- PDE
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://ac.nowcoder.com/acm/contest/57364/H)

**题意**

定义函数 $f_k(x) = (x ^ 2 - 1)f'_{k - 1}(x)$，其中 $f_0(x) = x$，求 $f_n(x_0)$ 对 $998\,244\,353$ 取模。

$0\le n \le 2 \times 10 ^ 5, 0 \le x_0 \le 998\,244\,352$

**分析：**

首先看到递推式含有导数，我们可以考虑序列 $f_0(x),f_1(x),f_2(x),\cdots$ 的 EGF。由于 $x$ 变量已存在，我们用二元函数 $F(x, y)$ 来刻画，即：

$$
F(x,y) = \sum_{i = 0} ^ {\infty}\frac{f_i(x)}{i!} y ^ i
$$

我们希望在 $F(x, y)$ 中看到 $f'_i(x)$，于是对 $F(x, y)$ 求一次 $x$ 的偏导。

$$
\frac{\partial F(x,y)}{\partial x} = \sum_{i = 0} ^ {\infty}\frac{f'_i(x)}{i!} y ^ i
$$

这样我们就可以把 $f'_{i}(x)$ 替换为 $\dfrac{f_{i + 1}(x)}{x ^ 2 - 1}$ (此时分母必然不为 $0$，否则不满足样例解释)

$$
\frac{\partial F(x,y)}{\partial x} = \sum_{i = 0} ^ {\infty}\frac{f_{i + 1}(x)}{(x ^ 2 - 1) \times i!} y ^ i
$$

我们希望右边也凑成 $\sum\limits_{i = 0} ^ {\infty}\dfrac{f_i(x)}{i!} y ^ i$，所以把 $(x ^ 2 - 1)$ 乘到左边。

$$
(x ^ 2 - 1)\frac{\partial F(x,y)}{\partial x} = \sum_{i = 0} ^ {\infty}\frac{f_{i + 1}(x)}{i!} y ^ i
$$

此时差别为 $f_{i + 1}(x)$ 产生的错位，所以我们可以对 $F(x, y)$ 求一次 $y$ 的偏导。

$$
\begin{array}{c}
&&\dfrac{\partial F(x,y)}{\partial y} &=& \sum\limits_{i = 1} ^ {\infty}\dfrac{f_i(x)}{(i - 1)!} y ^ {i - 1}
\\\\
&\Leftrightarrow& \dfrac{\partial F(x,y)}{\partial y} &=& \sum\limits_{i = 0} ^ {\infty}\dfrac{f_{i + 1}(x)}{i!} y ^ i
\end{array}
$$

这样两个等式右边就一样了，于是直接联立。

$$
\frac{\partial F(x,y)}{\partial y} = (x ^ 2 - 1)\frac{\partial F(x,y)}{\partial x}
$$

问题变为求解这个一阶线性偏微分方程，考虑用[特征线法](https://zhuanlan.zhihu.com/p/340161952)。方程改写为：

$$
a(x,y,F)F_x+b(x,y,F)F_y=c(x,y,F)
$$

其中 $a(x,y,F) = 1 - x ^ 2, b(x,y,F) = 1,c(x,y,F) = 0$，故有特征系统：

$$
\begin{array}{c}
\dfrac{\mathrm{d}x}{\mathrm{d}t}  &=& 1 - x ^ 2 &(1)\\\\
\dfrac{\mathrm{d}y}{\mathrm{d}t}  &=& 1 & (2) \\\\
\dfrac{\mathrm{d}F}{\mathrm{d}t}  &=& 0 & (3)
\end{array}
$$

将 $(1), (2)$ 方程联立消掉 $\mathrm{d}t$ 得 $\mathrm{d}y = \dfrac{1}{1 - x ^ 2} \mathrm{d}x$，两边积分得 $2y + C_1 = \ln\left|\dfrac{1 + x}{1 - x}\right|$

再由 $(3)$ 得 $F = C_2$，与上式结合得：

$$
F(x, y) = h\left(\ln\left|\dfrac{1 + x}{1 - x}\right| - 2y\right)
$$

其中 $h$ 为任意函数，此时由边界条件 $F(x, 0) = x$ 代入得： 

$$
x = h\left(\ln\left|\dfrac{1 + x}{1 - x}\right|\right)
$$

于是令 $u = \ln\left|\dfrac{1 + x}{1 - x}\right|$，反解 $x$ 得 $\dfrac{e ^ u - 1}{e ^ u + 1} = x$，所以 $h(u) = \dfrac{e ^ u - 1}{e ^ u + 1}$，即 $h(x) = \dfrac{e ^ x - 1}{e ^ x + 1}$

将 $\ln\left|\dfrac{1 + x}{1 - x}\right| - 2y$ 代入到 $h(x)$，整理出 $F(x, y)$：

$$
F(x, y) = \dfrac{1 + x - (1 - x)e ^ {2y}}{1 + x + (1 - x)e ^ {2y}}
$$

故

$$
\dfrac{1 + x - (1 - x)e ^ {2y}}{1 + x + (1 - x)e ^ {2y}} = \sum_{i = 0} ^ {\infty}\frac{f_i(x)}{i!} y ^ i
$$

直接代入 $x_0$ 并求 $f_n(x_0)$，那么答案就为

$$
f_n(x_0) = n!\times [y ^ n] \left(\dfrac{1 + x_0 - (1 - x_0) e ^ {2y}}{1 + x_0 + (1 - x_0)e ^ {2y}}\right)
$$

展开 $e ^ {2y}$ 到 $n$ 项，进行多项式求逆再卷积即可。

## 代码：

```cpp
#include <bits/stdc++.h>
using namespace std;
using i64 = long long;
constexpr int mod = 998244353;
template<class T>
T power(T a, int b) {
    T res = 1;
    for (; b; b /= 2, a *= a) {
        if (b % 2) {
            res *= a;
        }
    }
    return res;
}
template<int mod>
struct ModInt {
    int x;
    ModInt() : x(0) {}
    ModInt(i64 y) : x(y >= 0 ? y % mod : (mod - (-y) % mod) % mod) {}
    ModInt &operator+=(const ModInt &p) {
        if ((x += p.x) >= mod) x -= mod;
        return *this;
    }
    ModInt &operator-=(const ModInt &p) {
        if ((x += mod - p.x) >= mod) x -= mod;
        return *this;
    }
    ModInt &operator*=(const ModInt &p) {
        x = (int)(1LL * x * p.x % mod);
        return *this;
    }
    ModInt &operator/=(const ModInt &p) {
        *this *= p.inv();
        return *this;
    }
    ModInt operator-() const {
        return ModInt(-x);
    }
    ModInt operator+(const ModInt &p) const {
        return ModInt(*this) += p;
    }
    ModInt operator-(const ModInt &p) const {
        return ModInt(*this) -= p;
    }
    ModInt operator*(const ModInt &p) const {
        return ModInt(*this) *= p;
    }
    ModInt operator/(const ModInt &p) const {
        return ModInt(*this) /= p;
    }
    bool operator==(const ModInt &p) const {
        return x == p.x;
    }
    bool operator!=(const ModInt &p) const {
        return x != p.x;
    }
    ModInt inv() const {
        int a = x, b = mod, u = 1, v = 0, t;
        while (b > 0) {
            t = a / b;
            swap(a -= t * b, b);
            swap(u -= t * v, v);
        }
        return ModInt(u);
    }
    ModInt pow(i64 n) const {
        ModInt res(1), mul(x);
        while (n > 0) {
            if (n & 1) res *= mul;
            mul *= mul;
            n >>= 1;
        }
        return res;
    }
    friend ostream &operator<<(ostream &os, const ModInt &p) {
        return os << p.x;
    }
    friend istream &operator>>(istream &is, ModInt &a) {
        i64 t;
        is >> t;
        a = ModInt<mod>(t);
        return (is);
    }
    int val() const {
        return x;
    }
    static constexpr int val_mod() {
        return mod;
    }
};
using Z = ModInt<mod>;
vector<Z> fact, infact;
void init(int n) {
    fact.resize(n + 1), infact.resize(n + 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i <= n; i ++) {
        fact[i] = fact[i - 1] * i;
    }
    infact[n] = fact[n].inv();
    for (int i = n; i; i --) {
        infact[i - 1] = infact[i] * i;
    }
}
Z C(int n, int m) {
    if (n < 0 || m < 0 || n < m) return Z(0);
    return fact[n] * infact[n - m] * infact[m];
}
vector<int> rev;
vector<Z> roots{0, 1};
void dft(vector<Z> &a) {
    int n = a.size();
    if (int(rev.size()) != n) {
        int k = __builtin_ctz(n) - 1;
        rev.resize(n);
        for (int i = 0; i < n; i ++) {
            rev[i] = rev[i >> 1] >> 1 | (i & 1) << k;
        }
    }
    for (int i = 0; i < n; i ++) {
        if (rev[i] < i) {
            swap(a[i], a[rev[i]]);
        }
    }
    if (int(roots.size()) < n) {
        int k = __builtin_ctz(roots.size());
        roots.resize(n);
        while ((1 << k) < n) {
            Z e = power(Z(3), (mod - 1) >> (k + 1));
            for (int i = 1 << (k - 1); i < (1 << k); i ++) {
                roots[i << 1] = roots[i];
                roots[i << 1 | 1] = roots[i] * e;
            }
            k ++;
        }
    }
    for (int k = 1; k < n; k *= 2) {
        for (int i = 0; i < n; i += 2 * k) {
            for (int j = 0; j < k; j ++) {
                Z u = a[i + j], v = a[i + j + k] * roots[k + j];
                a[i + j] = u + v, a[i + j + k] = u - v;
            }
        }
    }
}
void idft(vector<Z> &a) {
    int n = a.size();
    reverse(a.begin() + 1, a.end());
    dft(a);
    Z inv = (1 - mod) / n;
    for (int i = 0; i < n; i ++) {
        a[i] *= inv;
    }
}
struct Poly {
    vector<Z> a;
    Poly() {}
    Poly(const vector<Z> &a) : a(a) {}
    Poly(const initializer_list<Z> &a) : a(a) {}
    int size() const {
        return a.size();
    }
    void resize(int n) {
        a.resize(n);
    }
    Z operator[](int idx) const {
        if (idx < size()) {
            return a[idx];
        } else {
            return 0;
        }
    }
    Z &operator[](int idx) {
        return a[idx];
    }
    Poly mulxk(int k) const {
        auto b = a;
        b.insert(b.begin(), k, 0);
        return Poly(b);
    }
    Poly modxk(int k) const {
        k = min(k, size());
        return Poly(vector<Z>(a.begin(), a.begin() + k));
    }
    Poly divxk(int k) const {
        if (size() <= k) {
            return Poly();
        }
        return Poly(vector<Z>(a.begin() + k, a.end()));
    }
    friend Poly operator+(const Poly &a, const Poly &b) {
        vector<Z> res(max(a.size(), b.size()));
        for (int i = 0; i < int(res.size()); i ++) {
            res[i] = a[i] + b[i];
        }
        return Poly(res);
    }
    friend Poly operator-(const Poly &a, const Poly &b) {
        vector<Z> res(max(a.size(), b.size()));
        for (int i = 0; i < int(res.size()); i ++) {
            res[i] = a[i] - b[i];
        }
        return Poly(res);
    }
    friend Poly operator*(Poly a, Poly b) {
        if (a.size() == 0 || b.size() == 0) {
            return Poly();
        }
        int sz = 1, tot = a.size() + b.size() - 1;
        while (sz < tot) {
            sz *= 2;
        }
        a.a.resize(sz);
        b.a.resize(sz);
        dft(a.a);
        dft(b.a);
        for (int i = 0; i < sz; i ++) {
            a.a[i] = a[i] * b[i];
        }
        idft(a.a);
        a.resize(tot);
        return a;
    }
    friend Poly operator*(Z a, Poly b) {
        for (int i = 0; i < int(b.size()); i ++) {
            b[i] *= a;
        }
        return b;
    }
    friend Poly operator*(Poly a, Z b) {
        for (int i = 0; i < int(a.size()); i ++) {
            a[i] *= b;
        }
        return a;
    }
    Poly &operator+=(Poly b) {
        return (*this) = (*this) + b;
    }
    Poly &operator-=(Poly b) {
        return (*this) = (*this) - b;
    }
    Poly &operator*=(Poly b) {
        return (*this) = (*this) * b;
    }
    Poly deriv() const {
        if (a.empty()) {
            return Poly();
        }
        vector<Z> res(size() - 1);
        for (int i = 0; i < size() - 1; i ++) {
            res[i] = a[i + 1] * (i + 1);
        }
        return Poly(res);
    }
    Poly integr() const {
        vector<Z> res(size() + 1);
        for (int i = 0; i < size(); i ++) {
            res[i + 1] = a[i] / (i + 1);
        }
        return Poly(res);
    }
    Poly inv(int m) const {
        Poly x{a[0].inv()};
        int k = 1;
        while (k < m) {
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
        while (k < m) {
            k *= 2;
            x = (x * (Poly{1} - x.log(k) + modxk(k))).modxk(k);
        }
        return x.modxk(m);
    }
    Poly pow(int k, int m) const {
        int i = 0;
        while (i < size() && a[i].val() == 0) {
            i ++;
        }
        if (i == size() || 1LL * i * k >= m) {
            return Poly(vector<Z>(m));
        }
        Z v = a[i];
        auto f = divxk(i) * v.inv();
        return (f.log(m - i * k) * k).exp(m - i * k).mulxk(i * k) * power(v, k);
    }
    Poly sqrt(int m) const {
        Poly x{1};
        int k = 1;
        while (k < m) {
            k *= 2;
            x = (x + (modxk(k) * x.inv(k)).modxk(k)) * ((mod + 1) / 2);
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
    vector<Z> eval(vector<Z> x) const {
        if (size() == 0) {
            return vector<Z>(x.size(), 0);
        }
        const int n = max(int(x.size()), size());
        vector<Poly> q(n << 2);
        vector<Z> ans(x.size());
        x.resize(n);
        function<void(int, int, int)> build = [&](int p, int l, int r) {
            if (r - l == 1) {
                q[p] = Poly{1, -x[l]};
            } else {
                int m = l + r >> 1;
                build(p << 1, l, m);
                build(p << 1 | 1, m, r);
                q[p] = q[p << 1] * q[p << 1 | 1];
            }
        };
        build(1, 0, n);
        function<void(int, int, int, const Poly &)> work = [&](int p, int l, int r, const Poly &num) {
            if (r - l == 1) {
                if (l < int(ans.size())) {
                    ans[l] = num[0];
                }
            } else {
                int m = (l + r) / 2;
                work(p << 1, l, m, num.mulT(q[p << 1 | 1]).modxk(m - l));
                work(p << 1 | 1, m, r, num.mulT(q[p << 1]).modxk(r - m));
            }
        };
        work(1, 0, n, mulT(q[1].inv(n)));
        return ans;
    }
    Poly inter(const Poly &y) const {
        vector<Poly> Q(a.size() << 2), P(a.size() << 2);
        function<void(int, int, int)> dfs1 = [&](int p, int l, int r) {
            int m = l + r >> 1;
            if (l == r) {
                Q[p].a.push_back(-a[m]);
                Q[p].a.push_back(Z(1));
                return;
            }
            dfs1(p << 1, l, m), dfs1(p << 1 | 1, m + 1, r);
            Q[p] = Q[p << 1] * Q[p << 1 | 1];
        };
        dfs1(1, 0, a.size() - 1);
        Poly f;
        f.a.resize((int)(Q[1].size()) - 1);
        for (int i = 0; i + 1 < Q[1].size(); i ++) {
            f[i] = Q[1][i + 1] * (i + 1);
        }
        Poly g = f.eval(a);
        function<void(int, int, int)> dfs2 = [&](int p, int l, int r) {
            int m = l + r >> 1;
            if (l == r) {
                P[p].a.push_back(y[m] * power(g[m], mod - 2));
                return;
            }
            dfs2(p << 1, l, m), dfs2(p << 1 | 1, m + 1, r);
            P[p].a.resize(r - l + 1);
            Poly A = P[p << 1] * Q[p << 1 | 1];
            Poly B = P[p << 1 | 1] * Q[p << 1];
            for (int i = 0; i <= r - l; i ++) {
                P[p][i] = A[i] + B[i];
            }
        };
        dfs2(1, 0, a.size() - 1);
        return P[1];
    }
};
Poly toFPP(vector<Z> &a) {
    int n = a.size();
    vector<Z> b(n);
    iota(b.begin(), b.end(), 0);
    auto F = Poly(a).eval(b);
    vector<Z> f(n), g(n);
    for (int i = 0, sign = 1; i < n; i ++, sign *= -1) {
        f[i] = F[i] * infact[i];
        g[i] = Z(sign) * infact[i];
    }
    return Poly(f) * Poly(g);
}
Poly toOP(vector<Z> &a) {
    int n = a.size();
    vector<Z> g(n);
    for (int i = 0; i < n; i ++) {
        g[i] = infact[i];
    }
    auto F = Poly(a) * Poly(g);
    for (int i = 0; i < n; i ++) {
        F[i] *= fact[i];
    }
    vector<Z> p(n);
    iota(p.begin(), p.end(), 0);
    return Poly(p).inter(F);
}
Poly FPPMul(Poly a, Poly b) {
    int n = a.size() + b.size() - 1;
    Poly p;
    p.resize(n);
    for (int i = 0; i < n; i ++) {
        p[i] = infact[i];
    }
    a *= p, b *= p;
    for (int i = 0; i < n; i ++) {
        a[i] *= b[i] * fact[i];
    }
    for (int i = 1; i < n; i += 2) {
        p[i] = -p[i];
    }
    a *= p;
    a.resize(n);
    return a;
}
Poly Lagrange2(vector<Z> &f, int m, int k) {
    int n = f.size() - 1;
    vector<Z> a(n + 1), b(n + 1 + k);
    for (int i = 0; i <= n; i ++) {
        a[i] = f[i] * ((n - i) & 1 ? -1 : 1) * infact[n - i] * infact[i];
    }
    for (int i = 0; i <= n + k; i ++) {
        b[i] = Z(1) / (m - n + i);
    }
    Poly ans = Poly(a) * Poly(b);
    for (int i = 0; i <= k; i ++) {
        ans[i] = ans[i + n];
    }
    ans.resize(k + 1);
    Z sum = 1;
    for (int i = 0; i <= n; i ++) {
        sum *= m - i;
    }
    for (int i = 0; i <= k; i ++) {
        ans[i] *= sum;
        sum *= Z(m + i + 1) / (m - n + i);
    }
    return ans;
}
Poly Chirp_Z_Transform(vector<Z> &a, int c, int m) {
    int n = a.size();
    a.resize(n + m - 1);
    Poly f, g;
    f.resize(n + m - 1), g.resize(n + m - 1);
    for (int i = 0; i < n + m - 1; i ++) {
        f[n - 1 + m - 1 - i] = power(Z(c), i * (i - 1LL) / 2 % (mod - 1));
        g[i] = power(Z(c), mod - 1 - i * (i - 1LL) / 2 % (mod - 1)) * a[i];
    }
    Poly res = f * g, ans;
    ans.resize(m);
    for (int i = 0; i < m; i ++) {
        ans[i] = res[n - 1 + m - 1 - i] * power(Z(c), mod - 1 - i * (i - 1LL) / 2 % (mod - 1));
    }
    return ans;
}
Poly S2_row;
void S2_row_init(int n) {
    vector<Z> f(n + 1), g(n + 1);
    for (int i = 0; i <= n; i ++) {
        f[i] = power(Z(i), n) * infact[i];
        g[i] = Z(i & 1 ? -1 : 1) * infact[i];
    }
    S2_row = Poly(f) * Poly(g);
}
Poly S2_col;
void S2_col_init(int n, int k) {
    n ++;
    vector<Z> f(n);
    for (int i = 1; i < n; i ++) {
        f[i] = infact[i];
    }
    auto ans = Poly(f).pow(k, n);
    S2_col.resize(n + 1);
    for (int i = 0; i < n; i ++) {
        S2_col[i] = ans[i] * fact[i] * infact[k];
    }
}
Poly Bell;
void Bell_init(int n) {
    vector<Z> f(n + 1);
    for (int i = 1; i <= n; i ++) {
        f[i] = infact[i];
    }
    auto ans = Poly(f).exp(n + 1);
    Bell.resize(n + 1);
    for (int i = 0; i <= n; i ++) {
        Bell[i] = ans[i] * fact[i];
    }
}
vector<Z> p;
void p_init(int n) {
    vector<int> f(n + 1);
    p.resize(n + 1);
    p[0] = 1;
    f[0] = 1, f[1] = 2, f[2] = 5, f[3] = 7;
    for (int i = 4; f[i - 1] <= n; i ++) {
        f[i] = 3 + 2 * f[i - 2] - f[i - 4];
    }
    for (int i = 1; i <= n; i ++) {
        for (int j = 0; f[j] <= i; j ++) {
            p[i] += Z(j & 2 ? -1 : 1) * p[i - f[j]];
        }
    }
}
Poly P;
void p_init(int n, int m) {
    vector<Z> a(n + 1);
    for (int i = 1; i <= m; i ++) {
        for (int j = i; j <= n; j += i) {
            a[j] += Z(j / i).inv();
        }
    }
    P = Poly(a).exp(n + 1);
}
signed main() {
    cin.tie(0) -> sync_with_stdio(0);
    int n, x0;
    cin >> n >> x0;
    init(n);
    vector<Z> f(n + 1), g(n + 1);
    for (int i = 0; i <= n; i ++) {
        f[i] = g[i] = power(Z(2), i) * infact[i];
        f[i] *= x0 - 1, g[i] *= 1 - x0;
    }
    f[0] += 1 + x0, g[0] += 1 + x0;
    Poly ans = Poly(f) * Poly(g).inv(n + 1);
    cout << fact[n] * ans[n] << "\n";
}
```