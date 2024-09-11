---
title: "[2021 CCPC 广州 A] Math Ball"
date: 2022-10-16 00:32:23
tags:
- 生成函数
- 多项式
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://codeforces.com/gym/103415/problem/A)

**题意**

给定 $n$ 个权值为 $c_1,c_2,\cdots,c_n$ 的物品，总共最多取 $W$ 个，求
$$
\sum_{k_1+k_2+\cdots+k_n \le W} \prod_{i = 1} ^ {n} k_i ^ {c_i}
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
现在考虑化简 $\sum\limits_{j = 0} ^ {\infty}\dbinom{j}{k} x ^ j$，我们根据广义二项式定理知道 $\sum\limits_{i = 0} ^ {\infty} \dbinom{i + k - 1}{i}x ^ i$ 的封闭形式为 $\dfrac{1}{(1 - x) ^ k}$，那么 $\sum\limits_{i = 0} ^ {\infty} \dbinom{i + k}{i}x ^ i = \dfrac{1}{(1 - x) ^ {k + 1}}$

由组合数性质有 $\dbinom{i + k}{i}=\dbinom{i + k}{k}$，所以 $\sum\limits_{i = 0} ^ {\infty} \dbinom{i + k}{i}x ^ i = \sum\limits_{i = 0} ^ {\infty} \dbinom{i + k}{k}x ^ i = \sum\limits_{i = k} ^ {\infty} \dbinom{i}{k}x ^ {i - k} = \dfrac{1}{(1 - x) ^ {k + 1}}$ 再等式两边同乘 $x ^ k$，得出结论
$$
\sum_{i = 0} ^ {\infty} \binom{i}{k}x ^ i = \frac{x ^ k}{(1 - x) ^ {k + 1}}
$$
(由于 $\dbinom{n}{m}$ 在 $n < m$ 时为 $0$，所以 $i$ 从 $0$ 或从 $k$ 开始都一样)

所以进一步化简了 $f_i(x)$，为
$$
f_i(x) = \sum_{k = 0} ^ {c_i} {c_i \brace k} \times k! \times \frac{x ^ k}{(1 - x) ^ {k + 1}}
$$
尽管如此，$f_i(x)$ 仍然不好算，我们注意到题目条件 $\sum\limits_{i = 1} ^ {n} c_i \le 10 ^ 5$，一般会往分治 $\texttt{NTT}$ 上考虑，我们不妨将 $f_i(x)$ 的形式变成
$$
f_i(x) = \sum\limits_{k = 0} ^ {c_i} {c_i \brace k} \times k! \times (\frac{x}{1-x}) ^ {k} \times \frac{1}{1 - x}
$$
此时如果令 $y = \dfrac{x}{1 - x},f_i(x) = g_i(y) \times \dfrac{1}{1 - x}$，其中 $g_i(y) = \sum\limits_{k = 0} ^ {c_i} {c_i \brace k} \times k! \times y ^ {k}$ 那么答案就为
$$
[x ^ W] \frac{\prod_{j = 1} ^ {n} g_j(y)}{(1 - x) ^ {n + 1}}
$$
这样的话 $\prod\limits_{j = 1} ^ {n} g_j(y)$ 是可以用分治 $\texttt{NTT}$ 求解的，其中需要用到快速求解第二类斯特林数的每一行，求出之后考虑计算第 $W$ 项的系数。

令 $F(x) = \prod\limits_{j = 1} ^ {n} g_j(y)$，那么 $F(x)$ 的第 $k$ 项就形如 $a_k \times (\dfrac{x}{1 - x}) ^ k$，其中 $a_k$ 为 $F(x)$ 的第 $k$ 项系数，那么 $a_k \times (\dfrac{x}{1 - x}) ^ k = a_k \times x ^ k \times \dfrac{1}{(1 - x) ^ k}$，把 $\dfrac{1}{(1 - x) ^ k}$ 拿到下面，变为 $\dfrac{1}{(1 - x) ^ {n + k + 1}}$

由于 $F(x)$ 项数较少，所以考虑枚举 $F(x)$ 的每一项，即答案为 $a_k$ 与 $\dfrac{1}{(1 - x) ^ {n + k + 1}}$ 的第 $W - k$ 项的乘积之和，考虑展开 $\dfrac{1}{(1 - x) ^ {n + k + 1}} = \sum\limits_{i = 0} ^ {\infty} \dbinom{n + k + i}{i}x ^ i$，所以第 $W - k$ 项为 $\dbinom{W + n}{W - k}$，虽然 $W$ 较大，但 $n$ 很小，经典维护 $W$ 的下降幂即可。

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
    init(1e5);
    cin.tie(0) -> sync_with_stdio(0);
    int n;
    Z w;
    cin >> n >> w;
    vector<int> c(n + 1);
    for (int i = 1; i <= n; i ++) {
        cin >> c[i];
    }
    function<Poly(int, int)> dc = [&](int l, int r) {
        if (l == r) {
            S2_row_init(c[l]);
            vector<Z> f(c[l] + 1);
            for (int i = 0; i <= c[l]; i ++) {
                f[i] = fact[i] * S2_row[i];
            }
            return Poly(f);
        }
        int mid = l + r >> 1;
        return dc(l, mid) * dc(mid + 1, r);
    };
    auto ans = dc(1, n);
    Z sum = infact[n];
    for (i64 i = w.val() + n; i > w.val(); i --) {
        sum *= Z(i);
    }
    Z res;
    for (int i = 0; i < ans.size() && i <= w.val(); i ++) {
        res += ans[i] * sum;
        sum *= (w - i) / (n + i + 1);
    }
    cout << res << "\n";
}
```