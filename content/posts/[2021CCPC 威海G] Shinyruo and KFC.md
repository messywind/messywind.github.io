---
title: "[2021CCPC 威海G] Shinyruo and KFC"
date: 2022-10-04 15:21:43
tags:
- 下降幂多项式
- 多项式多点求值
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://codeforces.com/gym/103428/problem/G)

**题意**

给定 $n$ 个正整数 $a_1,a_2,\cdots,a_n$，并给定正整数 $m$，对于每个 $k \in [1, m]$，计算 $\prod\limits_{i = 1} ^ {n} \dbinom{k}{a_i}$

对 $998\,244\,353$ 取模。

$(1 \le n, m \le 5 \times 10 ^ 4, \sum\limits_{i = 1} ^ {n}a_i \le 10 ^ 5)$

**分析：**

考虑拆组合数
$$
\prod_{i = 1} ^ {n}\binom{k}{a_i}=\prod_{i = 1} ^ {n}\frac{k!}{a_i! \times (k - a_i)!} \\
= \frac{1}{\prod\limits_{i = 1} ^ {n}a_i!} \times \prod_{i = 1} ^ {n} k ^ {\underline {a_i}}
$$
所以可以把 $k ^ {\underline{a_i}}$ 看作一个下降幂多项式，那么使用分治下降幂多项式乘法可以求出 $\prod\limits_{i = 1} ^ {n} k ^ {\underline {a_i}}$，再转为普通幂多项式，再对 $(1, 2, \cdots,m)$ 使用多项式多点求值即可求出答案，时间复杂度 $O(n\log ^ 2 n)$

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
signed main() {
    init(2e5);
    cin.tie(0) -> sync_with_stdio(0);
    int n, m;
    cin >> n >> m;
    Z inv = 1;
    vector<int> a(n + 1);
    vector<vector<Z>> num(n + 1);
    for (int i = 1; i <= n; i ++) {
        cin >> a[i];
        inv *= infact[a[i]];
        num[i].resize(a[i] + 1);
        num[i][a[i]] = 1;
    }
    function<Poly(int, int)> dc = [&](int l, int r) {
        if (l == r) {
            return Poly(num[l]);
        }
        int mid = l + r >> 1;
        return FPPMul(dc(l, mid), dc(mid + 1, r));
    };
    vector<Z> q(m + 1);
    iota(q.begin(), q.end(), 0);
    auto ans = dc(1, n).a;
    auto res = toOP(ans).eval(q);
    for (int i = 1; i <= m; i ++) {
        cout << res[i] * inv << "\n";
    }
}
```