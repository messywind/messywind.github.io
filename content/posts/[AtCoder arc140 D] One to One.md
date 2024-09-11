---
title: "[AtCoder arc140 D] One to One"
date: 2022-05-18 12:48:28
tags:
- 图论
- 组合计数
- 分治 NTT
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://atcoder.jp/contests/arc140/tasks/arc140_d)

**题意**

初始有 $n$ 个点，给定一个长度为 $n$ 的数组 $a_i$，若 $a_i \ne -1$，则有无向边 $(i, a_i)$，若 $a_i = -1$，则点 $i$ 可以连向 $1 \sim n$ 任意点，求所有图的联通块个数之和

$1 \le n \le 2 \times 10 ^ 3, a_i \in [1, n] \cup \{-1\}$

对 $998244353$ 取模。

**分析：**

首先考虑忽略 $a_i = -1$ 的所有边，那么图中会有若干个连通块，这些连通块分为三种情况：

- 树
- 基环树
- 环

对于环和基环树来说，因为是 $n$ 个点和 $n$ 条边，所以他们不可能有一条出边，换句话说，里边的点不可能包含 $a_i = -1$，而对于树来说，因为是 $n$ 个点 $n - 1$ 条边，所以**有且仅有**一条出边，也就是树里面只有一个 $a_i = -1$

这就代表树可以和其他连通块组成一个新的连通块，但是无论树如何连边，环和基环树的连通性都不会发生变化，也就是他始终有一个环，所以可以先计算出这部分的贡献，设图中环和基环树的数量为 $u$，树的数量为 $v$，则这部分贡献就为 $u \times n ^ {v}$

接下来考虑树的所有连边情况，我们枚举 $k$ 条边组成一个环，设第 $i$ 棵树的大小为 $f_i$，每棵树则有生成函数 $1 + f_ix$，记 $F(x)$ 为选若干个树构成一个环的方案数，可以用分治 $\text{NTT}$ 快速求出。
$$
F(x) = \prod_{i = 1} ^ {v} (1 + f_ix)
$$
每个点构成一个 $k$ 元环是有顺序的，第一个点可以有 $k - 1$ 种选择，第二个点有 $k - 2$ 种选择，所以总共构成一个 $k$ 元环的方案数为 $(k - 1)!$，还要考虑剩下没有被选出来的点，那么可以随便连，都不影响这个环，方案数就为 $n ^ {v - k}$，那么答案就是
$$
u \times n ^ {v} + \sum_{k = 1} ^ {v} (k - 1)! \times [x ^ k]F(x) \times n ^ {v - k}
$$
时间复杂度 $O(n\log ^ 2n)$

## 代码：

```cpp
#include <bits/stdc++.h>
using namespace std;
using i64 = long long;
constexpr int mod = 998244353;
struct DSU {
    vector<int> p, Size;
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
            Size[v] += Size[u];
            p[u] = v;
        }
    }
};
int norm(int x) {
    if (x < 0) {
        x += mod;
    }
    if (x >= mod) {
        x -= mod;
    }
    return x;
}
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
    Z &operator*=(const Z &rhs) {
        x = i64(x) * rhs.x % mod;
        return *this;
    }
    Z &operator+=(const Z &rhs) {
        x = norm(x + rhs.x);
        return *this;
    }
    Z &operator-=(const Z &rhs) {
        x = norm(x - rhs.x);
        return *this;
    }
    Z &operator/=(const Z &rhs) {
        return *this *= rhs.inv();
    }
    friend Z operator*(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res *= rhs;
        return res;
    }
    friend Z operator+(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res += rhs;
        return res;
    }
    friend Z operator-(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res -= rhs;
        return res;
    }
    friend Z operator/(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res /= rhs;
        return res;
    }
    friend istream &operator>>(istream &is, Z &a) {
        i64 v;
        is >> v;
        a = Z(v);
        return is;
    }
    friend ostream &operator<<(ostream &os, const Z &a) {
        return os << a.val();
    }
};
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
            res[i] = (i + 1) * a[i + 1];
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
};
signed main() {
    cin.tie(0) -> sync_with_stdio(0);
    int n;
    cin >> n;
    DSU p(n + 1);
    vector<Z> fact(n + 1);
    fact[0] = 1;
    for (int i = 1; i <= n; i ++) {
        fact[i] = fact[i - 1] * i;
    }
    vector<bool> st(n + 1);
    for (int i = 1; i <= n; i ++) {
        int x;
        cin >> x;
        if (x != -1) {
            int flag = 0;
            if (p.same(i, x)) {
                flag = 1;
            }
            p.merge(i, x);
            if (flag) {
                st[p.find(i)] = 1;
            }
        }
    }
    int v = 0, u = 0;
    vector<Poly> f(n + 1);
    for (int i = 1; i <= n; i ++) {
        if (p.find(i) == i) {
            if (!st[i]) {
                v ++;
                f[v].resize(2);
                f[v][0] = 1, f[v][1] = p.Size[i];
            } else {
                u ++;
            }
        }
    }
    Z res = u * power(Z(n), v);
    if (v) {
        function<Poly(int, int)> dc = [&](int l, int r) {
            if (l == r) return f[l];
            int mid = l + r >> 1;
            return dc(l, mid) * dc(mid + 1, r);
        };
        auto ans = dc(1, v);
        for (int k = 1; k <= v; k ++) {
            res += fact[k - 1] * ans[k] * power(Z(n), v - k);
        }
    }
    cout << res << "\n";
}
```