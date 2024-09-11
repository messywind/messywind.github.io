---
title: "[2021ICPC济南 L] Strange Series"
date: 2022-08-30 13:26:27
tags:
- Bell 数
- 多项式
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

**题意**

$T$ 组输入，给定一个 $n$ 次多项式 $f(x) = a_0 + a_1x + \cdots + a_nx ^ n$，定义 $S = \sum\limits_{i = 0} ^ {\infty} \dfrac{f(i)}{i!}$，可以证明 $S$ 一定是 $e$ 的倍数，即 $S = p \times e$，求 $p$ 对 $998\,244\,353$ 取模。

$1 \le T \le 100, 0 \le n \le 10 ^ 5,0 \le a_i < 998\,244\,353$

**分析：**

首先将 $f(x)$ 代入 $S$ 得
$$
\sum_{i = 0} ^ {\infty}\frac{1}{i!} \sum_{j = 0} ^ {n}a_j \times i ^ j 
$$
看到自然数幂想到展开 $i ^ k = \sum\limits_{j = 0} ^ {k} {k \brace j} i ^{\underline j}$，代入得
$$
\sum_{i = 0} ^ {\infty} \frac{1}{i!} \sum_{j = 0} ^ {n}a_j \sum_{k = 0} ^ {j} {j \brace k} i ^ {\underline k}
$$
交换求和次序，先对 $i$ 求和
$$
\sum_{j = 0} ^ {n}a_j \sum_{k = 0} ^ {j} {j \brace k} \sum_{i = 0} ^ {\infty} \frac{i ^ {\underline k}}{i!}
$$
把下降幂消掉
$$
\sum_{j = 0} ^ {n}a_j \sum_{k = 0} ^ {j} {j \brace k} \sum_{i = k} ^ {\infty} \frac{1}{(i-k)!}
$$
做变换 $(i - k) \rightarrow i$
$$
\sum_{j = 0} ^ {n}a_j \sum_{k = 0} ^ {j} {j \brace k} \sum_{i = 0} ^ {\infty} \frac{1}{i!}
$$
由于 $e = \sum\limits_{i = 0} ^ {\infty} \dfrac{1}{i!}$，所以原式为 $e$ 的倍数得证，那么式子变为
$$
\sum_{j = 0} ^ {n}a_j \sum_{k = 0} ^ {j} {j \brace k} \times e
$$
事实上 $\text{Bell} _ {n} = \sum \limits_{i = 0} ^ {n} {n \brace i}$，其中 $\text{Bell}_{n}$ 为第 $n$ 项贝尔数，代表 $n$ 个元素的集合划分为任意非空子集的方案数，所以答案就为

$$
\sum_{i = 0} ^ {n} a_i \times \text{Bell}_{i}
$$

考虑快速求解贝尔数，设贝尔数的 $\textbf{EGF}$ 为 $B(x) = \sum\limits_ {i = 0} ^ {\infty} \dfrac{F(x) ^ i}{i!}$，其中 $F(x) = \sum\limits_{i = 1} ^ {\infty}\dfrac{x ^ i}{i!} = e ^ x - 1$，那么 $B(x) = \sum\limits_{i = 0} ^ {\infty} \dfrac{(e ^ x - 1) ^ i}{i!} = e ^ {e ^ {x} - 1}$，直接多项式 $\text{exp}$ 就好了。

## 代码：

```cpp
#include <bits/stdc++.h>
using namespace std;
using i64 = long long;
constexpr int mod = 998244353;
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
vector<Z> fact, infact, f;
Poly bell;
void init(int n) {
    fact.resize(n + 1), infact.resize(n + 1), f.resize(n + 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i <= n; i ++) {
        fact[i] = fact[i - 1] * i;
    }
    infact[n] = fact[n].inv();
    for (int i = n; i; i --) {
        infact[i - 1] = infact[i] * i;
    }
    for (int i = 1; i <= n; i ++) {
        f[i] = infact[i];
    }
    bell = Poly(f).exp(n + 1);
    for (int i = 1; i <= n; i ++) {
    	bell[i] *= fact[i];
    }
}
void solve() {
	int n;
	cin >> n;
	Z res;
	for (int i = 0; i <= n; i ++) {
		int x;
		cin >> x;
		res += bell[i] * x;
	}
	cout << res << "\n";
}
signed main() {
    init(1e5);
    cin.tie(0) -> sync_with_stdio(0);
    int T;
    cin >> T;
    while (T --) {
        solve();
    }
}
```