---
title: "[Educational Codeforces Round 133 F] Bags with Balls"
date: 2022-08-05 15:37:55
tags:
- 组合计数
- 推式子
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://codeforces.com/contest/1716/problem/F)

**题意**

给定三个正整数 $n,m,k$，有 $n$ 个盒子，每个盒子有 $m$ 个标号分别为 $1 \sim m$ 的球，现从每个盒子选出恰好一个球，将奇数编号的球的个数记为 $F$，求所有方案的 $F ^ k$ 之和对 $998 \, 244 \, 353$ 取模。 

$(1 \le n, m \le 998244352, 1 \le k \le 2 \times 10 ^ 3)$

**分析：**

首先每个盒子有 $\lceil \dfrac{m}{2} \rceil$ 个奇数球和 $\lfloor \dfrac{m}{2} \rfloor$ 个偶数球，那么所有方案数为 $m ^ n = (\lceil \dfrac{m}{2} \rceil + \lfloor \dfrac{m}{2} \rfloor) ^ n$，根据二项式定理，所以每个 $F$ 的贡献就为 $F ^ k \times \dbinom{n}{F} \times \lceil \dfrac{m}{2} \rceil ^ F \times \lfloor \dfrac{m}{2} \rfloor ^ {n - F}$ 所以总答案为
$$
\sum_{i = 0} ^ {n} i ^ k \times \dbinom{n}{i} \times \lceil \dfrac{m}{2} \rceil ^ i \times \lfloor \dfrac{m}{2} \rfloor ^ {n - i}
$$
由于 $n \le 998244352$，没法直接求，但是看到 $i ^ k$ 想到自然数幂展开
$$
i ^ k = \sum_{j = 0} ^ {k} {k \brace j} \times j! \times \binom{i}{j}
$$
带入得
$$
\sum_{i = 0} ^ {n} \dbinom{n}{i} \times \lceil \dfrac{m}{2} \rceil ^ i \times \lfloor \dfrac{m}{2} \rfloor ^ {n - i} \sum_{j = 0} ^ {k} {k \brace j} \times j! \times \binom{i}{j}
$$
将 $\dbinom{n}{i}$ 放到后面的求和号化简：$\dbinom{n}{i} \times j! \times \dbinom{i}{j} = \dfrac{n!}{i! \times (n - i)!} \times j! \times \dfrac{i!}{j! \times (i - j)!} = \dfrac{n!}{(n - i)! \times (i - j)!}$，那么式子变为
$$
\sum_{i = 0} ^ {n} \lceil \dfrac{m}{2} \rceil ^ i \times \lfloor \dfrac{m}{2} \rfloor ^ {n - i} \sum_{j = 0} ^ {k} {k \brace j} \times \dfrac{n!}{(n - i)! \times (i - j)!}
$$
交换求和次序，注意 $i$ 要从 $j$ 开始，因为要保证 $i - j \ge 0$
$$
\sum_{j = 0} ^ {k} {k \brace j} \sum_{i = j} ^ {n} \dfrac{n!}{(n - i)! \times (i - j)!} \times \lceil \dfrac{m}{2} \rceil ^ i \times \lfloor \dfrac{m}{2} \rfloor ^ {n - i}
$$
对第二个和式做变换 $i - j \rightarrow i$
$$
\sum_{j = 0} ^ {k} {k \brace j} \sum_{i = 0} ^ {n - j} \dfrac{n!}{(n - i - j)! \times i!} \times \lceil \dfrac{m}{2} \rceil ^ {i + j} \times \lfloor \dfrac{m}{2} \rfloor ^ {n - i - j}
$$
到这里发现第二个和式比较像二项式展开了，即 $(a + b) ^ n = \sum\limits_{i = 0} ^ {n} \dbinom{n}{i} a ^ {i} b ^ {n - i}$，那么考虑往这个方向凑式子，首先要解决的是组合数，发现第二个和式上界为 $n - j$，那么我们就要凑一个 $\dbinom{n - j}{i} = \dfrac{(n - j)!}{(n - i - j) \times i!}$ 的组合数，发现恰好多了 $n \times(n - 1) \times \cdots \times (n - j + 1) = n ^ {\underline j}$，那么后面也多了 $\lceil \dfrac{m}{2} \rceil ^ {j}$，提出来之后为
$$
\sum_{j = 0} ^ {k} {k \brace j} \times n ^ {\underline j} \times \lceil \dfrac{m}{2} \rceil ^ {j} \sum_{i = 0} ^ {n - j} \binom{n - j}{i} \times \lceil \dfrac{m}{2} \rceil ^ {i} \times \lfloor \dfrac{m}{2} \rfloor ^ {n - i - j}
$$
由二项式定理得 $\sum\limits_{i = 0} ^ {n - j} \dbinom{n - j}{i} \times \lceil \dfrac{m}{2} \rceil ^ {i} \times \lfloor \dfrac{m}{2} \rfloor ^ {n - i - j} = (\lceil \dfrac{m}{2} \rceil + \lfloor \dfrac{m}{2} \rfloor) ^ {n - j} = m ^ {n - j}$，那么答案为
$$
\sum_{j = 0} ^ {k} {k \brace j} \times n ^ {\underline j} \times \lceil \dfrac{m}{2} \rceil ^ {j} \times m ^ {n - j}
$$
所以只需要预处理 $2 \times 10 ^ 3$ 以内的第二类斯特林数，再 $O(k)$ 维护下降幂即可。

## 代码：

```cpp
#include <bits/stdc++.h>
using namespace std;
using i64 = long long;
constexpr int mod = 998244353, N = 2e3;
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
vector<vector<Z>> stirling(N + 1, vector<Z>(N + 1));
void init() {
    stirling[0][0] = 1;
    for (int i = 1; i <= N; i ++) {
        for (int j = 1; j <= i; j ++) {
            stirling[i][j] = stirling[i - 1][j - 1] + j * stirling[i - 1][j];
        }
    }
}
void solve() {
    int n, m, k;
    cin >> n >> m >> k;
    Z res, sum = 1;
    for (int i = 0, cnt = n; i <= k; i ++, cnt --) {
        res += stirling[k][i] * power(Z((m + 1) / 2), i) * power(Z(m), n - i) * sum;
        sum *= cnt;
    }
    cout << res << "\n";
}
signed main() {
    init();
    cin.tie(0) -> sync_with_stdio(0);
    int T;
    cin >> T;
    while (T --) {
        solve();
    }
}
```