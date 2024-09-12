---
title: "DP 题目选讲"
date: 2022-07-07 10:52:14
tags:
- DP
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

## 背包：Acwing 1454 异或和是质数的子集数

题目链接 https://www.acwing.com/problem/content/description/1456/

**分析：**

考虑 $01$ 背包，$1 \sim n$ 中对于每件物品 $1 \le i \le n$ 的体积为 $i$

定义状态 $dp_{i, j}$ 为从前 $i$ 件物品中选，异或和为 $j$ 的方案数，那么有转移
$$
dp_{i,j} = dp_{i - 1,j} + dp_{i - 1,j \oplus a_i}
$$
由于按位异或会变小，所以状态必须开二维，但第一维可以用滚动数组优化。

注意到 $1 \le a_i \le 5 \times 10 ^ 3$，$2 ^ {12} = 4096 < 5 \times 10 ^ 3$，所以总体积最大为 $\sum_{i = 0} ^ {12} 2 ^ i = 2 ^ {13} - 1$

那么最后的答案为
$$
\sum_{i \in \text{prime}} dp_i
$$
时间复杂度 $O(2 ^{13} \times n)$

## 代码：

```cpp
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int mod = 1e9 + 7, N = 1 << 13;
signed main() {
    cin.tie(0) -> sync_with_stdio(0);
    int cnt = 0;
    vector<int> primes(N + 1);
    vector<bool> st(N + 1);
    auto sieve = [&](int n) {
        st[1] = 1;
        for (int i = 2; i <= n; i ++) {
            if (!st[i]) {
                primes[cnt ++] = i;
            }
            for (int j = 0; i * primes[j] <= n; j ++) {
                int t = i * primes[j];
                st[t] = 1;
                if (i % primes[j] == 0) {
                    break;
                }
            }
        }
    };
    sieve(N);
    int n;
    cin >> n;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i ++) {
        cin >> a[i];
    }
    vector<vector<int>> dp(2, vector<int>(N + 1));
    dp[0][0] = 1;
    for (int i = 1; i <= n; i ++) {
        for (int j = 0; j <= N; j ++) {
            dp[i & 1][j] = dp[(i - 1) & 1][j];
            if ((j ^ a[i]) <= N) {
                dp[i & 1][j] = (dp[i & 1][j] + dp[(i - 1) & 1][j ^ a[i]]) % mod;
            }
        }
    }
    int res = 0;
    for (int i = 1; i <= N; i ++) {
        if (!st[i]) {
            res = (res + dp[n & 1][i]) % mod;
        }
    }
    cout << res << "\n";
}
```

## 区间DP：CF 1114 D Flood Fill

题目链接 https://codeforces.com/problemset/problem/1114/D

 **题意：**

有 $n$ 个砖块排成一排，从左到右编号为 $1 \sim n$

其中，第 $i$ 个砖块的初始颜色为 $c_i$

我们规定，如果编号范围 $[i,j]$ 内的所有砖块的颜色都相同，则砖块 $i$ 和 $j$ 属于同一个连通块。

现在，要对砖块进行涂色操作。

开始所有操作之前，你需要任选一个砖块作为**起始砖块**。

每次操作：

1. 任选一种颜色。
2. 将最开始选定的**起始砖块**所在连通块中包含的所有砖块都涂为选定颜色，

请问，至少需要多少次操作，才能使所有砖块都具有同一种颜色。

**分析：**

首先把所有砖块进行缩点，也就是相邻相同颜色的砖块进行合并。

考虑区间 DP，定义状态 $dp_{i, j}$ 为将区间 $[i, j]$ 染成同色的最小次数，转移分为两种情况：

对于每个区间 $[l, r]$，每个端点 $i$ 的颜色为 $a_i$

若 $a_l = a_r$，那么 $dp_{l, r} = dp_{l + 1, r - 1} + 1$

若 $a_l \ne a_r$，那么 $dp_{l, r} = \min(dp_{l + 1, r}, dp_{l, r - 1}) + 1$

最终答案为 $dp_{1, n}$

时间复杂度 $O(n ^ 2)$

## 代码：

```cpp
#include <bits/stdc++.h>
using namespace std;
signed main() {
    cin.tie(0) -> sync_with_stdio(0);
    int n;
    cin >> n;
    vector<int> a{0};
    for (int i = 1; i <= n; i ++) {
        int x;
        cin >> x;
        if (a.back() != x) {
            a.push_back(x);
        }
    }
    n = a.size() - 1;
    vector<vector<int>> dp(n + 1, vector<int>(n + 1));
    for (int len = 2; len <= n; len ++) {
        for (int l = 1; l + len - 1 <= n; l ++) {
            int r = l + len - 1;
            if (a[l] == a[r]) {
                dp[l][r] = dp[l + 1][r - 1] + 1;
            } else {
                dp[l][r] = min(dp[l + 1][r], dp[l][r - 1]) + 1;
            }
        }
    }
    cout << dp[1][n] << endl;
}
```

## 数位DP：洛谷 P2657 windy 数

题目链接 https://www.luogu.com.cn/problem/P2657

数位 DP 学习笔记：https://www.acwing.com/blog/content/7944/

**题意：** 找到区间 $[L,R]$ 相邻数字之差至少为 $2$ 的数的个数

**分析：** 搜索初始条件第二个参数 $pre$ 必须填一个 $\le -2$ 的数来保证可以搜索下去，不然会出错。此题需要记录前导零，不然忽视前导零的影响会造成最高位数 $-0<2$ 无法继续搜索的情况。

## 代码：

```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -> sync_with_stdio(0);
    int l, r;
    cin >> l >> r;
    auto cal = [&](int x) {
        vector<int> a{0};
        while (x) {
            a.push_back(x % 10);
            x /= 10;
        }
        vector<vector<int>> dp(a.size(), vector<int>(10, -1));
        function<int(int, int, int, int)> dfs = [&](int pos, int pre, int lead, int limit) {
            if (!pos) return 1ll;
            if (!limit && !lead && dp[pos][pre] != -1) return dp[pos][pre];
            int res = 0, up = limit ? a[pos] : 9;
            for (int i = 0; i <= up; i ++) {
                if (abs(pre - i) < 2) continue;
                if (lead && !i) {
                    res += dfs(pos - 1, -2, lead && !i, limit && i == up);
                } else {
                    res += dfs(pos - 1, i, lead && !i, limit && i == up);
                }
            }
            return limit ? res : (lead ? res : dp[pos][pre] = res);
        };
        return dfs(a.size() - 1, -2, 1, 1);
    };
    cout << cal(r) - cal(l - 1) << "\n";
}
```



## 状压 + 树形DP：武汉科技大学校赛 B 杰哥的树

题目链接 https://ac.nowcoder.com/acm/contest/35746/B

**分析：**

首先边权只有 $6$ 种颜色可以想到状态压缩，路径的所有颜色必须出现偶数次，可以想到按位异或操作，每次增加颜色时令当前状态异或 $2 ^ i$，$i$ 为该颜色对应的二进制位。

每种颜色都出现偶数次就对应 $0$ 这个状态，那么只有两个状态相同时才可以异或成为 $0$，定义每个点 $u$ 的点权为从 $1$ 到 $u$ 的异或和，问题就转化为树上有多少点对点权相同。直接 $\text{dfs}$ 遍历一遍树记录状态，$dp_i$ 就为状态 $i$ 的个数，那么最后答案就为
$$
\sum_{i = 0} ^ {2 ^ 6} \binom{dp_i}{2}
$$

## 代码：

```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -> sync_with_stdio(0);
    int n;
    cin >> n;
    vector<vector<pair<int, int>>> g(n + 1);
    for (int i = 1; i < n; i ++) {
        int u, v;
        char ch;
        cin >> u >> v >> ch;
        g[u].push_back({v, ch - 'a'}), g[v].push_back({u, ch - 'a'});
    }
    vector<int> dp(1ll << 6);
    function<void(int, int, int)> dfs = [&](int u, int fa, int st) {
        dp[st] ++;
        for (auto [v, w] : g[u]) {
            if (v == fa) {
                continue;
            }
            dfs(v, u, st ^ (1ll << w));
        }
    };
    dfs(1, -1, 0);
    int res = 0;
    for (int i = 0; i < 1ll << 6; i ++) {
        res += dp[i] * (dp[i] - 1) / 2;
    }
    cout << res << "\n";
}
```

