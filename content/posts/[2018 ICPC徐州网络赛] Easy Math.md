---
title: "[2018 ICPC徐州网络赛] Easy Math"
date: 2021-09-18 09:14:03
tags:
- 杜教筛
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://nanti.jisuanke.com/t/A2003)

**题意**

求

$$\sum_{i=1}^{m} \mu(in)$$

$m \le 2×10^9,n\le 10^{12}$

**分析：**

首先分析 $n$ 的因子中有没有平方，如果有那么答案就是 $0$

如果 $n$ 的因子没有平方，设某个因子为 $p$，原式就可以拆成

$$\sum_{i=1} ^{m}\mu(i\cdot\frac{n}{p}\cdot p)$$

莫比乌斯函数是积性函数，考虑把 $p$ 分出去，那么 $p$ 与 $\dfrac{n}{p}$ 一定是互质的，但 $i\cdot \dfrac{n}{p}$ 并不一定互质，那么就考虑 $i$ 与 $p$ 的关系，在 $[1,m]$ 中只有 $p$ 的倍数才与 $p$ 不互质，所以要加上这一部分。

$$\sum_{i=1}^{m}\mu(i\cdot\frac{n}{p})\mu(p)+\sum_{i=1}^{\lfloor \frac{m}{p} \rfloor}\mu(i\cdot p \cdot \frac{n}{p})$$

由于 $p$ 是单因子，所以 $\mu(p)=-1$

$$\sum_{i=1}^{\lfloor \frac{m}{p} \rfloor}\mu(i n) - \sum_{i=1}^{m}\mu(i\cdot\frac{n}{p})$$

设 $S(n,m)=\sum\limits_{i=1}^{m}\mu(in)$，那么得到递推式

$$S(n,m)=S(n,\frac{m}{p})-S(\frac{n}{p},m)$$

那么就可以每次枚举 $n$ 的质因子 $p$，递归求解，那么递归边界就是 $S(0,n)$ 和 $S(1,m)$，也就是莫比乌斯函数前缀和，用杜教筛处理一下就好了

## 代码：
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 2e6 + 5;
int t, n, m, mobius[N], primes[N], cnt, sum[N];
bool st[N];
unordered_map<int, int> mp;
void get_mobius(int n) {
    mobius[1] = 1;
    for (int i = 2; i <= n; i ++) {
        if (!st[i]) {
            primes[cnt ++] = i;
            mobius[i] = -1;
        }
        for (int j = 0; primes[j] * i <= n; j ++) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                mobius[t] = 0;
                break;
            }
            mobius[t] = -mobius[i];
        }
    }
    for (int i = 1; i <= n; i ++) sum[i] = sum[i - 1] + mobius[i];
}
int Sum(int n) {
    if (n < N) return sum[n];
    if (mp[n]) return mp[n];
    int res = 1;
    for (int l = 2, r; l <= n; l = r + 1) {
        r = n / (n / l);
        res -= Sum(n / l) * (r - l + 1);
    }
    return mp[n] = res;
}
int S(int n, int m) {
    if (m == 0) return 0;
    if (n == 1) return Sum(m);
    int flag = 0;
    for (int i = 2; i * i <= n; i ++) {
        if (n % i == 0) {
            flag = 1;
            return S(n, m / i) - S(n / i, m);
        }
    }
    if (!flag) return S(n, m / n) - S(1, m);
}
signed main() {
    get_mobius(N - 1);
    cin >> m >> n;
    t = n;
    for (int i = 2; i * i <= t; i ++) {
        if (t % i == 0) {
            int cnt = 0;
            while (t % i == 0) {
                t /= i;
                cnt ++;
                if (cnt == 2) {
                    cout << 0 << endl;
                    return 0;
                }
            }
        }
    }
    cout << S(n, m) << endl;
}
```