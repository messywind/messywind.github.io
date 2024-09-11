---
title: "[NC 200008] Lady Layton with Math"
date: 2021-09-23 08:58:54
tags:
- 杜教筛
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://ac.nowcoder.com/acm/problem/200008)

**题意**

求

$$\sum_{i=1}^{n}\sum_{j=1}^{n} \varphi(\gcd(i,j))$$

$1 \le n \le 10^9$，对 $10^9+7$ 取模

**分析：**

枚举 $\gcd(i,j)$

$$\sum_{d=1}^{n}\varphi(d)\sum_{i=1}^{n}\sum_{j=1}^{n}[\gcd(i,j)=d]$$

将 $d$ 拿到上界

$$\sum_{d=1}^{n}\varphi(d)\sum_{i=1}^{\lfloor \frac{n}{d} \rfloor}\sum_{j=1}^{\lfloor \frac{n}{d} \rfloor}[\gcd(i,j)=1]$$

因为 $\sum\limits_{i=1}^{n}\sum\limits_{i=1}^{n}[\gcd(i,j)=1]=\sum\limits_{i=1}^{n}2\varphi(i)-1$，所以

$$\sum_{d=1}^{n}\varphi(d)(\sum_{i=1}^{\lfloor \frac{n}{d} \rfloor}2\varphi(i)-1)$$

再用一下杜教筛就好了

## 代码：
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 2e6 + 5, mod = 1e9 + 7;
int T, n, euler[N], primes[N], cnt, sum[N];
bool st[N];
unordered_map<int, int> mp;
void get_eulers(int n) {
    euler[1] = 1;
    for (int i = 2; i <= n; i ++) {
        if (!st[i]) {
            primes[cnt ++] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                euler[t] = primes[j] * euler[i];
                break;
            }
            euler[t] = (primes[j] - 1) * euler[i];
        }
    }
    for (int i = 1; i <= n; i ++) sum[i] = (sum[i - 1] + euler[i]) % mod;
}
int Sum(int n) {
    if (n < N) return sum[n];
    if (mp[n]) return mp[n];
    int res = n * (n + 1) / 2 % mod;
    for (int l = 2, r; l <= n; l = r + 1) {
        r = n / (n / l);
        res = (res - Sum(n / l) * (r - l + 1) % mod + mod) % mod;
    }
    return mp[n] = res;
}
signed main() {
    get_eulers(N - 1);
    cin >> T;
    while (T --) {
        int res = 0;
        cin >> n;
        for (int l = 1, r; l <= n; l = r + 1) {
            r = n / (n / l);
            res = (res + (Sum(r) - Sum(l - 1)) * (2 * Sum(n / l) - 1) % mod + mod) % mod;
        }
        cout << res << endl;
    }
}
```