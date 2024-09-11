---
title: "[洛谷 P1829] Crash的数字表格"
date: 2021-09-09 11:48:54
tags:
- 莫比乌斯反演
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[题目链接](https://www.luogu.com.cn/problem/P1829)

**题意：** 

求

$$\sum_{i = 1} ^{n} \sum_{j = 1} ^{m} \text{lcm}(i, j)$$

对 $20101009$ 取模

**分析：** 

首先 $\text{lcm}(i, j) = \dfrac{i \cdot j}{\gcd(i,j)}$ 代入：

$$\sum_{i = 1} ^{n} \sum_{j = 1} ^{m} \frac{i \cdot j}{\gcd(i,j)}$$

枚举 $\gcd(i,j)$

$$\sum_{d = 1} ^{\min(n,m)}\sum_{i = 1} ^{n} \sum_{j = 1} ^{m} \frac{i \cdot j}{d}[\gcd(i,j)=d] $$

根据 $\gcd$ 的性质：

$$\sum_{d = 1} ^{\min(n,m)}\sum_{i = 1} ^{n} \sum_{j = 1} ^{m} \frac{i \cdot j}{d}[\gcd(\frac{i}{d}, \frac{j}{d}) = 1] $$

在 $\dfrac{i \cdot j}{d}$ 中 除一个 $d$ 乘一个 $d$，来凑形式一致。

$$\sum_{d = 1} ^{\min(n,m)}d \sum_{i = 1} ^{n} \sum_{j = 1} ^{m} \frac{i}{d}\cdot \frac{j}{d} [\gcd(\frac{i}{d}, \frac{j}{d}) = 1] $$

替换 $\dfrac{i}{d},\dfrac{j}{d}$

$$\sum_{d = 1} ^{\min(n,m)}d \sum_{i = 1} ^{ \lfloor \frac{n}{d} \rfloor } \sum_{j = 1} ^{\lfloor \frac{m}{d} \rfloor }i\cdot j [\gcd(i, j) = 1]$$

用单位函数替换

$$\sum_{d = 1} ^{\min(n,m)}d \sum_{i = 1} ^{ \lfloor \frac{n}{d} \rfloor } \sum_{j = 1} ^{\lfloor \frac{m}{d} \rfloor }i\cdot j \cdot \varepsilon (\gcd(i, j) = 1)$$

莫比乌斯反演

$$\sum_{d = 1} ^{\min(n,m)}d \sum_{i = 1} ^{ \lfloor \frac{n}{d} \rfloor } \sum_{j = 1} ^{\lfloor \frac{m}{d} \rfloor }i\cdot j \sum_{k \mid \gcd(i,j)} \mu(k)$$

交换求和次序

$$\sum_{d = 1} ^{\min(n,m)}d \sum_{k =1} ^{\min(\lfloor \frac{n}{d} \rfloor,\lfloor \frac{m}{d} \rfloor)} \mu(k) \sum_{i = 1} ^{ \lfloor \frac{n}{dk} \rfloor } i \cdot k \sum_{j = 1} ^{\lfloor \frac{m}{dk} \rfloor }  j \cdot k$$

整理式子

$$\frac{1}{4} \sum_{d = 1} ^{\min(n,m)}d \sum_{k =1} ^{\min(\lfloor \frac{n}{d} \rfloor,\lfloor \frac{m}{d} \rfloor)} k^2 \mu(k) (\lfloor \frac{n}{dk} \rfloor ^2 + \lfloor \frac{n}{dk} \rfloor) \cdot (\lfloor \frac{m}{dk} \rfloor ^2 + \lfloor \frac{m}{dk} \rfloor)  $$

时间复杂度 $O(N\sqrt{N})$

## 代码：
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 1e7 + 5, mod = 20101009;
int n, m, mobius[N], primes[N], cnt, res, sum[N];
bool st[N];
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
    for (int i = 1; i <= n; i ++) sum[i] = (sum[i - 1] + i * i * mobius[i] % mod + mod) % mod;
}
signed main() {
    get_mobius(N - 1);
    cin >> n >> m;
    for (int d = 1; d <= min(n, m); d ++) {
        int x = n / d, y = m / d, Sum = 0;
        for (int l = 1, r; l <= min(x, y); l = r + 1) {
            r = min(x / (x / l), y / (y / l));
            int p = ((x / l) * (x / l) + x / l) / 2 % mod, q = ((y / l) * (y / l) + y / l) / 2 % mod;
            Sum += (sum[r] - sum[l - 1]) % mod * p % mod * q % mod;
            Sum = (Sum % mod + mod) % mod;
        }
        res = (res + d * Sum) % mod;
    }
    cout << res << endl;
}
```