---
title: "[2019 ICPC西安邀请赛] Product"
date: 2021-09-17 20:17:33
tags:
- 莫比乌斯反演
- 杜教筛
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

[原题链接](https://nanti.jisuanke.com/t/39269)

**题意**

求

$$\prod_{i=1} ^{n} \prod_{j=1}^{n}\prod_{k=1}^{n}m^{\gcd(i,j)[k \mid \gcd(i,j)]} \bmod p$$

$n \le10^9,m \le 2 ×10^9,p\le 2×10^9$ ，$p$是质数

**分析：**

相乘变为指数相加

$$m ^ {\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{n}\sum\limits_{k=1}^{n}\gcd(i,j)[k \mid \gcd(i,j)]}$$

看一下指数部分

$$\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n}\gcd(i,j)[k \mid \gcd(i,j)]$$

发现 $\sum\limits_{k=1}^{n}[k \mid \gcd(i,j)]$ 就是 $d(\gcd(i,j))$

$$\sum_{i=1}^{n}\sum_{j=1}^{n}\gcd(i,j)d(\gcd(i,j))$$

枚举 $\gcd(i,j)$

$$\sum_{k=1}^{n}k \cdot d(k)\sum_{i=1}^{n}\sum_{j=1}^{n}[\gcd(i,j)=k]$$

$k$ 拿到上界

$$\sum_{k=1}^{n}k \cdot d(k)\sum_{i=1}^{\lfloor \frac{n}{k} \rfloor }\sum_{j=1}^{\lfloor \frac{n}{k} \rfloor}[\gcd(i,j)=1]$$

因为 $\sum\limits_{i=1}^{n}\sum\limits_{i=1}^{n}[\gcd(i,j)=1]=\sum\limits_{i=1}^{n}2\varphi(i)-1$，所以

$$\sum_{k=1}^{n}k \cdot d(k)(\sum_{i=1}^{\lfloor \frac{n}{k} \rfloor }2\varphi(i)-1)$$

对于 $\sum\limits_{k=1}^{n}k \cdot d(k)$

$$\sum_{k=1}^{n}k \cdot d(k)=\sum_{k=1} ^ {n} \sum_{d \mid k}k=\sum_{d=1}^{n}d\sum_{k=1} ^{\lfloor \frac{n}{d} \rfloor}k=\sum_{d=1}^{n}d\frac{\lfloor \frac{n}{d}\rfloor^2+\lfloor \frac{n}{d}\rfloor}{2}$$

替换得

$$\sum_{d =1}^{n}d \frac{\lfloor \frac{n}{d}\rfloor^2+\lfloor \frac{n}{d}\rfloor}{2}(\sum_{i=1}^{\lfloor \frac{n}{d} \rfloor }2\varphi(i)-1)$$

整除分块+杜教筛

那么在求原式的时候用一下欧拉降幂就好了

## 代码：
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 2e6 + 5;
int n, m, mod, primes[N], euler[N], cnt, res, sum[N], d[N], num[N];
bool st[N];
unordered_map<int, int> mp, Mp;
void get_eulers(int n) {
    euler[1] = d[1] = 1;
    for (int i = 2; i <= n; i ++) {
        if (!st[i]) {
            primes[cnt ++] = i;
            euler[i] = i - 1;
            d[i] = 2;
            num[i] = 1;
        }
        for (int j = 0; primes[j] <= n / i; j ++) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                num[t] = num[i] + 1;
                euler[t] = primes[j] * euler[i];
                d[t] = (d[i] / num[t] * (num[t] + 1)) % mod;
                break;
            }
            euler[t] = (primes[j] - 1) * euler[i];
            num[t] = 1;
            d[t] = (d[i] * 2) % mod;
        }
    }
    for (int i = 1; i <= n; i ++) {
        sum[i] = (sum[i - 1] + euler[i]) % mod;
        d[i] = (d[i - 1] + i * d[i]) % mod;
    }
}
int qmi(int a, int b) {
    int res = 1;
    while (b) {
        if (b & 1) res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}
int Sum_euler(int n) {
    if (n < N) return sum[n];
    if (mp[n]) return mp[n];
    int res = n * (n + 1) / 2 % mod;
    for (int l = 2, r; l <= n; l = r + 1) {
        r = n / (n / l);
        res = (res - Sum_euler(n / l) * (r - l + 1) + mod) % mod;
    }
    return mp[n] = res;
}
int Sum(int n) {
    if (n < N) return d[n];
    if (Mp[n]) return Mp[n];
    int res = 0;
    for (int l = 1, r; l <= n; l = r + 1) {
        r = n / (n / l);
        res = (res + (l + r) * (r - l + 1) / 2 % mod * (n / l) * (n / l + 1) / 2 % mod) % mod;
    }
    return Mp[n] = res;
}
signed main() {
    cin >> n >> m >> mod;
    mod --;
    get_eulers(N - 1);
    for (int l = 1, r; l <= n; l = r + 1) {
        r = n / (n / l);
        res = (res + (2 * Sum_euler(n / l) - 1) * (Sum(r) - Sum(l - 1) + mod) % mod) % mod;
    }
    mod ++;
    cout << qmi(m, res) << endl;
}
```