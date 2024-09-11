---
title: "[2021 CCPC威海热身赛] Number Theory"
date: 2021-11-20 18:53:18
tags:
- 打表
- 推式子
categories:
- 算法竞赛
code:
  maxShownLines: 11
---

**题意**

求

$$\sum_{k = 1}^{n}\sum_{i \mid k} \sum_{j \mid i} \lambda(i) \lambda(j)$$

对 $998244353$ 取模

其中 $\lambda(x) = (-1)^{\sum\limits_{i}e_i},x=\prod\limits_{i}p_i^{e_i}$

**分析：**

$\lambda(x)$ 为刘维尔函数，可以打表发现 $$\sum_{d \mid n}\lambda(d) =[n = a^2,a \in N^+]$$

也就是 $n$ 是否为完全平方数

把式子中的 $\lambda(i)$ 提到前面

$$\sum_{k = 1}^{n}\sum_{i \mid k} \lambda(i)\sum_{j \mid i}  \lambda(j)$$

那么就变为

$$\sum_{k = 1}^{n}\sum_{i \mid k} \lambda(i)[i= a^2,a \in N^+]$$

那么完全平方数的刘维尔函数为 $1$，再设 $f(x)=[i= a^2,a \in N^+]$ 得

$$\sum_{i = 1}^{n}\sum_{d \mid i}f(d)$$

交换求和次序

$$\sum_{d = 1}^{n}f(d) \lfloor\frac{n}{d}\rfloor$$

这样直接枚举平方数即可，时间复杂度 $O(\sqrt{n})$

## 代码：
```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;
const int mod = 998244353;
int n, res;
signed main() {
    cin >> n;
    for (int i = 1; i * i <= n; i ++) {
        res = (res + n / (i * i)) % mod;
    }
    cout << res << endl;
}
```