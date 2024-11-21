# [洛谷 P1829] Crash的数字表格


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

$$\frac{1}{4} \sum_{d = 1} ^{\min(n,m)}d \sum_{k =1} ^{\min(\lfloor \frac{n}{d} \rfloor,\lfloor \frac{m}{d} \rfloor)} k^2 \mu(k) (\lfloor \frac{n}{dk} \rfloor ^2 &#43; \lfloor \frac{n}{dk} \rfloor) \cdot (\lfloor \frac{m}{dk} \rfloor ^2 &#43; \lfloor \frac{m}{dk} \rfloor)  $$

时间复杂度 $O(N\sqrt{N})$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 1e7 &#43; 5, mod = 20101009;
int n, m, mobius[N], primes[N], cnt, res, sum[N];
bool st[N];
void get_mobius(int n) {
    mobius[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            mobius[i] = -1;
        }
        for (int j = 0; primes[j] * i &lt;= n; j &#43;&#43;) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                mobius[t] = 0;
                break;
            }
            mobius[t] = -mobius[i];
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = (sum[i - 1] &#43; i * i * mobius[i] % mod &#43; mod) % mod;
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; n &gt;&gt; m;
    for (int d = 1; d &lt;= min(n, m); d &#43;&#43;) {
        int x = n / d, y = m / d, Sum = 0;
        for (int l = 1, r; l &lt;= min(x, y); l = r &#43; 1) {
            r = min(x / (x / l), y / (y / l));
            int p = ((x / l) * (x / l) &#43; x / l) / 2 % mod, q = ((y / l) * (y / l) &#43; y / l) / 2 % mod;
            Sum &#43;= (sum[r] - sum[l - 1]) % mod * p % mod * q % mod;
            Sum = (Sum % mod &#43; mod) % mod;
        }
        res = (res &#43; d * Sum) % mod;
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E6%B4%9B%E8%B0%B7-p1829-crash%E7%9A%84%E6%95%B0%E5%AD%97%E8%A1%A8%E6%A0%BC/  

