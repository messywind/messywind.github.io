# [CQOI2015] 选数


[原题链接](https://www.luogu.com.cn/problem/P3172)

**题意**

求从区间 $[L,R]$ 选出 $n$ 个数使得最大公约数为 $k$ 的方案数，对 $10^9 &#43; 7$ 取模

$1 \le n,k \le 10^9$
$1 \le L \le R \le10^9$

**分析：**

根据题意

$$\sum_{a_1=L}^{R}\sum_{a_2=L}^{R}\cdots\sum_{a_n=L}^{R}[\gcd(a_1,a_2,\cdots,a_n)=k]$$

把 $k$ 拿到上下界

$$\sum_{a_1=\lfloor \frac{L-1}{k} \rfloor &#43;1}^{\lfloor \frac{R}{k} \rfloor}\sum_{a_2=\lfloor \frac{L - 1}{k} \rfloor &#43; 1}^{\lfloor \frac{R}{k} \rfloor}\cdots\sum_{a_n=\lfloor \frac{L - 1}{k} \rfloor &#43; 1}^{\lfloor \frac{R}{k} \rfloor}[\gcd(a_1,a_2,\cdots,a_n)=1]$$

莫比乌斯反演

$$\sum_{a_1=\lfloor \frac{L-1}{k} \rfloor &#43;1}^{\lfloor \frac{R}{k} \rfloor}\sum_{a_2=\lfloor \frac{L-1}{k} \rfloor &#43;1}^{\lfloor \frac{R}{k} \rfloor}\cdots\sum_{a_n=\lfloor \frac{L-1}{k} \rfloor &#43;1}^{\lfloor \frac{R}{k} \rfloor}\sum_{d \mid \gcd(a_1,a_2,\cdots,a_n)} \mu(d)$$

交换求和次序

$$\sum_{d=1}^{\lfloor \frac{R}{k} \rfloor}\mu(d) (\lfloor \frac{R}{kd}\rfloor - \lfloor \frac{L - 1}{kd}\rfloor)^n$$

然后用杜教筛做

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5, mod = 1e9 &#43; 7;
int n, k, L, R, primes[N], mobius[N], cnt, res, sum[N];
bool st[N];
unordered_map&lt;int, int&gt; mp;
int qmi(int a, int b) {
    int res = 1;
    while (b) {
        if (b &amp; 1) res = res * a % mod;
        a = a * a % mod;
        b &gt;&gt;= 1;
    }
    return res;
}
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
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = (sum[i - 1] &#43; mobius[i] &#43; mod) % mod;
}
int Sum(int n) {
    if (n &lt; N) return sum[n];
    if (mp[n]) return mp[n];
    int res = 1;
    for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res = (res - Sum(n / l) * (r - l &#43; 1) % mod &#43; mod) % mod;
    }
    return mp[n] = res;
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; n &gt;&gt; k &gt;&gt; L &gt;&gt; R;
    L = (L - 1) / k, R /= k;
    for (int l = 1, r; l &lt;= R; l = r &#43; 1) {
        r = min(R / (R / l), L / l ? L / (L / l) : mod);
        res = (res &#43; (Sum(r) - Sum(l - 1)) * qmi(R / l - L / l, n) % mod &#43; mod) % mod;
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cpcqoi2015-%E9%80%89%E6%95%B0/  

