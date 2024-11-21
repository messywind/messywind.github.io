# [NC 200008] Lady Layton With Math


[原题链接](https://ac.nowcoder.com/acm/problem/200008)

**题意**

求

$$\sum_{i=1}^{n}\sum_{j=1}^{n} \varphi(\gcd(i,j))$$

$1 \le n \le 10^9$，对 $10^9&#43;7$ 取模

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
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5, mod = 1e9 &#43; 7;
int T, n, euler[N], primes[N], cnt, sum[N];
bool st[N];
unordered_map&lt;int, int&gt; mp;
void get_eulers(int n) {
    euler[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; primes[j] &lt;= n / i; j &#43;&#43;) {
            int t = primes[j] * i;
            st[t] = 1;
            if (i % primes[j] == 0) {
                euler[t] = primes[j] * euler[i];
                break;
            }
            euler[t] = (primes[j] - 1) * euler[i];
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = (sum[i - 1] &#43; euler[i]) % mod;
}
int Sum(int n) {
    if (n &lt; N) return sum[n];
    if (mp[n]) return mp[n];
    int res = n * (n &#43; 1) / 2 % mod;
    for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res = (res - Sum(n / l) * (r - l &#43; 1) % mod &#43; mod) % mod;
    }
    return mp[n] = res;
}
signed main() {
    get_eulers(N - 1);
    cin &gt;&gt; T;
    while (T --) {
        int res = 0;
        cin &gt;&gt; n;
        for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
            r = n / (n / l);
            res = (res &#43; (Sum(r) - Sum(l - 1)) * (2 * Sum(n / l) - 1) % mod &#43; mod) % mod;
        }
        cout &lt;&lt; res &lt;&lt; endl;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/nc-200008-lady-layton-with-math/  

