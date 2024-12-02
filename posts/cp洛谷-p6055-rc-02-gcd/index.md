# [洛谷 P6055] [RC-02] GCD


[原题链接](https://www.luogu.com.cn/problem/P6055)

**题意**

求

$$\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{p=1}^{\lfloor \frac{n}{j} \rfloor} \sum_{q=1}^{\lfloor \frac{n}{j} \rfloor}[\gcd(i,j)=1][\gcd(p,q)=1]$$

对 $998244353$ 取模

**分析：**

常规套用莫比乌斯反演式子会很麻烦，所以这里反向把上界拿下来

$$\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{p=1}^{n} \sum_{q=1}^{n}[\gcd(i,j)=1][\gcd(p,q)=j]$$

那么就是

$$\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n} [\gcd(i,j,k)=1]$$

再来莫比乌斯反演

$$\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n} \sum_{d \mid \gcd(i,j,k)}\mu(d)$$

化简一下

$$\sum_{d=1}^{n}\mu(d) \lfloor\frac{n}{d}\rfloor^3$$

就可以用杜教筛了

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5, mod = 998244353;
int n, mobius[N], primes[N], cnt, sum[N], res;
bool st[N];
unordered_map&lt;int, int&gt; mp;
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
    cin &gt;&gt; n;
    for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res = (res &#43; (Sum(r) - Sum(l - 1)) * (n / l) % mod * (n / l) % mod * (n / l) % mod &#43; mod) % mod;
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E6%B4%9B%E8%B0%B7-p6055-rc-02-gcd/  

