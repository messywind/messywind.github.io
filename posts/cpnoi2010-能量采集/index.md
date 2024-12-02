# [NOI2010] 能量采集


[题目链接](https://www.luogu.com.cn/problem/P1447)

**题意：**

求

$$2\sum_{i=1}^{n}\sum_{j=1}^{m}\gcd(i,j)-nm$$

**分析：**

## 莫比乌斯反演：

那么只需要求 

$$\sum_{i=1}^{n}\sum_{j=1}^{m}\gcd(i,j)$$

枚举每个 $\gcd(i,j)$ 的值

$$\sum_{d=1}^{n}d\sum_{i=1}^{n}\sum_{j=1}^{m}[\gcd(i,j)=d]$$

再用莫比乌斯反演，和 **problem b** 这题是一样的

$$\sum_{d=1}^{n}d\sum_{i=1}^{\left \lfloor \frac{\min(n,m)}{d} \right \rfloor }\mu(i)\left \lfloor \frac{n}{di} \right \rfloor \left \lfloor \frac{m}{di} \right \rfloor $$

后面可用整除分块。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 1e5 &#43; 5;
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
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = sum[i - 1] &#43; mobius[i];
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; n &gt;&gt; m;
    for (int d = 1; d &lt;= n; d &#43;&#43;) {
        int k = min(n, m) / d, ans = 0;
        for (int l = 1, r; l &lt;= k; l = r &#43; 1) {
            r = min(n / (n / l), m / (m / l));
            ans &#43;= (sum[r] - sum[l - 1]) * (n / (d * l)) * (m / (d * l));
        }
        res &#43;= d * ans;
    }
    cout &lt;&lt; 2 * res - n * m &lt;&lt; endl;
}
```

## 欧拉反演：
**定理：**

$$n=\sum_{d \mid n} \varphi(d)$$

原式为

$$\sum_{i=1}^{n}\sum_{j=1}^{m}\gcd(i,j)$$

将 $\gcd(i,j)$ 套用欧拉反演定理

$$\sum_{i=1}^{n}\sum_{j=1}^{m} \sum_{d \mid \gcd(i,j)}\varphi(d)$$

将 $\varphi(d)$ 提到前面，枚举 $d$

$$\sum_{d=1}^{n}\varphi(d)\sum_{i=1}^{n}\sum_{j=1}^{m}[d \mid \gcd(i,j)]$$

化简后面

$$\sum_{d=1}^{n}\varphi(d)\left \lfloor \frac{n}{d} \right \rfloor \left \lfloor \frac{m}{d} \right \rfloor $$

预处理欧拉函数前缀和，用整除分块即可。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 1e5 &#43; 5;
int n, m, primes[N], euler[N], cnt, res, sum[N];
bool st[N];
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
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = sum[i - 1] &#43; euler[i];
}
signed main() {
    get_eulers(N - 1);
    cin &gt;&gt; n &gt;&gt; m;
    for (int l = 1, r; l &lt;= min(n, m); l = r &#43; 1) {
        r = min(n / (n / l), m / (m / l));
        res &#43;= (sum[r] - sum[l - 1]) * (n / l) * (m / l);
    }
    cout &lt;&lt; 2 * res - n * m &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cpnoi2010-%E8%83%BD%E9%87%8F%E9%87%87%E9%9B%86/  

