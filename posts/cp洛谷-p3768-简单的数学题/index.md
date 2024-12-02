# [洛谷 P3768] 简单的数学题


[题目链接](https://www.luogu.com.cn/problem/P3768)

**题意：**

求

$$\sum_{i=1}^{n}\sum_{j=1}^{n}ij\gcd(i,j)$$

对 $p$ 取模，$n \le10^{10}, 5 ×10^8\le p \le1.1 ×10^{9}$

**分析：**

$$\sum_{i=1}^{n}\sum_{j=1}^{n}ij\gcd(i,j)$$

枚举 $\gcd(i,j)$

$$\sum_{d=1} ^{n} d \sum_{i=1}^{n}\sum_{j=1}^{n}ij[\gcd(i,j)=d]$$

利用 $\gcd$ 的性质

$$\sum_{d=1} ^{n} d \sum_{i=1}^{n}\sum_{j=1}^{n}ij[\gcd(\frac{i}{d},\frac{j}{d})=1]$$

让式子除 $d^2$ 再乘 $d^2$

$$\sum_{d=1} ^{n} d^3 \sum_{i=1}^{n}\sum_{j=1}^{n} \frac{i}{d} \cdot \frac{j}{d}[\gcd(\frac{i}{d},\frac{j}{d})=1]$$

换一下上界

$$\sum_{d=1} ^{n} d^3 \sum_{i=1}^{\lfloor \frac{n}{d} \rfloor}\sum_{j=1}^{\lfloor \frac{n}{d} \rfloor} i j[\gcd(i,j)=1]$$

莫比乌斯反演

$$\sum_{d=1} ^{n} d^3 \sum_{i=1}^{\lfloor \frac{n}{d} \rfloor}\sum_{j=1}^{\lfloor \frac{n}{d} \rfloor} i j \sum_{k \mid \gcd(i,j)} \mu(k)$$

交换求和次序

$$\sum_{d=1} ^{n} d^3 \sum_{k=1} ^{\lfloor \frac{n}{d} \rfloor} \mu(k) \sum_{i=1}^{\lfloor \frac{n}{dk} \rfloor} ik\sum_{j=1}^{\lfloor \frac{n}{dk} \rfloor} jk$$

化简后半部分

$$\sum_{d=1} ^{n} d^3 \sum_{k=1} ^{\lfloor \frac{n}{d} \rfloor} k^2 \mu(k) (\sum_{i=1} ^{\lfloor \frac{n}{dk} \rfloor}i)^2$$

设 $T=dk$

$$\sum_{T=1} ^{n} T^2 \sum_{d \mid T}  d\mu(\frac{T}{d}) (\sum_{i=1} ^{\lfloor \frac{n}{T} \rfloor}i)^2$$

因为 $\mu * Id=\varphi$，替换 $\sum \limits_{d \mid T}  d\mu(\frac{T}{d})$ 得

$$\sum_{T=1} ^{n} T^2 \varphi(T) (\sum_{i=1} ^{\lfloor \frac{n}{T} \rfloor}i)^2$$

根据 $(\dfrac{n^2 &#43;n}{2})^2=1^3&#43;2^3&#43;\cdots&#43;n^3$

$$\sum_{T=1} ^{n} T^2 \varphi(T) \sum_{i=1} ^{\lfloor \frac{n}{T} \rfloor}i^3$$

设 $f(x)= x^2\varphi(x)$ 套用杜教筛，$S(n)$ 为 $\sum \limits_{i=1}^{n} f(i)$

$$g(1)S(n)=\sum_{i=1} ^{n}h(i)-\sum_{i=1}^{n}g(i)S(\lfloor \frac{n}{i} \rfloor)$$

那么 $h=f *g$，也就是 $h(n)=\sum \limits _{d \mid n}f(n)g(\dfrac{n}{d})$，带入 $f(n)$ 得

$$h(n)=\sum_{d \mid n} d^2\varphi(d)g(\frac{n}{d})$$

考虑把 $d^2$ 消去，所以设 $g(x)=x^2$，故

$$h(n)=n^2\sum_{d \mid n}\varphi(d)$$

根据 $\varphi *I=Id$

$$h(n)=n^3$$

那么 $f(x)$ 的前缀和就是

$$S(n)=(\frac{n^2&#43;n}{2})^2-\sum_{i=2} ^{n}i^2S(\lfloor \frac{n}{i} \rfloor)$$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 5e6 &#43; 5;
int mod, n, euler[N], primes[N], cnt, sum[N], inv, res;
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
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = (sum[i - 1] &#43; i * i % mod * euler[i] % mod) % mod;
}
int s2(int n) {
    n %= mod;
    return n * (n &#43; 1) % mod * (2 * n &#43; 1) % mod * inv % mod;
}
int s3(int n) {
    n %= mod;
    return (n * (n &#43; 1) / 2) % mod * ((n * (n &#43; 1) / 2) % mod) % mod;
}
int Sum(int n) {
    if (n &lt; N) return sum[n];
    if (mp[n]) return mp[n];
    int res = s3(n);
    for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res -= (s2(r) - s2(l - 1) &#43; mod) % mod * Sum(n / l) % mod;
        res = (res &#43; mod) % mod;
    }
    return mp[n] = res;
}
signed main() {
    cin &gt;&gt; mod &gt;&gt; n;
    get_eulers(N - 1);
    inv = qmi(6, mod - 2);
    for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res &#43;= (Sum(r) - Sum(l - 1) &#43; mod) % mod * s3(n / l) % mod;
        res = (res % mod &#43; mod) % mod;
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E6%B4%9B%E8%B0%B7-p3768-%E7%AE%80%E5%8D%95%E7%9A%84%E6%95%B0%E5%AD%A6%E9%A2%98/  

