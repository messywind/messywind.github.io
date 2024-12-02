# [洛谷 P1891] 疯狂 LCM


[题目链接](https://www.luogu.com.cn/problem/P1891)

**题意：**

求

$$\sum_{i=1} ^{n} \text{lcm}(i,n)$$

**分析：**

## 法一：欧拉函数
拆一下 $\text{lcm}(i,n) = \dfrac{i \cdot n}{\gcd{(i,n)}}$ 变为：

$$\sum_{i=1} ^{n} \frac{i \cdot n}{\gcd{(i,n)}}$$

枚举 $\gcd(i,n)$：

$$n \sum_{d \mid n} \sum_{i=1} ^{n} \frac{i }{d}[\gcd{(i,n)} = d  ]$$

利用 $\gcd$ 的性质：

$$n \sum_{d \mid n} \sum_{i=1} ^{n} \frac{i }{d}[\gcd{(\frac{i}{d},\frac{n}{d})} = 1  ]$$

把 $d$ 拿到上界

$$n \sum_{d \mid n} \sum_{i=1} ^{ \lfloor \frac{n}{d} \rfloor } i[\gcd{(i,\frac{n}{d})} = 1  ]$$

$\lfloor \dfrac{n}{d} \rfloor$ 等价于 $d$

$$n \sum_{d \mid n} \sum_{i=1} ^{ d } i[\gcd{(i,d)} = 1  ]$$

由于 $\gcd(i, d) = \gcd(d - i,d)$ ，所以因子必成对出现（除了1），那么总共出现了 $\dfrac{\varphi(d)}{2}$ 次，$d - i &#43; i =d$，所以就是

$$n \sum_{d \mid n} \frac{\varphi(d)}{2} d$$

这样时间复杂度是 $O(N&#43;T\sqrt{n})$，但是可以用狄利克雷卷积优化，可以做到 $O(N \log{N} &#43; T)$

设 $F(x) = \dfrac{x  \cdot \varphi(x)}{2}$

则答案为 $n \cdot F * \textbf{1}$，注意处理 $d=1$ 的情况。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
int cnt;
vector&lt;int&gt; primes, euler, f;
vector&lt;bool&gt; st;
void init(int n) {
    f.resize(n &#43; 1), primes.resize(n &#43; 1), euler.resize(n &#43; 1), st.resize(n &#43; 1);
    euler[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; i * primes[j] &lt;= n; j &#43;&#43;) {
            int t = i * primes[j];
            st[t] = 1;
            if (i % primes[j] == 0) {
                euler[t] = euler[i] * primes[j];
                break;
            }
            euler[t] = euler[i] * (primes[j] - 1);
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = i; j &lt;= n; j &#43;= i) {
            f[j] &#43;= (euler[i] * i &#43; 1) / 2;
        }
    }
}
void solve() {
    int n;
    cin &gt;&gt; n;
    cout &lt;&lt; n * f[n] &lt;&lt; &#34;\n&#34;;
}
signed main() {
    init(1e6);
    cin.tie(0) -&gt; sync_with_stdio(0);
    int T;
    cin &gt;&gt; T;
    while (T --) {
        solve();
    }
}
```

## 法二：莫比乌斯反演
还是法一的式子，推到这一步

$$n \sum_{d \mid n} \sum_{i=1} ^{ d } i[\gcd{(i,d)} = 1  ]$$

用单位函数替换

$$n \sum_{d \mid n} \sum_{i=1} ^{ d } i \cdot \varepsilon (\gcd{(i,d)})$$

莫比乌斯反演

$$n \sum_{d \mid n} \sum_{i=1} ^{ d } i \sum_{k \mid \gcd(i,d) } \mu(k)$$

交换枚举次序

$$n \sum_{d \mid n} \sum_{k \mid d} k\mu(k) \sum_{i=1} ^ { \lfloor \frac{d}{k} \rfloor } i$$

对后半部分求和

$$\frac{n}{2} \sum_{d \mid n} \sum_{k \mid d} k\mu(k) (\lfloor \frac{d}{k} \rfloor ^ 2 &#43; \lfloor \frac{d}{k} \rfloor) $$

可以用狄利克雷卷积优化到 $O(N\log N &#43;T)$

设 $f(x)=x \cdot \mu(x)$ ，$g(x)=x^2&#43;x$，$F(x) = f * g$

那么答案就为：

$$\frac{n}{2} \cdot F * \textbf {1}$$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 1e6 &#43; 5;
int T, n, mobius[N], primes[N], cnt, F[N], ans[N];
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
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = i; j &lt;= n; j &#43;= i) {
            F[j] &#43;= i * mobius[i] * ((j / i) * (j / i) &#43; (j / i));
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = i; j &lt;= n; j &#43;= i) {
            ans[j] &#43;= F[i];
        } 
    }
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; T;
    while (T --) {
        cin &gt;&gt; n;
        cout &lt;&lt; n * ans[n] / 2 &lt;&lt; endl;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E6%B4%9B%E8%B0%B7-p1891-%E7%96%AF%E7%8B%82-lcm/  

