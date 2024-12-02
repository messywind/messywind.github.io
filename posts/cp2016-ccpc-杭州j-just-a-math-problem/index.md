# [2016 CCPC 杭州J] Just a Math Problem


[原题链接](https://acm.hdu.edu.cn/showproblem.php?pid=5942)

**题意**

$T$ 组输入，每次给定一个正整数 $n$，定义 $f(k)$ 为 $k$ 的质因子个数，$g(k) = 2 ^ {f(i)}$，求
$$
\sum_{i = 1} ^ {n} g(i)
$$
$1 \le T \le 50, 1 \le n \le 10 ^ {12}$

**分析：**

首先 $2 ^ {f(i)}$ 不好直接计算，考虑组合意义，发现 $2 ^ {f(i)}$ 就是从 $i$ 的所有质因子中选出若干个子集的方案数，假设 $i = p_1 ^ {\alpha_1} p_2 ^ {\alpha_2}\cdots p_k ^ {\alpha_k}$，那么我们可以枚举 $i$ 的所有约数 $d$，将约数带入莫比乌斯函数，那么就去掉了所有存在大于等于 $2$ 次的质因子，所以每个 $\alpha_i$ 只能取 $0$ 或 $1$，但如果有奇数个质因子莫比乌斯函数值为负数，所以平方一下即可，即

$$\sum_{i = 1} ^ {n}2 ^ {f(i)} = \sum_{i = 1} ^ {n} \sum_{d \mid i} \mu ^ 2(d)$$

交换求和次序

$$
\sum_{d = 1} ^ {n} \mu ^ 2(d) \lfloor\frac{n}{d}\rfloor
$$
此时就可以套用[完全平方数](https://www.acwing.com/solution/content/67097/)这个题的公式
$$
\sum_{i=1}^{n} \mu^2(i)=\sum_{i=1} ^{n}\sum_{d^2 \mid i} \mu(d)
$$
代入可得
$$
\sum_{i = 1} ^ {n} \lfloor \frac{n}{i} \rfloor \sum_{d ^ 2 \mid i} \mu(d)
$$
交换求和次序
$$
\sum_{d = 1} ^ {\sqrt n} \mu(d) \sum_{i = 1} ^ {\lfloor \frac{n}{d ^ 2} \rfloor} \lfloor \frac{n}{i \times d ^ 2} \rfloor = \sum_{d = 1} ^ {\sqrt n} \mu(d) \sum_{i = 1} ^ {\lfloor \frac{n}{d ^ 2} \rfloor} \lfloor \frac{\lfloor \frac{n}{d ^ 2} \rfloor}{i} \rfloor
$$
后面式子直接分块即可，注意优化当莫比乌斯函数非 $0$ 时才计算答案，并且分块要记忆化一下答案，复杂度比较玄学，此题给了 $15$ 秒。

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int mod = 1e9 &#43; 7;
int cnt;
vector&lt;int&gt; primes(1e6 &#43; 1), mobius(1e6 &#43; 1), block(1e6 &#43; 1);
vector&lt;bool&gt; st(1e6 &#43; 1);
void init(int n) {
    mobius[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            mobius[i] = -1;
        }
        for (int j = 0; i * primes[j] &lt;= n; j &#43;&#43;) {
            int t = i * primes[j];
            st[t] = 1;
            if (i % primes[j] == 0) {
                mobius[t] = 0;
                break;
            }
            mobius[t] = -mobius[i];
        }
    }
};
void solve() {
    int n;
    cin &gt;&gt; n;
    int res = 0;
    auto sum = [&amp;](int n) {
        if (n &lt; 1e6 &amp;&amp; block[n]) {
            return block[n];
        }
        int res = 0;
        for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
            r = n / (n / l);
            res = (res &#43; (n / l) * (r - l &#43; 1) % mod) % mod;
        }
        if (n &gt; 1e6) {
            return res;
        } else {
            return block[n] = res;
        }
    };
    for (int i = 1; i * i &lt;= n; i &#43;&#43;) {
        if (mobius[i]) {
            res = (res &#43; mobius[i] * sum(n / (i * i)) % mod &#43; mod) % mod;
        }
    }
    cout &lt;&lt; res &lt;&lt; &#34;\n&#34;;
}
signed main() {
    init(1e6);
    cin.tie(0) -&gt; sync_with_stdio(0);
    int T;
    cin &gt;&gt; T;
    for (int t = 1; t &lt;= T; t &#43;&#43;) {
        cout &lt;&lt; &#34;Case #&#34; &lt;&lt; t &lt;&lt; &#34;: &#34;;
        solve();
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2016-ccpc-%E6%9D%AD%E5%B7%9Ej-just-a-math-problem/  

