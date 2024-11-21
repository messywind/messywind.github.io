# [2021 CCCC天梯赛] 可怜的简单题


**题意**

每次从 $[1,n]$ 中选择一个数加到一个序列末尾，当 $\gcd(a_1,\cdots,a_n)=1$ 时停止，求期望长度，对 $p$ 取模

$1\le n \le 10^{11},n&lt; p \le 10 ^{12}$ 

**分析：**

设 $E(x)$ 为长度为 $x$ 的期望，那么根据期望定义

$$E(x)=\sum_{i=1}^{\infty}P(x=i) \times i$$

把 $i$ 改为 $\sum\limits_{j=1} ^{i}$

$$E(x)=\sum_{i=1}^{\infty}P(x=i) \sum_{j=1}^{i}$$

交换求和次序

$$\sum_{i=1}^{\infty}\sum_{j = i}^{\infty}P(x=j)$$

等价于

$$\sum_{i=1}^{\infty}P(x\ge i)$$

化简一下

$$\sum_{i=1}^{\infty}P(x\ge i)=1&#43;\sum_{i=1}^{\infty}P(x&gt; i)$$

考虑 $P(x&gt; i)$，进行容斥 $1-P(x \le i)$ 就等价于

$$1-P(\gcd(a_1,\cdots,a_i)=1)$$

枚举 $a_i$ 在 $[1,n]$ 中的取值

$$1-\sum_{a_1=1}^{n}\cdots\sum_{a_i=1}^{n}\frac{[\gcd(a_1,\cdots,a_i)=1]}{n^{i}}$$

莫比乌斯反演

$$1-\sum_{a_1=1}^{n}\cdots\sum_{a_i=1}^{n}\frac{\sum\limits_{d \mid\gcd(a_1,\cdots,a_i) }\mu(d)}{n^{i}}$$

交换求和次序

$$1-\frac{\sum\limits_{d=1}^{n}\mu(d)\lfloor \dfrac{n}{d} \rfloor^i}{n^i}$$

把 $1$ 拿到分子，和第一项抵消了

$$-\frac{\sum\limits_{d=2}^{n}\mu(d)\lfloor \dfrac{n}{d} \rfloor^i}{n^{i}}$$

代入到 $1&#43;\sum\limits_{i=1}^{\infty}P(len &gt; i)$ 得

$$1-\sum_{i=1}^{\infty}\frac{\sum\limits_{d=2}^{n}\mu(d)\lfloor \dfrac{n}{d} \rfloor^i}{n^{i}}$$

交换求和次序

$$1-\sum_{d=2}^{n}\mu(d)\sum_{i=1}^{\infty}(\frac{\lfloor \dfrac{n}{d} \rfloor}{n})^i$$

$\sum\limits_{i=1}^{\infty}(\dfrac{\lfloor \dfrac{n}{d} \rfloor}{n})^i$ 这是个等比级数，极限为 $\dfrac{首项}{1-公比}$

$$1-\sum_{d=2}^{n}\mu(d)\frac{\lfloor \dfrac{n}{d} \rfloor}{n-\lfloor \dfrac{n}{d} \rfloor}$$

就可以用杜教筛了

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n, mod;
    cin &gt;&gt; n &gt;&gt; mod;
    int cnt = 0, N = 2.2e7 &#43; 5;
    vector&lt;int&gt; primes(N), mobius(N), sum(N);
    vector&lt;bool&gt; st(N);
    auto sieve = [&amp;](int n) {
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
        for (int i = 1; i &lt;= n; i &#43;&#43;) {
        	sum[i] = (sum[i - 1] &#43; mobius[i] &#43; mod) % mod;
        }
    };
    sieve(N - 1);
    auto qmul = [&amp;](int a, int b) {
    	return (__int128)a * b % mod;
    };
    auto qmi = [&amp;](int a, int b) {
    	int res = 1;
    	while (b) {
    		if (b &amp; 1) res = qmul(res, a);
    		a = qmul(a, a);
    		b &gt;&gt;= 1;
    	}
    	return res;
    };
    unordered_map&lt;int, int&gt; mp;
    function&lt;int(int)&gt; Sum = [&amp;](int n) {
    	if (n &lt; N) return sum[n];
    	if (mp[n]) return mp[n];
    	int res = 1;
    	for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
    		r = n / (n / l);
    		res = (res - qmul(r - l &#43; 1, Sum(n / l)) % mod &#43; mod) % mod;
    	}
    	return mp[n] = res;
    };
    int res = 1;
    for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
    	r = n / (n / l);
    	int t = qmul(n / l, qmi(n - n / l, mod - 2));
    	res = (res - qmul(Sum(r) - Sum(l - 1), t) &#43; mod) % mod;
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2021-cccc%E5%A4%A9%E6%A2%AF%E8%B5%9B-%E5%8F%AF%E6%80%9C%E7%9A%84%E7%AE%80%E5%8D%95%E9%A2%98/  

