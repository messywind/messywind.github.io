# [Wannafly 28] Msc的背包


[原题链接](https://ac.nowcoder.com/acm/problem/21207)

**题意**

有 $n$ 种体积为 $1$ 的物品和 $m$ 种体积为 $2$ 的物品，求选择物品的体积为 $k$ 的方案数

对 $998244353$ 取模

$(1 \le n, m \le 10 ^ 6,1 \le k \le 9 \times 10 ^ 8)$

**分析：**

所有体积为 $1$ 的生成函数为
$$
F(x) = \left ( \sum_{i = 0} ^ {\infty} x ^ i \right ) ^ n
$$
所有体积为 $2$ 的生成函数为
$$
G(x) = \left ( \sum_{i = 0} ^ {\infty} x ^ {2i} \right ) ^ m
$$
那么组成的所有体积方案数为 $F(x) \times G(x)$

把 $F(x)$ 和 $G(x)$ 写成形式幂级数的逆的形式就为
$$
\frac{1}{(1-x) ^ n(1 - x ^ 2) ^ m}
$$
为了使分母的形式一致，对分数上下乘 $(1 &#43; x)  ^ n$
$$
\frac{(1 &#43; x) ^ n}{(1 - x ^ 2) ^ {n &#43; m}}
$$
再把 $\dfrac{1}{(1 - x ^ 2) ^ {n &#43; m}}$ 转为一般形式 $\sum\limits_{j = 0} ^ {\infty} \binom{j &#43; n &#43; m - 1}{n &#43; m - 1} x ^ {2j}$

那么 $(1 &#43; x) ^ n$ 也对应二项式展开 $\sum\limits_{i = 0} ^ {n} \binom{n}{i} x ^ i$

因为我们要求第 $k$ 项的系数，所以考虑 $x ^ i$ 与 $x ^ {2j}$ 凑出 $x ^ k$ 的所有项，

也就是 $i &#43; 2 j = k$，整理出 $j = \dfrac{k - i}{2}$

由于 $i \in [0, n]$ 所以可以 $O(n)$ 枚举 $i$ 的范围，即
$$
\sum_{i = 0} ^ {n} [(k - i) \bmod 2 = 0] \binom{n}{i} \binom{\dfrac{k - i}{2} &#43; n &#43; m - 1}{n &#43; m - 1} 
$$
 每次 $\dfrac{k - i}{2}$ 只会减少 $1$，所以对第二个的组合数可以递推求解，分母一定是 $(n &#43; m - 1)!$，那么分子每次必定会减少 $1$，所以只需要维护 $a$ 为第一次的 $\dfrac{k - i}{2} &#43; 1$，之后每次乘 $a &#43; n &#43; m$ 的逆元再乘 $a - 1$ 就是答案 ($a$ 每次自减 $1$)

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int mod = 998244353;
int qmi(int a, int b) {
    int res = 1;
    while (b) {
        if (b &amp; 1) res = res * a % mod;
        a = a * a % mod;
        b &gt;&gt;= 1;
    }
    return res;
}
vector&lt;int&gt; fact, infact;
void init(int n) {
    fact.resize(n &#43; 1), infact.resize(n &#43; 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        fact[i] = fact[i - 1] * i % mod;
    }
    infact[n] = qmi(fact[n], mod - 2);
    for (int i = n; i; i --) {
        infact[i - 1] = infact[i] * i % mod;
    }
}
int C(int n, int m) {
    if (n &lt; 0 || m &lt; 0 || n &lt; m) return 0;
    return fact[n] * infact[n - m] % mod * infact[m] % mod;
}
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    init(1e6);
    int n, m, k;
    cin &gt;&gt; n &gt;&gt; m &gt;&gt; k;
    int res = 0;
    int inv = 1;
    for (int i = 1; i &lt;= n &#43; m - 1; i &#43;&#43;) {
        inv = inv * i % mod;
    }
    inv = qmi(inv, mod - 2);
    int flag = 0, sum = 1, last = 0;
    for (int i = 0; i &lt;= n; i &#43;&#43;) {
        if ((k - i) % 2 == 0) {
            if (!flag) {
                flag = 1;
                for (int j = 1; j &lt;= n &#43; m - 1; j &#43;&#43;) {
                    sum = sum * ((k - i) / 2 &#43; j) % mod;
                }
                last = (k - i) / 2 &#43; 1;
            } else {
                last --;
                sum = sum * qmi((k - i) / 2 &#43; n &#43; m, mod - 2) % mod;
                sum = sum * last % mod;
            }
            res = (res &#43; C(n, i) * sum % mod * inv % mod) % mod;
        }
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/wannafly-28-msc%E7%9A%84%E8%83%8C%E5%8C%85/  

