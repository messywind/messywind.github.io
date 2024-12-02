# [NOI2020统一省选 A] 组合数问题


[原题链接](https://www.luogu.com.cn/problem/P6620)

**题意**

给定四个整数 $n,x,p,m$，求
$$
\sum_{i=0}^{n}f(i)\times x^i\times \binom{n}{i}
$$
对 $p$ 取模，其中 $f(x) = a_0 &#43; a_1x &#43; a_2x ^ 2 &#43; \cdots &#43; a_mx ^ m$

$1 \le n,x,p \le 10 ^ 9, 0 \le a_i \le 10 ^ 9, 0 \le m \le \min(n, 10 ^ 3)$

**分析：**

首先把 $f(i)$ 带入原式
$$
\sum_{i=0}^{n} x^i\times \binom{n}{i} \sum_{j = 0} ^ {m} a_j \times i ^ {j}
$$
看到 $i ^ j$，故想到展开 $i ^ j = \sum\limits_{k = 0} ^ {j} {j \brace k} i ^ {\underline k}$

$$
\sum_{i=0}^{n} x^i\times \binom{n}{i} \sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k} \times \frac{i!}{(i - k)!}
$$

把前面的 $\dbinom{n}{i}$ 放到最后面化简

$$
\sum_{i=0}^{n} x^i \sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k} \times\dfrac{n!}{i! \times (n - i)!} \times \frac{i!}{(i - k)!} \\\\
= \sum_{i=0}^{n} x^i \sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k} \times\dfrac{n!}{(n - i)! \times (i - k)!}
$$

考虑凑组合数 $\dbinom{n - k}{n - i} = \dfrac{(n - k)!}{(n - i)! \times (i - k)!}$，所以分式上下同乘 $(n - k)!$，即
$$
\sum_{i=0}^{n} x^i \sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k} \times \binom{n - k}{n - i} \times n ^ {\underline k}
$$
交换求和次序，将 $i$ 放到最后求和

$$
\sum_{j = 0} ^ {m} a_{j} \sum_{k = 0} ^ {j} {j \brace k} \times n ^ {\underline k} \sum_{i=0}^{n} x^i \times \binom{n - k}{n - i} \\\\
= \sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k} \times n ^ {\underline k} \sum_{i=0}^{n} x^i \times \binom{n - k}{i - k} \\\\
= \sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k} \times n ^ {\underline k} \sum_{i=k}^{n} x^i \times \binom{n - k}{i - k}
$$

做变换 $(i - k) \rightarrow i$

$$
\sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k}  \times n ^ {\underline k} \sum_{i=0}^{n - k} x^{i &#43; k} \times \binom{n - k}{i} \\\\
= \sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k}  \times n ^ {\underline k} \times x ^ {k} \sum_{i=0}^{n - k} x^{i} \times \binom{n - k}{i}
$$

考虑二项式展开 $(a &#43; b) ^ n = \sum\limits_{i = 0} ^ {n} \dbinom{n}{i} a ^ {i} b ^ {n - i}$，所以 $\sum\limits_{i=0}^{n - k} x^{i} \times \dbinom{n - k}{i} = (1 &#43; x) ^ {n - k}$，故式子变为
$$
\sum_{j = 0} ^ {m} a_j \sum_{k = 0} ^ {j} {j \brace k}  \times n ^ {\underline k} \times x ^ {k} \times (1 &#43; x) ^ {n - k}
$$
这样式子就变为 $O(m ^ 2)$ 了，第二类斯特林数可以预处理，下降幂可以线性维护。

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
using i64 = long long;
constexpr int N = 1e3;
int mod;
int norm(int x) {
    if (x &lt; 0) {
        x &#43;= mod;
    }
    if (x &gt;= mod) {
        x -= mod;
    }
    return x;
}
template&lt;class T&gt;
T power(T a, int b) {
    T res = 1;
    for (; b; b /= 2, a *= a) {
        if (b % 2) {
            res *= a;
        }
    }
    return res;
}
struct Z {
    int x;
    Z(int x = 0) : x(norm(x)) {}
    int val() const {
        return x;
    }
    Z operator-() const {
        return Z(norm(mod - x));
    }
    Z inv() const {
        assert(x != 0);
        return power(*this, mod - 2);
    }
    Z &amp;operator*=(const Z &amp;rhs) {
        x = i64(x) * rhs.x % mod;
        return *this;
    }
    Z &amp;operator&#43;=(const Z &amp;rhs) {
        x = norm(x &#43; rhs.x);
        return *this;
    }
    Z &amp;operator-=(const Z &amp;rhs) {
        x = norm(x - rhs.x);
        return *this;
    }
    Z &amp;operator/=(const Z &amp;rhs) {
        return *this *= rhs.inv();
    }
    friend Z operator*(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res *= rhs;
        return res;
    }
    friend Z operator&#43;(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res &#43;= rhs;
        return res;
    }
    friend Z operator-(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res -= rhs;
        return res;
    }
    friend Z operator/(const Z &amp;lhs, const Z &amp;rhs) {
        Z res = lhs;
        res /= rhs;
        return res;
    }
    friend istream &amp;operator&gt;&gt;(istream &amp;is, Z &amp;a) {
        i64 v;
        is &gt;&gt; v;
        a = Z(v);
        return is;
    }
    friend ostream &amp;operator&lt;&lt;(ostream &amp;os, const Z &amp;a) {
        return os &lt;&lt; a.val();
    }
};
vector&lt;vector&lt;Z&gt;&gt; stirling(N &#43; 1, vector&lt;Z&gt;(N &#43; 1));
void init() {
    stirling[0][0] = 1;
    for (int i = 1; i &lt;= N; i &#43;&#43;) {
        for (int j = 1; j &lt;= i; j &#43;&#43;) {
            stirling[i][j] = stirling[i - 1][j - 1] &#43; j * stirling[i - 1][j];
        }
    }
}
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n, x, m;
    cin &gt;&gt; n &gt;&gt; x &gt;&gt; mod &gt;&gt; m;
    init();
    vector&lt;Z&gt; a(m &#43; 1);
    for (int i = 0; i &lt;= m; i &#43;&#43;) {
        cin &gt;&gt; a[i];
    }
    Z res;
    for (int j = 0; j &lt;= m; j &#43;&#43;) {
        Z sum = 1;
        for (int k = 0, cnt = n; k &lt;= j; k &#43;&#43;, cnt --) {
            res &#43;= a[j] * stirling[j][k] * power(Z(x), k) * sum * power(Z(1 &#43; x), n - k);
            sum *= cnt;
        }
    }
    cout &lt;&lt; res &lt;&lt; &#34;\n&#34;;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cpnoi2020%E7%BB%9F%E4%B8%80%E7%9C%81%E9%80%89-a-%E7%BB%84%E5%90%88%E6%95%B0%E9%97%AE%E9%A2%98/  

