# [2022 广东省赛M] 拉格朗日插值


**题意**

求在满足 $\sum\limits_{i = 1} ^ {k}\dfrac{x_i ^ 2}{a_i ^ 2} = 1$ 的条件下，从长度为 $m$ 的数组 $b$ 中选 $k$ 个数组成 $a_1,a_2,\cdots,a_k$，$\prod\limits_{i = 1} ^{k} x_i$ 的最大值的期望，$k$ 为偶数。

$(1 \le k \le m \le 10 ^ 5, 0  &lt; b_i &lt; 10 ^ 9)$ 

**分析：**

首先求解最大值需要用到高等数学中多元函数条件极值的拉格朗日乘数法，设
$$
L(x_1,x_2,\cdots,x_k, \lambda) = \prod_{i = 1} ^{k} x_i &#43;  \lambda(\sum\limits_{i = 1} ^ {k}\dfrac{x_i ^ 2}{a_i ^ 2} - 1)
$$
对每个变量求偏导数，令偏导数为 $0$ 得
$$
\frac{\partial L}{\partial x_1} = \frac{\prod\limits_{i = 1} ^{k} x_i}{x_1} &#43; \frac{2\lambda x_1}{a_1 ^ 2} = 0
\\\\
\frac{\partial L}{\partial x_2} = \frac{\prod\limits_{i = 1} ^{k} x_i}{x_2} &#43; \frac{2\lambda x_2}{a_2 ^ 2} = 0
\\\\
\cdots
\\\\
\frac{\partial L}{\partial x_k} = \frac{\prod\limits_{i = 1} ^{k} x_i}{x_k} &#43; \frac{2\lambda x_k}{a_k ^ 2} = 0
\\\\
\frac{\partial L}{\partial \lambda} =  \sum_{i = 1} ^ {k}\dfrac{x_i ^ 2}{a_i ^ 2} - 1 = 0
$$
那么稍微化简一下，对于 $1 \le i \le k$ 都有
$$
\prod_{i = 1} ^ {k}x_i = \frac{-2\lambda x_i ^ 2}{a_i ^ 2}
$$
通过任意两式 $1 \le i, j \le k$ 联立消掉 $\lambda$
$$
\frac{a_i ^ 2\prod\limits_{i = 1} ^ {k}x_i}{-2x_i ^ 2} = \frac{a_j ^ 2\prod\limits_{i = 1} ^ {k}x_i}{-2x_j ^ 2}
$$
化简得
$$
\frac{x_i}{a_i} = \frac{x_j}{a_j}
$$
所以当且仅当 $\dfrac{x_1}{a_1} = \dfrac{x_2}{a_2}=\cdots=\dfrac{x_k}{a_k}$ 时取得最大值，且 $\sum\limits_{i = 1} ^ {k}\dfrac{x_i ^ 2}{a_i ^ 2} = 1$，所以对任意 $1 \le i \le k$ 都有 $\dfrac{x_i}{a_i} = \pm \sqrt{\dfrac{1}{k}}$，那么 $\prod\limits_{i = 1} ^{k} x_i = k ^ {- \frac{k}{2}}\prod\limits_{i = 1} ^ {k} a_i$，因为 $k$ 为偶数，所以一定为正，且 $\dfrac{k}{2}$ 一定是整数。

求从 $b$ 数组中选出 $k$ 个数的所有乘积之和，考虑构造生成函数
$$
F(x) = \prod_{i = 1} ^ {k} (1 &#43; b_ix)
$$
那么 $[x ^ k]F(x)$ 就是选出 $k$ 个数的所有乘积之和，总共有 $\dbinom{m}{k}$ 种选法，所以期望就为
$$
k ^ {-\frac{k}{2}} \times \frac{[x ^ k]F(x)}{\dbinom{m}{k}}
$$
$F(x)$ 可用分治 $\text{NTT}$ 计算，总时间复杂度 $O(n\log ^ 2n)$

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
using i64 = long long;
constexpr int mod = 998244353;
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
vector&lt;int&gt; rev;
vector&lt;Z&gt; roots{0, 1};
void dft(vector&lt;Z&gt; &amp;a) {
    int n = a.size();
    if (int(rev.size()) != n) {
        int k = __builtin_ctz(n) - 1;
        rev.resize(n);
        for (int i = 0; i &lt; n; i &#43;&#43;) {
            rev[i] = rev[i &gt;&gt; 1] &gt;&gt; 1 | (i &amp; 1) &lt;&lt; k;
        }
    }
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        if (rev[i] &lt; i) {
            swap(a[i], a[rev[i]]);
        }
    }
    if (int(roots.size()) &lt; n) {
        int k = __builtin_ctz(roots.size());
        roots.resize(n);
        while ((1 &lt;&lt; k) &lt; n) {
            Z e = power(Z(3), (mod - 1) &gt;&gt; (k &#43; 1));
            for (int i = 1 &lt;&lt; (k - 1); i &lt; (1 &lt;&lt; k); i &#43;&#43;) {
                roots[i &lt;&lt; 1] = roots[i];
                roots[i &lt;&lt; 1 | 1] = roots[i] * e;
            }
            k &#43;&#43;;
        }
    }
    for (int k = 1; k &lt; n; k *= 2) {
        for (int i = 0; i &lt; n; i &#43;= 2 * k) {
            for (int j = 0; j &lt; k; j &#43;&#43;) {
                Z u = a[i &#43; j], v = a[i &#43; j &#43; k] * roots[k &#43; j];
                a[i &#43; j] = u &#43; v, a[i &#43; j &#43; k] = u - v;
            }
        }
    }
}
void idft(vector&lt;Z&gt; &amp;a) {
    int n = a.size();
    reverse(a.begin() &#43; 1, a.end());
    dft(a);
    Z inv = (1 - mod) / n;
    for (int i = 0; i &lt; n; i &#43;&#43;) {
        a[i] *= inv;
    }
}
struct Poly {
    vector&lt;Z&gt; a;
    Poly() {}
    Poly(const vector&lt;Z&gt; &amp;a) : a(a) {}
    Poly(const initializer_list&lt;Z&gt; &amp;a) : a(a) {}
    int size() const {
        return a.size();
    }
    void resize(int n) {
        a.resize(n);
    }
    Z operator[](int idx) const {
        if (idx &lt; size()) {
            return a[idx];
        } else {
            return 0;
        }
    }
    Z &amp;operator[](int idx) {
        return a[idx];
    }
    Poly mulxk(int k) const {
        auto b = a;
        b.insert(b.begin(), k, 0);
        return Poly(b);
    }
    Poly modxk(int k) const {
        k = min(k, size());
        return Poly(vector&lt;Z&gt;(a.begin(), a.begin() &#43; k));
    }
    Poly divxk(int k) const {
        if (size() &lt;= k) {
            return Poly();
        }
        return Poly(vector&lt;Z&gt;(a.begin() &#43; k, a.end()));
    }
    friend Poly operator&#43;(const Poly &amp;a, const Poly &amp;b) {
        vector&lt;Z&gt; res(max(a.size(), b.size()));
        for (int i = 0; i &lt; int(res.size()); i &#43;&#43;) {
            res[i] = a[i] &#43; b[i];
        }
        return Poly(res);
    }
    friend Poly operator-(const Poly &amp;a, const Poly &amp;b) {
        vector&lt;Z&gt; res(max(a.size(), b.size()));
        for (int i = 0; i &lt; int(res.size()); i &#43;&#43;) {
            res[i] = a[i] - b[i];
        }
        return Poly(res);
    }
    friend Poly operator*(Poly a, Poly b) {
        if (a.size() == 0 || b.size() == 0) {
            return Poly();
        }
        int sz = 1, tot = a.size() &#43; b.size() - 1;
        while (sz &lt; tot) {
            sz *= 2;
        }
        a.a.resize(sz);
        b.a.resize(sz);
        dft(a.a);
        dft(b.a);
        for (int i = 0; i &lt; sz; i &#43;&#43;) {
            a.a[i] = a[i] * b[i];
        }
        idft(a.a);
        a.resize(tot);
        return a;
    }
    friend Poly operator*(Z a, Poly b) {
        for (int i = 0; i &lt; int(b.size()); i &#43;&#43;) {
            b[i] *= a;
        }
        return b;
    }
    friend Poly operator*(Poly a, Z b) {
        for (int i = 0; i &lt; int(a.size()); i &#43;&#43;) {
            a[i] *= b;
        }
        return a;
    }
    Poly &amp;operator&#43;=(Poly b) {
        return (*this) = (*this) &#43; b;
    }
    Poly &amp;operator-=(Poly b) {
        return (*this) = (*this) - b;
    }
    Poly &amp;operator*=(Poly b) {
        return (*this) = (*this) * b;
    }
    Poly deriv() const {
        if (a.empty()) {
            return Poly();
        }
        vector&lt;Z&gt; res(size() - 1);
        for (int i = 0; i &lt; size() - 1; i &#43;&#43;) {
            res[i] = (i &#43; 1) * a[i &#43; 1];
        }
        return Poly(res);
    }
    Poly integr() const {
        vector&lt;Z&gt; res(size() &#43; 1);
        for (int i = 0; i &lt; size(); i &#43;&#43;) {
            res[i &#43; 1] = a[i] / (i &#43; 1);
        }
        return Poly(res);
    }
    Poly inv(int m) const {
        Poly x{a[0].inv()};
        int k = 1;
        while (k &lt; m) {
            k *= 2;
            x = (x * (Poly{2} - modxk(k) * x)).modxk(k);
        }
        return x.modxk(m);
    }
    Poly log(int m) const {
        return (deriv() * inv(m)).integr().modxk(m);
    }
    Poly exp(int m) const {
        Poly x{1};
        int k = 1;
        while (k &lt; m) {
            k *= 2;
            x = (x * (Poly{1} - x.log(k) &#43; modxk(k))).modxk(k);
        }
        return x.modxk(m);
    }
    Poly pow(int k, int m) const {
        int i = 0;
        while (i &lt; size() &amp;&amp; a[i].val() == 0) {
            i &#43;&#43;;
        }
        if (i == size() || 1LL * i * k &gt;= m) {
            return Poly(vector&lt;Z&gt;(m));
        }
        Z v = a[i];
        auto f = divxk(i) * v.inv();
        return (f.log(m - i * k) * k).exp(m - i * k).mulxk(i * k) * power(v, k);
    }
    Poly sqrt(int m) const {
        Poly x{1};
        int k = 1;
        while (k &lt; m) {
            k *= 2;
            x = (x &#43; (modxk(k) * x.inv(k)).modxk(k)) * ((mod &#43; 1) / 2);
        }
        return x.modxk(m);
    }
    Poly mulT(Poly b) const {
        if (b.size() == 0) {
            return Poly();
        }
        int n = b.size();
        reverse(b.a.begin(), b.a.end());
        return ((*this) * b).divxk(n - 1);
    }
};
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n, k;
    cin &gt;&gt; n &gt;&gt; k;
    vector&lt;Z&gt; fact(n &#43; 1), infact(n &#43; 1);
    fact[0] = infact[0] = 1;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        fact[i] = fact[i - 1] * i;
    }
    infact[n] = fact[n].inv();
    for (int i = n; i; i --) {
        infact[i - 1] = infact[i] * i;
    }
    vector&lt;int&gt; b(n &#43; 1);
    vector&lt;Poly&gt; f(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        cin &gt;&gt; b[i];
        f[i].resize(2);
        f[i][0] = 1, f[i][1] = b[i];
    }
    function&lt;Poly(int, int)&gt; dc = [&amp;](int l, int r) {
        
        if (l == r) return f[l];
        int mid = l &#43; r &gt;&gt; 1;
        return dc(l, mid) * dc(mid &#43; 1, r);
    };
    auto ans = dc(1, n);
    Z res = 1;
    auto C = [&amp;](int n, int m) {
        if (n &lt; 0 || m &lt; 0 || n &lt; m) return Z(0);
        return fact[n] * infact[n - m] * infact[m];
    };
    cout &lt;&lt; power(Z(k), k / 2).inv() * ans[k] * C(n, k).inv() &lt;&lt; &#34;\n&#34;;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/2022-%E5%B9%BF%E4%B8%9C%E7%9C%81%E8%B5%9Bm-%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E6%8F%92%E5%80%BC/  

