# [洛谷 P4318] 完全平方数


[原题链接](https://www.luogu.com.cn/problem/P4318)

**题意**

$T$ 组询问，回答第 $K_i$ 个不是完全平方数的正整数倍的数。

$1\le K_i \le 10^9,T \le 50$

**分析：**

## 法一：

如果一个数 $n$ 不是完全平方数，那么 $n=p_1^{\alpha_1}p_2^{\alpha_2} \cdots p_k^{\alpha_k}$ 中 $0 \le \alpha_i \le 1$，所以就想到了莫比乌斯函数，那么题目要询问第 $K$ 个数是什么，可以用二分来解决，但是必须要有单调性，莫比乌斯函数前缀和可能存在负数，所以就想到把莫比乌斯函数做一个平方，这样前缀和就没有负数了，就有了单调性。

现在考虑如何计算 $\sum\limits_{i=1} ^{n} \mu^2(i)$，根据数据范围来看必须要用杜教筛来快速求前缀和，设 $f(n)=\mu^2(n)$，那么设 $g(n)=[n=k ^ 2,k \in N^&#43;]$，发现 $f*g=1$，所以 

$$S(n)=n-\sum_{i=2}^{n}g(i)S(\lfloor \frac{n}{i} \rfloor)$$

改为枚举平方

$$S(n)=n-\sum_{i=2}^{\sqrt{n}}S(\lfloor \frac{n}{i^2} \rfloor)$$

## 代码($O_2$优化)：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5;
unordered_map&lt;int,int&gt; mp;
int T, n, mobius[N], primes[N], cnt, sum[N];
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
    for (int i = 1; i &lt;= n; i &#43;&#43;) sum[i] = sum[i - 1] &#43; mobius[i] * mobius[i];
}
int Sum(int n) {
    if (n &lt; N) return sum[n];
    if (mp[n]) return mp[n];
    int res = n;
    for (int l = 2, r; l * l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res -= Sum(n / (l * l));
    }
    return mp[n] = res;
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; T;
    while (T --) {
        cin &gt;&gt; n;
        int l = 1, r = n &lt;&lt; 1;
        while (l &lt; r) {
            int mid = l &#43; r &gt;&gt; 1;
            if (Sum(mid) &lt; n) {
                l = mid &#43; 1;
            } else {
                r = mid;
            }
        }
        cout &lt;&lt; l &lt;&lt; endl;
    }
}
```

## 法二：
$$\sum_{i=1}^{n} \mu^2(i)=\sum_{i=1} ^{n}\sum_{d^2 \mid i} \mu(d)=\sum_{d=1} ^{\sqrt{n}} \mu(d)\lfloor \frac{n}{d^2}\rfloor$$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5;
int T, n, mobius[N], primes[N], cnt, sum[N];
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
int Sum(int n) {
    int res = 0;
    for (int l = 1, r; l * l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res &#43;= (sum[r] - sum[l - 1]) * (n / (l * l));
    }
    return res;
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; T;
    while (T --) {
        cin &gt;&gt; n;
        int l = 1, r = n &lt;&lt; 1;
        while (l &lt; r) {
            int mid = l &#43; r &gt;&gt; 1;
            if (Sum(mid) &lt; n) {
                l = mid &#43; 1;
            } else {
                r = mid;
            }
        }
        cout &lt;&lt; l &lt;&lt; endl;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E6%B4%9B%E8%B0%B7-p4318-%E5%AE%8C%E5%85%A8%E5%B9%B3%E6%96%B9%E6%95%B0/  

