# 数论提高


## 前置：数学符号介绍

- $\sum\limits_{i = 1} ^ {n}$：求和符号，例如 $\sum\limits_{i = 1} ^ {n}i = \dfrac{n \times (n &#43; 1)}{2}$ 代表 $1 &#43; 2 &#43; 3 &#43; 4 &#43; \cdots &#43; n$ 
- $\prod\limits_{i = 1} ^ {n}$：连乘符号，例如 $\prod\limits_{i = 1} ^ {n} i = n!$ 代表 $1 \times 2 \times 3 \times 4 \times \cdots \times n$
- $\lfloor \dfrac{x}{y} \rfloor$：向下取整符号，例如 $\lfloor \dfrac{5}{2} \rfloor = 2$
- $[]$：艾弗森括号，例如 $[n = 1]$ 只有 $n = 1$ 时才取值为 $1$
- $x \mid y$：整除符号，表示 $x$ 整除 $y$，也就是 $x$ 是 $y$ 的约数，例如 $2 \mid 4$

## 欧拉函数

定义：$\varphi(x)$ 为小于等于 $x$ 与 $x$ 互质的数，即
$$
\varphi(x) = \sum_ {i = 1} ^ {x} [\gcd(x, i) = 1]
$$

### 性质：

积性函数：$\varphi(x \times y) = \varphi(x) \times \varphi(y)$

展开式：设 $x = p_1 ^ {\alpha_1}p_2 ^ {\alpha_2}\cdots p_k ^ {\alpha_k}$ 则 $\varphi(x) = x \times \prod\limits_{i = 1} ^ {k} \dfrac{p_i - 1}{p_i}$

欧拉反演：$n = \sum\limits_{d \mid n} \varphi(d)$

### $O(\sqrt n)$ 求欧拉函数值：

```cpp
int phi(int x) {
    int res = x;
    for (int i = 2; i * i &lt;= x; i &#43;&#43;) {
        if (x % i == 0) {
            res = res / i * (i - 1);
            while (x % i == 0) {
                x /= i;
            }
        }
    }
    if (x &gt; 1) {
        res = res / x * (x - 1);
    }
    return res;
}
```

### $O(n)$ 筛欧拉函数

```cpp
void get_eulers(int n) {
    euler[1] = 1;
    for (int i = 2; i &lt;= n; i &#43;&#43;) {
        if (!st[i]) {
            primes[cnt &#43;&#43;] = i;
            euler[i] = i - 1;
        }
        for (int j = 0; i * primes[j] &lt;= n; j &#43;&#43;) {
            int t = primes[j] * i;
            st[t] = true;
            if (i % primes[j] == 0) {
                euler[t] = euler[i] * primes[j];
                break;
            }
            euler[t] = euler[i] * (primes[j] - 1);
        }
    }
}
```

## 习题

### [2021 ICPC North American Qualifier Contest] Common Factors

题目链接：https://open.kattis.com/problems/commonfactors

**题意**

给定一个正整数 $n$，求
$$
\max\limits_{i = 2} ^ {n} \dfrac{i - \varphi(i)}{i}
$$
并输出其最简分数形式。$(1 \le n \le 10 ^ {18})$

**分析：**

将式子化简得 $\max\limits_{i = 2} ^ {n} \dfrac{i - \varphi(i)}{i} = 1 -\min_{i = 2} ^ {n} \dfrac{\varphi(i)}{i}$

其中 $\dfrac{\varphi(x)}{x} = \prod\limits_{i = 1} ^ {k}\dfrac{p_i - 1}{p_i}$，故转换为了求分数最小值。

由于 $\dfrac{p_i - 1}{p_i} &lt; 1$，所以要尽可能多的含有质因子，那么最好情况就是 $2 \times3\times\cdots$

注意到 $2 \times 3 \times 5 \times 7 \times 11 \times 13 \times 17 \times 19 \times 23 \times 29 \times 31 \times 37 \times 41 \times 43 \times 47 \times 53 &gt; 10 ^ {18}$ 所以只需要找到 $53$ 以内的素数，不断乘起来即可，注意判断时可能爆 `long long`，要转 `__int128`

**代码：**

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n;
    cin &gt;&gt; n;
    auto prime = [&amp;](int x) {
        for (int i = 2; i * i &lt;= x; i &#43;&#43;) {
            if (x % i == 0) {
                return false;
            }
        }
        return true;
    };
    auto phi = [&amp;](int x) {
        int res = x;
        for (int i = 2; i * i &lt;= x; i &#43;&#43;) {
            if (x % i == 0) {
                res = res / i * (i - 1);
                while (x % i == 0) {
                    x /= i;
                }
            }
        }
        if (x &gt; 1) {
            res = res / x * (x - 1);
        }
        return res;
    };
    int sum = 1;
    for (int i = 2; i &lt;= 53; i &#43;&#43;) {
        if (prime(i)) {
            if ((__int128)i * sum &gt; n) {
                int up = sum - phi(sum), down = sum;
                int Gcd = __gcd(up, down);
                up /= Gcd, down /= Gcd;
                cout &lt;&lt; up &lt;&lt; &#34;/&#34; &lt;&lt; down &lt;&lt; &#34;\n&#34;;
                return 0;
            }
            sum *= i;
        }
    }
}
```

**思考：** 求 $\dfrac{\varphi(n)}{n}$ 的最大值。

即只需要找到一个小于等于 $n$ 的最大质数。

### [洛谷 P1891] 疯狂 LCM

题目链接：https://www.luogu.com.cn/problem/P1891

**题意：**

$T$ 组输入，每次给定一个正整数 $n$，求

$$
\sum_{i=1} ^{n} \text{lcm}(i,n)
$$
$1 \le T \le 3 \times 10 ^ 5, 1 \le n \le 10 ^ 6$

**分析：**

由 $\text{lcm}(i,n) = \dfrac{i \times n}{\gcd{(i,n)}}$ 得

$$
\sum_{i=1} ^{n} \frac{i \times n}{\gcd{(i,n)}}
$$
枚举 $\gcd(i,n)$

$$
n \sum_{d \mid n} \sum_{i=1} ^{n} \frac{i }{d}[\gcd{(i,n)} = d]
$$
利用 $\gcd$ 的性质：

$$
n \sum_{d \mid n} \sum_{i=1} ^{n} \frac{i }{d}[\gcd{(\frac{i}{d},\frac{n}{d})} = 1]
$$
把 $d$ 拿到上界，也就是做变换 $\dfrac{i}{d} \rightarrow i$

$$
n \sum_{d \mid n} \sum_{i=1} ^{ \lfloor \frac{n}{d} \rfloor } i[\gcd{(i,\frac{n}{d})} = 1  ]
$$
由于约数成对出现，所以第二层和式的 $\lfloor \dfrac{n}{d} \rfloor$ 等价于 $d$

$$
n \sum_{d \mid n} \sum_{i=1} ^{ d } i[\gcd{(i,d)} = 1]
$$
由于 $\gcd(i, d) = \gcd(d - i,d)$ ，所以必成对出现，那么总共出现了 $\dfrac{\varphi(d)}{2}$ 次，$d - i &#43; i =d$，所以就是

$$
n \sum_{d \mid n} \frac{\varphi(d)}{2} d
$$
这样时间复杂度是 $O(N&#43;T\sqrt{n})$，但是可以用狄利克雷卷积优化，可以做到 $O(N \log{N} &#43; T)$

设 $F(x) = \dfrac{x  \times \varphi(x)}{2}$

则答案为 $n \times  F * \textbf{1}$，注意处理 $d \le 2$ 的情况。

**代码：**

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
            if (i &lt;= 2) {
                f[j] &#43;&#43;;
            } else {
                f[j] &#43;= euler[i] / 2 * i;
            }
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

## 数论分块

### 引入：$\sum\limits_{i = 1} ^ {n} \lfloor \dfrac{n}{i} \rfloor$

假设取 $n = 100$，我们可以通过打表发现每一项的分布情况，发现结果的取值都是连续一段的，并且有 $\sqrt n$ 块

对于每一块 $i$ 的右端点为 $\left \lfloor \dfrac{n}{\lfloor \dfrac{n}{i} \rfloor} \right \rfloor$，每一块的值都为 $\lfloor \dfrac{n}{i} \rfloor$

**代码:**

```cpp
for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res &#43;= (r - l &#43; 1) * (n / l);
}
```

这样就做到了 $O(\sqrt n)$ 求和

## 习题

•https://www.luogu.com.cn/problem/P2424

### [洛谷 P2261] 余数求和

题目链接：https://www.luogu.com.cn/problem/P2261

**题意**

给定正整数 $n, k$，求
$$
\sum_{i = 1} ^ {n} k \bmod i
$$
$(1 \le n, k \le 10 ^ 9)$

**分析：**

根据余数的定义：$k \bmod i = k - i \times \lfloor \dfrac{k}{i} \rfloor$

式子等价于
$$
\sum_{i = 1} ^ {n}(k - i \times \lfloor \dfrac{k}{i} \rfloor) =n \times k - \sum_{i = 1} ^ {n} i \times \lfloor \dfrac{k}{i} \rfloor
$$
注意到 $\lfloor \dfrac{k}{i} \rfloor$，可以利用整除分块

**代码：**

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n, k;
    cin &gt;&gt; n &gt;&gt; k;
    int res = 0;
    for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
        if (k / l != 0) {
            r = min(k / (k / l), n);
        } else {
            r = n;
        }
        res &#43;= (r - l &#43; 1) * (l &#43; r) / 2 * (k / l);
    }
    cout &lt;&lt; n * k - res &lt;&lt; &#34;\n&#34;;
}
```

### [洛谷 P2424] 约数和

题目链接：https://www.luogu.com.cn/problem/P2424

**题意**

定义 $f(x)$ 为 $x$ 的所有约数和，

给定两个正整数 $l, r$，求 $\sum\limits_{i = l} ^ {r} f(i)$

$(1 \le l &lt; r \le 2 \times 10 ^ 9)$

**分析：**

一个数 $x$ 的所有约数和可以表示为 $f(x) = \sum\limits_{d \mid x}d$

那么
$$
\sum\limits_{i = 1} ^ {n} f(i) = \sum\limits_{i = 1} ^ {n}\sum\limits_{d \mid i}d
$$
交换一下枚举顺序
$$
\sum\limits_{i = 1} ^ {n}\sum\limits_{d \mid i}d = \sum_{d = 1} ^ {n} d \times \lfloor \frac{n}{d} \rfloor
$$
可以利用整除分块求解

**代码：**

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
signed main() {
    int l, r;
    cin &gt;&gt; l &gt;&gt; r;
    auto f = [&amp;](int n) {
        if (!n) return 0ll;
        int res = 0;
        for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
            r = n / (n / l);
            res &#43;= (l &#43; r) * (r - l &#43; 1) * (n / l) / 2;
        }
        return res;
    };
    cout &lt;&lt; f(r) - f(l - 1) &lt;&lt; endl;
}
```

## 狄利克雷卷积

### 定义：

对于两个数论函数 $f(x),g(x)$ 那么它们的卷积 $h(x)$ 记作 $f(x) * g(x)$，式子如下：

$$
f(x) * g(x) = h(x) = \sum_{d \mid n} f(d)g(\frac{n}{d})
$$
简记为 $h = f * g$

### 性质：

**交换律：** $f * g = g * f$

**结合律：** $(f * g) * h = f * (g * h)$

**分配律：** $(f &#43; g) * h = f * h &#43; g * h$

**两个积性函数的狄利克雷卷积还是积性函数**

### 常见积性函数：

 **1. 莫比乌斯函数：$\mu(x)$**

 &gt;设 $n=p_1^{c_1}\cdots p_k^{c_k}$
 &gt;$$\mu(n)=\begin{cases}
 &gt;0,&amp;\exists i \in[1,k],c_i &gt;1 \\\\
 &gt;1,&amp;k \equiv 0\pmod2,\forall i \in[1,k],c_i=1\\\\
 &gt;-1,&amp;k\equiv1\pmod2,\forall i\in [1,k],c_i=1
 &gt;\end{cases}$$

 **2. 欧拉函数：$\varphi(x)$**
 &gt;$\varphi(n) = \sum \limits_{i=1} ^{n}[\gcd(i,n) = 1]$

 **3. 单位函数：$\varepsilon(x)$**
&gt;$\varepsilon(n) = [n = 1]$

 **4. 恒等函数：$Id(x)$**
 &gt;$Id(n) = n$

 **5. 常数函数：$I(x)$**
 &gt;$I(n)=1$

**6. 约数个数函数：$d(x)$**

&gt;$d(n)=\sum \limits_{i \mid n}1$

**7. 约数和函数：$\sigma(x)$**

&gt;$\sigma(n)=\sum \limits_{d \mid n} d$

### 常见卷积：

**1. $\varepsilon = \mu * 1$**

&gt;$\varepsilon = [n=1]=\sum \limits _{d \mid n} \mu (d)$

**2. $d = 1 * 1$**

&gt;$d(n)=\sum \limits_{i \mid n}1$

**3. $Id * 1 = \sigma$**
&gt;$\sigma(n)=\sum \limits_{d \mid n} d$

**4. $\mu * Id = \varphi$**

&gt;$\varphi(n)=\sum \limits _{d \mid n} d \times \mu(\dfrac{n}{d})$

**5. $\varphi * 1 = Id$**

&gt;$Id(n)=\sum \limits _{d \mid n} \varphi(d)$

狄利克雷卷积可做到 $O(n \log n)$，利用枚举倍数法。
$$
\frac{n}{1} &#43; \frac{n}{2} &#43; \cdots  &#43; \frac{n}{n} = n \times (\frac{1}{1} &#43; \frac{1}{2} &#43; \cdots  &#43; \frac{1}{n})
$$
其中 $\dfrac{1}{1} &#43; \dfrac{1}{2} &#43; \cdots  &#43; \dfrac{1}{n}$ 为调和级数，约为 $\ln n$

**枚举倍数法代码：**

```cpp
for (int i = 1; i &lt;= n; i &#43;&#43;) {
    for (int j = i; j &lt;= n; j &#43;= i) {
        
    }
}
```

## 习题

### [牛客小白月赛 40] 来点gcd

题目链接：https://ac.nowcoder.com/acm/problem/229589

**题意**

$T$ 组输入，给定一个有 $n$ 个元素的多重集 $S$，有 $m$ 个询问，对于每个询问，给出一个整数 $x$，问是否能选择 $S$ 的一个非空子集，满足这个子集的 $\gcd$ 等于 $x$，当集合只有一个数时，这个集合的 $\gcd$ 就等于这个数

$\sum n, m \le 10 ^ 6$

**分析：**

$S$ 中能取 $\gcd$ 成为 $x$ 的只可能是 $x$ 的倍数，故直接卷积 $O(n \log n)$ 即可。

 **代码：**

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
void solve() {
    int n, m;
    cin &gt;&gt; n &gt;&gt; m;
    vector&lt;int&gt; Gcd(n &#43; 1), mp(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        int x;
        cin &gt;&gt; x;
        mp[x] = 1;
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = i; j &lt;= n; j &#43;= i) {
            if (mp[j]) {
                Gcd[i] = __gcd(Gcd[i], j);
            }
        }
    }
    while (m --) {
        int x;
        cin &gt;&gt; x;
        cout &lt;&lt; (Gcd[x] == x ? &#34;YES&#34; : &#34;NO&#34;) &lt;&lt; &#34;\n&#34;;
    }
}
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int T;
    cin &gt;&gt; T;
    while (T --) {
        solve();
    }
}
```

### [AtCoder Beginner Contest 206] Divide Both

题目链接：https://atcoder.jp/contests/abc206/tasks/abc206_e

**题意**

给定两个正整数 $l, r$，找到区间 $[l, r]$ 满足下列条件的二元组 $(x, y)$

- $l \le x,y \le r$
- $\gcd(x, y) \ne 1$ 且 $\dfrac{x}{\gcd(x, y)} \ne 1$ 且 $\dfrac{y}{\gcd(x, y)} \ne 1$

**分析：**

转换一下题意，即找到区间不互质也不互为倍数的二元组。

考虑枚举两个数的公约数 $x$，设 $x$ 及 $x$ 的倍数在区间 $[l, r]$ 的个数为 $\text{cnt}_x$，那么两两组合的方案数为 $\text{cnt}_x ^ 2$，这样会有重复，所以减去 $x$ 的倍数所有方案，即 $\text{cnt}^ 2_x - \sum\limits_{x \mid d ,d\ne x} \text{cnt}^2_d$。此外，还需要减去与 $x$ 互为倍数的数量，即 $2 \times \text{cnt}_x - 1$，$-1$ 是因为 $(x, x)$ 重复减掉了一次。

**代码：**

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int l, r;
    cin &gt;&gt; l &gt;&gt; r;
    vector&lt;int&gt; cnt(r &#43; 1);
    for (int i = 2; i &lt;= r; i &#43;&#43;) {
        for (int j = i; j &lt;= r; j &#43;= i) {
            if (j &gt;= l) {
                cnt[i] &#43;&#43;;
            }
        }
    }
    int res = 0;
    vector&lt;int&gt; cnt2(r &#43; 1);
    for (int i = r; i &gt;= 2; i --) {
        cnt2[i] = cnt[i] * cnt[i];
        for (int j = i &lt;&lt; 1; j &lt;= r; j &#43;= i) {
            cnt2[i] -= cnt2[j];
        }
        res &#43;= cnt2[i];
        if (i &gt;= l) {
            res -= 2 * cnt[i] - 1;
        }
    }
    cout &lt;&lt; res &lt;&lt; &#34;\n&#34;;
}
```


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E6%95%B0%E8%AE%BA%E6%8F%90%E9%AB%98/  

