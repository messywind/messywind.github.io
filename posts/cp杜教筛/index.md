# 杜教筛

## 杜教筛
杜教筛可以在 $O(n ^{\frac{2}{3}})$ 的复杂度内求解积性函数的前缀和

设数论函数 $f(x)$，求

$$\sum_{i=1}^{n}f(i)$$

令 $S(n) = \sum \limits_{i=1} ^{n} f(i)$，找一个数论函数 $g(x)$ 与 $f(x)$ 做卷积

$$\sum_{i=1} ^ {n} f * g = \sum_{i=1} ^{n}g *f$$

展开卷积

$$\sum_{i=1}^{n} \sum_{d \mid i}g(d)f(\frac{i}{d})$$

交换求和次序

$$\sum_{d=1} ^{n}g(d) \sum_{i=1}^{\lfloor\frac{n}{d} \rfloor} f(i)$$

后半部分其实就是 $S(\lfloor\dfrac{n}{d} \rfloor)$，换一下变量名得

$$\sum_{i=1} ^{n}g(i)S(\lfloor\frac{n}{i} \rfloor)$$

那么就有递推公式

$$g(1)S(n) = \sum_{i=1} ^{n}f*g-\sum_{i=2}^{n} g(i)S(\lfloor\frac{n}{i} \rfloor)$$

假设 $\sum\limits _{i=1} ^{n}f*g$ 可以快速求出，后半部分则可以用数论分块求解

## 莫比乌斯函数前缀和

由狄利克雷卷积

$$\varepsilon = \mu * I$$

设 $h=\varepsilon,f=\mu,g=I$ 套用杜教筛

$$S(n) = \sum_{i=1} ^{n}\varepsilon(i)-\sum_{i=2}^{n} I(i)S(\lfloor\frac{n}{i} \rfloor)$$

化简可得

$$S(n) = 1 - \sum_{i=2}^{n} S(\lfloor\frac{n}{i} \rfloor)$$

## 欧拉函数前缀和
**法一:** 

先对欧拉函数用莫比乌斯反演

$$\sum_{i=1} ^{n}\varphi(i)$$

展开欧拉函数

$$\sum_{i=1} ^{n} \sum_{j = 1} ^{n} [\gcd(i,j)=1]$$

莫比乌斯反演

$$\sum_{i=1}^{n} \sum_{j = 1} ^{n} \sum_{d \mid \gcd(i,j)}\mu(d)$$

交换求和次序

$$\sum_{d=1} ^{n}\mu(d) \lfloor \frac{n}{d} \rfloor ^2$$

那么就可以用前面的莫比乌斯函数前缀和来求解了

注意要减去 $i=1,j=1$ 的情况，并且 $/2$

**法二：**

由狄利克雷卷积

$$Id = \varphi * I$$

设 $h = Id,f=\varphi,g=I$ 套用杜教筛

$$S(n) = \sum_{i=1} ^{n}Id(i)-\sum_{i=2}^{n} I(i)S(\lfloor\frac{n}{i} \rfloor)$$

$$S(n) = \sum_{i=1} ^{n}i-\sum_{i=2}^{n} S(\lfloor\frac{n}{i} \rfloor)$$

等差数列求和

$$S(n) = \frac{n (n &#43; 1)}{2} - \sum_{i=2}^{n} S(\lfloor\frac{n}{i} \rfloor)$$


## 代码(两个函数一起求)：
题目：[洛谷P4213](https://www.luogu.com.cn/problem/P4213)
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5;
int T, n, mobius[N], primes[N], cnt, sum[N];
bool st[N];
unordered_map&lt;int, int&gt; mp;
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
int sum_mobius(int n) {
    if (n &lt; N) return sum[n];
    if (mp[n]) return mp[n];
    int res = 1;
    for (int l = 2, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res -= sum_mobius(n / l) * (r - l &#43; 1);
    }
    return mp[n] = res;
}
int sum_phi(int n) {
    int res = 0;
    for (int l = 1, r; l &lt;= n; l = r &#43; 1) {
        r = n / (n / l);
        res &#43;= (sum_mobius(r) - sum_mobius(l - 1)) * (n / l) * (n / l);
    }
    return (res - 1) / 2 &#43; 1;
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; T;
    while (T --) {
        cin &gt;&gt; n;
        cout &lt;&lt; sum_phi(n) &lt;&lt; &#34; &#34; &lt;&lt; sum_mobius(n) &lt;&lt; endl;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E6%9D%9C%E6%95%99%E7%AD%9B/  

