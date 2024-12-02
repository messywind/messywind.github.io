# [HAOI2011] Problem B


**题意：**
求

$$\sum_{i=a}^{b}\sum_{j=c}^{d}[\gcd(i,j)=k]$$

**分析：**

用二维前缀和的思想，把答案分为$4$个部分

$$令S(x,y)=\sum_{i=1}^{x}\sum_{j=1}^{y}[\gcd(i,j)=k]$$

则

$$\sum_{i=a}^{b}\sum_{j=b}^{d}[\gcd(i,j)=k]=S(b,d)-S(a-1,d)-S(b,c-1)&#43;S(a-1,c-1)$$

问题转化为求 $S(x,y)$

构造

$$f(n)=\sum_{i=1}^{x}\sum_{j=1}^{y}[n|\gcd(i,j)]$$

$$g(n)=\sum_{i=1}^{x}\sum_{j=1}^{y}[\gcd(i,j)=n]$$

可以发现

$$f(n)=\sum_{n \mid d}g(d)$$

所以就可以莫比乌斯反演

$$g(n)=\sum_{n \mid d}\mu(\frac{d}{n})f(d)$$

$$\because d\mid\gcd(i,j),\therefore d\mid i,d \mid j$$

$$\therefore f(d)=\lfloor\frac{x}{d}\rfloor\lfloor\frac{y}{d}\rfloor$$

$$g(n)=\sum_{n \mid d}\mu(\frac{d}{n})\lfloor\frac{x}{d}\rfloor\lfloor\frac{y}{d}\rfloor$$

设

$$t=\frac{d}{n},d=nt$$

则

$$g(n)=\sum_{t=1}^{\min(x,y)}\mu(t)\lfloor\frac{x}{nt}\rfloor\lfloor\frac{y}{nt}\rfloor$$

所以只需要求 $g(k)$，可以枚举 $t$，用整除分块，加上筛莫比乌斯函数前缀和，时间复杂度 $O(N&#43;T\sqrt n)$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
const int N = 5e5 &#43; 5;
int T, a, b, c, d, k, mobius[N], primes[N], cnt, sum[N];
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
int f(int x, int y) {
    int res = 0;
    for (int l = 1, r; l &lt;= min(x, y); l = r &#43; 1) {
        r = min(x / (x / l), y / (y / l));
        res &#43;= (sum[r] - sum[l - 1]) * ((x / k) / l) * ((y / k) / l);
    }
    return res;
}
signed main() {
    get_mobius(N - 1);
    cin &gt;&gt; T;
    while (T --) {
        cin &gt;&gt; a &gt;&gt; b &gt;&gt; c &gt;&gt; d &gt;&gt; k;
        cout &lt;&lt; f(b, d) - f(a - 1, d) - f(b, c - 1) &#43; f(a - 1, c - 1) &lt;&lt; endl;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cphaoi2011-problem-b/  

