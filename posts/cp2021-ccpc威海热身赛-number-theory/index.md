# [2021 CCPC威海热身赛] Number Theory


**题意**

求

$$\sum_{k = 1}^{n}\sum_{i \mid k} \sum_{j \mid i} \lambda(i) \lambda(j)$$

对 $998244353$ 取模

其中 $\lambda(x) = (-1)^{\sum\limits_{i}e_i},x=\prod\limits_{i}p_i^{e_i}$

**分析：**

$\lambda(x)$ 为刘维尔函数，可以打表发现 $$\sum_{d \mid n}\lambda(d) =[n = a^2,a \in N^&#43;]$$

也就是 $n$ 是否为完全平方数

把式子中的 $\lambda(i)$ 提到前面

$$\sum_{k = 1}^{n}\sum_{i \mid k} \lambda(i)\sum_{j \mid i}  \lambda(j)$$

那么就变为

$$\sum_{k = 1}^{n}\sum_{i \mid k} \lambda(i)[i= a^2,a \in N^&#43;]$$

那么完全平方数的刘维尔函数为 $1$，再设 $f(x)=[i= a^2,a \in N^&#43;]$ 得

$$\sum_{i = 1}^{n}\sum_{d \mid i}f(d)$$

交换求和次序

$$\sum_{d = 1}^{n}f(d) \lfloor\frac{n}{d}\rfloor$$

这样直接枚举平方数即可，时间复杂度 $O(\sqrt{n})$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int mod = 998244353;
int n, res;
signed main() {
    cin &gt;&gt; n;
    for (int i = 1; i * i &lt;= n; i &#43;&#43;) {
        res = (res &#43; n / (i * i)) % mod;
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2021-ccpc%E5%A8%81%E6%B5%B7%E7%83%AD%E8%BA%AB%E8%B5%9B-number-theory/  

