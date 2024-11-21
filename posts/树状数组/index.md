# 树状数组

## 树状数组的基本应用：
$O(\log n)$单点修改、区间查询

## 原理：
设区间 $[1,R]$，对区间右端点 $R$ 做二进制拆分，有：

$$R=2^{x_1}&#43;2^{x_2}&#43;\cdots&#43;2^{x_k}$$

假设 $x_1\sim x_k$ 严格单调递减，那么可以把区间 $[1,R]$ 拆分成 $\log R$ 个区间

$$[1,2^{x_1}],\\ 
[2^{x_1}&#43;1,2^{x_1}&#43;2^{x_2}],\\ 
\cdots, \\ [2^{x_1}&#43;2^{x_2}&#43; \cdots &#43; 2^{x_{k-1}}&#43;1,2^{x_1}&#43;2^{x_2}&#43;\cdots&#43;2^{x_k}]$$

可以发现每个区间的长度就等于每个区间结尾的 $\text{lowbit}$，所以可以建立一个数组 $tr$，保存区间 $[R-\text{lowbit}(R)&#43;1,R]$ 的和，也就是树状数组。

## 查询区间和：
对区间 $[L,R]$ 只需要求出 $\sum_{i=1}^{R}-\sum_{i=1}^{L-1}$。

所以目标只要计算区间 $[1,i]$ 的和：设 $i$ 的二进制下的最后一位 $1$ 是第 $k$ 位，那么只需要求出 $k-1$ 个子节点的和加上 $tr_i$，访问每个子节点只需要减去 $\text{lowbit}(i)$，一共 $\log i$ 次，所以时间复杂度为 $O(\log n)$

## 代码：
```cpp
int ask(int x) {
    int res = 0;
    for (int i = x; i; i -= lowbit(i)) res &#43;= tr[i];
    return res;
}
```

## 单点修改
假设令 $a_i$ 增加 $c$，考虑只有 $tr_i$ 及其祖先节点保存 $a_i$ 的值，所以只需要每次加上 $\text{lowbit}(i)$，就可以一直修改祖先节点，最多 $\log n$ 次，所以时间复杂度为 $O(\log n)$

## 代码：
```cpp
void update(int x, int c) {
    for (int i = x; i &lt;= n; i &#43;= lowbit(i)) tr[i] &#43;= c;
}
```

## 树状数组的扩展应用：

## $1.$求某个数前面或后面有几个数比它大或小
 [AcWing 788 逆序对的数量](https://www.acwing.com/problem/content/790/)

**分析：** 令$tr_x$ 定义为 $x$ 出现的次数，那么 $\sum_{i=L}^{R} tr[i]$ 就表示在区间 $[L,R]$ 中出现的数有多少个，那么相当于在 $x$ 的数值范围上建立一个树状数组。所以求逆序对时可以倒序统计 $i$ 之后比 $a_i$ 小的数，每次将 $tr_{a_i} &#43; 1$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
#define lowbit(x) x &amp; -x
#define find(x) lower_bound(num.begin(), num.end(), x) - num.begin()
using namespace std;
const int N = 1e5 &#43; 5;
int n, a[N], tr[N], res;
vector&lt;int&gt; num;
void modify(int x, int c) {
    for (int i = x; i &lt; N; i &#43;= lowbit(i)) tr[i] &#43;= c; 
}
int ask(int x) {
    int res = 0;
    for (int i = x; i; i -= lowbit(i)) res &#43;= tr[i];
    return res;
}
signed main() {
    cin &gt;&gt; n;
    for (int i = 1; i &lt;= n; i &#43;&#43;) cin &gt;&gt; a[i], num.push_back(a[i]);
    sort(num.begin(), num.end());
    num.erase(unique(num.begin(), num.end()), num.end());
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        res &#43;= ask(N - 1) - ask(find(a[i]) &#43; 1);
        modify(find(a[i]) &#43; 1, 1);
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```


 [AcWing 241 楼兰图腾](https://www.acwing.com/problem/content/243/)

**分析：** 与逆序对一样，在取值范围建立树状数组。求比 $a_i$ 小直接用 $ask(a_i-1)$，求比 $a_i$ 大的数可以用 $ask(n)-ask(a_i)$ 这一前缀和技巧处理。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
#define lowbit(x) x &amp; -x
using namespace std;
const int N = 2e5 &#43; 5;
int n, a[N], tr[N], res1, res2, high[N], low[N];
void update(int x, int c) {
    for (int i = x; i &lt;= n; i &#43;= lowbit(i)) tr[i] &#43;= c;
}
int ask(int x) {
    int res = 0;
    for (int i = x; i; i -= lowbit(i)) res &#43;= tr[i];
    return res;
}
signed main() {
    cin &gt;&gt; n;
    for (int i = 1; i &lt;= n; i &#43;&#43;) cin &gt;&gt; a[i];
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        high[i] = ask(n) - ask(a[i]);
        low[i] = ask(a[i] - 1);
        update(a[i], 1);
    }
    memset(tr, 0, sizeof tr);
    for (int i = n; i; i --) {
        res1 &#43;= high[i] * (ask(n) - ask(a[i]));
        res2 &#43;= low[i] * ask(a[i] - 1);
        update(a[i], 1);
    }
    cout &lt;&lt; res1 &lt;&lt; &#34; &#34; &lt;&lt; res2 &lt;&lt; endl;
}
```

## $2.$区间修改，单点查询

[AcWing 242 一个简单的整数问题](https://www.acwing.com/problem/content/248/)

**分析：** 可以利用差分的思想，在区间 $[L,R]$ 加上某一个数 $c$，那么就是在差分数组 $b$ 上将 $b_L&#43;c,b_{R&#43;1}-c$，所以可以用树状数组维护 $a_i$ 的差分。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
#define lowbit(x) x &amp; -x
using namespace std;
const int N = 1e5 &#43; 5;
int n, m, a[N], tr[N], l, r, x, d;
char op;
int ask(int x) {
    int res = 0;
    for (int i = x; i; i -= lowbit(i)) res &#43;= tr[i];
    return res;
}
void update(int x, int c) {
    for (int i = x; i &lt;= n; i &#43;= lowbit(i)) tr[i] &#43;= c;
}
signed main() {
    cin &gt;&gt; n &gt;&gt; m;
    for (int i = 1; i &lt;= n; i &#43;&#43;) cin &gt;&gt; a[i];
    for (int i = 1; i &lt;= n; i &#43;&#43;) update(i, a[i] - a[i - 1]);
    while (m --) {
        cin &gt;&gt; op;
        if (op == &#39;Q&#39;) {
            cin &gt;&gt; x;
            cout &lt;&lt; ask(x) &lt;&lt; endl;
        } else if (op == &#39;C&#39;) {
            cin &gt;&gt; l &gt;&gt; r &gt;&gt; d;
            update(l, d), update(r &#43; 1, -d);
        }
    }
}
```

## $3.$ 区间修改，区间查询
[AcWing 243 一个简单的整数问题2](https://www.acwing.com/problem/content/244/)

**分析：** 区间修改可以用差分维护，那么如果查询区间 $[1,R]$，就等价于求

$$\sum_{i=1}^{R}\sum_{j=1}^{i}b_{j}=\sum_{i=1}^{R}(R-i&#43;1) * b_{i}=(R&#43;1)\sum_{i=1}^{R}b_{i}-\sum_{i=1}^{R}i * b_{i}$$

所以只需要再增加一个树状数组维护 $i*b_i$ 的前缀和即可。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
#define lowbit(x) x &amp; -x
using namespace std;
const int N = 1e5 &#43; 5;
int n, m, a[N], l, r, d, tr1[N], tr2[N];
char op;
void update(int tr[], int x, int c) {
    for (int i = x; i &lt;= n; i &#43;= lowbit(i)) tr[i] &#43;= c;
}
int ask(int tr[], int x) {
    int res = 0;
    for (int i = x; i; i -= lowbit(i)) res &#43;= tr[i];
    return res;
}
int sum(int x) {
    return ask(tr1, x) * (x &#43; 1) - ask(tr2, x);
}
signed main() {
    cin &gt;&gt; n &gt;&gt; m;
    for (int i = 1; i &lt;= n; i &#43;&#43;) cin &gt;&gt; a[i];
    for (int i = 1; i &lt;= n; i &#43;&#43;) update(tr2, i, i * (a[i] - a[i - 1])), update(tr1, i, a[i] - a[i - 1]);
    while (m --) {
        cin &gt;&gt; op;
        if (op == &#39;Q&#39;) {
            cin &gt;&gt; l &gt;&gt; r;
            cout &lt;&lt; sum(r) - sum(l - 1) &lt;&lt; endl;
        } else if (op == &#39;C&#39;) {
            cin &gt;&gt; l &gt;&gt; r &gt;&gt; d;
            update(tr1, l, d), update(tr2, l, d * l);
            update(tr1, r &#43; 1, -d), update(tr2, r &#43; 1, (r &#43; 1) * -d);
        }
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84/  

