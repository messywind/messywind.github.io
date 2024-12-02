# DP


## 动态规划的引入：斐波那契数列

[题目链接](https://www.luogu.com.cn/problem/U198005)

已知斐波那契数列 $f_1 = 1, f_2 = 1, f_i = f_{i - 1} &#43; f_{i - 2}, i \ge 3$  ，给定正整数 $n$ ，求 $f_n \bmod 10 ^ 9 &#43; 7$ 

$3 \le n \le 2 \times 10^6$

直观上可以采用最暴力的方法，递归。

#### 递归代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int mod = 1e9 &#43; 7;
int n;
int f(int n) {
    if (n == 1 || n == 2) return 1;
    return (f(n - 1) &#43; f(n - 2)) % mod;
}
signed main() {
    cin &gt;&gt; n;
    cout &lt;&lt; f(n) &lt;&lt; endl;
}
```

可以试一下，这种方法在 $n = 45$ 时就会超时，原因是递归耗费了大量的时间，计算了许多重复的子问题。

假设 $n = 5$ 那么计算 $f_5$ 会先计算 $f_3, f_4$ ，计算 $f_3$ 会计算 $f_1,f_2$ ，计算 $f_4$ 会计算 $f_2,f_3$ ，这样 $f_3$ 就被计算了 $2$ 次，如果规模更大，会被重复计算很多次，我们称为**重复子问题 **。

![alt](https://uploadfiles.nowcoder.com/images/20220115/877350534_1642238826007/8A679F8E90087D25394508F4387CF863)

那么要避免重复计算情况我们可以用一个数组来保存计算的结果，使得每个值只会被计算 $1$ 次，我们称为**记忆化搜索**。

#### 记忆化搜索代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5, mod = 1e9 &#43; 7;
int n, dp[N];
int f(int n) {
    if (dp[n]) return dp[n];
    return dp[n] = (f(n - 1) &#43; f(n - 2)) % mod;
}
signed main() {
    dp[1] = dp[2] = 1;
    cin &gt;&gt; n;
    cout &lt;&lt; f(n) &lt;&lt; endl;
}
```

也可以直接用数组转移，这是一般 $\text{DP}$ 的常规写法

#### DP代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 2e6 &#43; 5, mod = 1e9 &#43; 7;
int n, dp[N];
signed main() {
    dp[1] = dp[2] = 1;
    for (int i = 3; i &lt; N; i &#43;&#43;) dp[i] = (dp[i - 1] &#43; dp[i - 2]) % mod;
    cin &gt;&gt; n;
    cout &lt;&lt; dp[n] &lt;&lt; endl;
}
```

如此可以对动态规划做一个总结：

1.最优子结构：$f(n)$ 的最优解可以由 $f(n−2)$ 和 $f(n−1)$ 的最优解推出。

2.无后效性：要求 $f(n)$，只需要求 $f(n−1)$ 和 $f(n−2)$，无需关心 $f(n - 1)$ 和 $f(n - 2)$ 是怎么得到的。

3.状态：求解过程进行到了哪一步，可以理解为一个子问题。

4.转移：$f_i = f_{i - 1} &#43; f_{i - 2}$

## 数字三角形：

[题目链接](https://www.acwing.com/problem/content/900/)

给定一个由 $n$ 行数字组成的数字三角形 $a_{i, j} (j \le i)$，计算出从三角形的顶至底的一条路径，使该路径经过的数字总和最大。$(1 \le n \le 5 \times 10 ^2, - 10^4 \le a_{i, j} \le 10 ^ 4)$

![](https://acm.sdut.edu.cn/image/1730.png)

状态表示：$dp_{i,j}$ 表示从起点走到 $(i, j)$ 的路径总和最小值。

状态转移：在点 $(i, j)$ 处只可能由两个点转移过来，分别是 $(i - 1, j - 1)$ 和 $(i - 1, j)$，所以该点的状态转移方程就是
$$
dp_{i,j} = \max(dp_{i - 1, j - 1}, dp_{i - 1, j}) &#43; a_{i, j}
$$
初始状态：因为我们要取最大值，所以一开始把所有状态更新为 $-\infty$ 并且 $(0, 0)$ 点的 $dp_{0, 0} = 0$ 

答案：根据状态和题意，我们需要找到底部的一个最优值，所以就遍历一遍最底部的一层 (枚举列)，来得到答案，即
$$
\max_{i = 1} ^ {n} dp_{n, i}
$$

#### 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 5e2 &#43; 5;
int n, a[N][N], dp[N][N], res = -2e9;
signed main() {
    memset(dp, -0x3f, sizeof dp);
    dp[0][0] = 0;
    cin &gt;&gt; n;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = 1; j &lt;= i; j &#43;&#43;) {
            cin &gt;&gt; a[i][j];
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = 1; j &lt;= i; j &#43;&#43;) {
            dp[i][j] = max(dp[i - 1][j - 1], dp[i - 1][j]) &#43; a[i][j];
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) res = max(res, dp[n][i]);
    cout &lt;&lt; res &lt;&lt; endl;
}
```

## 最长上升子序列

[题目链接](https://www.acwing.com/problem/content/897/)

给定一个长度为 $n$ 的数列 $a$，求**数值严格单调递增的非空子序列**的最长长度。

$(1 \le n \le 10^3,-10^9 \le a_i \le 10 ^ 9)$

状态表示：$dp_i$ 表示以 $a_i$ 结尾的最长上升子序列的长度。

状态转移：对于每个 $a_i$，枚举 $j \in [1, i - 1]$ 让 $a_i$ 接到 $a_j$ 后面的最优解，也就是
$$
dp_i = \max_{j = 1} ^{i - 1} (dp_j &#43; 1, dp_i)[a_j &lt; a_i]
$$
初始状态：$dp_{1 \sim n} = 1$ 每个数都是一个长度为 $1$ 的子序列。

答案：枚举可能作为最长上升子序列结尾的 $a_i$，取一遍最大值，即
$$
\max_{i = 1} ^ {n} dp_i
$$

#### 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 1e3 &#43; 5;
int n, dp[N], a[N], res;
signed main() {
    cin &gt;&gt; n;
    for (int i = 1; i &lt;= n; i &#43;&#43;) cin &gt;&gt; a[i];
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        dp[i] = 1;
        for (int j = 1; j &lt; i; j &#43;&#43;) {
            if (a[j] &lt; a[i]) {
                dp[i] = max(dp[j] &#43; 1, dp[i]);
            }
        }
        res = max(res, dp[i]);
    }
    cout &lt;&lt; res &lt;&lt; endl;
}
```

## 最长公共子序列

[题目链接](https://www.acwing.com/problem/content/899/)

给定两个长度分别为 $n$ 和 $m$ 的字符串 $a$ 和 $b$，求既是 $a$ 的子序列又是 $b$ 的子序列的字符串的最长长度。

$(1 \le n,m \le 10^3)$

状态表示：$dp_{i, j}$ 表示 $a$ 的前 $i$ 个字母和 $b$ 的前 $j$ 个字母的最长公共子序列。

状态转移：从小到大枚举 $a, b$ 所有不同位置 $i, j$ 的情况，如果 $a_i = b_j$ 说明遇到了公共部分，那么 $dp_{i, j}$ 应该从 $dp_{i - 1, j - 1}$ 转移过来，否则，就一定要舍弃 $a_i, b_j$ 中的一个字母，就取两种情况的最大值，即
$$
\begin{cases}
dp_{i, j} = \max(dp_{i,j}, dp_{i - 1, j - 1} &#43; 1) &amp;a_i = a_j \\
dp_{i, j} = \max(dp_{i - 1, j}, dp_{i, j - 1}) &amp;a_i \ne a_j
\end{cases}
$$
初始状态：所有状态初始为 $0$

答案：$a$ 中全部的字母和 $b$ 中全部的字母组成的最长上升子序列，即 $dp_{n, m}$

#### 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 1e3 &#43; 5;
int n, m, dp[N][N];
char a[N], b[N];
signed main() {
    cin &gt;&gt; n &gt;&gt; m &gt;&gt; a &#43; 1 &gt;&gt; b &#43; 1;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = 1; j &lt;= m; j &#43;&#43;) {
            if (a[i] == b[j]) {
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] &#43; 1);
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    cout &lt;&lt; dp[n][m] &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cpdp/  

