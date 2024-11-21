# 数位DP(DFS做法)

由于算法提高课的数位 DP 的非搜索做法比较难想，所以总结一下数位 DP 的 DFS 写法。

数位 DP 问题一般给定一个区间 $[L,R]$，问区间满足的条件的数有多少个。

可以利用前缀和来求解答案：$\sum \limits  _ {i=1}^{R} ans_i - \sum \limits_{i=1}^{L - 1} ans_i$

## 模板：

```cpp
int dfs(int pos, int pre, int lead, int limit) {
    if (!pos) {
    	边界条件
    }
    if (!limit &amp;&amp; !lead &amp;&amp; dp[pos][pre] != -1) return dp[pos][pre];
    int res = 0, up = limit ? a[pos] : 无限制位;
    for (int i = 0; i &lt;= up; i &#43;&#43;) {
        if (不合法条件) continue;
        res &#43;= dfs(pos - 1, 未定参数, lead &amp;&amp; !i, limit &amp;&amp; i == up);
    }
    return limit ? res : (lead ? res : dp[pos][sum] = res);
}
int cal(int x) {
    memset(dp, -1, sizeof dp);
    len = 0;
    while (x) a[&#43;&#43; len] = x % 进制, x /= 进制;
    return dfs(len, 未定参数, 1, 1);
}
signed main() {
    cin &gt;&gt; l &gt;&gt; r;
    cout &lt;&lt; cal(r) - cal(l - 1) &lt;&lt; endl;
}
```
## $\text{cal}$函数（一般情况）：

**注意每次初始化 DP 数组为 $-1$，长度 $len=0$**


## 基本参数：

$len:$ 数位长度，一般根据这个来确定数组范围

$a_i:$ 每个数位具体数字

返回值 $\text{return}$ 根据题目的初始条件来确定前导 $0$ 以及 $pre$


## DFS 函数（一般情况）：
变量 $res$ 来记录答案，初始化一般为 $0$

变量 $up$ 表示能枚举的最高位数

采用记忆化搜索的方式：

`if (!limit &amp;&amp; !lead &amp;&amp; dp[pos][pre] != -1) return dp[pos][pre];`

&gt;只有无限制、无前导零才算，不然都是未搜索完的情况。


`return limit ? res : dp[pos][pre] = res;`

&gt;如果最后还有限制，那么返回 `res`，否则返回 `dp[pos][pre]`
## 基本参数：
假设数字 $x$ 位数为 $a_1\cdots a_n$

 **必填参数：**

$pos:$ 表示数字的位数
&gt;从末位或第一位开始，要根据题目的数字构造性质来选择顺序，一般选择从 $a_1$ 到 $a_n$ 的顺序。初始从 $len$ 开始的话，边界条件应该是 $pos = 0$，限制位数应该是 $a_{pos}$，DFS 时 $pos-1$；初始从$1$开始的话，边界条件应该是 $pos &gt; len$，限制位数应该是 $a_{len - pos &#43; 1}$，DFS 时 $pos&#43;1$。两种都可以，看个人习惯。

$limit:$ 可以填数的限制（无限制的话 $(limit=0)$ $0\sim 9$ 随便填，否则只能填到 $a_i$）
&gt;如果搜索到 $a_1\cdots a_{pos} \cdots a_n$，原数位为 $a_1\cdots a_k \cdots a_n$，那么我们必须对接下来搜索的数加以限制，也就是不能超过区间右端点 $R$，所以要引入 $limit$ 这个参数，如果 $limit=1$，那么最高位数 $up \le a_{pos&#43;1}$，如果没有限制，那么 $up=9$（十进制下）这也就是确定搜索位数上界的语句 `limit ? a[pos] : 9;`
&gt;如果 $limit=1$ 且已经取到了能取到的最高位时 $(a_{pos}=a_k)$，那么下一个 $limit=1$
&gt;如果 $limit=1$ 且没有取到能取到的最高位时 $(a_{pos} &lt; a_k)$，那么下一个 $limit=0$
&gt;如果 $limit=0$，那么下一个 $limit=0$，因为前一位没有限制后一位必定没有限制。
&gt;所以我们可以把这 $3$ 种情况合成一个语句进行下一次搜索：`limit &amp;&amp; i == up`
$(i$为当前枚举的数字$)$


**可选参数：**

$pre:$ 表示上一个数是多少
&gt;有些题目会用到前面的数

$lead:$ 前导零是否存在，$lead=1$ 存在前导零，否则不存在。
&gt;一般来说有些题目不加限制前导零会影响数字结构，所以 $lead$ 是一个很重要的参数。
&gt;如果 $lead=1$ 且当前位为 $0$，那么说明当前位是前导 $0$，继续搜索 $pos&#43;1$，其他条件不变。
&gt;如果 $lead=1$ 且当前位不为 $0$，那么说明当前位是最高位，继续搜索 $pos&#43;1$，条件变动。
&gt;如果 $lead=0$，则不需要操作。

$sum:$ 搜索到当前所有数字之和

&gt;有些题目会出现数字之和的条件

$cnt:$ 某个数字出现的次数
&gt;有些题目会出现某个数字出现次数的条件

## 参数基本的差不多这些，有些较难题目会用到更多方法或改变$\text{DP}$状态

## [题目1：不要62](https://www.acwing.com/problem/content/1087/)

**题意：** 找到区间 $[L,R]$ 不能出现 $4$ 和 $62$ 的数的个数

**分析：** 首先此题不需要 $lead$，其次有 $62$ 所以要记前驱 $pre$
## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
const int N = 15;
int l, r, dp[N][N], len, a[N];
int dfs(int pos, int pre, int limit) {
    if (!pos) return 1;
    if (!limit &amp;&amp; dp[pos][pre] != -1) return dp[pos][pre];
    int res = 0, up = limit ? a[pos] : 9;
    for (int i = 0; i &lt;= up; i &#43;&#43;) {
        if (i == 4 || (i == 2 &amp;&amp; pre == 6)) continue;
        res &#43;= dfs(pos - 1, i, limit &amp;&amp; i == up);
    }
    return limit ? res : dp[pos][pre] = res;
}
int cal(int x) {
    memset(dp, -1, sizeof dp);
    len = 0;
    while (x) a[&#43;&#43; len] = x % 10, x /= 10;
    return dfs(len, 0, 1);
}
signed main() {
    while (cin &gt;&gt; l &gt;&gt; r, l || r) {
        cout &lt;&lt; cal(r) - cal(l - 1) &lt;&lt; endl;
    }
}
```

## [ 题目2：windy数](https://www.acwing.com/problem/content/1085/)

**题意：** 找到区间 $[L,R]$ 相邻数字之差至少为 $2$ 的数的个数

**分析：** 搜索初始条件第二个参数 $pre$ 必须填一个 $\le -2$ 的数来保证可以搜索下去，不然会出错。此题需要记录前导零，不然忽视前导零的影响会造成最高位数 $-0&lt;2$ 无法继续搜索的情况。

## 代码：
```
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
const int N = 15;
int l, r, a[N], len, dp[N][N];
int dfs(int pos, int pre, int lead, int limit) {
    if (!pos) return 1;
    if (!limit &amp;&amp; !lead &amp;&amp; dp[pos][pre] != -1) return dp[pos][pre];
    int res = 0, up = limit ? a[pos] : 9;
    for (int i = 0; i &lt;= up; i &#43;&#43;) {
        if (abs(pre - i) &lt; 2) continue;
        if (lead &amp;&amp; !i) {
            res &#43;= dfs(pos - 1, -2, lead &amp;&amp; !i, limit &amp;&amp; i == up);
        } else {
            res &#43;= dfs(pos - 1, i, lead &amp;&amp; !i, limit &amp;&amp; i == up);
        }
    }
    return limit ? res : (lead ? res : dp[pos][pre] = res);
}
int cal(int x) {
    memset(dp, -1, sizeof dp);
    len = 0;
    while (x) a[&#43;&#43; len] = x % 10, x /= 10;
    return dfs(len, -2, 1, 1);
}
signed main() {
    cin &gt;&gt; l &gt;&gt; r;
    cout &lt;&lt; cal(r) - cal(l - 1) &lt;&lt; endl;
}
```

## [题目3：数字游戏](https://www.acwing.com/problem/content/1084/)

**题意：** 找到区间 $[L,R]$ 各位数字非严格单调递增的数的个数

**分析：** 前导零不影响，所以不需要 $lead$。所以只需要判断枚举的位数是不是非严格递增来判断是否继续搜索。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
const int N = 15;
int l, r, a[N], len, dp[N][N];
int dfs(int pos, int pre, int limit) {
    if (!pos) return 1;
    if (!limit &amp;&amp; dp[pos][pre] != -1) return dp[pos][pre];
    int res = 0, up = limit ? a[pos] : 9;
    for (int i = 0; i &lt;= up; i &#43;&#43;) {
        if (i &lt; pre) continue;
        res &#43;= dfs(pos - 1, i, limit &amp;&amp; i == up);
    }
    return limit ? res : dp[pos][pre] = res;
}
int cal(int x) {
    memset(dp, -1, sizeof dp);
    len = 0;
    while (x) a[&#43;&#43; len] = x % 10, x /= 10;
    return dfs(len, 0, 1);
}
signed main() {
    while (cin &gt;&gt; l &gt;&gt; r) {
        cout &lt;&lt; cal(r) - cal(l - 1) &lt;&lt; endl;
    }
}
```

## [题目4：数字游戏Ⅱ](https://www.acwing.com/problem/content/1086/)

**题意：** 找到区间 $[L,R]$ 各位数字之和 $\mod n=0$ 的数的个数

**分析：** 前导零不影响，所以不需要 $lead$。此题涉及到数字和 ，所以要用到 $sum$，不需要记录前驱 $pre$，所以 $\text{dp}$ 状态变为了 $\text{dp}[pos][sum]$。边界条件为 $sum \bmod n=0$，返回 $1$，否则返回 $0$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
const int N = 1e3 &#43; 5;
int l, r, p, len, a[N], dp[N][N];
int dfs(int pos, int sum, int limit) {
    if (!pos) return sum % p == 0;
    if (!limit &amp;&amp; dp[pos][sum] != -1) return dp[pos][sum];
    int res = 0, up = limit ? a[pos] : 9;
    for (int i = 0; i &lt;= up; i &#43;&#43;) {
        res &#43;= dfs(pos - 1, sum &#43; i, limit &amp;&amp; i == up);
    }
    return limit ? res : dp[pos][sum] = res;
}
int cal(int x) {
    memset(dp, -1, sizeof dp);
    len = 0;
    while (x) a[&#43;&#43; len] = x % 10, x /= 10;
    return dfs(len, 0, 1);
}
signed main() {
    while (cin &gt;&gt; l &gt;&gt; r &gt;&gt; p) {
        cout &lt;&lt; cal(r) - cal(l - 1) &lt;&lt; endl;
    }
}
```

## [题目5：度的数量](https://www.acwing.com/problem/content/1083/)

**题意：** 找到区间$[L,R]$恰好为$K$个$B$的幂次方之和的数的个数

**分析：** 前导零不影响，所以不需要$lead$。因为要记录数量，所以要增加变量$cnt$。前驱$pre$不需要记录。判断边界时只要最后数量$cnt=k$，返回$1$，否则返回$0$。同时枚举数字时如果前面系数不为$1$或者没搜索完就已经$K$个了，那么就`continue`

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
const int N = 35;
int l, r, k, b, a[N], len, dp[N][N];
int dfs(int pos, int cnt, int limit) {
    if (!pos) return cnt == k;
    if (!limit &amp;&amp; dp[pos][cnt] != -1) return dp[pos][cnt];
    int res = 0, up = limit ? a[pos] : b - 1;
    for (int i = 0; i &lt;= up; i &#43;&#43;) {
        if ((i == 1 &amp;&amp; cnt == k) || i &gt; 1) continue;
        res &#43;= dfs(pos - 1, cnt &#43; (i == 1), limit &amp;&amp; i == up);
    }
    return limit ? res : dp[pos][cnt] = res;
}
int cal(int x) {
    memset(dp, -1, sizeof dp);
    len = 0;
    while (x) a[&#43;&#43; len] = x % b, x /= b;
    return dfs(len, 0, 1);
}
signed main() {
    cin &gt;&gt; l &gt;&gt; r &gt;&gt; k &gt;&gt; b;
    cout &lt;&lt; cal(r) - cal(l - 1) &lt;&lt; endl;
}
```

## [题目6：计数问题](https://www.acwing.com/problem/content/340/)
**题意：** 统计区间$[L,R]$出现$0123456789$的各个数字总次数

**分析：** 需要用到$lead$，需要用到次数总和$sum$，还有哪个数字$num$。基本上可以套模板，注意边界条件。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
using namespace std;
const int N = 15;
int l, r, len, a[N], dp[N][N];
int dfs(int pos, int sum, int num, int lead, int limit) {
    if (!pos) {
        if (lead &amp;&amp; !num) return 1;
        return sum;
    }
    if (!limit &amp;&amp; !lead &amp;&amp; dp[pos][sum] != -1) return dp[pos][sum];
    int res = 0, up = limit ? a[pos] : 9;
    for (int i = 0; i &lt;= up; i &#43;&#43;) {
        int t;
        if (i == num) {
            if (!num) {
                t = sum &#43; (lead == 0);
            } else {
                t = sum &#43; 1;
            }
        } else {
            t = sum;
        }
        res &#43;= dfs(pos - 1, t, num, lead &amp;&amp; i == 0, limit &amp;&amp; i == up);
    }
    return limit ? res : (lead ? res : dp[pos][sum] = res);
}
int cal(int x, int num) {
    memset(dp, -1, sizeof dp);
    len = 0;
    while (x) a[&#43;&#43; len] = x % 10, x /= 10;
    return dfs(len, 0, num, 1, 1);
}
signed main() {
    while (cin &gt;&gt; l &gt;&gt; r, l || r) {
        if (l &gt; r) swap(l, r);
        for (int i = 0; i &lt;= 9; i &#43;&#43;) cout &lt;&lt; cal(r, i) - cal(l - 1, i) &lt;&lt; &#34; &#34;;
        cout &lt;&lt; endl;
    }
}
```


## 以上只是模板题用来熟悉数位$\text{DP}$，当然做这些题还远远不够，需要更多练习。
## **题单：**

[CodeForce1036C Classy Numbers](https://codeforces.com/problemset/problem/1036/C)

找到区间$[L,R]$有不超过$3$个非$0$的数的个数

[洛谷P4127 同类分布](https://www.luogu.com.cn/problem/P4127)

找到区间$[L,R]$各位数字之和能整除原数的数的个数

[洛谷P4317 花神的数论题](https://www.luogu.com.cn/problem/P4317)

设 $\text{sum}(i)$ 表示 $i$ 的二进制表示中 $1$ 的个数。给出一个正整数 $N$ ，求 $\prod _{i=1}^{N}\text{sum}(i)$
​
**较难：**

[HDU 3693 Math teacher&#39;s homework ](https://vjudge.z180.cn/problem/HDU-3693)

[HDU 4352 XHXJ&#39;s LIS ](https://vjudge.z180.cn/problem/HDU-4352)

[CodeForce 55D Beautiful numbers](https://codeforces.com/problemset/problem/55/D)

[AcWing 1086 恨7不成妻](https://www.acwing.com/problem/content/1088/)

[POJ 3252 Round Numbers](https://vjudge.z180.cn/problem/POJ-3252)

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/%E6%95%B0%E4%BD%8Ddpdfs%E5%81%9A%E6%B3%95/  

