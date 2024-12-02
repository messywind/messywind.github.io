# CF 题目选讲


## CF 题目选讲

## Gym 103736 D Tree Problem

题目链接 https://codeforces.com/gym/103736/problem/D

**题意**

给定一颗 $n$ 个点的树，有 $q$ 次询问，每次给定一个点 $x$，计算有多少长度至少为 $1$ 的简单路径经过点 $x$

**分析：**

经过点 $u$ 的路径总数为以 $u$ 为根的各子树大小两两乘积和再加上剩下 $n - 1$ 个点，也就是
$$
n - 1 &#43; \sum\limits_{i \in u} \sum\limits_{j \in u} S_i \times S_j[i &lt; j]
$$
其中 $S_i$ 为点 $i$ 子树大小。

将右边式子化简一下
$$
n - 1 &#43; \frac{\sum\limits_{i \in u} \sum\limits_{j \in u} S_i \times S_j - \sum\limits_{i \in u} S_i ^ 2}{2}
$$
其中 $\sum\limits_{i \in u} \sum\limits_{j \in u} S_i \times S_j =\sum\limits_{i \in u} S_i \sum\limits_{j \in u} S_j = \left (\sum\limits_{j \in u} S_j \right ) ^ 2 = (n - 1) ^ 2$

故答案为
$$
n - 1 &#43; \frac{(n - 1) ^ 2 - \sum\limits_{i \in u} S_i ^ 2}{2}
$$
其中 $\sum\limits_{i \in u} S_i ^ 2$ 可以用一遍 $\text{dfs}$ 预处理。

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n;
    cin &gt;&gt; n;
    vector&lt;vector&lt;int&gt;&gt; g(n &#43; 1);
    for (int i = 1; i &lt; n; i &#43;&#43;) {
        int u, v;
        cin &gt;&gt; u &gt;&gt; v;
        g[u].push_back(v), g[v].push_back(u);
    }
    vector&lt;int&gt; Size(n &#43; 1), dp(n &#43; 1);
    function&lt;void(int, int)&gt; dfs = [&amp;](int u, int fa) {
        Size[u] = 1;
        int sum = 0;
        for (auto v : g[u]) {
            if (v == fa) {
                continue;
            }
            dfs(v, u);
            Size[u] &#43;= Size[v];
            sum &#43;= Size[v] * Size[v];
        }
        sum &#43;= (n - Size[u]) * (n - Size[u]);
        dp[u] = ((n - 1) * (n - 1) - sum) / 2;
    };
    dfs(1, -1);
    int m;
    cin &gt;&gt; m;
    while (m --) {
        int u;
        cin &gt;&gt; u;
        cout &lt;&lt; dp[u] &#43; n - 1 &lt;&lt; &#34;\n&#34;;
    }
}
```

## CF1517 C Fillomino 2

题目链接 https://codeforces.com/contest/1517/problem/C

**题意**

给定长度为 $n \space (1 \le n \le 500)$ 的排列 $p$，要求构造一个三角形，满足以下条件：

1. 三角形共 $n$ 行，第 $i$ 行有 $i$ 个数。第 $i$ 行最后一个数是 $p_i$
2. 接下来构造 $n$ 个连通块。对于第 $x \space (1 \le x \le n)$ 个连通块，每个元素、连通块大小都必须等于 $x$
3. 三角形每个格子必须恰好填一个数。

**分析：**

每次能向左延伸就向左延伸，不能往左延伸就向下延伸。

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n;
    cin &gt;&gt; n;
    vector&lt;int&gt; p(n &#43; 1);
    vector&lt;vector&lt;int&gt;&gt; a(n &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        cin &gt;&gt; p[i];
        a[i].resize(i &#43; 1);
        a[i][i] = p[i];
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        int x = i, y = i;
        for (int j = 1; j &lt;= p[i] - 1; j &#43;&#43;) {
            if (y - 1 &gt;= 1 &amp;&amp; !a[x][y - 1]) {
                a[x][y - 1] = p[i];
                y --;
            } else if (x &#43; 1 &lt;= n &amp;&amp; !a[x &#43; 1][y]) {
                a[x &#43; 1][y] = p[i];
                x &#43;&#43;;
            }
        }
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        for (int j = 1; j &lt;= i; j &#43;&#43;) {
            cout &lt;&lt; a[i][j] &lt;&lt; &#34; \n&#34;[j == i];
        }
    }
}
```

## CF1649 D Integral Array

题目链接 https://codeforces.com/contest/1649/problem/D

**题意**

给定一个数组 $a$，该数组完整的定义 ：对数组 $a$ 中任意两数 $x, y$ $(y \le x)$ 满足 $\lfloor \dfrac{x}{y} \rfloor$ 也在数组中

数组中每个数 $a_i \le c$，判断数组 $a$ 是否完整。

$(1 \le n, c \le 10 ^ 6)$

**分析：**

首先朴素想法是对于每个数可以用整除分块判断是否在数组中，但时间复杂度 $O(n \sqrt n)$，会超时。

所以考虑 $O(n \log n)$ 的做法，可以用枚举倍数法。

枚举 $1 \le i \le c$，那么 $i$ 的倍数为 $j$，对于 $j \sim j &#43; i - 1$ 这一段数来说除 $i$ 下取整得到的结果都是 $\dfrac{j}{i}$，所以枚举 $i$ 就相当于枚举 $y$，$j \sim j &#43; i - 1$ 中的所有数都是 $x$，可以用前缀和快速判断区间里的数是否存在，如果区间存在一个数并且 $i$ 也在数组出现过，并且 $\dfrac{j}{i}$ 在数组中不存在，那么数组 $a$ 就是不完整的。

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
void solve() {
    int n, c;
    cin &gt;&gt; n &gt;&gt; c;
    vector&lt;int&gt; a(n &#43; 1), sum(c &#43; 1), mp(c &#43; 1);
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        cin &gt;&gt; a[i];
        sum[a[i]] &#43;&#43;;
        mp[a[i]] = 1;
    }
    for (int i = 1; i &lt;= c; i &#43;&#43;) {
        sum[i] &#43;= sum[i - 1];
    }
    string ans = &#34;Yes&#34;;
    for (int i = 1; i &lt;= c; i &#43;&#43;) {
        for (int j = i; j &lt;= c; j &#43;= i) {
            int l = j, r = min(c, j &#43; i - 1);
            if (sum[r] - sum[l - 1] &amp;&amp; mp[i] &amp;&amp; !mp[j / i]) {
                ans = &#34;No&#34;;
            }
        }
    }
    cout &lt;&lt; ans &lt;&lt; endl;
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

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cpcf-%E9%A2%98%E7%9B%AE%E9%80%89%E8%AE%B2/  

