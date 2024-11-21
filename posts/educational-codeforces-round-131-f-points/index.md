# [Educational Codeforces Round 131 F] Points


[原题链接](https://codeforces.com/contest/1701/problem/F)

**题意**

给定两个正整数 $q, d$，定义三元组 $(i, j, k)$ 满足 $i &lt; j &lt; k, k - i \le d$，为**美丽三元组**，现在有一个空集和 $q$ 组询问，每次给定一个正整数 $x$，若 $x$ 不在集合，那么将 $x$ 加入集合，若 $x$ 在集合中，那么将 $x$ 从集合中删除，每次询问计算集合中**美丽三元组**的个数。

**分析：**

考虑每个数从集合加入或删除的贡献，对于一个数 $x$，从区间 $[x, x &#43; d]$ 中选出任意两个不同的数都可以组成美丽三元组(假设 $x$ 为三元组中的最小值)，记区间中在集合的数量为 $cnt$，那么方案数为 $\dbinom{cnt}{2}$，那么考虑区间 $[x - d, x - 1]$，对区间中的每个数 $i$，考虑 $x$ 加入后的影响，设区间 $[i, i &#43; d]$ 在集合中的个数为 $a_i$，那么美丽三元组的个数为 $\dbinom{a_i}{2}$，则 $x$ 加入后的美丽三元组数量为 $\dbinom{a_i &#43; 1}{2}$，设整个集合为 $S$，那么在区间 $[x - d, x - 1]$ 中新增的美丽三元组数量就为 $\sum\limits_{i = x - d} ^ {x - 1} \left (\dbinom{a_i &#43; 1}{2} - \dbinom{a_i}{2} \right ) [i \in S] = \sum\limits_{i = x - d} ^ {x - 1}a_i [i \in S]$，对于 $x$ 删除后的影响就为 $\sum\limits_{i = x - d} ^ {x - 1} \left (\dbinom{a_i}{2} - \dbinom{a_i - 1}{2} \right ) [i \in S] = \sum\limits_{i = x - d} ^ {x - 1} (a_i - 1) [i \in S]$

考虑使用线段树，我们重点要维护的是每个数 $x$ 在区间 $[x, x &#43; d]$ 中在集合里的个数，那么每次加入或删除操作就相当于对区间 $[x - d, x - 1]$ 进行区间 $&#43;1$ 或 $-1$ 操作，线段树中维护四个值：$\text{cnt}$ 代表区间里在集合中的数的个数，$\text{add}$ 代表区间加的懒标记，$\text{val}$ 代表每个数 $i$ 在区间 $[i, i &#43; d]$ 中在集合里的个数，$\text{sum}$ 代表**存在集合中的每个数** $i$ 在区间 $[i, i &#43; d]$ 中在集合里的个数。因为每个数是否存在于集合中由 $\text{cnt}$ 是否为 $1$ 来决定，相当于 $\text{val}$ 是全部的值，也就是说无论区间 $[x - d, x - 1]$ 的某个数存不存在于集合，我们都要维护，那么真正的答案是 $\text{sum}$，也就是那些存在于集合里的数的值，通过懒标记用 $\text{cnt} \times \text{val}$ 来下传，这样就巧妙地算出了一段区间存在于集合中的数对答案的贡献，至于区间 $[x, x &#43; d]$ 的贡献可以直接查询 $\text{val}_x$ 的单点值并给答案贡献 $\dbinom{\text{val}_x}{2}$

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
constexpr int N = 2e5;
struct SegmentTree {
    struct Info {
        int l, r, cnt, add, val, sum;
    };
    vector&lt;Info&gt; tr;
    SegmentTree(int n) : tr(n &lt;&lt; 2) {
        function&lt;void(int, int, int)&gt; build = [&amp;](int u, int l, int r) {
            if (l == r) {
                tr[u] = {l, r};
            } else {
                tr[u] = {l, r};
                int mid = l &#43; r &gt;&gt; 1;
                build(u &lt;&lt; 1, l, mid), build(u &lt;&lt; 1 | 1, mid &#43; 1, r);
                pushup(u);
            }
        };
        build(1, 1, n);
    }
    void pushdown(int u) {
        if (tr[u].add) {
            tr[u &lt;&lt; 1].add &#43;= tr[u].add, tr[u &lt;&lt; 1 | 1].add &#43;= tr[u].add;
            tr[u &lt;&lt; 1].val &#43;= (tr[u &lt;&lt; 1].r - tr[u &lt;&lt; 1].l &#43; 1) * tr[u].add;
            tr[u &lt;&lt; 1 | 1].val &#43;= (tr[u &lt;&lt; 1 | 1].r - tr[u &lt;&lt; 1 | 1].l &#43; 1) * tr[u].add;
            tr[u &lt;&lt; 1].sum &#43;= tr[u &lt;&lt; 1].cnt * tr[u].add;
            tr[u &lt;&lt; 1 | 1].sum &#43;= tr[u &lt;&lt; 1 | 1].cnt * tr[u].add;
            tr[u].add = 0;
        }
    }
    void pushup(int u) {
        tr[u].cnt = tr[u &lt;&lt; 1].cnt &#43; tr[u &lt;&lt; 1 | 1].cnt;
        tr[u].val = tr[u &lt;&lt; 1].val &#43; tr[u &lt;&lt; 1 | 1].val;
        tr[u].sum = tr[u &lt;&lt; 1].sum &#43; tr[u &lt;&lt; 1 | 1].sum;
    }
    void modifycnt(int u, int pos, int c) {
        if (!pos) return ;
        if (tr[u].l == tr[u].r) {
            tr[u].cnt &#43;= c;
            if (!tr[u].cnt) {
                tr[u].sum = 0;
            } else {
                tr[u].sum = tr[u].val;
            }
            return ;
        }
        pushdown(u);
        int mid = tr[u].l &#43; tr[u].r &gt;&gt; 1;
        if (pos &lt;= mid) {
            modifycnt(u &lt;&lt; 1, pos, c);
        } else {
            modifycnt(u &lt;&lt; 1 | 1, pos, c);
        }
        pushup(u);
    }
    void modifysum(int u, int l, int r, int c) {
        if (l &gt; r) return ;
        if (tr[u].l &gt;= l &amp;&amp; tr[u].r &lt;= r) {
            tr[u].val &#43;= (tr[u].r - tr[u].l &#43; 1) * c;
            tr[u].sum &#43;= tr[u].cnt * c;
            tr[u].add &#43;= c;
            return ;
        }
        pushdown(u);
        int mid = tr[u].l &#43; tr[u].r &gt;&gt; 1;
        if (l &lt;= mid) modifysum(u &lt;&lt; 1, l, r, c);
        if (r &gt; mid) modifysum(u &lt;&lt; 1 | 1, l, r, c);
        pushup(u);
    }
    int askval(int u, int pos) {
        if (!pos) return 0;
        if (tr[u].l == tr[u].r) return tr[u].val;
        pushdown(u);
        int mid = tr[u].l &#43; tr[u].r &gt;&gt; 1, res = 0;
        if (pos &lt;= mid) {
            return askval(u &lt;&lt; 1, pos);
        } else {
            return askval(u &lt;&lt; 1 | 1, pos);
        }
    }
    int asksum(int u, int l, int r) {
        if (l &gt; r) return 0;
        if (tr[u].l &gt;= l &amp;&amp; tr[u].r &lt;= r) return tr[u].sum;
        pushdown(u);
        int mid = tr[u].l &#43; tr[u].r &gt;&gt; 1, res = 0;
        if (l &lt;= mid) res &#43;= asksum(u &lt;&lt; 1, l, r);
        if (r &gt; mid) res &#43;= asksum(u &lt;&lt; 1 | 1, l, r);
        return res;
    }
};
signed main() {
    cin.tie(0) -&gt; sync_with_stdio(0);
    int n, d;
    cin &gt;&gt; n &gt;&gt; d;
    vector&lt;int&gt; st(N &#43; 1);
    SegmentTree tr(N &#43; 1);
    int ans = 0;
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        int x;
        cin &gt;&gt; x;
        int l = max(1ll, x - d), r = x - 1;
        if (!st[x]) {
            ans &#43;= tr.asksum(1, l, r);
            tr.modifysum(1, l, r, 1);
            int cnt = tr.askval(1, x);
            ans &#43;= cnt * (cnt - 1) / 2;
            tr.modifycnt(1, x, 1);
        } else if (st[x]) {
            tr.modifysum(1, l, r, -1);
            ans -= tr.asksum(1, l, r);
            int cnt = tr.askval(1, x);
            ans -= cnt * (cnt - 1) / 2;
            tr.modifycnt(1, x, -1);
        }
        st[x] ^= 1;
        cout &lt;&lt; ans &lt;&lt; &#34;\n&#34;;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/educational-codeforces-round-131-f-points/  

