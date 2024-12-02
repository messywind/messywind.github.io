# [算法竞赛进阶指南] 雨天的尾巴


[原题链接](https://www.luogu.com.cn/problem/P4556)

**题意**

给定一棵 $n$ 个节点的树和 $m$ 次操作，每次操作把 $u$ 到 $v$ 路径上的节点加上一个颜色 $z$，最后询问每个点最多颜色的编号(如果相同取编号最小)

$1 \le n,m,z \le 10^5$

**分析：**
此题是线段树合并模板题，这里给出树链剖分的做法。
每次操作修改树上的路径，可以用树链剖分维护一下，注意到 $z$ 的范围是 $10^5$ ，所以我们不能在树上的每个节点上开一个桶记录颜色，所以可以用权值线段树的动态开点。不过这里有更优做法，因为树链剖分出来的序列对应树上的唯一路径，所以题目的操作就相当于：给定一个序列，每次在 $[l,r]$ 区间添加一个颜色，询问每个点最多颜色的编号。这样就可以用差分的思想，每次在 $l$ 点 $&#43;1$，$r &#43; 1$ 点 $-1$，我们把 $l$ 排序，扫描 $1 \sim N$ 的每个点，每次遍历这个点的询问，把对这个点的修改在权值线段树上操作，然后查询一下最大的下标。
此题在 $\text{acwing}$ 上 $z$ 的数据范围为 $10^9$ 所以最好离散化一下。

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
#define find(x) lower_bound(num.begin(), num.end(), x) - num.begin()
using namespace std;
const int N = 1e5 &#43; 5, M = N &lt;&lt; 1;
int z[N], a[N], u[N], v[N], n, m, h[N], e[M], ne[M], idx, id[N], ans[N], mp[N], cnt, dep[N], Size[N], top[N], fa[N], son[N];
vector&lt;int&gt; x[N], num;
struct SegmentTree {
    int l, r, mx, val;
} tr[N &lt;&lt; 2];
void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx &#43;&#43;;
}
void dfs1(int u, int father, int depth) {
    dep[u] = depth, fa[u] = father, Size[u] = 1;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == father) continue;
        dfs1(j, u, depth &#43; 1);
        Size[u] &#43;= Size[j];
        if (Size[son[u]] &lt; Size[j]) son[u] = j;
    }
}
void dfs2(int u,int t) {
    id[u] = &#43;&#43; cnt, top[u] = t, mp[cnt] = u;
    if (!son[u]) return ;
    dfs2(son[u], t);
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa[u] || j == son[u]) continue;
        dfs2(j, j);
    }
}
void pushup(int u) {
    if (tr[u &lt;&lt; 1].mx &gt;= tr[u &lt;&lt; 1 | 1].mx) {
        tr[u].mx = tr[u &lt;&lt; 1].mx;
        tr[u].val = tr[u &lt;&lt; 1].val;
    } else {
        tr[u].mx = tr[u &lt;&lt; 1 | 1].mx;
        tr[u].val = tr[u &lt;&lt; 1 | 1].val;
    }
}
void build(int u, int l, int r) {
    if (l == r) {
        tr[u] = {l, r, 0, l};
    } else {
        tr[u] = {l, r};
        int mid = l &#43; r &gt;&gt; 1;
        build(u &lt;&lt; 1, l, mid), build(u &lt;&lt; 1 | 1, mid &#43; 1, r);
        pushup(u);
    }
}
void modify(int u, int pos, int c) {
    if (tr[u].l == pos &amp;&amp; tr[u].r == pos) {
        tr[u].mx &#43;= c;
    } else {
        int mid = tr[u].l &#43; tr[u].r &gt;&gt; 1;
        if (pos &lt;= mid) {
            modify(u &lt;&lt; 1, pos, c);
        } else {
            modify(u &lt;&lt; 1 | 1, pos, c);
        }
        pushup(u);
    }
}
void modify_path(int u, int v, int k) {
    while (top[u] != top[v]) {
        if (dep[top[u]] &lt; dep[top[v]]) swap(u, v);
        x[id[top[u]]].push_back(k), x[id[u] &#43; 1].push_back(-k);
        u = fa[top[u]];
    }
    if (dep[u] &lt; dep[v]) swap(u, v);
    x[id[v]].push_back(k), x[id[u] &#43; 1].push_back(-k);
}
signed main() {
    memset(h, -1, sizeof h);
    cin &gt;&gt; n &gt;&gt; m;
    for (int i = 0; i &lt; n - 1; i &#43;&#43;) {
        cin &gt;&gt; u[i] &gt;&gt; v[i];
        add(u[i], v[i]), add(v[i], u[i]);
    }
    dfs1(1, -1, 1), dfs2(1, 1);
    build(1, 1, N - 1);
    for (int i = 1; i &lt;= m; i &#43;&#43;) {
        cin &gt;&gt; u[i] &gt;&gt; v[i] &gt;&gt; z[i];
        num.push_back(z[i]);
    }
    sort(num.begin(), num.end());
    num.erase(unique(num.begin(), num.end()), num.end());
    for (int i = 1; i &lt;= m; i &#43;&#43;) {
        modify_path(u[i], v[i], find(z[i]) &#43; 1);
    }
    for (int i = 1; i &lt; N; i &#43;&#43;) {
        for (int j = 0; j &lt; x[i].size(); j &#43;&#43;) {
            if (x[i][j] &gt; 0) {
                modify(1, x[i][j], 1);
            } else {
                modify(1, -x[i][j], -1);
            }
        }
        ans[mp[i]] = tr[1].mx ? num[tr[1].val - 1] : 0;
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) cout &lt;&lt; ans[i] &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E7%AE%97%E6%B3%95%E7%AB%9E%E8%B5%9B%E8%BF%9B%E9%98%B6%E6%8C%87%E5%8D%97-%E9%9B%A8%E5%A4%A9%E7%9A%84%E5%B0%BE%E5%B7%B4/  

