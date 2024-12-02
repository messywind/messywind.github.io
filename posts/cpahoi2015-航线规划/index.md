# [AHOI2015] 航线规划


[原题链接](https://www.luogu.com.cn/problem/P2542)

**题意**

给定一个 $n$ 个点 $m$ 条边的图，有不超过 $40000$ 次的操作，每次操作有三个参数 $op, u, v$

若 $op =0$ ，表示删除点 $u,v$ 之间的边

若 $op = 1$，表示询问 $u, v$ 之间有多少**关键边**

**关键边：** $u, v$ 联通，若删除该边 $u, v$ 不连通，则为关键边

**分析：**

最朴素的想法是每次删完边之后 $\text{tarjan}$ 缩点，再维护树上两点距离，但这样显然会超时。

所以考虑逆序离线处理。

我们发现如果是一棵树，那么所有的边都是关键边，$u, v$ 两点的关键边数量就等于 $u, v$ 树上两点距离。并且在树上任意两点加一条边都会形成一个环，环内所有的边都不可能成为关键边，所以此时只需要将这两点路径上的每一条边的权值都变为 $0$ ，这个操作可以用树链剖分线段树维护。

先在图中随意找一颗生成树，可以用并查集维护，把这些树边标记一下，那么其他未标记的边就是要成环的边，也就是要进行路径上赋值为 $0$ 的点对。

之后读入操作，把要删除的边以点对的形式用 $\text{STL map}$ 维护，这些点对相当于**覆盖**一开始要进行路径上赋值为 $0$ 的点对，最后逆序地处理询问，逆序输出结果即可。

## 代码：

```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
#define x first
#define y second
#define PII pair&lt;int, int&gt;
using namespace std;
const int N = 1e5 &#43; 5, M = N &lt;&lt; 1;
int n, m, u[N], v[N], op, cnt, h[N], e[M], ne[M], idx, Size[N], top[N], dep[N], fa[N], son[N], id[N], Cnt, p[N];
bool st[N];
map&lt;PII, int&gt; mp;
vector&lt;int&gt; ans;
struct Query {
    int op, u, v;
} q[N];
struct SegmentTree {
    int l, r, add, sum;
} tr[N &lt;&lt; 2];
int find(int x) {
    return p[x] == x ? p[x] : p[x] = find(p[x]);
}
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
void dfs2(int u, int t) {
    id[u] = &#43;&#43; Cnt, top[u] = t;
    if (!son[u]) return ;
    dfs2(son[u], t);
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa[u] || j == son[u]) continue;
        dfs2(j, j);
    }
}
void pushup(int u) {
    tr[u].sum = tr[u &lt;&lt; 1].sum &#43; tr[u &lt;&lt; 1 | 1].sum;
}
void pushdown(int u) {
    if (tr[u].add != -1) {
        tr[u &lt;&lt; 1].add = tr[u &lt;&lt; 1 | 1].add = tr[u].add;
        tr[u &lt;&lt; 1].sum = (tr[u &lt;&lt; 1].r - tr[u &lt;&lt; 1].l &#43; 1) * tr[u].add;
        tr[u &lt;&lt; 1 | 1].sum = (tr[u &lt;&lt; 1 | 1].r - tr[u &lt;&lt; 1 | 1].l &#43; 1) * tr[u].add;
        tr[u].add = -1;
    }
}
void build(int u, int l, int r) {
    if (l == r) {
        tr[u] = {l, r, -1, 1};
    } else {
        tr[u] = {l, r, -1};
        int mid = l &#43; r &gt;&gt; 1;
        build(u &lt;&lt; 1, l, mid), build(u &lt;&lt; 1 | 1, mid &#43; 1, r);
        pushup(u);
    }
}
void modify(int u, int l, int r, int c) {
    if (tr[u].l &gt;= l &amp;&amp; tr[u].r &lt;= r) {
        tr[u].add = c;
        tr[u].sum = (tr[u].r - tr[u].l &#43; 1) * c;
        return ;
    }
    pushdown(u);
    int mid = tr[u].l &#43; tr[u].r &gt;&gt; 1;
    if (l &lt;= mid) modify(u &lt;&lt; 1, l, r, c);
    if (r &gt; mid) modify(u &lt;&lt; 1 | 1, l, r, c);
    pushup(u);
}
int ask(int u, int l, int r) {
    if (tr[u].l &gt;= l &amp;&amp; tr[u].r &lt;= r) return tr[u].sum;
    pushdown(u);
    int mid = tr[u].l &#43; tr[u].r &gt;&gt; 1, res = 0;
    if (l &lt;= mid) res &#43;= ask(u &lt;&lt; 1, l, r);
    if (r &gt; mid) res &#43;= ask(u &lt;&lt; 1 | 1, l, r);
    return res;
}
void modify_path(int u, int v, int k) {
    while (top[u] != top[v]) {
        if (dep[top[u]] &lt; dep[top[v]]) swap(u, v);
        modify(1, id[top[u]], id[u], k);
        u = fa[top[u]];
    }
    if (dep[u] &lt; dep[v]) swap(u, v);
    modify(1, id[v] &#43; 1, id[u], k);
}
int ask_path(int u, int v) {
    int res = 0;
    while (top[u] != top[v]) {
        if (dep[top[u]] &lt; dep[top[v]]) swap(u, v);
        res &#43;= ask(1, id[top[u]], id[u]);
        u = fa[top[u]];
    }
    if (dep[u] &lt; dep[v]) swap(u, v);
    res &#43;= ask(1, id[v] &#43; 1, id[u]);
    return res;
}
signed main() {
    memset(h, -1, sizeof h);
    cin &gt;&gt; n &gt;&gt; m;
    for (int i = 1; i &lt;= n; i &#43;&#43;) p[i] = i;
    for (int i = 1; i &lt;= m; i &#43;&#43;) cin &gt;&gt; u[i] &gt;&gt; v[i];
    for (int i = 1; i &lt;= m; i &#43;&#43;) {
        int pu = find(u[i]), pv = find(v[i]);
        if (pu != pv) {
            p[pu] = pv;
            add(u[i], v[i]), add(v[i], u[i]);
            st[i] = 1;
        }
    }
    dfs1(1, -1, 1), dfs2(1, 1);
    build(1, 1, n);
    while (1) {
        cin &gt;&gt; op;
        if (op == -1) break;
        cnt &#43;&#43;;
        q[cnt].op = op;
        cin &gt;&gt; q[cnt].u &gt;&gt; q[cnt].v;
    }
    for (int i = 1; i &lt;= cnt; i &#43;&#43;) {
        if (!q[i].op) {
            mp[{q[i].u, q[i].v}] = mp[{q[i].v, q[i].u}] = 1;
        }
    }
    for (int i = 1; i &lt;= m; i &#43;&#43;) {
        if (!st[i] &amp;&amp; !mp[{u[i], v[i]}]) {
            modify_path(u[i], v[i], 0);
        }
    }
    for (int i = cnt; i; i --) {
        if (!q[i].op) {
            modify_path(q[i].u, q[i].v, 0);
        } else if (q[i].op == 1) {
            ans.push_back(ask_path(q[i].u, q[i].v));
        }
    }
    for (int i = ans.size() - 1; ~i; i --) cout &lt;&lt; ans[i] &lt;&lt; endl;
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cpahoi2015-%E8%88%AA%E7%BA%BF%E8%A7%84%E5%88%92/  

