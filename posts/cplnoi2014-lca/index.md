# [LNOI2014] LCA


[原题链接](https://www.luogu.com.cn/problem/P4211)

**题意**

给定一颗 $n$ 个节点并且根为 $1$ 的树和 $q$ 次询问，每次询问给定 $l,r,z$ 求 

$$\sum_{i=l}^{r} \text{dep}(\text{lca}(i,z))$$

对 $201314$ 取模

$\text{dep}(x)$ 表示点 $x$ 的深度，$\text{lca}(u,v)$ 表示 $u,v$ 的最近公共祖先

$1 \le n,q \le 5×10^4$

**分析：**

对式子进行一步转化，那就相当于在 $[l,r]$ 区间内的每个点，每次将该点和树根的路径上点权 $&#43;1$，最后查询 $z$ 到树根路径的点权之和。这样做显然是超时的，所以就考虑怎么优化式子，我们发现每次询问是在 $[1,n]$ 区间的，所以就可以想到前缀和，也就是 

$$\sum_{i=l}^{r} \text{dep}(\text{lca}(i,z))=\sum_{i=1}^{r} \text{dep}(\text{lca}(i,z)) - \sum_{i=1}^{l - 1} \text{dep}(\text{lca}(i,z))$$

这样就可以将一个询问分为两个询问，那只需要处理每个询问的 $r$ 和 $l-1$ 最后相减就可以了，所以我们可以考虑离线来做：

对于每个修改：维护一个差分，就相当于在区间 $[l-1, r]$ 都 $&#43;1$，表示要在区间里的每个点加一次对树根的贡献(也就是 $1$)。

对于每个查询：在 $r$ 和 $l - 1$ 挂上 $z$ 的询问，左端点打上 $0$ 的标记， 右端点打上 $1$ 的标记。

最后从 $1$ 到 $n$ 扫一遍，先判断差分，如果是大于$0$的，那么表示这个点要修改一次。在判断询问，如果遇到有标记的点就查询 $z$ 到 树根的路径和，设当前的询问编号为 $id$， $0$ 的话就记到 $L_{id}$ ，$1$ 的话就记到 $R_{id}$，每个询问的答案就是 $R_{id} - L_{id}$

## 代码：
```cpp
#include &lt;bits/stdc&#43;&#43;.h&gt;
#define int long long
using namespace std;
const int N = 5e4 &#43; 5, M = N &lt;&lt; 1, mod = 201314;
int L[N], R[N], sum[N], n, m, h[N], e[M], ne[M], idx, l, v, r, z, top[N], Size[N], fa[N], son[N], dep[N], cnt, id[N];
struct SegmentTree {
    int l, r, add, sum;
} tr[N &lt;&lt; 2];
struct node {
    int id, z, type;
};
vector&lt;node&gt; num[N];
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
    id[u] = &#43;&#43; cnt, top[u] = t;
    if (!son[u]) return ;
    dfs2(son[u], t);
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == fa[u] || j == son[u]) continue;
        dfs2(j, j);
    }
}
void pushup(int u) {
    tr[u].sum = (tr[u &lt;&lt; 1].sum &#43; tr[u &lt;&lt; 1 | 1].sum) % mod;
}
void pushdown(int u) {
    if (tr[u].add) {
        tr[u &lt;&lt; 1].add &#43;= tr[u].add;
        tr[u &lt;&lt; 1 | 1].add &#43;= tr[u].add;
        tr[u &lt;&lt; 1].sum &#43;= (tr[u &lt;&lt; 1].r - tr[u &lt;&lt; 1].l &#43; 1) * tr[u].add % mod;
        tr[u &lt;&lt; 1 | 1].sum &#43;= (tr[u &lt;&lt; 1 | 1].r - tr[u &lt;&lt; 1 | 1].l &#43; 1) * tr[u].add % mod;
        tr[u].add = 0;
    }
}
void build(int u, int l, int r) {
    if (l == r) {
        tr[u] = {l, r};
    } else {
        tr[u] = {l, r};
        int mid = l &#43; r &gt;&gt; 1;
        build(u &lt;&lt; 1, l, mid), build(u &lt;&lt; 1 | 1, mid &#43; 1, r);
        pushup(u);
    }
}
void modify(int u, int l, int r, int c) {
    if (tr[u].l &gt;= l &amp;&amp; tr[u].r &lt;= r) {
        tr[u].add &#43;= c;
        tr[u].sum &#43;= (tr[u].r - tr[u].l &#43; 1) * c % mod;
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
    if (l &lt;= mid) res = (res &#43; ask(u &lt;&lt; 1, l, r)) % mod;
    if (r &gt; mid) res = (res &#43; ask(u &lt;&lt; 1 | 1, l, r)) % mod;
    return res;
}
void modify_path(int u, int v, int k) {
    while (top[u] != top[v]) {
        if (dep[top[u]] &lt; dep[top[v]]) swap(u, v);
        modify(1, id[top[u]], id[u], k);
        u = fa[top[u]];
    }
    if (dep[u] &lt; dep[v]) swap(u, v);
    modify(1, id[v], id[u], k);
}
int ask_path(int u, int v) {
    int res = 0;
    while (top[u] != top[v]) {
        if (dep[top[u]] &lt; dep[top[v]]) swap(u, v);
        res = (res &#43; ask(1, id[top[u]], id[u])) % mod;
        u = fa[top[u]];
    }
    if (dep[u] &lt; dep[v]) swap(u, v);
    res = (res &#43; ask(1, id[v], id[u])) % mod;
    return res;
}
signed main() {
    memset(h, -1, sizeof h);
    cin &gt;&gt; n &gt;&gt; m;
    for (int u = 2; u &lt;= n; u &#43;&#43;) {
        cin &gt;&gt; v, v &#43;&#43;;
        add(u, v), add(v, u);
    }
    dfs1(1, -1, 1), dfs2(1, 1);
    build(1, 1, n);
    for (int i = 1; i &lt;= m; i &#43;&#43;) {
        cin &gt;&gt; l &gt;&gt; r &gt;&gt; z, l &#43;&#43;, r &#43;&#43;, z &#43;&#43;;
        sum[l] &#43;&#43;, sum[r &#43; 1] --;
        num[l - 1].push_back({i, z, 0});
        num[r].push_back({i, z, 1});
    }
    for (int i = 1; i &lt;= n; i &#43;&#43;) {
        sum[i] &#43;= sum[i - 1];
        if (sum[i]) modify_path(i, 1, 1);
        for (int j = 0; j &lt; num[i].size(); j &#43;&#43;) {
            if (num[i][j].type == 0) {
                L[num[i][j].id] = ask_path(num[i][j].z, 1);
            } else {
                R[num[i][j].id] = ask_path(num[i][j].z, 1);
            }
        }
    }
    for (int i = 1; i &lt;= m; i &#43;&#43;) {
        cout &lt;&lt; (R[i] - L[i] &#43; mod) % mod &lt;&lt; endl;
    }
}
```

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cplnoi2014-lca/  

