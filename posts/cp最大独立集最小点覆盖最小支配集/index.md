# 最大独立集、最小点覆盖、最小支配集


## 最大独立集
`选出最多的点,使得所有点都是不相邻的`

**状态表示：** $dp_{i,j}$ 表示以 $i$ 为根的树，如果 $j$ 为 $0$ ，表示不选这个点，如果 $j$ 为 $1$，表示选这个点

**属性：** $\text{Max}$

**状态计算：**

如果当前点 $i$ 不选，那么子节点 $j$ 可以被选或不被选：

$$dp_{i,0}=\sum_{k=1}^{n}\max (dp_{j_k,0},dp_{j_k,1})$$

如果当前点 $i$ 被选，那么子节点 $j$ 一定不能被选：

$$dp_{i,1}=\sum_{k=1}^{n}dp_{j_k,0}$$

[$\text{AcWing}$ $285$ 没有上司的舞会(带点权)](https://www.acwing.com/problem/content/description/287/)

## 最小点覆盖
`选出最少的点,覆盖所有的边`

**状态表示：** $dp_{i,j}$ 表示以$i$为根的树，如果 $j$ 为 $0$ ，表示不选这个点，如果 $j$ 为 $1$ ，表示选这个点

**属性：**$\text{Min}$

**状态计算：**

如果当前点 $i$ 不选，那么子节点 $j$ 一定被选：

$$dp_{i,0}=\sum_{k=1}^{n} dp_{j_k,1}$$

如果当前点 $i$ 被选，那么子节点 $j$ 可以被选或者不选：

$$dp_{i,1}=\sum_{k=1}^{n} \min (dp_{j_k,0},dp_{j_k,1})$$

[$\text{AcWing}$ $323$ 战略游戏(带点权)](https://www.acwing.com/problem/content/325/)

## 最小支配集

`选出最少的点,使得每个点要么被选、要么被它的相邻点支配`

**状态表示：** $dp_{i,j}$ 表示以$i$为根的树，如果 $j$ 为 $0$，表示在点 $i$不被支配，且将要被父节点支配，如果 $j$ 为 $1$，表示在点 $i$ 不被支配，且将要被子节点支配，如果 $j$ 为 $2$，表示在点 $i$ 支配

**属性：**$\text{Min}$

**状态计算：**

如果当前点 $i$ 要被父节点支配，那么可以选择子节点或者选择该节点：

$$dp_{i,0}=\sum_{k=1}^{n}\min(dp_{j_k,1},dp_{j_k,2})$$

如果当前的点 $i$ 要被子节点支配，那么就要枚举是哪个子节点 $j$ 被选的方案最小($u_k$ 代表子节点的第 $k$ 个子节点)：

$$dp_{i,1}= \min( dp_{i,1},dp_{j_k,2}&#43;dp_{i,0}-\sum_{k=1}^{n}\min (dp_{u_k,1},dp_{u_k,2}))$$

如果选当前的点 $i$，那么子节点 $j$ 被 $i$ 支配，或者选择子节点 $j$，或者子节点 $j$ 被子节点的子节点 $u$ 支配：

$$dp_{i,2}=\sum_{k=1}^{n}\min(dp_{j_k,0},dp_{j_k,1},dp_{j_k,2})$$


[$\text{SDUT}$ $4831$ 树的染色](https://acm.sdut.edu.cn/onlinejudge3/problems/4831)

[$\text{AcWing}$ $1077$ 皇宫看守(带点权)](https://www.acwing.com/problem/content/description/1079/)

**参考：**

[AcWing算法提高课](https://www.acwing.com/activity/content/16/)

[树上dp的一些总结](https://www.acwing.com/blog/content/3582/)

[没有上司的舞会题解(小呆呆)](https://www.acwing.com/solution/content/7920/)

[战略游戏题解(小呆呆)](https://www.acwing.com/solution/content/8294/)

[皇宫看守题解(小呆呆)](https://www.acwing.com/solution/content/22109/)

[$\text{SDUT}$ $4831$ 树的染色题解(lxw)](https://acm.sdut.edu.cn/sdutacm_files/%E5%B1%B1%E4%B8%9C%E7%90%86%E5%B7%A5%E5%A4%A7%E5%AD%A6%E7%AC%AC%E5%8D%81%E4%B8%89%E5%B1%8A%20ACM%20%E7%A8%8B%E5%BA%8F%E8%AE%BE%E8%AE%A1%E7%AB%9E%E8%B5%9B%20-%20%E8%A7%A3%E9%A2%98%E6%8A%A5%E5%91%8A.pdf)

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp%E6%9C%80%E5%A4%A7%E7%8B%AC%E7%AB%8B%E9%9B%86%E6%9C%80%E5%B0%8F%E7%82%B9%E8%A6%86%E7%9B%96%E6%9C%80%E5%B0%8F%E6%94%AF%E9%85%8D%E9%9B%86/  

