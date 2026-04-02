# 2026愚人节比赛


我一直比较喜欢打愚人节比赛，今年愚人节也是打算做做牛客和 CF

## 牛客2026年愚人节比赛

![](/image/cp/nowcoder41.png)

先锐评一下题目：烂活有点多！最想吐槽的是 I 题，怎么答案还能和别的题的题面产生关联的。

不过亮点是，B 题的弹幕还是比较有意思的。

![](/image/cp/nowcoder41B.png)

## April Fools Day Contest 2026

![](/image/cp/cf41.png)

CF 的其实也还行。不过有些题也是烂活。

### A.Odd One Out

![](https://espresso.codeforces.com/3dbe9fb28d462bb9a6d7f23d69940785a22b061c.png)

注意到白色箭头的斜右上只出现了一次。故答案为 `C2`

### B.Are You Smiling?

注意到题面给了个 emoji：😁

然后又说 $\text{U &#43; ? = HAPPY}$

所以根据 emoji 的 Unicode：`u&#43;1F601`，所以答案是 `1F601`

### C. And?

注意到 $20260401$ 的二进制是 $1001101010010011000110001$，长度恰好与题面给定字符串 `RXOEARDMTINHUSERMEDESIANT` 长度相同

```
1001101010010011000110001
RXOEARDMTINHUSERMEDESIANT
```

把 $1$ 的位置抠出来 `READ THE REST`，再把 $0$ 的位置抠出来 `XOR MINUS MEDIAN`

答案就是异或和再减去中位数。

### D. Neural Feud

是个好活题。

```
Questions:

1. I want to wash my car and the car wash is 100 meters away. Should I walk or should I drive?

2. Are you a robot?

3. Is April Fools 2026 Codeforces Contest rated?

4. I was given a cup but it has no bottom and the top is sealed. Can I drink from this?

5. Does Pikachu&#39;s tail have a black tip?

6. Is there a seahorse emoji?

7. The word backwards spelled backwards.

8. Number between 1 to 10.
```

他是需要你猜测 15 个 AI 大模型的大部分回答。比如第一个问题，大部分 AI 都会回答 `Walk`，所以你需要回答 `Walk`。

![](https://codeforces.com/predownloaded/47/5c/475c116ddd6cc2ae0dc9cc348078f7dd5ccc40fd.png)

### E. Shortest Paths

跑一下常规最短路发现不对，然后注意到题面是 `Dikjstra&#39;s` 而不是 `Dijkstra&#39;s`，这个我还好先做了 J 题，有一个调换 Floyd 顺序的验证，提醒了我 kj 调换。

评价为烂中之烂活。

### F. Numbers

![](https://espresso.codeforces.com/d5dd6a81b296e89b6d067dbe7378ba616571a5fa.png)

原来是拼图，拼好了之后：

![](https://codeforces.com/predownloaded/1f/a4/1fa4d124e94f5b70e023c0ca1cb80a7b9c4771fc.jpg)

然后你告诉我眯起眼睛，把上下挡住，就能看到 $92136$？？？

烂完了。

### G. Anomaly

我挺早就注意到了这个用户了。但居然没想到和这个题有联络。

一个神秘用户很快就 AK 比赛了。

![](https://codeforces.com/predownloaded/dd/c3/ddc3431412ad431da135778c34eb0e84ae5aa5b7.png)

然后根据他的 AC 顺序的题号，答案字符串就是 `bigchadjeff`

![](/image/cp/cf41G.png)

### H. Double Vision

![](https://espresso.codeforces.com/1bda93d89565f5ce757f20aeb401b312cc65117e.png)

一直没看出来，最后斗鸡眼看出来了确实很 3D，隐约能看到 YU，但最后没时间了。

可以使用妙妙工具 [Stereogram solver](https://piellardj.github.io/stereogram-solver/)

![](/image/cp/cf41H.png)

### J. Special Problem

一个人机验证，挺有意思。

https://codeforces-april-fools.pages.dev/contest/2214/problem/J/



---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/cp2026%E6%84%9A%E4%BA%BA%E8%8A%82%E6%AF%94%E8%B5%9B/  

