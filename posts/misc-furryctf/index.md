# FurryCTF WP


# furryCTF WP

[比赛链接](https://hydro.ac/contest/6728fe9aa325b9e5ba5f9489)

## 前言

业余选手，这次比赛算是做的最爽的一次 CTF 比赛了，感谢 furryCTF，一开始在俊杰群里看到福瑞控居然能搞出个 CTF 比赛，比较好奇想去试试，早在大一的时候就想好好学习一下 CTF，但被钩八 ACM 占用了时间，现在工作了有摸鱼时间可以好好研究一下了！从一开始啥都不会只会做签到题和 osint 题到慢慢学习逐步会做后面的题，虽然可能大佬都是秒切的，但我确实学到了很多东西和技巧，以下是按照过题顺序的题解：

## A [beginner]签到题

网易云识曲，造梦西游 3

## B [misc]猫猫的故事

F12 发现零宽字符，解密工具：[在线零宽字符unicode文字隐写工具-BFW工具箱](https://tool.bfw.wiki/tool/1695021695027599.html)

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc04.png)

## D [misc]丢失的文档

改成 `.zip` 然后解压，记事本打开 WordDocument 找到 flag


![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc05.png)

## o [osint]人文风景

看了图片填了马家老式油茶，错误。但是注意审题是**对面**！所以高德搜位置然后看对面饭店，老刘传统蒸菜馆。

## p [osint]循迹


我也不知道怎么过的.jpg，观察到有一头牛，百度识图他叫墨小牛，然后他是即墨古城的吉祥物，然后搜一下即墨古城附近火锅店就出来了。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc06.png)

## w [osint]归去

一看群主 QQ 资料是阜阳的，所以锁定阜阳西站。

[阜阳西](https://qq.ip138.com/train/anhui/fuyangxi.htm)

观察到高铁时间是从早到晚的，所以去官网找到这俩站：


![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc07.png)

然后查找到：

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc08.png)

答案是上海站。

## u [osint]旅行照片

百度识图，日月双塔。

## K [misc]Windows 1.0



直接用 [010editor](https://hexed.it/) 打开一下搜索 furryCTF

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc09.png)


## l [hardware]Charge

直接用 AI 识图，加 AI 提问，`furryCTF{20000_50_PPJL65C}`

## Q [misc]求而不得


打开压缩包发现有密码，然后文件头 `90` 改 `00` 发现不行，不是伪加密。

发现 `1.txt 2.txt 3.txt 4.txt 5.txt` 原始大小很小，考虑 CRC32 碰撞。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc0a.png)

用几个工具，字节为 2 和 3 的很容易破解。

```

&#43;--------------遍历指定压缩包的CRC值----------------&#43;
[OK] 1.txt: 0xd6794fcc
[OK] 2.txt: 0xa4923bd1
[OK] 3.txt: 0x5150ce86
[OK] 4.txt: 0x269fce5d
[OK] 5.txt: 0x327e383
[OK] flag.txt: 0x77093881
&#43;-------------对输出的CRC值进行碰撞-----------------&#43;
[Success] 0xd6794fcc: 1qa
[Success] 0xa4923bd1: 4?u
[Success] 0x327e383: !w
```

得到以下三个文件的内容：

```
1.txt: 1qa
2.txt: 4?u
5.txt: !w
```



考虑 6 字节的，有工具：https://github.com/theonlypwner/crc32

所有可能如下：
```
!;F#5.
!j$BY2
$#UO.@
$nxrCH
%n9CXQ
)Al1&lt;V
)]#m=B
&#43;a&amp;sd_
&#43;}i/eK
,x!MNt
-5MA8e
-Eq0=5
-Y&gt;l&lt;!
/e;re&lt;
/yt.d(
07}LZt
0GA=_$
1&#43;s!@y
17&lt;}Am
1[OPE)
2*&amp;Bj%
2FUona
3*gsq&lt;
36(/p(
5_RQDJ
6.;CkF
7.zrp_
725.qK
8!$sWH
9!eBLQ
;PMaxD
&lt;%9rV&#43;
&lt;9v.W?
&lt;IJ_Ro
=%xCM2
=UD2Hb
?8#M}c
@*RYc8
@Zn(fh
BG5&amp;V9
B[zzW-
Bfk6?u
Bz$j&gt;a
C7HfHp
C[;KL4
Cze[%x
D.OXb[
E2A5xV
FC(&#39;WZ
F_g{VN
G_&amp;JMW
HPxKj@
ILv&amp;pM
IP9zqY
J!Ph^U
Jl}U3]
Kl&lt;d(D
Kps8)P
L$Y;ns
LTeJk#
MHk&#39;q.
MT${p:
N%Mi_6
NI&gt;D[r
Oh!e)&#39;
Otn9(3
PwJf{g
R&#39;&lt;U&amp;&gt;
S&#39;}d=&#39;
S;28&lt;3
SjPYP/
V#!T&#39;]
VrC5KA
W?/9=P
WnMXQL
YaRhmB
[]Wv4_
]eOil!
^)77&#43;u
^5xk*a
^xUVGi
_59Z1x
_YJw5&lt;
_d[;]d
atB0/8
b$u2ix
b8:nhl
bTICl(
c8{_su
cHG.v%
d=3=XJ
e!=PBG
ep_1.[
fPTBmK
gLZ/wF
h_KrQE
j.cQeP
l7%^Tb
l[VsP&amp;
m&#43;&#43;3No
m7doO{
nZB!ac
oFLL{n
p5Tb-j
pY&#39;O).
pxy_@b
qdw2Zo
qx8n[{
sy,&lt;j&gt;
t]:N(M
uA4#2@
va?PqP
w}1=k]
xro&lt;LJ
y#Ll;O
ynaQVG
}&#39;Qm:,
```

5 字节的，直接暴力跑 (跑了几十分钟)：

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc0b.png)


最后都拼成字典用 archpr 暴力破解。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc0c.png)

得到密码，解压即可。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc0d.png)


## M [misc]此时无声胜有声

直接用 Audacity 打开。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc0f.png)


看图发现是 `furryCTF{B1ack_Pi2no}`

## E [misc]黑暗

一开始没什么头绪，但是必须要考虑数字顺序 (受求而不得启发)。

`010editor` 发现 8 张图片的文件尾不一样，依次是：

```
ZnVy
cnlD
VEZ7
SGVs
bG9f
SUVO
RF9B
d0F9
```

拼起来是 `ZnVycnlDVEZ7SGVsbG9fSUVORF9Bd0F9`，Base64 解码。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc10.png)


## L [misc]春风得意

题面提示**趴在电脑上**，所以考虑键盘加密。发现前面按 8 个字符分组是：

```
egtrcvge
6i87hji6
3t54dft3
3t54dft3
5u76ghu5
VSFSFFXD
4Y65FGY4
EGTRCVGE
0]=-;&#39;]0
```

对应 `furryCTF{`

按照密文脑补明文 (疯狂看电脑键盘)。


![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc11.png)


得到答案 `furryCTF{keyboard_with_random_cout}`

## R [misc]图片的秘密

标题提示图片，考虑使用 SSTV，音频变成图片。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc13.png)



枚举凯撒加密，字母往前移 10 位 `furryCTF{WELAOME_TO_FURRYCTF!}`

## _ [mobile]登录

解包搜 furryCTF

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc14.png)


## \` [mobile]认证系统

继续解包搜 furryCTF

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc15.png)


## C [misc]安装包

解压安装包发现 flag

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc16.png)


![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc17.png)


凯撒解不出来，考虑维吉尼亚密码，密钥是 `furryCTF`


![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc18.png)


## d [rev]烦人的黑框框

发现程序图标是 Python 编写，使用 `pyinstxtractor.py` 反编译。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc19.png)



反编译 `trojan.pyc`

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc1a.png)

Base32 解密。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc1b.png)

## f [crypto]亡羊补牢

枚举一下栅栏，发现 9 不对劲。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc1c.png)


然后发现每五个字符进行分组，前两组按照

```
swap(s[1], s[2])
swap(s[0], s[2])
swap(s[2], s[3])
```

的交换规则可以得到 `furryCTF{A`，于是模拟一下得到答案。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc1d.png)

## H [misc]乱码

开始想统计一下字符集，没想到字符集有意外收获 (出现 `furryCTF` 和 `{}` 等)，然后词频统计一下。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc1e.png)


尝试按词频排序。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc1f.png)

补一个 r 然后得到答案：`furryCTF{How_Man1-times}`

## Z [web]剑走偏锋

发现 `furry` 和 `CTF` 参数可以有 `./`，又发现 `CTF` 参数可以有问号，于是模糊匹配。

![](https://md.sdutacm.cn/uploads/adc29f004b3328858ded8dc20.png)

正确 flag 为：`furryCTF{Hundred_Secrets_An6_4_Mere_Care1essness}`


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/misc-furryctf/  

