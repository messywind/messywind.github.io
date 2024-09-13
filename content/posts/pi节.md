---
title: "今天π节，出了一道题，高中生 大佬勿喷"
date: 2020-03-14 16:39:00
tags:
- 高考数学
categories:
- 杂项
code:
  maxShownLines: 11
---
![2](/image/pi.png)

分析：

$(1)$ $g\'(x)=\dfrac{π^{x}}{x^{π}}(\lnπ-\frac{π}{x})$

$∴g\'(1)=π(\lnπ-π)$

$(2)$ $f(x)=\int x \text{d} \frac{π^{x}}{\lnπ} = \dfrac{xπ^{x}}{\lnπ}-\int \dfrac{π^{x}}{\lnπ}\text{d}x = \dfrac{xπ^{x}}{\lnπ}-\dfrac{π^{x}}{\ln ^ 2π}$

$∴h(x)=\dfrac{x}{\lnπ}-\dfrac{1}{\ln^2π}-\dfrac{1}{x^{π}}$

$∴h(1)=\dfrac{1}{\lnπ}-\dfrac{1}{\ln^2π}-1$

令 $t=\dfrac{1}{\lnπ}\in(0,1)$，$h(1)=t-t^{2}-1\leq-\frac{3}{4}<0$

$h(π)+\dfrac{1}{π^π}=\dfrac{π}{\lnπ}-\dfrac{1}{\ln^2π}=\dfrac{π\lnπ-1}{\ln^2π}$

令 $u=\lnπ \in (1,2)$，$h(π)+\dfrac{1}{π^π}=\dfrac{ue^u-1}{u^2} = \varphi(u)$

$\varphi'(u)=\dfrac{(u^2-u)e^{u}+2}{u^{3}}$，令 $\phi(u)=(u^2-u)e^{u}+2, \phi'(u)=(u^2+u-1)e^u$

易证 $\phi'(u)>0\ (u\in(1,2))$，$∴\varphi'(u)>0$，$\varphi(u)$ 在 $u\in(1,2)$ 单调递增。

$\varphi(1)=e-1>0,\varphi(2)=\dfrac{2e^2-1}{4}>0,∴h(π)+\dfrac{1}{π^π}>0$

$∴h(1)·[h(π)+\dfrac{1}{π^π}]<0$ 得证。