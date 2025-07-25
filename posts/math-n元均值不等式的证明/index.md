# N元均值不等式的证明


**求证：**
$$
\frac{n}{\sum\limits_{i = 1} ^ {n} \dfrac{1}{x_i}} \le  \sqrt [n] {\prod_{i = 1} ^ {n} x_i } \le \frac{\sum\limits_{i = 1} ^ {n} x_i}{n} \le \sqrt{\frac{\sum\limits_{i = 1} ^ {n} x_i ^ 2}{n}}
$$
**分析：**

引理：琴生不等式

若 $f(x)$ 是区间 $[a,b]$ 的凹函数，则对任意 $x_1,x_2,\cdots,x_n \in [a,b]$ 有不等式
$$
f\left(\frac{\sum\limits_{i = 1} ^ {n} x_i}{n}\right) \le \frac{\sum\limits_{i = 1} ^ {n} f(x_i)}{n}
$$
若 $f(x)$ 是区间 $[a,b]$ 的凸函数，则对任意 $x_1,x_2,\cdots,x_n \in [a,b]$ 有不等式
$$
\frac{\sum\limits_{i = 1} ^ {n} f(x_i)}{n}\le f\left(\frac{\sum\limits_{i = 1} ^ {n} x_i}{n}\right)
$$
设 $f(x) = \ln x$，易得 $\ln x$ 为凸函数，所以有

$$
 \frac{\sum\limits_{i = 1} ^ {n} \ln x_i}{n} \le \ln \frac{\sum\limits_{i = 1} ^ {n} x_i}{n} \\\\
\Leftrightarrow \frac{\ln \prod\limits_{i = 1} ^ {n} x_i}{n} \le \ln \frac{\sum\limits_{i = 1} ^ {n} x_i}{n} \\\\
\Leftrightarrow \ln \prod\limits_{i = 1} ^ {n} x_i \le \ln \left (\frac{\sum\limits_{i = 1} ^ {n} x_i}{n} \right) ^ n \\\\
\Leftrightarrow \prod\limits_{i = 1} ^ {n} x_i \le \left (\frac{\sum\limits_{i = 1} ^ {n} x_i}{n} \right) ^ n \\\\
\Leftrightarrow \sqrt[n]{\prod\limits_{i = 1} ^ {n} x_i} \le \frac{\sum\limits_{i = 1} ^ {n} x_i}{n}
$$

第二个不等式得证。

若对于第二个不等式做变换 $x_i \rightarrow \dfrac{1}{x_i}$，有

$$
 \sqrt[n]{\prod\limits_{i = 1} ^ {n} \frac{1}{x_i}} \le \frac{\sum\limits_{i = 1} ^ {n} \dfrac{1}{x_i}}{n} \\\\
\Leftrightarrow \frac{1}{\sqrt[n]{\prod\limits_{i = 1} ^ {n} x_i}} \le \frac{\sum\limits_{i = 1} ^ {n} \dfrac{1}{x_i}}{n} \\\\
\Leftrightarrow \frac{n}{\sum\limits_{i = 1} ^ {n} \dfrac{1}{x_i}} \le \sqrt[n]{\prod\limits_{i = 1} ^ {n} x_i}
$$

第一个不等式得证。

再设 $f(x) = x ^ 2$，易得 $f(x)$ 在 $x \in [0, \infty]$ 为凹函数，所以有
$$
 \left(\frac{\sum\limits_{i = 1} ^ {n} x_i}{n}\right) ^ 2 \le \frac{\sum\limits_{i = 1} ^ {n} x_i ^ 2}{n} \\\\
\Leftrightarrow \frac{\sum\limits_{i = 1} ^ {n} x_i}{n} \le \sqrt{\frac{\sum\limits_{i = 1} ^ {n} x_i ^ 2}{n}}
$$
第三个不等式得证。

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/math-n%E5%85%83%E5%9D%87%E5%80%BC%E4%B8%8D%E7%AD%89%E5%BC%8F%E7%9A%84%E8%AF%81%E6%98%8E/  

