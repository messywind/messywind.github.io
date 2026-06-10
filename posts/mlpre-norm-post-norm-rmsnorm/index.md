# Pre-Norm &amp; Post-Norm &amp; RMSNorm

## 模块一：核心数学公式与基础理论

标准化算法的核心目的是将隐藏层特征向量的数值范围约束在健康的区间内。假设单样本的隐藏层特征向量为 $x \in \mathbb{R}^d$，其中 $d$ 为特征维度。

### 1. 经典 LayerNorm (LN)

LayerNorm 通过计算整条向量的**均值**与**方差**，执行“平移”与“缩放”两个微观动作：


$$y = \frac{x - \mu}{\sigma &#43; \epsilon} \odot \gamma &#43; \beta$$

* **均值（平移项）**：$\mu = \frac{1}{d} \sum_{i=1}^d x_i$ （将数据中心平移到原点 $0$）
* **方差（缩放项）**：$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$
* **可学习参数**：$\gamma, \beta \in \mathbb{R}^d$（仿射变换参数），$\epsilon$ 为防止分母为 0 的极小常数。

### 2. RMSNorm (Root Mean Square Normalization)

RMSNorm 基于“均值平移对稳定训练贡献极小”的学术发现，大刀阔斧地砍掉了均值 $\mu$ 的计算，强行假设 $\mu = 0$：


$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x) &#43; \epsilon} \odot \gamma$$

* **均方根（RMS）**：$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}$

### 💡 核心数理桥梁：方差与 RMS 的物理等价性

在数理统计中，存在恒等式：$\text{Var}(X) = E[X^2] - (E[X])^2$。
在大模型前向传播空间中，随着网络稳定或受到对称性初始化影响，特征均值 $E[x] \approx 0$。代入恒等式：


$$\text{Var}(x) = E[x^2] - 0 = \text{RMS}^2(x) \implies \text{RMS}(x) = \sqrt{\text{Var}(x)}$$


**物理本质**：当均值为 0 时，向量的均方根 $\text{RMS}(x)$ 与其标准差（方差开根号）在数学上完全等价。

---

## 模块二：Post-Norm (后归一化) —— 消失与爆炸的深渊

### 1. 架构公式与前向流

后归一化将标准化算子挂在残差连接的**最外层**：


$$x_{l&#43;1} = \text{LN}(x_l &#43; \text{SubLayer}(x_l))$$

### 2. 反向梯度流机理（微积分推导）

假设最终的顶层损失函数为 $\mathcal{L}$，为了计算第 $l$ 层输入 $x_l$ 的梯度，使用链式法则（Chain Rule）拆解：


$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l&#43;1}} \cdot \frac{\partial x_{l&#43;1}}{\partial y_l} \cdot \frac{\partial y_l}{\partial x_l}$$


其中 $y_l = x_l &#43; \text{SubLayer}(x_l)$ 为残差相加的未归一化中间结果。

1. 由于 $\text{LN}(y_l) \approx \frac{y_l}{\sigma_l}$，根据微积分商法则，LayerNorm 对输入的偏导数其量级**反比于其标准差**：$\frac{\partial x_{l&#43;1}}{\partial y_l} \propto \frac{1}{\sigma_l}$。
2. 残差项导数：$\frac{\partial y_l}{\partial x_l} = I &#43; \frac{\partial \text{SubLayer}(x_l)}{\partial x_l}$。

若将梯度从最顶层（第 $L$ 层）一路连乘回传到最底层（第 1 层）：


$$\frac{\partial \mathcal{L}}{\partial x_1} \propto \left( \prod_{l=1}^{L-1} \frac{1}{\sigma_l} \right) \cdot \frac{\partial \mathcal{L}}{\partial x_L}$$

### 💥 致命痛点：高层爆炸，底层消失

* **前向方差重置**：由于 Post-Norm 每层最外层都套了 LN，前向传播时每一层的输出方差都被强行重置为 1（即 $\text{Var}(x_l)=1$）。
* **残差叠加导致中间方差膨胀**：虽然 $x_l$ 方差为 1，但经过 $\text{SubLayer}$ 叠加后，$y_l$ 的方差 $\sigma_l^2 &gt; 1$。
* **底层梯度消失**：因为 $\sigma_l &gt; 1$，导致连乘项中的分母 $\frac{1}{\sigma_l} &lt; 1$。当模型有 $L=100$ 层时，底层的梯度必须连续乘以 100 个小于 1 的小数（如 $0.8^{100} \to 0$）。底层参数被“层层剥削”，完全训不动。
* **高层梯度爆炸**：靠近顶层的网络由于距离 Loss 最近，缺少足够的 $\frac{1}{\sigma_l}$ 算子去稀释和中和，其特征矩阵及参数导数体量极度庞大，开局极易引发数值溢出（出现 `NaN`）。

---

## 模块三：Pre-Norm (前归一化) —— 现代大模型的钢铁骨架

### 1. 架构公式与前向流

前归一化将标准化算子移入残差分支的**内部（小路）**，而残差主干（大路）不加任何拦截：


$$x_{l&#43;1} = x_l &#43; \text{SubLayer}(\text{LN}(x_l))$$

若将该公式从第 0 层（Embedding）一路前向展开到第 $L$ 层，会呈现出清晰的宏观加法链：


$$x_L = x_0 &#43; \sum_{l=0}^{L-1} \text{SubLayer}_l(\text{LN}(x_l))$$

### 2. 反向梯度流机理（“梯度高铁”的诞生）

根据上述宏观展开式，依据导数的加法法则，顶层对底层输入 $x_0$ 直接求偏导，会将主干上的 $x_0$ 拆出项：


$$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \left( I &#43; \sum_{l=0}^{L-1} \frac{\partial \text{SubLayer}_l}{\partial x_0} \right)$$

* **单位矩阵 $I$（即常数 1）的本质**：它代表残差主干那条“毫无阻碍的直线通道”。
* **100% 梯度直达**：无论中间各层的分支 $\text{SubLayer}$ 导数缩水到多么小（哪怕全部衰减为 0），顶层的梯度 $\frac{\partial \mathcal{L}}{\partial x_L}$ 乘以单位矩阵 $I$，依然能够 **100% 完整无损地灌回最底层**，从数学结构上免疫了梯度消失。

### 📉 隐蔽代价：方差滚雪球与容量坍塌（混日子效应）

Pre-Norm 带来了绝对的训练稳定性，但天下没有免费的午餐，它付出了高层特征稀释的惨痛代价：

#### ① 前向方差滚雪球定律

在概率论中，若随机变量 $A$ 与 $B$ 独立不相关，则 $\text{Var}(A&#43;B) = \text{Var}(A) &#43; \text{Var}(B)$。
在前向主干 $x_{l&#43;1} = x_l &#43; f_l$ 中，假设分支吐出的新特征方差稳定为常数 $v$，且初始词嵌入方差 $\text{Var}(x_0) = 1$。由于主干没有标准化拦截，方差随层数线性累加：


$$\text{Var}(x_l) \approx 1 &#43; l \cdot v$$


越往高层走，主干数据范围越膨胀，在高层形成了一片方差规模极其宏大的“特征太平洋”。

#### ② 高层梯度被无情稀释

当数据准备进入高层子层内部时，必须先通过 LN，此时 LN 算出的标准差为 $\sigma_l = \sqrt{\text{Var}(x_l)} \approx \sqrt{l \cdot v}$。
将此项带入反向传播的分支链式展开中：


$$\frac{\partial \text{SubLayer}_l}{\partial x_0} = \frac{\partial \text{SubLayer}_l}{\partial \text{LN}(x_l)} \cdot \underset{\text{极小项 } \mathcal{O}(\frac{1}{\sqrt{l}})}{\frac{\partial \text{LN}(x_l)}{\partial x_l}} \cdot \frac{\partial x_l}{\partial x_0}$$


由于 $\frac{\partial \text{LN}(x_l)}{\partial x_l} \propto \frac{1}{\sigma_l} \approx \frac{1}{\sqrt{l}}$：

* 当网络堆叠到第 400 层时，该项导数被庞大的分母稀释为原来的 $\frac{1}{\sqrt{400}} = \frac{1}{20}$。
* **最终后果**：越往高层，子层分支拿到的参数更新梯度越微弱，高层 Attention/FFN 权重几乎更新不动；在前向传播中，新算出的轻量特征加进太平洋主干也如同“一滴水融入大海”。高层网络退化为等价映射（Identity Mapping），开始“混日子”，引发**模型容量坍塌**。

---

## 模块四：RMSNorm 升级 —— 工程与速度的极致榨干

现代大模型（如 LLaMA、DeepSeek）在保留 Pre-Norm 骨架的前提下，无一例外将 LayerNorm 升级为了 **RMSNorm**，彻底解决了大模型在大规模训练中的算力与硬件痛点。

### 1. 解决的痛点：打破显存带宽限制 (Memory-Bound)

* **LayerNorm 的硬件地狱**：计算经典 LN 需要两次遍历特征向量。第一次遍历所有元素计算均值 $\mu$，写入显存；硬件再次读取整条向量计算方差 $\sigma^2$。大模型隐藏层维度动辄上万（如 8192），这种**重复读写**造成了极大的 GPU 显存带宽浪费。
* **RMSNorm 的单次遍历**：因为去掉了均值 $\mu$，硬件内核（CUDA Kernel）可以在单次遍历特征时，一边读取数据，一边在寄存器里累加平方值，直接算懂 $\text{RMS}(x)$。消除了重复读写，整体计算速度直接**飙升 10% ~ 50%**。

### 2. 完美的数学继承：梯度控温能力不减

将均值归零的 RMS 引入反向传播，它对高层方差根号级膨胀的投影依然完好：


$$\text{RMS}(x_l) \approx \sqrt{\text{Var}(x_l)} \approx \sqrt{l \cdot v}$$


其偏导数依然雷打不动地保留了倒数缩放特性：


$$\frac{\partial \text{RMSNorm}(x_l)}{\partial x_l} \propto \frac{1}{\text{RMS}(x_l)} \approx \frac{1}{\sqrt{l}}$$


它完美继承了经典 LN 的“梯度控温”能力，既确保了高层混日子、主干不崩溃的绝对安全，又卸下了沉重的硬件带宽包袱。

---

## 模块五：三大标准化配置纵向对比总结表

| 维度 | Post-Norm (经典 Transformer) | Pre-Norm (GPT-3 时代) | RMSNorm &#43; Pre-Norm (现代大模型标准) |
| --- | --- | --- | --- |
| **算子定义** | $y = \frac{x-\mu}{\sigma}\cdot\gamma &#43; \beta$ | $y = \frac{x-\mu}{\sigma}\cdot\gamma &#43; \beta$ | $y = \frac{x}{\text{RMS}(x)}\cdot\gamma$ （无 $\mu$、无 $\beta$） |
| **前向前流** | $x_{l&#43;1} = \text{LN}(x_l &#43; f_l)$ | $x_{l&#43;1} = x_l &#43; f_l(\text{LN}(x_l))$ | $x_{l&#43;1} = x_l &#43; f_l(\text{RMSNorm}(x_l))$ |
| **主干方差** | 恒定约束为 1 | 线性滚雪球累加：$\text{Var} \approx l \cdot v$ | 线性滚雪球累加：$\text{Var} \approx l \cdot v$ |
| **反向导数** | 连乘机制：$\prod \frac{1}{\sigma_l}$ | 加法独立项机制：$I &#43; \sum \text{f&#39;}_l$ | 加法独立项机制：$I &#43; \sum \text{f&#39;}_l$ |
| **底层硬伤** | 极易梯度消失与顶层爆炸 | 高层表达能力被压制（容量坍塌） | 高层表达能力被压制（容量坍塌） |
| **硬件效率** | 慢（低效两次遍历） | 慢（低效两次遍历） | **极快（高效单次遍历，节省显存带宽）** |
| **工业地位** | 弃用（层数上双即崩溃） | 基石底座（适合百亿规模） | **绝对统治（现代 LLM 工业界默认死配）** |
| **工程补丁** | 必须配合严格的 Learning Rate Warmup | **必须在输出 Head 前加 Final Norm 收尾** | **必须在输出 Head 前加 Final Norm 收尾** |


---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mlpre-norm-post-norm-rmsnorm/  

