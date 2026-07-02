# LoRA


# 大模型有监督微调 SFT-LoRA 笔记

&gt; 面试导向 · 参数高效微调（PEFT）
&gt; 结合结构图 &#43; 概念追问整理，覆盖：结构/公式、为什么用加法、SFT 训练循环与损失、Self-Instruct 是否改 $W$、LoRA 与全参微调的关系。

---

## 0. 一句话本质

LoRA 不去改原来那个巨大的权重 $W$，而是在旁边并联一条「小旁路」，用两个小矩阵 $B$、$A$ 的乘积去拟合微调带来的**增量** $\Delta W$。原权重冻结，只训练这两个小矩阵。

---

## 1. 为什么需要 LoRA（三个动机）

以 Qwen3-8b（80 亿参数）为例，全参数微调要更新全部 80 亿参数，且优化器还要为每个参数额外存动量，显存动辄几十上百 G。LoRA 解决三类痛点：

1. **显存小** —— 只训练百万级参数，消费级显卡能跑。
2. **防过拟合** —— 可训练参数大幅减少，等于加了强约束，小数据上不易过拟合。
3. **样本足够少** —— 下游数据不多时，不需要也不应该动整个大模型。

---

## 2. 核心思想：低秩性（low-rank）

记微调前权重 $W$（图里写 $W_{old}$），微调后是 $W &#43; \Delta W$。LoRA 的关键假设：

&gt; **$\Delta W$ 是低秩的（low-rank）。**

直觉：适应一个下游任务，真正「有用的变化方向」很少，大部分维度是冗余的。这与 **PCA / SVD** 的思想一脉相承——一个看起来很大的矩阵，信息集中在少数主方向上，可用低秩近似。

既然 $\Delta W$（满秩时是 $d \times d$）信息量低，就用秩为 $k$ 的乘积表示：

$$\Delta W_q = B A, \qquad B \in \mathbb{R}^{d \times k}, \quad A \in \mathbb{R}^{k \times d}, \quad k \ll d$$

&gt; 注意：公式里的 $r$ 和图里的 $k$ 是同一个东西 —— **秩 rank**。

---

## 3. 公式逐步拆解

$$
\begin{aligned}
Q = Q_1 &#43; Q_2
&amp;= W_q X &#43; \frac{\alpha}{r}\,\Delta W_q\, X \\
&amp;= W_q X &#43; \frac{\alpha}{r}\,B A\, X \\
&amp;= \left(W_q &#43; \frac{\alpha}{r}\,B A\right) X \\
&amp;= W_q^{\text{Merged}}\, X
\end{aligned}
$$

逐项看：

- **$Q_1 = W_q X$**：主干路径，用**冻结**的原始权重（基座参数 $W_q$）。
- **$Q_2 = \dfrac{\alpha}{r} B A X$**：LoRA 旁路，是**唯一**参与梯度更新的部分。
- 两条路相加（图中 $\oplus$）得到最终的 $Q$。
- 训练完可合并为 $W_q^{\text{Merged}} = W_q &#43; \dfrac{\alpha}{r} B A$，推理时就是普通矩阵，**零额外延迟**。

---

## 4. 结构与初始化（面试高频）

| 矩阵 | 形状 | 初始化 |
|---|---|---|
| $A$ | $k \times d$ | **高斯随机初始化** |
| $B$ | $d \times k$ | **全 0 初始化** |

**为什么一个随机、一个全 0？** 关键在训练**起点**：

- $B = 0 \;\Rightarrow\; \Delta W = B A = 0 \;\Rightarrow\; Q_2 = 0 \;\Rightarrow\;$ 整个模型**完全等价于原始预训练模型**。微调从「原模型」平滑出发，不破坏预训练知识，训练稳。
- $A$ 用高斯而非也置 0：若 $A$、$B$ 同时为 0，梯度无法打破对称，$B$ 永远学不动。$A$ 给随机值是为了**让梯度能流起来**。

&gt; 核心记忆点：**基座 $W_q$ 冻结，只训练 $A$、$B$。** 这是 LoRA 省的根本原因，面试别说反。

---

## 5. 到底省多少？（算一遍）

假设某个权重：

- $\Delta W \in \mathbb{R}^{1万 \times 2万}$ → 参数量 $= 2 \text{ 亿}$
- 取秩 $k = 8$：
  - $A \in \mathbb{R}^{8 \times 2万} = 8 \times 20000 =$ **16 万**
  - $B \in \mathbb{R}^{1万 \times 8} = 10000 \times 8 =$ **8 万**
  - 合计 $=$ **24 万**
- 压缩比：$\dfrac{2\text{亿}}{24\text{万}} \approx 833 \approx 830 \text{ 倍}$

本来要训 2 亿个参数，现在只训 24 万，**少约 830 倍**。这就是「参数高效」中「高效」的来源。

---

## 6. $\alpha/r$ 缩放因子

$\dfrac{\alpha}{r}$ 是对 LoRA 旁路输出的标量缩放：

- $r$ 是秩，$\alpha$ 是超参。比值 $\dfrac{\alpha}{r}$ 控制旁路 $BA$ 对输出的**影响强度**。
- 工程上常把 $\alpha$ 设为 $r$ 的固定倍数（如 $r=8,\ \alpha=16$），好处：**调整 $r$ 时不用重新大幅调学习率**，因为 $\dfrac{\alpha}{r}$ 大致稳定，LoRA 贡献的尺度不随 $r$ 剧烈漂移。

---

## 7. 推理合并：零延迟

训练阶段是「主干 &#43; 旁路」两条路，有额外计算。推理时提前算好：

$$W_q^{\text{Merged}} = W_q &#43; \frac{\alpha}{r} B A$$

直接当普通权重用，**推理速度与原模型完全一样**，无额外开销。这是 LoRA 相比 Adapter 类方法的一大优势（Adapter 串在网络里，推理甩不掉）。

&gt; 因为能合并，可为不同任务训不同 $BA$，推理时按需「挂载 / 卸载」——一个底座配多套 LoRA。

---

## 8. 易混淆点：为什么微调是 $W &#43; \Delta W$？加法凭什么成立？

**结论：加法不是设计选择，也不是「把两份知识倒进一个桶」，而是数学恒等式 &#43; 线性层的分配律。**

### 8.1 $\Delta W$ 是被「定义」出来的，不是「加」上去的

梯度下降训练 $T$ 步后：

$$W_{final} = W_{init} - \eta \sum_{t=0}^{T-1} \nabla L(W_t)$$

把「训练带来的总改变量」记作：

$$\Delta W \triangleq W_{final} - W_{init}$$

所以 $W_{final} = W_{init} &#43; \Delta W$ **恒等成立**，不含任何假设。任何一次微调（含全参微调）都满足。$\Delta W$ 不是独立的「知识矩阵」，只是权重在参数空间里**移动的位移量**。

&gt; 不是「先有 $W$，再造 $\Delta W$ 加上去」，而是「$W$ 动了一下，把动的那部分起名叫 $\Delta W$」。全参微调与 LoRA 一样，区别仅在：全参让 $\Delta W$ 自由（满秩），LoRA 规定 $\Delta W = BA$（低秩）。

### 8.2 输出能拆成两项相加：靠分配律

$$(W &#43; \Delta W) X = W X &#43; \Delta W X$$

因为单层做的是**线性变换**（$y = Wx$），天然可拆。所以 $W X =$ 原始响应，$\Delta W X =$ 微调修正量。**这里加的是「对输出的修正向量」，不是「知识」。**

### 8.3 知识真的能直接相加吗？分两个尺度

- **单个权重矩阵这一层**：严格线性可加（分配律），数学上精确，无近似。
- **整个网络这一层**：**不是简单相加，而是高度非线性的。** LLM 是「线性层 &#43; 非线性激活（SwiGLU、Softmax）」层层堆叠，每层给 $W$ 加的微小位移 $\Delta W$，会被逐层非线性**放大、耦合**，最终行为改变绝非各层 $\Delta W$ 的简单求和。

&gt; 正确心智模型：不是「相加两份知识」，而是**在每个权重上加一个很小的线性位移，再让模型自身的非线性机器把这些位移翻译成行为/知识变化。** 知识在权重里的存储本就是分布式 &#43; 非线性的，没有「一个矩阵 = 一份知识」的对应。

### 8.4 补充：Task Arithmetic（任务向量）

学术界有一条线：把「微调后参数 $-$ 原始参数」当成任务向量 $\tau$，做 $\theta &#43; \tau_1 &#43; \tau_2$ 获得多任务能力，甚至 $\theta - \tau$ 遗忘某能力，实验上居然能 work。但这是**经验性涌现现象**，不是 LoRA 用加法的原因；只在「同底座、参数空间对齐」前提下大致成立，且有干扰。它侧面说明「微调位移方向」一定程度可组合——这也是「一底座多 LoRA」「LoRA 合并」的理论基础。

---

## 9. SFT 训练循环：微调什么？输入是什么？损失跟谁比？

**关键澄清：SFT 没有新的损失函数，也没有「老师模型」对比。目标与预训练一致——预测下一个 token。**

### 9.1 「线性位移」= 从预训练 checkpoint 继续做 $T$ 步梯度下降

- 全参微调：直接对 $W$ 做梯度下降，$\Delta W = W_{final} - W_{init}$。
- LoRA：$W$ 冻结，梯度下降只更新 $A$、$B$：

$$A \leftarrow A - \eta\,\frac{\partial L}{\partial A}, \qquad B \leftarrow B - \eta\,\frac{\partial L}{\partial B}$$

$T$ 步后 $A$、$B$ 累积位移，于是 $\Delta W = \dfrac{\alpha}{r} B A \neq 0$。LoRA 的约束是：**强迫位移只能活在低秩子空间 $BA$ 里。**

### 9.2 微调「什么」

架构、目标函数、forward 全不变，唯一变的是这个条件概率分布：

$$P_\theta(\text{下一个 token} \mid \text{上文})$$

预训练让它「会接龙、懂世界知识」；SFT 把它推成「会按 chat 模板对话、服从指令、风格贴合数据」。

### 9.3 输入：套了 chat template 的指令对（teacher forcing）

$(\text{prompt}, \text{response})$ 套进 chat template 拼成一整条序列，一次 forward 走完，causal mask 保证每位置只看左边。**teacher forcing**：输入永远是数据里的真实 token。

```
&lt;|im_start|&gt;user
把这句话翻译成英文：今天天气很好&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
The weather is nice today.&lt;|im_end|&gt;
```

### 9.4 损失跟谁对比？—— 数据自己的真实下一个 token

$$L = -\sum_{i} \log P_\theta\big(\text{token}_{i&#43;1} \mid \text{token}_{\le i}\big)$$

**模型预测的分布 $\leftrightarrow$ 数据里真实写着的那个 token**，算交叉熵。与预训练损失完全一致，无新东西。**不是跟另一个模型比，也不是跟原始模型比。**

### 9.5 Loss Masking：只在「答案」部分算 loss

整条序列都喂进去，但损失**只在 response（assistant 答案）段算**，prompt 段 label 全设 $-100$（交叉熵忽略）：

```
位置:  [user 指令 tokens .......][assistant 答案 tokens .....]
label:  -100 -100 ... -100         t1  t2  t3 ...  tn
        ↑ 不算 loss                ↑ 只有这部分跟模型预测对比
```

原因：要教模型「给定问题该怎么答」，而非复述用户问题。

### 9.6 一个 SFT step 全流程

1. **取数据**：一条 $(\text{prompt}, \text{response})$ → 套 chat template → 一整条 token 序列。
2. **Forward**（teacher forcing）：整条喂入，每位置吐 next-token 分布。
3. **算 loss**：答案区每位置预测 vs 真实下一个 token，交叉熵；问题区 $-100$ 跳过。
4. **Backward**：梯度只流到 $A$、$B$（LoRA），$W$ 冻结。
5. **Update**：AdamW 更新 $A$、$B$。
6. **重复 $T$ 步** → $A$、$B$ 累积位移 → $\Delta W = \dfrac{\alpha}{r} B A$ 长出来。

### 9.7 边界：SFT ≠ 蒸馏

| | 损失对比对象 | 有老师模型吗 |
|---|---|---|
| **SFT** | 数据里硬写死的真实 token（one-hot 交叉熵） | ❌ 无 |
| **知识蒸馏** | 另一个「老师模型」输出的软分布 logits（KL 散度） | ✅ 有 |

看到「对比 logits / KL 散度 / teacher model」是蒸馏，不是 SFT，别串。

---

## 10. Self-Instruct 会改 $W$ 吗？

**不会。Self-Instruct 只是「造数据」的方法；真正改 $W$ 的是拿这批数据去跑 SFT 的那一步。**

### 10.1 Self-Instruct 是数据生产线，不是训练算法

流程：种子指令 → LLM 自动生成新指令 → LLM 生成对应答案 → 过滤去重 → 产出大规模 $(\text{instruction}, \text{response})$ 数据集。全程是**推理（inference / forward）**，**无反向传播、无梯度、无参数更新**，$W$ 一点没动。

### 10.2 $W$ 在哪一步被改？—— SFT 训练那一步

```
Self-Instruct（造数据，纯推理，不改 W）
        │ 产出 (instruction, response) 数据集
        ▼
   SFT 训练（吃这批数据，backward，改 W / 改 A,B）
```

Self-Instruct 对 $W$ 的影响是**间接的**（决定数据长什么样），程序本身从不碰梯度。

### 10.3 生成数据的模型 ≠ 被训练的模型

- **生成器**（造数据的强模型）：只推理，$W$ 不变。
- **被训练的目标模型**：$W$ 在 SFT 阶段被改。
- 即使同一个模型既当生成器又当被训对象，也是**先「生成数据」一遍不更新参数，后「训练」一遍才更新**，两阶段时间上分离。

### 10.4 「哪一步才改 $W$」对照表

| 概念 | 是什么 | 改 $W$ 吗 |
|---|---|---|
| **Self-Instruct** | 自动**造**指令数据 | ❌ 不改（纯推理） |
| **Chat Template** | 把数据拼成对话格式 | ❌ 不改（数据格式化） |
| **Loss Masking** | 只在答案区算 loss | —（规则，影响梯度怎么算） |
| **SFT 训练** | 吃数据跑梯度下降 | ✅ 改 $W$（LoRA 时改 $A$、$B$） |

&gt; 记法：带「instruct / template / 数据」字样的基本都在**数据侧**，不改 $W$；真正改 $W$ 的永远是那个有 **backward** 的训练 step。

---

## 11. 手撕版 LoRA Linear（面试可能让你写）

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16):
        super().__init__()
        # 基座权重：冻结
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W.weight.requires_grad = False

        self.r = r
        self.scaling = alpha / r                          # α/r 缩放

        # A: 高斯初始化;B: 全 0 初始化
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)  # [k, d]
        self.B = nn.Parameter(torch.zeros(out_dim, r))        # [d, k]

    def forward(self, x):
        # Q1 主干 &#43; Q2 旁路
        base = self.W(x)                                       # W_q · X
        lora = (x @ self.A.T @ self.B.T) * self.scaling        # (α/r)·BA·X
        return base &#43; lora

    @torch.no_grad()
    def merge(self):
        # 推理前合并:W_merged = W &#43; (α/r)·BA
        self.W.weight &#43;= self.scaling * (self.B @ self.A)
```

---

## 12. 面试速记（总）

- **本质**：冻结 $W$，用低秩 $BA$ 近似增量 $\Delta W$。
- **假设**：$\Delta W$ 低秩（任务适应的有效方向很少），思想同 PCA/SVD。
- **初始化**：$A$ 高斯、$B$ 全 0 $\Rightarrow$ 起点等于原模型，训练稳。
- **缩放**：$\dfrac{\alpha}{r}$，解耦秩与学习率。
- **省多少**：$\Delta W$ 2 亿 → $A&#43;B$ 24 万 $\approx 830$ 倍。
- **推理**：可合并 $W^{\text{Merged}}$，零额外延迟，一底座多 LoRA。
- **为什么加法**：$\Delta W \triangleq W_{new} - W_{old}$ 是恒等定义 &#43; 线性层分配律；单层可加，整网非线性；知识不是直接相加。
- **SFT 损失**：next-token 交叉熵，对比**数据自己的真实下一个 token**，经 loss masking 只算答案区；无老师模型（有老师的是蒸馏）。
- **Self-Instruct**：只造数据、纯推理、**不改 $W$**；改 $W$ 的是随后的 SFT step。
- **SFT 改 $W$ 范围**：只有**全参微调**全量改；LoRA 冻结 $W$，只训 $A$、$B$。
- **LoRA 定位**：属 **PEFT**，是低秩**约束**（非加速），有损近似，效果上限略低于全参微调；效果对 $r$、$\alpha$、插在哪些层敏感。

## 13. LoRA 初始化:为什么 B=0、A=随机,不能反过来?

### 1. 问题设定

LoRA(Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*)的前向形式:

$$h = W_0 x &#43; \Delta W x = W_0 x &#43; \frac{\alpha}{r} BAx$$

其中 $A \in \mathbb{R}^{r \times d}$(高斯随机初始化),$B \in \mathbb{R}^{d&#39; \times r}$(全 0 初始化),$r \ll \min(d, d&#39;)$。

**初始化的核心原则:训练起点必须满足 $\Delta W = BA = 0$**,即第 0 步模型输出与预训练模型逐位一致,微调从原模型平滑出发,不向输出注入随机噪声、不破坏预训练能力。

要让 $BA = 0$,只需 A、B 之一为 0,于是有三种候选方案:

| 方案 | A | B | $BA=0$? | 可训练? |
|------|-----|-----|---------|---------|
| ① LoRA 采用 | 随机高斯 | 全 0 | ✅ | ✅,效果好 |
| ② 反向初始化 | 全 0 | 随机高斯 | ✅ | ✅,但动力学更差 |
| ③ 双零 | 全 0 | 全 0 | ✅ | ❌ 鞍点,训不动 |

### 2. 为什么不能 A、B 都为 0:鞍点问题

记 $G = \dfrac{\partial L}{\partial (\Delta W)}$,由链式法则:

$$\frac{\partial L}{\partial B} = G A^\top, \qquad \frac{\partial L}{\partial A} = B^\top G$$

若 A = B = 0:

- $\partial L / \partial B = G \cdot 0 = 0$
- $\partial L / \partial A = 0 \cdot G = 0$

两个梯度恒为 0,初始点是一个**鞍点(saddle point)**,梯度下降永远无法离开,LoRA 分支完全学不到东西。这与普通神经网络不能全零初始化是同一个道理:必须有一方非零来**打破对称性**,让另一方获得非零梯度启动训练。

```
B=0, A=随机 时的启动过程:

step 0:  ∂L/∂B = G·Aᵀ ≠ 0  →  B 获得有效梯度,更新后 B ≠ 0
         ∂L/∂A = Bᵀ·G = 0   →  A 第一步不动(其随机值充当投影方向)
step 1&#43;: B ≠ 0  →  A 也开始获得梯度,双方正常协同更新
```

### 3. 为什么是 B=0、A=随机,而不是反过来?

反向初始化(A=0、B=随机)同样能打破鞍点、同样可训练,所以这不是&#34;能不能&#34;的问题,而是**训练动力学好不好**的问题。三个层面的理由:

**(1) 输入侧 / 输出侧的角色不对称(直觉)**

- A 在**输入侧**:$Ax$ 把 d 维输入压缩到 r 维。随机 A 等价于一个随机投影(random projection),由 Johnson–Lindenstrauss 引理,随机投影能较好保留高维输入的信息多样性,为 B 提供有信息量的&#34;特征基&#34;。
- B 在**输出侧**:直接决定往残差流里加什么。B=0 意味着输出方向从零开始、完全由梯度学出,更新&#34;干净&#34;。

若反过来 B 随机:A 一旦更新为非零,r 维信号就会被**未经训练的随机方向**混合后注入输出,早期训练噪声大、方向由随机初始化主导。

**(2) 理论分析:两种初始化的动力学不对称**

Hayou, Ghosh &amp; Yu, 2024, *The Impact of Initialization on LoRA Finetuning Dynamics*(NeurIPS 2024)在大宽度极限下系统分析了两种方案(记 Init[A]:A 随机 B=0;Init[B]:B 随机 A=0),结论:

- **Init[A] 允许使用更大的稳定学习率**,特征学习(feature learning)更充分,但可能伴随一定&#34;内部不稳定&#34;;总体上最终效果更好,实验(GLUE、LLM 微调)验证了这一点。
- **Init[B]** 稳定训练所允许的学习率更小,B 的随机方向主导更新、A 学习不充分,效率较低。
- 直觉:A 在窄边($r \times d$,承接高维输入),B 在宽边($d&#39; \times r$)。让高维输入侧先有固定的随机压缩、输出侧从零生长,数值上更稳。

**(3) 工程一致性**

B=0 保证一个**未训练的 adapter 加载后模型行为严格不变**,这带来:

- sanity check 方便:step 0 输出必须与原模型逐位一致,可直接验证接入正确性;
- 多任务 adapter 热插拔安全:挂载空 adapter 零副作用,适合 Agent/服务化场景中按需切换 LoRA 权重。

### 4. 相关延伸:scaling 系数与初始化的关系

- 原始 LoRA 用 $\frac{\alpha}{r}$ 缩放,目的之一是调 r 时不必重调学习率;
- **rsLoRA**(Kalajdzievski, 2023, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*)指出 $\frac{\alpha}{r}$ 在大 r 下会过度抑制更新,导致大 rank 学不动,应改为 $\frac{\alpha}{\sqrt{r}}$ 才能保持各 rank 下梯度尺度稳定;
- **PiSSA**(Meng et al., 2024, *PiSSA: Principal Singular Values and Singular Vectors Adaptation*)是另一条初始化改进路线:用 $W_0$ 的主奇异值/奇异向量初始化 A、B(同时从 $W_0$ 中减去该部分保持起点等价),让 LoRA 一开始就沿主方向更新,收敛更快——说明&#34;起点等价于原模型&#34;这个原则不变,但**如何分配初始信息**仍有优化空间。

### 5. 面试 Q&amp;A

**Q1:LoRA 为什么 B 初始化为 0、A 随机初始化?**

&gt; 初始化要保证 $BA=0$,第 0 步输出与预训练模型一致,不注入噪声。因此 A、B 至少一个为 0;但不能都为 0,否则 $\partial L/\partial B = GA^\top = 0$、$\partial L/\partial A = B^\top G = 0$,陷入鞍点训不动。选 A 随机、B=0:随机 A 相当于对高维输入做随机投影,保留信息、给 B 提供有效梯度;B=0 保证输出方向完全由梯度学出,不被随机初始化污染。

**Q2:反过来(A=0、B 随机)真的不能训吗?**

&gt; 能训——只要打破鞍点就能训。但动力学不对称:输出侧被随机 B 的方向主导,理论分析(Hayou et al., NeurIPS 2024)表明该方案允许的稳定学习率更小、特征学习不充分,实验效果更差。所以是&#34;次优&#34;而非&#34;不可行&#34;。

**Q3:这和普通网络不能全零初始化是一个道理吗?**

&gt; 本质相同:都是零梯度/对称性问题。LoRA 中体现为 B、A 互相把对方梯度乘成 0;普通网络中体现为同层神经元对称、更新完全相同。

**Q4:B=0 在工程上有什么额外好处?(Agent 岗加分)**

&gt; 未训练 adapter 加载后模型行为严格不变——便于 step-0 sanity check,也让多任务场景下 LoRA adapter 的热插拔零副作用,适合按需切换任务 adapter 的 Agent 服务化部署。

**参考文献**

1. Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
2. Hayou, Ghosh &amp; Yu, 2024. *The Impact of Initialization on LoRA Finetuning Dynamics.* NeurIPS 2024. arXiv:2406.08447
3. Kalajdzievski, 2023. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA.* arXiv:2312.03732
4. Meng et al., 2024. *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.* arXiv:2404.02948

---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/mllora/  

