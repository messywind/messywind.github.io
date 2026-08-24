# 多模态 RAG


## 1. 什么是多模态 RAG

传统 RAG 的知识通常是纯文本。多模态 RAG（Multimodal RAG）把知识范围扩展到：

- 文本：正文、标题、OCR 结果、字幕、ASR 转写。
- 图片：照片、示意图、流程图、截图。
- 表格：财务报表、实验结果、参数对比。
- 音频：语音、会议录音、音乐和环境声音。
- 视频：画面、关键帧、字幕和音轨。
- 版面：PDF 页面中元素的位置、字体、颜色和阅读顺序。

核心流程仍然是 RAG：

```text
用户问题 → 检索相关知识 → 组织多模态上下文 → 模型生成答案
```

区别在于，多模态 RAG 需要同时解决三个问题：

1. **如何表示**：不同模态怎样转换成可检索的表示。
2. **如何对齐和检索**：文本问题怎样找到相关图片、音频、表格或页面。
3. **如何生成**：命中后把摘要交给文本 LLM，还是把原始模态交给多模态 LLM。

多模态 RAG 通常包含两条链路：

```mermaid
flowchart LR
    subgraph Indexing[&#34;索引链路&#34;]
        A[&#34;PDF / 文本 / 图片 / 音频 / 视频&#34;] --&gt; B[&#34;解析、OCR、ASR、抽帧&#34;]
        B --&gt; C[&#34;切块、分页、区域裁剪&#34;]
        C --&gt; D[&#34;文本 Embedding / 跨模态 Embedding / 多向量表示&#34;]
        D --&gt; E[&#34;一个或多个检索索引&#34;]
        C --&gt; F[&#34;原始内容存储&#34;]
    end

    subgraph Querying[&#34;查询链路&#34;]
        Q[&#34;用户 Query&#34;] --&gt; QE[&#34;Query 编码与改写&#34;]
        QE --&gt; E
        E --&gt; R[&#34;候选内容&#34;]
        R --&gt; RR[&#34;融合与 Rerank&#34;]
        RR --&gt; RAW[&#34;回取原始文本、图片、页面、音频片段&#34;]
        F --&gt; RAW
        RAW --&gt; M[&#34;LLM / Multimodal LLM&#34;]
        M --&gt; O[&#34;答案与引用&#34;]
    end
```

一个重要原则是：

&gt; **用于检索的表示，不一定等于最终交给生成模型的内容。**

例如，可以用图片描述进行向量检索，但检索命中后回取原始图片，交给多模态 LLM 阅读。这样既利用了成熟的文本检索能力，又避免 caption 丢失图片细节。

---

## 2. 跨模态语义对齐：CLIP 的基本原理

### 2.1 为什么不同模态可以互相检索

普通文本 Embedding 只能把文本映射为向量，不能直接理解图片。CLIP 一类模型包含两个编码器：

```text
文本 → Text Encoder  → 文本向量 T
图片 → Image Encoder → 图片向量 I
```

训练时使用大量“图片—文本描述”配对数据，让匹配的图片和文本在同一向量空间中接近，让不匹配的样本远离。

假设一个 Batch 中有 N 对图片和文本，可以计算相似度矩阵：

```text
              T1      T2      T3      ...     TN
I1           I1·T1   I1·T2   I1·T3    ...    I1·TN
I2           I2·T1   I2·T2   I2·T3    ...    I2·TN
...           ...     ...     ...      ...     ...
IN           IN·T1   IN·T2   IN·T3    ...    IN·TN
```

训练目标是让对角线上的正确配对分数较高，其他错误配对分数较低。这属于对比学习（Contrastive Learning）。

向量归一化后，图片 `I` 和文本 `T` 的相似度可以写成：

```text
similarity(I, T) = I · T = cosine(I, T)
```

实际模型还会使用可学习的温度系数调整分数分布。

### 2.2 CLIP 能做什么

因为图片和文本处于对齐后的共享空间，所以可以实现：

- 文本搜索图片：输入“草地上的小狗”，召回相关图片。
- 图片搜索文本：输入图片，找到匹配的描述或文档。
- 图片分类：计算图片与多个类别文本的相似度。
- 多模态路由：比较 Query 与不同候选模态的相关程度。

### 2.3 音频怎样进入统一语义空间

本项目的 CLIP Notebook 没有直接编码音频，而是采用：

```text
音频 → Whisper ASR → 转写文本 → CLIP Text Encoder
```

这种方式适合“语音内容”检索，但会丢失音色、旋律、环境声等非语言信息。

如果要直接检索音频，可以使用 CLAP 一类 Audio-Text 模型：

```text
音频 → Audio Encoder ┐
                     ├→ 共享 Audio-Text 语义空间
文本 → Text Encoder  ┘
```

项目代码中已经留下了 `ClapModel` 和 `ClapProcessor` 的注释示例，但当前运行链路仍然是 ASR 文本化。

---

## 3. 多模态 RAG 的三种主要范式

课程图和项目代码可以归纳为三条主要路线。

### 3.1 范式一：统一或对齐向量空间检索

#### 核心思路

使用 CLIP、CLAP 等模型，把不同模态映射到可比较的语义空间，再使用 Query 向量检索相关内容。

```mermaid
flowchart LR
    T[&#34;文本&#34;] --&gt; TE[&#34;Text Encoder&#34;]
    I[&#34;图片&#34;] --&gt; IE[&#34;Image Encoder&#34;]
    A[&#34;音频&#34;] --&gt; AE[&#34;Audio Encoder 或 ASR&#34;]
    TE --&gt; V[&#34;向量索引&#34;]
    IE --&gt; V
    AE --&gt; V
    Q[&#34;Query&#34;] --&gt; QE[&#34;Query Encoder&#34;]
    QE --&gt; V
    V --&gt; K[&#34;Top-K 原始模态&#34;]
    K --&gt; M[&#34;多模态 LLM&#34;]
    M --&gt; O[&#34;Answer&#34;]
```

#### 优点

- 可以直接完成文本到图片、文本到音频等跨模态检索。
- 不必先为每一张图片生成完整描述。
- 检索阶段保留视觉或听觉语义。
- 适合图库、商品、媒体资产和音视频内容库。

#### 局限

- 不同跨模态模型的能力边界不同，CLIP 不负责音频，CLAP 不负责图片。
- 通用视觉向量对密集文字、复杂表格和精确数字的理解有限。
- 即使模型声称处于共享空间，不同模态的分数分布也可能不完全可比。
- OpenAI `clip-vit-base-patch32` 主要使用英文图文数据训练，中文 Query 的效果可能不稳定。

### 3.2 范式二：模态文本化，再使用文本 RAG

#### 核心思路

先把图片、表格、音频和视频转换为文本，再复用成熟的文本切块、Embedding、向量库和 LLM 链路。

```text
图片 → Image Caption / OCR ┐
表格 → Markdown / 摘要      │
音频 → ASR 转写             ├→ 文本切块 → Text Embedding → Vector DB
视频 → 字幕 &#43; 关键帧描述     │
正文 → 原始文本             ┘
```

检索后有两种生成方式：

1. **只发送文本描述给 LLM**：实现简单、成本较低，但受描述信息损失影响。
2. **通过描述定位原始模态，再发送给多模态 LLM**：信息更完整，是更推荐的生产实现。

#### Multi-Vector Retriever 思想

一份原始内容可以对应多个用于召回的“小表示”：

```text
原始图片 image_3_1.png
├── 全局 caption
├── OCR 文本
├── 物体标签
└── 问题式摘要
```

这些摘要分别生成向量，提高召回覆盖率；命中任意一个子向量后，通过统一的 `doc_id` 回取同一个原始对象。

因此，工程上经常使用两个存储：

- **Vector Store**：保存 caption、摘要、OCR、子块及其向量。
- **Document/Object Store**：保存原始文本、表格、图片、页码、文件路径或对象存储 URI。

#### 优点

- 可以直接复用文本 RAG 技术栈。
- 易于调试，检索到的 caption 和 OCR 文本可读。
- 容易使用 BM25、混合检索和文本 Reranker。
- 对 API、模型和向量数据库的要求相对低。

#### 局限

- Caption 是有损压缩，可能漏掉小字、颜色、数量、位置关系和图表趋势。
- Caption 模型的幻觉会污染索引。
- OCR 和表格解析错误会被继续传播。
- 预处理每张图片会增加离线成本。

### 3.3 范式三：原生视觉文档检索

#### 核心思路

把 PDF 页面直接渲染为图片，使用视觉文档检索模型编码整个页面，不依赖先提取文本或生成 caption。

ColPali 的典型流程是：

```mermaid
flowchart LR
    P[&#34;PDF 页面图片&#34;] --&gt; VE[&#34;VLM 视觉编码&#34;]
    VE --&gt; PV[&#34;多个图像 Patch 向量&#34;]
    Q[&#34;文本 Query&#34;] --&gt; QE[&#34;Query Token 编码&#34;]
    QE --&gt; QV[&#34;多个 Query Token 向量&#34;]
    QV --&gt; S[&#34;Late Interaction / MaxSim&#34;]
    PV --&gt; S
    S --&gt; K[&#34;Top-K 页面&#34;]
    K --&gt; M[&#34;多模态 LLM 阅读页面&#34;]
    M --&gt; O[&#34;Answer&#34;]
```

ColPali 不是把整页压缩成单个向量，而是为 Query Token 和图片 Patch 保留多个向量。其 ColBERT 风格的迟交互评分可以简化表示为：

```text
score(Q, P) = Σᵢ maxⱼ(qᵢ · pⱼ)
```

含义是：对每个 Query Token，寻找与它最相关的页面区域，再把各 Token 的最佳匹配累加起来。

#### 优点

- 保留版面、字体、图表、表格和空间关系。
- 避免复杂且容易出错的 PDF 解析、OCR、表格重建链路。
- 对视觉丰富的 PDF、扫描件和复杂报表很有优势。
- 检索命中的页面可以直接交给多模态 LLM。

#### 局限

- 多向量索引比单向量索引占用更多存储和计算资源。
- 页面级召回可能包含较多无关区域。
- 模型和框架对 GPU、MPS、显存及版本兼容性要求较高。
- 当前项目使用的 `colpali-v1.3` 主要面向 PDF 和高资源语言，中文 Query 需要实际评测。
- 许多通用向量数据库没有原生支持 ColBERT 风格的多向量迟交互。

---

## 4. 项目方案一：CLIP 跨模态相关性与模态路由

对应 Notebook：

```text
multi_modal_rag_with_clip.ipynb
```

### 4.1 数据准备

Notebook 加载了三类数据：

- 图片：`clip_resource/Lorenz_Ro28-200px.png`
- 音频：`clip_resource/audio.mp3`
- 文本：`clip_resource/Wiki.txt`

音频由 `librosa` 读取为 16 kHz 波形，之后使用 Whisper 转写：

```python
processor = WhisperProcessor.from_pretrained(&#34;openai/whisper-base&#34;)
model = WhisperForConditionalGeneration.from_pretrained(&#34;openai/whisper-base&#34;)

input_features = processor(
    audio_array,
    sampling_rate=sample_rate,
    return_tensors=&#34;pt&#34;,
).input_features

predicted_ids = model.generate(input_features)
audio_transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True,
)[0]
```

### 4.2 CLIP 编码

图片经过 Image Encoder，Query、音频转写和普通文本经过 Text Encoder：

```python
model = CLIPModel.from_pretrained(&#34;openai/clip-vit-base-patch32&#34;)
processor = CLIPProcessor.from_pretrained(&#34;openai/clip-vit-base-patch32&#34;)

image_inputs = processor(images=image, return_tensors=&#34;pt&#34;)
image_embeddings = model.get_image_features(**image_inputs)

text_inputs = processor(
    text=[query, audio_transcription, text_data],
    return_tensors=&#34;pt&#34;,
    padding=True,
)
text_embeddings = model.get_text_features(**text_inputs)
```

Notebook 输出的向量形状为：

```text
image_embeddings: [1, 512]
text_embeddings:  [3, 512]
```

### 4.3 相似度和路由

代码计算 Query 与三类候选的余弦相似度：

```python
cos_sim_query_image = cosine_similarity(query_embedding, image_embedding)
cos_sim_query_audio = cosine_similarity(query_embedding, audio_embedding)
cos_sim_query_text = cosine_similarity(query_embedding, text_embedding)
```

本次运行结果：

| 候选模态 | 余弦相似度 |
| --- | ---: |
| 图片 | 0.1928 |
| 音频转写 | 0.7653 |
| 普通文本 | 0.6411 |

音频转写得分最高，因此系统把 Query 和 ASR 文本交给 `gemini-2.5-flash`。最终识别出用户喜欢的竖琴演奏家是 **Turlough O&#39;Carolan**。

### 4.4 这份 Notebook 的准确定位

它演示了：

- 跨模态共享向量空间。
- Query 与不同模态候选的相似度比较。
- 根据最高分候选选择生成上下文。

但它还不是完整的大规模 RAG：

- 每种模态只有一个候选，没有真正建立向量索引。
- 只选择得分最高的一个模态，没有 Top-K、融合和 Rerank。
- 音频经过 ASR 后本质上仍按文本检索。
- 原始图片、音频、文本的分数分布可能存在偏差，直接比较原始余弦值不一定可靠。

要扩展成完整系统，应把每个资源切成可定位单元，写入向量数据库，并保存：

```text
id、modality、embedding、source、page、timestamp、raw_uri、metadata
```

检索后再回取 Top-K 原始内容，而不是只做三选一。

---

## 5. 项目方案二：Image Caption &#43; 文本向量库

对应 Notebook：

```text
multi_model_rag_with_captioning.ipynb
```

### 5.1 PDF 内容抽取

Notebook 使用 PyMuPDF 打开 `transformer.pdf`，逐页提取文本和内嵌图片：

```python
with fitz.open(&#34;transformer.pdf&#34;) as pdf_file:
    for page_number in range(len(pdf_file)):
        page = pdf_file[page_number]

        text = page.get_text().strip()
        text_data.append({&#34;response&#34;: text, &#34;name&#34;: page_number &#43; 1})

        for image_index, img in enumerate(page.get_images(full=True)):
            # 提取并保存图片
            ...
```

示例 PDF 共提取出：

- 15 页文本。
- 3 张图片。

需要注意，`page.get_images()` 只能提取 PDF 中作为图片对象嵌入的内容。矢量图、绘制命令、复杂表格和整页版面不一定能被完整提取。若视觉结构很重要，可以直接把整页渲染成图片。

### 5.2 生成图片描述

每张图片交给 `gemini-2.5-flash`，提示模型生成“简洁、便于检索”的摘要：

```python
response = google_client.models.generate_content(
    model=&#34;gemini-2.5-flash&#34;,
    contents=[retrieval_summary_prompt, image],
)
```

示例中生成了以下类型的描述：

- Transformer 整体 Encoder-Decoder 架构。
- Scaled Dot-Product Attention 流程。
- Multi-Head Attention 流程。

这里的目标不是写一段漂亮的图片说明，而是尽量覆盖用户可能用于检索的术语、实体、数字和关系。

### 5.3 文本切块与向量化

原始正文和图片描述都被包装为 `Document`，使用同一个文本 Embedding 模型编码：

```python
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=400,
    chunk_overlap=50,
)

doc_splits = text_splitter.split_documents(docs_list)
img_splits = text_splitter.split_documents(img_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits &#43; img_splits,
    collection_name=&#34;multi_model_rag&#34;,
    embedding=embedding_model,
)
```

Notebook 使用：

- Embedding：`text-embedding-v4`，1024 维。
- Vector Store：Chroma。
- 文本块：36 个。
- 图片描述块：4 个。
- 召回数量：`k=1`。

### 5.4 检索与生成

示例问题：

```text
Transformer（base model）的 BLEU 得分是多少？
```

系统召回论文第 8 页的 Table 2，然后使用 `qwen3-max` 根据召回文本作答：

```text
EN-DE：27.3
EN-FR：38.1
```

### 5.5 当前实现与完整 Multi-Vector RAG 的差异

当前 Notebook 把图片 caption 直接作为普通文本块加入 Chroma，检索命中后也只把 `page_content` 交给文本 LLM。

因此它实际实现的是：

```text
图片 → Caption → Text Embedding → 检索 Caption → 文本 LLM
```

更完整的实现应该是：

```text
图片 → Caption / OCR / 摘要 → Text Embedding → 检索代理文本
                                      ↓
                                通过 doc_id 回取原图
                                      ↓
                              原图 &#43; Query → 多模态 LLM
```

建议元数据至少包含：

```python
metadata = {
    &#34;doc_id&#34;: &#34;transformer-pdf&#34;,
    &#34;element_id&#34;: &#34;image-4-2&#34;,
    &#34;modality&#34;: &#34;image&#34;,
    &#34;page&#34;: 4,
    &#34;raw_path&#34;: &#34;extracted_images/image_4_2.png&#34;,
}
```

这样才能做到“用文本表示检索，用原始模态生成”。

---

## 6. 项目方案三：ColPali 原生视觉 PDF RAG

对应 Notebook：

```text
multi_model_rag_with_colpali.ipynb
```

### 6.1 建立页面级视觉索引

Notebook 通过 Byaldi 加载 `vidore/colpali-v1.3`：

```python
from byaldi import RAGMultiModalModel

RAG = RAGMultiModalModel.from_pretrained(
    &#34;vidore/colpali-v1.3&#34;,
    verbose=1,
)
```

然后直接对 PDF 建索引：

```python
RAG.index(
    input_path=&#34;transformer.pdf&#34;,
    index_name=&#34;attention_is_all_you_need&#34;,
    store_collection_with_index=True,
    overwrite=True,
)
```

这一步会把 PDF 的 15 页作为视觉页面加入索引。`store_collection_with_index=True` 会同时保存页面的 Base64 表示，方便检索后直接取回页面图片。

### 6.2 文本 Query 检索页面

```python
query = &#34;Transformer（base model）的BLEU得分是多少？&#34;
results = RAG.search(query, k=1)
```

检索返回的不是某个 OCR 文本块，而是最相关的 PDF 页面及其分数、文档 ID、页码和 Base64 图片。

### 6.3 多模态生成

Notebook 把命中页面解码为 `image.jpg`，再将问题和页面图片一起交给 Gemini：

```python
response = google_client.models.generate_content(
    model=&#34;gemini-2.5-flash&#34;,
    contents=[query, image],
)
```

最终答案同样是：

```text
EN-DE：27.3
EN-FR：38.1
```

与方案二相比，这条链路没有依赖 PDF 文本抽取或表格转 Markdown，而是由检索模型定位包含表格的页面，再由多模态 LLM 直接阅读页面。

### 6.4 适用场景

ColPali 更适合：

- 扫描 PDF。
- 复杂表格和财务报表。
- 图文混排论文、说明书和宣传册。
- 信息依赖版面、颜色、位置关系的文档。

如果知识主要是结构清晰的纯文本，传统文本 RAG 往往成本更低、切块更细、引用更精确，不必强行使用视觉检索。

---

## 7. 分模态检索、结果融合与多模态 Rerank

课程第三张图给出了另一种更灵活的工程架构：不同模态使用各自擅长的编码器和索引。

```mermaid
flowchart LR
    Q[&#34;文本 Query&#34;] --&gt; AT[&#34;Audio-Text Query Encoder&#34;]
    Q --&gt; IT[&#34;Image-Text Query Encoder&#34;]
    Q --&gt; TT[&#34;Text Query Encoder&#34;]

    A[&#34;音频库&#34;] --&gt; AI[&#34;音频向量索引&#34;]
    I[&#34;图片库&#34;] --&gt; II[&#34;图片向量索引&#34;]
    T[&#34;文本库&#34;] --&gt; TI[&#34;文本向量索引&#34;]

    AT --&gt; AI
    IT --&gt; II
    TT --&gt; TI

    AI --&gt; C[&#34;候选集合&#34;]
    II --&gt; C
    TI --&gt; C
    C --&gt; R[&#34;多模态 Reranker&#34;]
    R --&gt; M[&#34;LLM / 多模态 LLM&#34;]
    M --&gt; O[&#34;生成回复&#34;]
```

### 7.1 为什么要分模态建索引

一个模型很难同时在所有模态上达到最佳效果：

- 图片使用 Image-Text 模型。
- 音频使用 Audio-Text 模型或 ASR &#43; Text Embedding。
- 文本使用专门的 Text Embedding。
- PDF 页面使用 ColPali 一类视觉文档检索模型。

这种方式能利用各领域最强的模型，并允许按模态单独扩缩容。

### 7.2 为什么不能直接混合原始分数

不同模型的余弦分数通常不在同一个分布中：

```text
CLIP 的 0.30、CLAP 的 0.30、文本 Embedding 的 0.30
并不一定表示相同相关程度。
```

更稳妥的候选融合方式包括：

1. **每个模态分别取 Top-K**，先保证召回覆盖率。
2. **分数归一化**，例如 Min-Max、Z-Score 或按历史分位数校准。
3. **RRF（Reciprocal Rank Fusion）**，按名次而不是原始分数融合：

```text
RRF(d) = Σ 1 / (k &#43; rankᵢ(d))
```

4. **学习式融合**，根据 Query 类型动态设置图片、文本、音频权重。
5. **统一 Reranker**，对合并后的候选进行跨模态精排。

### 7.3 多模态 Reranker

Reranker 接收 Query 和候选原始内容，输出统一相关性分数。它可以解决：

- 各索引分数不可比。
- 初始 Embedding 只捕获粗粒度语义。
- 问题要求精确数字、颜色、位置或跨元素关系。

可以使用多模态大模型作为 Listwise 或 Pointwise Reranker。例如：

```text
Query：埃菲尔铁塔是什么颜色？

候选：
1. 一张埃菲尔铁塔夜景图片
2. 一段介绍埃菲尔铁塔历史的文字
3. 一段提到铁塔涂装颜色的音频

任务：按“能否直接、准确回答 Query”排序，并给出相关性分数。
```

生产中可以先用轻量向量模型召回几十条，再用 Qwen-VL、Gemini 等多模态模型精排少量候选，控制成本。

---

## 8. 三种方案对比

| 维度 | 跨模态向量检索 | Caption &#43; 文本 RAG | ColPali 视觉文档检索 |
| --- | --- | --- | --- |
| 典型模型 | CLIP、SigLIP、CLAP | VLM Caption &#43; Text Embedding | ColPali / ColQwen |
| 检索对象 | 图片、文本、音频等资源 | Caption、OCR、摘要、正文 | PDF 页面图片 |
| 索引表示 | 单向量为主 | 文本单向量 | Query Token / Image Patch 多向量 |
| 是否需要 OCR | 可选 | 通常需要 | 不一定需要 |
| 是否保留版面 | 较弱 | 取决于解析结果 | 强 |
| 数字与表格能力 | 通常较弱 | 取决于 OCR/表格解析 | 通常更适合视觉表格页面 |
| 检索可解释性 | 中等 | 高，可查看文本代理 | 中等，可查看命中页面 |
| 计算与存储成本 | 中等 | 离线 Caption 成本较高，在线较低 | 较高 |
| 适合场景 | 图库、商品、音视频资产 | 通用企业知识库 | 复杂 PDF、扫描件、图文混排文档 |
| 项目 Notebook | `with_clip` | `with_captioning` | `with_colpali` |

实际系统不必三选一，常见做法是混合：

```text
正文 → 文本 RAG
PDF 页面 → ColPali
独立图片 → CLIP / Caption
音频 → ASR &#43; CLAP
所有候选 → 融合 &#43; 多模态 Rerank → 多模态 LLM
```

---

## 9. 一套更完整的生产架构

### 9.1 索引阶段

1. 为每个原始资源生成稳定的 `doc_id` 和 `element_id`。
2. 文档解析时保留页码、坐标、章节、时间戳、权限和来源。
3. 根据模态选择预处理：
   - 文本：结构化切块。
   - 图片：caption、OCR、标签、视觉向量。
   - 表格：Markdown/HTML、摘要、原始截图。
   - 音频：ASR、说话人、时间段、音频向量。
   - 视频：字幕、镜头切分、关键帧和音轨。
4. 将检索代理写入向量索引，把原始对象写入对象存储或文档存储。
5. 建立 `vector_id → element_id → raw_uri` 映射。

推荐的数据结构：

```python
{
    &#34;vector_id&#34;: &#34;vec-001&#34;,
    &#34;doc_id&#34;: &#34;transformer-paper&#34;,
    &#34;element_id&#34;: &#34;page-8-table-2&#34;,
    &#34;modality&#34;: &#34;table&#34;,
    &#34;retrieval_text&#34;: &#34;Table 2 compares BLEU scores...&#34;,
    &#34;source&#34;: &#34;transformer.pdf&#34;,
    &#34;page&#34;: 8,
    &#34;bbox&#34;: [72, 110, 520, 420],
    &#34;raw_uri&#34;: &#34;s3://bucket/transformer/page-8-table-2.png&#34;,
    &#34;permissions&#34;: [&#34;research-team&#34;],
}
```

### 9.2 查询阶段

1. 判断问题涉及的模态和答案类型。
2. 对 Query 做改写、翻译或补充检索关键词。
3. 并行查询文本、图片、音频和页面索引。
4. 每路分别取 Top-K。
5. 使用 RRF、分数校准或 Reranker 合并候选。
6. 回取原始对象，并控制总 Token、图片数和音频长度。
7. 把 Query、上下文和来源信息交给生成模型。
8. 输出答案、引用页码、时间戳或图片来源。

### 9.3 生成 Prompt 的基本要求

```text
你是一个基于检索资料回答问题的助手。

要求：
1. 只根据提供的文本、图片、表格或音频转写回答。
2. 如果资料不足，明确说明无法确定。
3. 数字、单位、颜色和专有名词必须与来源一致。
4. 给出引用来源，包括文件名、页码或时间戳。
5. 多个来源冲突时，指出冲突，不要自行拼凑结论。
```

---

## 10. 评估多模态 RAG

不要只看最终回答是否“听起来合理”，需要分阶段评估。

### 10.1 解析与索引质量

- OCR 字符准确率。
- 表格单元格和行列结构准确率。
- 图片与 caption 的事实一致性。
- 音频转写的 WER（Word Error Rate）。
- 页码、坐标和时间戳是否正确。

### 10.2 检索质量

- Recall@K：正确内容是否出现在前 K 条。
- Precision@K：前 K 条中有多少真正相关。
- MRR：第一个正确结果的排名。
- nDCG：整体排序质量。
- 按模态分别统计召回率，避免文本效果掩盖图片或音频问题。

### 10.3 生成质量

- Faithfulness：答案是否严格来自检索上下文。
- Answer Relevance：是否直接回答问题。
- 数字、单位、实体和颜色是否准确。
- 引用是否指向真正支持答案的页面或资源。
- 模型是否真的使用了图片，而不是只依赖旁边的文字。

### 10.4 端到端测试集

测试问题应覆盖：

- 纯文本问题。
- 只能从图片回答的问题。
- 只能从表格精确读取数字的问题。
- 需要结合正文和图片的问题。
- 需要定位音频时间段的问题。
- 无答案问题和冲突来源问题。

---

## 11. 当前 Notebook 的注意事项与改进建议

### 11.1 CLIP 方案

- `openai/clip-vit-base-patch32` 对中文支持有限，可使用多语言 CLIP、中文图文模型，或先把 Query 翻译成英文。
- CLIP 文本长度较短，不适合直接编码长文档，应先切块或生成检索摘要。
- Whisper 输入最好统一为单声道 16 kHz；`mono=False` 遇到立体声音频可能产生二维数组。
- Notebook 输出了 Whisper 的 `attention_mask` 警告，正式代码应向生成阶段传递有效的 attention mask。
- 使用 `argmax` 选择最佳候选比逐个比较浮点数更清晰。
- 不建议直接比较来自不同模型的原始余弦分数，应做校准或统一 Rerank。

### 11.2 Caption 方案

- Caption Prompt 应强调实体、数字、坐标关系、图例、趋势和单位，而不仅是概括主题。
- 图片描述和原始图片必须通过 `element_id` 绑定。
- `k=1` 容易漏召回，通常先取 5～20 条，再精排。
- 应把召回结果的 metadata 和来源一起传入生成链路。
- 对图片 caption 再使用通用文本切分器未必必要，短 caption 可以直接作为一个检索单元。
- `page.get_images()` 无法完整还原所有 PDF 视觉元素，必要时应渲染整页或使用版面解析器。

### 11.3 ColPali 方案

- 页面 Base64 全量存入索引会增加磁盘占用，大规模系统更适合只存对象存储 URI。
- 页面级召回后可以增加区域检测或裁剪，减少多模态 LLM 输入中的无关内容。
- 对中文、行业文档和特殊版式，需要构建自己的评测集，而不能只依赖公开 Benchmark。
- `overwrite=True` 会重建并覆盖同名索引，生产环境应谨慎使用。
- Byaldi、Transformers、Torch 和 ColPali 模型之间可能存在版本约束，部署前应锁定依赖版本。

### 11.4 通用安全与工程问题

- 多模态文件同样可能包含 Prompt Injection，图片和 PDF 中的指令不能被当成系统指令执行。
- 检索前后都要执行权限过滤，不能在生成后才隐藏敏感内容。
- 对象存储链接应使用短期签名或服务端读取，避免暴露私有资源。
- 记录检索候选、分数、Rerank 结果和模型版本，方便排查线上问题。
- 对重复图片、重复页面和近似音频进行去重，减少索引污染。

---

## 12. 方案选择速查

### 选择 Caption &#43; 文本 RAG，如果：

- 已经有成熟的文本 RAG 基础设施。
- 数据量不大，离线生成描述的成本可接受。
- 希望检索结果可读、易调试。
- 图片内容较简单，caption 能覆盖主要信息。

### 选择 CLIP / CLAP 跨模态检索，如果：

- 需要在大量独立图片或音频资产中搜索。
- 用户通常使用自然语言描述视觉或听觉内容。
- 任务更关心整体语义，而不是从表格读取精确数字。

### 选择 ColPali，如果：

- 主要数据是 PDF、扫描件、图文混排页面。
- OCR、表格解析和阅读顺序经常出错。
- 答案依赖页面视觉结构。
- 可以接受更高的索引、推理和存储成本。

### 选择分模态检索 &#43; Rerank，如果：

- 知识库同时包含文本、图片、音频和复杂 PDF。
- 不同模态的数据量和更新频率差异较大。
- 对召回率要求高，并且能承担候选融合与精排成本。

---

## 13. 项目文件与知识点对应关系

| 文件或目录 | 作用 | 对应知识点 |
| --- | --- | --- |
| `multi_modal_rag_with_clip.ipynb` | 图片、ASR 文本和普通文本统一比较 | CLIP、Whisper、余弦相似度、模态路由 |
| `multi_model_rag_with_captioning.ipynb` | PDF 抽取、图片描述、Chroma 检索 | 模态文本化、Text Embedding、Caption RAG |
| `multi_model_rag_with_colpali.ipynb` | PDF 页面视觉索引和检索 | ColPali、Multi-Vector、Late Interaction |
| `transformer.pdf` | 示例知识文档 | 复杂 PDF、文本、表格和图片混合内容 |
| `extracted_images/` | 从 PDF 提取的图片 | Image Caption 的原始输入 |
| `clip_resource/` | 图片、音频和文本示例 | 多模态输入与跨模态相似度 |
| `vidore/colpali-v1.3/` | 本地 ColPali 模型文件 | 视觉文档检索模型 |

---

## 14. 总结

多模态 RAG 的关键不是简单地“让 LLM 看图片”，而是建立完整的信息链路：

```text
原始多模态知识
→ 合理解析和切分
→ 为每种模态选择合适的检索表示
→ 跨模态召回或分模态召回
→ 候选融合与 Rerank
→ 回取高保真的原始内容
→ 多模态 LLM 基于证据生成答案
```

三种方案可以用一句话概括：

- **CLIP/CLAP**：把不同模态对齐后直接检索。
- **Caption RAG**：先把多模态内容转换成文本，再复用文本 RAG。
- **ColPali**：直接理解和检索整张文档页面的视觉结构。

工程上最重要的取舍是：

&gt; 用低成本、易检索的表示完成召回，用信息最完整的原始内容完成生成。



---

> 作者: [凌乱之风](https://github.com/messywind)  
> URL: https://blog.messywind.top/posts/ml%E5%A4%9A%E6%A8%A1%E6%80%81-rag/  

