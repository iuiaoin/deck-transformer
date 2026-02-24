# 演讲稿

## Slide 1 — 封面
欢迎大家。今天我们将从零开始，逐步建立对 Transformer 架构的直觉认知。目标是让 Attention 机制变得直观易懂，而非高深莫测。在演讲结束时，你将理解"Attention Is All You Need"论文中的每一个组件。

## Slide 2 — 议程
本次演讲分为三个部分。第一部分也是最核心的部分，讲解 Attention 机制——我们将从字典查找的类比出发，逐步推进到 Multi-Head Attention。第二部分，我们把这些组件组装成完整的 Transformer 架构。最后，我们看看 Transformer 是如何衍生出 BERT、GPT 和 T5 的。

## Slide 3 — 为什么需要 Attention？
在 Transformer 出现之前，RNN 主导着 NLP 领域。但 RNN 有两个根本性缺陷：它逐 token 顺序处理，无法充分利用 GPU 并行性；同时信号在长距离传播中逐渐衰减。Attention 一举解决了这两个问题——它在单步计算中完成所有 token 对之间的关系运算，任意两个 token 之间的路径长度仅为 O(1)。

## Slide 4 — 字典查找
这里有一个关键的直觉类比。普通字典执行精确匹配查找。但如果你的查询词是"水果"，而键分别是"苹果""香蕉""椅子"呢？Attention 执行的是软查找——它计算查询与每个键之间的相似度，通过 softmax 归一化为概率分布，然后返回值的加权求和。这就是 Attention 的全部核心思想。

## Slide 5 — Scaled Dot-Product Attention
现在让我们将其形式化。公式包含四个步骤：计算 Q 和 K 的点积相似度；除以维度的平方根以防止 softmax 饱和；用 softmax 归一化；最后对 V 做加权求和。用 PyTorch 实现只需三行代码。缩放因子至关重要——若缺少它，高维空间中的点积值会很大，导致 softmax 输出趋近于 one-hot 分布，进而使梯度消失。

## Slide 6 — Self-Attention
在 Self-Attention 中，Q、K、V 全部来自同一输入。输入经过三个独立的线性投影，分别生成 Q、K、V——同源数据，不同的可学习变换。这使每个 token 都能计算它与其他所有 token 之间的关系。观察 Attention 热力图——"cat"对自身、"sat"和"mat"有很强的注意力权重，捕获了语义层面的关联。

## Slide 7 — Masked Self-Attention
在生成任务中，模型必须在看不到未来信息的情况下预测下一个 token。我们通过因果掩码来实现这一约束——一个上三角矩阵，填充负无穷值。经过 softmax 后，负无穷变为零概率。其精妙之处在于：这种掩码使并行训练成为可能，所有位置可以同时训练——不再需要像 RNN 那样逐步训练。

## Slide 8 — Cross-Attention
Cross-Attention 在两个不同的序列之间建立桥梁。它与 Self-Attention 的关键区别在于：Q 来自 Decoder，而 K 和 V 来自 Encoder 的输出。可以理解为：Decoder 负责提问，Encoder 负责作答。在机器翻译中，Encoder 理解源语言，Decoder 在生成目标语言时不断查询 Encoder 所提供的理解。

## Slide 9 — Multi-Head Attention
单个 Attention 头只能捕获一种类型的关系。Multi-Head Attention 并行运行 h 个 Attention 计算，每个头专注于不同的模式——句法结构、语义相似性、位置邻近性等。我们将 d_model 维度拆分到 h 个头中，各自独立计算 Attention，然后拼接并投影回原始维度。关键在于，这种方式的总计算量与单头 Attention 完全相同。

## Slide 10 — Positional Encoding
Attention 本身对位置无感——"我爱你"和"你爱我"会得到完全相同的 Attention 权重。我们通过在输入 Embedding 上叠加正弦位置编码来解决这一问题。正弦/余弦设计满足三个关键性质：每个位置有确定的编码、相对距离保持一致、可以泛化到训练时未见过的序列长度。残差连接确保位置信息在所有层中得以保留。

## Slide 11 — Encoder
每个 Encoder 层包含两个子层：Multi-Head Self-Attention 和前馈网络（FFN）。每个子层都配有残差连接和 LayerNorm。Self-Attention 捕获不同位置之间的关系；FFN 则对每个位置独立处理，融合特征维度。我们堆叠 6 个相同的层。值得注意的是，实践中通常采用 Pre-Norm 变体——即在子层之前而非之后应用 LayerNorm。

## Slide 12 — Decoder
Decoder 在 Encoder 的基础上增加了第三个子层：位于 Masked Self-Attention 和 FFN 之间的 Cross-Attention。模块一（Masked Self-Attention）建模目标序列内部的依赖关系。模块二（Cross-Attention）读取 Encoder 的输出。模块三（FFN）执行同样的逐位置处理。在推理阶段，生成过程是 Autoregressive 的——每次生成一个 token。

## Slide 13 — 完整的 Transformer
现在来看完整的全景图。源序列经过 Embedding、Positional Encoding，再通过 N 个 Encoder 层，生成记忆表示。目标序列经过 Embedding、Positional Encoding，再通过 N 个 Decoder 层——Decoder 层还通过 Cross-Attention 接收 Encoder 的记忆。最后，经过线性层和 softmax 产生输出概率。在 PyTorch 中，这只需要 4 个类和 5 个超参数即可实现。

## Slide 14 — 三大范式
Transformer 催生了三种主流架构。BERT 只取 Encoder——双向 Attention、MLM 训练方式，擅长理解类任务。GPT 只取 Decoder——因果 Attention、CLM 训练方式，在生成任务中占据主导。T5 同时保留 Encoder 和 Decoder——采用文本到文本的统一格式，能够处理各类任务。当今的大语言模型（GPT-4、Claude、LLaMA）均采用 Decoder-only 架构，沿袭 GPT 范式进行大规模扩展。

## Slide 15 — 核心要点与问答
四个核心要点：Attention 是一种可学习的软字典查找机制，取代了顺序处理方式；Multi-Head Attention 在不增加计算开销的前提下捕获多样化模式；Transformer 将 Attention、FFN、残差连接和 LayerNorm 组装成模块化架构；BERT、GPT、T5 都是同一组基础构件的不同特化形式。欢迎大家提问。
