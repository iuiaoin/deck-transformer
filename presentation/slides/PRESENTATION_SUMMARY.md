# 演示文稿概要

## 深入理解 Transformer 架构：Attention Is All You Need

**幻灯片数量**：15
**语言**：中文
**目标受众**：具有有限深度学习背景的软件工程师
**日期**：2026 年 2 月

---

## 结构

| # | 幻灯片标题 | 类型 | 所属章节 |
|---|-----------|------|---------|
| 1 | 封面 | 封面 | — |
| 2 | 目录 | 目录 | — |
| 3 | 为什么需要 Attention？序列化瓶颈 | 内容（对比） | Attention 机制 |
| 4 | Attention = 加权字典查找 | 内容（类比） | Attention 机制 |
| 5 | Scaled Dot-Product Attention | 内容（公式 + 代码） | Attention 机制 |
| 6 | Self-Attention：Q = K = V 同源 | 内容（图解） | Attention 机制 |
| 7 | Masked Self-Attention：禁止窥探未来 | 内容（矩阵可视化） | Attention 机制 |
| 8 | Cross-Attention：连接两个序列 | 内容（图解） | Attention 机制 |
| 9 | Multi-Head Attention：多角度并行关注 | 内容（图解 + 代码） | Attention 机制 |
| 10 | Positional Encoding：注入顺序信息 | 内容（公式 + 热力图） | 架构 |
| 11 | Encoder：Self-Attention + FFN + Residual | 内容（架构图） | 架构 |
| 12 | Decoder：三模块流水线 | 内容（架构图） | 架构 |
| 13 | 完整 Transformer：全局组装 | 内容（架构图 + API） | 架构 |
| 14 | Transformer 的传承：三大范式 | 对比（三栏布局） | 下游应用 |
| 15 | 核心要点与问答 | 结语 | — |

## 设计规格

- **画布尺寸**：1280 x 720px（16:9）
- **背景色**：#F7F4EF（暖米色）
- **主文字色**：#2B2A27（深炭灰）
- **主强调色**：#E07A59（珊瑚橙）
- **副强调色**：#9FD3B8（薄荷绿）
- **字体**：Inter（标题 700，正文 400-500），Source Code Pro（代码）
- **动画**：fadeInContent、slideInUp，采用交错延迟

## 内容来源

- DataWhale HappyLLM 第二章 — Transformer 架构、Attention 机制
- DataWhale HappyLLM 第三章 — 预训练语言模型（BERT、GPT、T5）
- Transformer 架构补充材料 — PyTorch API 详解
- 原始论文："Attention Is All You Need"（Vaswani 等，2017）
