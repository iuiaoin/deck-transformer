# Presentation Summary

**Title:** Understanding Transformer Architecture: Attention Is All You Need
**Length:** 16 slides
**Language:** English
**Audience:** Software engineers with limited deep-learning background
**Canvas:** 1280 × 720, standalone HTML per slide

## Goals

1. Build Attention intuition from scratch — dictionary lookup → scaled dot-product → self / masked / cross / multi-head. (≈50% of deck)
2. Explain the Transformer encoder-decoder stack — Positional Encoding, FFN, LayerNorm, residual, causal masks, via `torch.nn.Transformer`.
3. Summarize downstream paradigms — BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder).

## Structure

| # | Title | Role |
|---|---|---|
| 1 | Cover | Title + author + date |
| 2 | Why Transformer? The 2017 inflection point | Motivation — RNN/CNN pain, attention as the fix |
| 3 | Part 1 opener — the mental model | Dictionary-lookup analogy for attention |
| 4 | Q, K, V — three projections from one input | Linear projections, shapes, intuition |
| 5 | Scaled Dot-Product Attention — the formula | `softmax(QKᵀ/√d_k)V` hero + 4-step pipeline |
| 6 | Attention in ~10 lines of PyTorch | Code slide: the only code until §2 |
| 7 | Self-Attention — tokens look at themselves | Example sentence with attention weights |
| 8 | Masked Self-Attention — no peeking | Upper-triangular causal mask visual |
| 9 | Cross-Attention — decoder reads encoder | Data flow between the two stacks |
| 10 | Multi-Head Attention — many perspectives | h=8 heads, reshape code, concat + project |
| 11 | Part 2 opener — Positional Encoding | Why order matters; sin/cos PE |
| 12 | FFN · LayerNorm · Residual | Three glue components + pre-norm pattern |
| 13 | The full picture — encoder-decoder stack | End-to-end block diagram (N=6) |
| 14 | Wiring it up with `torch.nn.Transformer` | PyTorch API + three mask types |
| 15 | Three paradigms from one architecture | BERT vs GPT vs T5 comparison |
| 16 | Key takeaways | Hero formula + 4 bullets + next step |

## Design Specs

- **Palette:** warm off-white `#F7F4EF` background, coral `#E07A59` accent, mint `#9FD3B8` secondary, charcoal `#2B2A27` headings, warm grey `#5D4E42` body.
- **Typography:** Inter (400/500/600/700) for UI + body; Source Code Pro for code; Noto Serif Italic for the hero formula on slides 5 and 16.
- **Animations:** `fadeInContent` and `slideInUp` with staggered delays; `pulseGlow` on the takeaway formula.
- **Consistent footer:** `Transformer · Attention Is All You Need` · `N / 16` on every content slide.

## Source Mapping

- Attention section (slides 3-10) — `resources/llm-from-scratch/chapter2/第二章 Transformer架构.md`
- Architecture + PyTorch API (slides 11-14) — `resources/transformer-architecture/readme.md` and chapter 2
- Paradigms (slide 15) — `resources/llm-from-scratch/chapter3/第三章 预训练语言模型.md`

## Deliverables

- `presentation/index.html` — viewer (grid/list + fullscreen presenter mode, keyboard nav)
- `presentation/slides/slide1.html` … `slide16.html` — 16 standalone slides
- `presentation/slides/PRESENTATION_SUMMARY.md` — this file
- `presentation/slides/PRESENTATION_SCRIPT.md` — speaker notes
