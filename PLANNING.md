**Task**: Generate an English technical-sharing deck titled "Understanding Transformer Architecture: Attention Is All You Need", building attention intuition from scratch, walking through the Encoder-Decoder architecture via PyTorch's `nn.Transformer` API, and mapping out the Encoder-only / Decoder-only / Encoder-Decoder paradigms.
**Slide count**: 16 (within 15 ± 2)
**Language**: English
**Audience**: Software engineers with limited deep-learning background
**Goals**:
- Build intuition for Attention from the ground up — dictionary-lookup analogy → Scaled Dot-Product → Self / Masked / Cross / Multi-Head Attention (≈ 50% of deck)
- Explain the end-to-end Transformer architecture — Positional Encoding, FFN, LayerNorm, Residual, causal mask — and how PyTorch's `nn.Transformer` API wires them
- Summarize the three downstream paradigms spawned by Transformer — Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5)

**Style**: Restrained, high information density, diagram-heavy, presentation-friendly. Code only for the two highest-value snippets (scaled dot-product attention core and multi-head attention core). Prefer formulas and architecture diagrams over prose. When a DL concept (softmax, embedding, gradient, logits, etc.) first appears, give a one-sentence plain-language gloss. Each slide holds one takeaway.

---

## Visual & Layout Guidelines

- **Overall tone**: Restrained, diagram-first, lots of whitespace; feels like a technical whitepaper slide deck
- **Background**: `#F7F4EF` warm beige (default Claude style)
- **Primary text**: `#2B2A27` deep charcoal for titles and body
- **Secondary text**: `#5D4E42` medium brown for captions and glosses
- **Accent (primary)**: `#E07A59` coral — borders on key shapes, attention arrows, formulas
- **Accent (secondary)**: `#9FD3B8` mint — secondary highlights, Encoder/Decoder block fills
- **Strong emphasis**: `#4A3F35` dark brown-black for bolded keywords
- **Typography**: Inter 600–700 for titles (48–64px), Inter 400–500 for body (18–24px), Source Code Pro for code (14–16px); math rendered inline with italic serif fallback for variables
- **Per-slide rule**: 1 takeaway + at most 3–4 supporting bullets / labeled diagram elements
- **Footer**: bottom-left "Transformer · Attention Is All You Need" + bottom-right page number `NN / 16`
- **Diagrams**: hand-assembled with CSS boxes, SVG arrows, and LaTeX-style formulas — no external images except where noted
- **Jargon boxes**: when a new DL term first appears, show a dotted-border "plain-language" callout one-liner

---

## Slide-by-Slide Outline

**Slide 1 | Cover**
- Title: "Understanding Transformer Architecture"
- Subtitle: "Attention Is All You Need — from dictionary lookup to self-attention to BERT/GPT"
- Footer: Technical Sharing Session · 2026
- Visual: oversized subdued "Q · K · V" glyph as a watermark; single coral underline under the title

**Slide 2 | Agenda & Why This Matters**
- Key takeaway: Transformer replaced RNNs by making long-range dependencies parallelizable — and it is the foundation of every modern LLM.
- Three-part roadmap shown as a vertical numbered stack:
  1. Attention from scratch (≈ half the deck)
  2. Putting a Transformer together (Encoder-Decoder + PyTorch `nn.Transformer`)
  3. Three paradigms it spawned (BERT / GPT / T5)
- Tiny side callout: "RNN was sequential → GPU-unfriendly; attention is a matmul → massively parallel."
- Visual: 3-column layout with a coral numeral per section and a thin divider

---

### Part 1 — Attention from Scratch (Slides 3 – 10, ≈ 50% of deck)

**Slide 3 | Attention Intuition: Start with a Dictionary**
- Key takeaway: Attention generalizes dictionary lookup from exact match to *weighted match*.
- Left: Python dict `{ "apple": 10, "banana": 5, "chair": 2 }` — query `"apple"` → hard match → value `10`.
- Right: query `"fruit"` has no exact key — we softly match to `apple` (0.6) and `banana` (0.4), chair (0.0) → value = 0.6·10 + 0.4·5 = 8.
- Jargon callout: *weight* = "how much each key contributes", must sum to 1.
- Visual: two panels side by side with arrow "exact → soft" between them.

**Slide 4 | From Soft Lookup to Q, K, V**
- Key takeaway: The three variables of attention — **Query**, **Key**, **Value** — are all just vectors; their relationships are learned.
- Definitions (one line each):
  - **Query** — what am I looking for? (e.g., "fruit")
  - **Key** — what does each entry advertise itself as? (e.g., "apple", "banana")
  - **Value** — what do I get back? (the payload)
- Jargon callout: *embedding* = "a learned dense vector that represents a word or token in N-dimensional space".
- Visual: three labeled lanes (Q / K / V), each a coral-bordered rectangle with a sample vector `[0.12, -0.03, …]`.

**Slide 5 | Scaled Dot-Product Attention — the Formula**
- Key takeaway: Attention is a three-step pipeline: *similarity → normalize → weighted sum*.
- Centerpiece formula:
  $$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\tfrac{QK^{\top}}{\sqrt{d_k}}\right) V$$
- Annotated step labels pointing at each piece of the formula:
  1. `QKᵀ` — dot product = similarity between every query and every key
  2. `/√d_k` — scale so variance stays stable when `d_k` is large (prevents tiny gradients)
  3. `softmax` — turn raw scores into probabilities that sum to 1
  4. `·V` — weighted sum of values → context vector
- Jargon callouts: *softmax* = "squashes a vector of numbers into probabilities that sum to 1"; *gradient* = "the signal used to update weights during training — tiny gradients stall learning".
- Visual: horizontal pipeline with 4 nodes + arrows.

**Slide 6 | Attention in Code — Five Lines**
- Key takeaway: The whole formula fits in a handful of PyTorch calls.
- Code snippet (syntax-highlighted, ~8 lines):
  ```python
  def attention(q, k, v, mask=None):
      d_k = q.size(-1)
      scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
      if mask is not None:
          scores = scores.masked_fill(mask == 0, float("-inf"))
      attn = F.softmax(scores, dim=-1)
      return torch.matmul(attn, v), attn
  ```
- Side bullet: "Masking = set blocked positions to `-inf` *before* softmax → their probability becomes 0."
- Visual: small inset diagram showing shape evolution `(B,T,d) × (B,d,T) → (B,T,T) · (B,T,d) → (B,T,d)`.

**Slide 7 | Self-Attention — Q, K, V All Come From the Same Sequence**
- Key takeaway: Let every token look at every other token in the same sentence — this is how Transformers capture context without recurrence.
- Left: equation `Q = XW_Q,  K = XW_K,  V = XW_V` where `X` is the input sequence.
- Right: a 4×4 attention heatmap for the sentence *"The cat sat down"* — the cell for *"sat"* lights up on *"cat"*.
- One-line why-it-matters: "Distance in the sentence no longer matters — position 0 and position 999 are one matmul apart."
- Visual: heatmap + input-token row/column labels.

**Slide 8 | Masked Self-Attention — Don't Peek at the Future**
- Key takeaway: During language-model training we must hide future tokens; a **causal mask** enforces this with an upper-triangular matrix of `-inf`.
- Step-by-step visualisation of predicting "I like you":
  - row 1: `<BOS> ? ? ?` — can only see `<BOS>`
  - row 2: `<BOS> I ? ?` — can see up to "I"
  - row 3: `<BOS> I like ?`
  - row 4: `<BOS> I like you`
- Formula hint: `scores += triu(-inf, diagonal=1)` then softmax.
- Why it exists: lets us train all positions *in parallel* instead of token-by-token.
- Visual: 4×4 grid with lower triangle coral-filled, upper triangle grey with `-∞`.

**Slide 9 | Cross-Attention — Letting the Decoder Read the Encoder**
- Key takeaway: Cross-attention is the bridge between two sequences — **Q** comes from the decoder, **K** & **V** come from the encoder output (called *memory*).
- Diagram: two parallel streams, encoder output (green) feeds K/V into the decoder block; decoder hidden state (coral) feeds Q.
- Contrast table (2 rows):
  | Mechanism | Q from | K,V from |
  | Self-attention | same sequence | same sequence |
  | Cross-attention | **decoder** | **encoder memory** |
- One-liner use case: "Translation — the French decoder attends to the English encoder."

**Slide 10 | Multi-Head Attention — Many Perspectives at Once**
- Key takeaway: Run attention `h` times in parallel with different projections, concatenate, then project — each head learns a different kind of relationship (syntax, coreference, etc.).
- Formula block:
  $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O$$
  $$\text{head}_i = \text{Attention}(QW^Q_i,\; KW^K_i,\; VW^V_i)$$
- Mini code snippet (core shape manipulation only, ~10 lines):
  ```python
  # split: (B, T, d) -> (B, h, T, d/h)
  xq = self.wq(q).view(B, T, h, d_k).transpose(1, 2)
  xk = self.wk(k).view(B, T, h, d_k).transpose(1, 2)
  xv = self.wv(v).view(B, T, h, d_k).transpose(1, 2)
  out = attention(xq, xk, xv, mask)
  # merge: (B, h, T, d/h) -> (B, T, d)
  out = out.transpose(1, 2).contiguous().view(B, T, d)
  return self.wo(out)
  ```
- Side note: `d_model = 512, h = 8 → each head has d_k = 64`.
- Visual: 8 small coral heads feeding into a single concat-and-project block.

---

### Part 2 — Putting a Transformer Together (Slides 11 – 14)

**Slide 11 | Positional Encoding — Attention Has No Sense of Order**
- Key takeaway: Attention is permutation-equivariant — "I love you" and "you love I" look identical — so we **inject position** by adding a position-specific vector to every embedding.
- Formulas (side by side):
  $$PE_{(pos,2i)} = \sin\!\left(\tfrac{pos}{10000^{2i/d}}\right) \quad PE_{(pos,2i+1)} = \cos\!\left(\tfrac{pos}{10000^{2i/d}}\right)$$
- Three design properties, single-line each:
  1. Deterministic — position 3 is always the same vector
  2. Relative distances preserved — `PE(pos+k)` is a linear combo of `PE(pos)`
  3. Generalizes beyond training length (sin/cos are defined everywhere)
- Visual: the classic sin/cos heat-stripe `(seq_len × d_model)` with alternating colors.

**Slide 12 | The Three Glue Components: FFN · LayerNorm · Residual**
- Key takeaway: Between attention layers, three small pieces do the heavy lifting of stable training.
- Three mini-cards side by side:
  - **FFN (2-layer MLP)** — `Linear(d→4d) → ReLU → Linear(4d→d)`. Mixes features *within* each position; attention mixes *across* positions.
  - **LayerNorm** — normalize across the feature dimension for each token, stabilizing activations across deep stacks. Jargon gloss: "rescales each token's vector to mean 0, variance 1."
  - **Residual** — `x + Sublayer(x)`. Guarantees the original signal (and the position encoding!) survives to the top of the stack.
- Short equation row: `x = x + Attention(LN(x));  x = x + FFN(LN(x))` — the Pre-Norm pattern used by modern LLMs.
- Visual: 3-column card grid; each card has one icon, one formula, one sentence.

**Slide 13 | The Full Picture — Encoder-Decoder Stack**
- Key takeaway: An Encoder-Decoder Transformer is `N=6` of each block type, connected by cross-attention at every decoder layer.
- Central architecture diagram (re-drawn from the paper):
  - Left column (Encoder × 6): `Self-Attn → Add&Norm → FFN → Add&Norm`
  - Right column (Decoder × 6): `Masked Self-Attn → Add&Norm → Cross-Attn(mem=enc_out) → Add&Norm → FFN → Add&Norm`
  - Top: `Linear → Softmax → next-token probabilities`
- Small data-flow callouts:
  - Source tokens → Embedding + PE → Encoder stack → **memory**
  - Target tokens → Embedding + PE → Decoder stack → logits
- Visual: two vertical stacks with an arrow labelled "memory (K,V)" spanning across.

**Slide 14 | Wiring It Up with `torch.nn.Transformer`**
- Key takeaway: PyTorch exposes the whole thing as one class — the tricky part is getting the **masks** right.
- Left panel, minimal usage:
  ```python
  model = nn.Transformer(d_model=512, nhead=8,
                         num_encoder_layers=6, num_decoder_layers=6,
                         dim_feedforward=2048)

  out = model(src, tgt,
              src_key_padding_mask=src_pad,   # batch: hide <pad> tokens
              tgt_mask=causal_mask,           # triangular: hide future tokens
              memory_key_padding_mask=src_pad) # cross-attn: hide <pad> in memory
  ```
- Right panel, three mask types as a compact table:
  | Mask | Shape | Purpose |
  | `src_key_padding_mask` | `(B, S)` | hide `<pad>` in the encoder |
  | `tgt_mask` (causal) | `(T, T)` upper triangular | prevent decoder from peeking ahead |
  | `memory_key_padding_mask` | `(B, S)` | hide `<pad>` in cross-attention |
- Bottom caption: "At inference time the decoder is still **autoregressive** — one token at a time — even though training is parallel."

---

### Part 3 — What Transformer Spawned (Slides 15 – 16)

**Slide 15 | Three Paradigms from One Architecture**
- Key takeaway: By keeping only half of the Transformer — or both halves — you get three families of models that dominate modern NLP.
- Three-column comparison table (tight, single row each):
  | | **Encoder-only** | **Decoder-only** | **Encoder-Decoder** |
  | Flagship | **BERT** (2018) | **GPT** (2018 → ChatGPT) | **T5** (2019) |
  | Attention | bidirectional self-attn | causal self-attn | enc: bidirectional; dec: causal + cross |
  | Pre-train task | MLM (masked fill-in) | CLM (next-token) | span corruption |
  | Best at | understanding (classify, NER, QA) | generation (chat, code) | seq2seq (translate, summarize) |
- Jargon callouts: *MLM* — "fill-in-the-blank"; *CLM* — "predict the next word".
- Visual: three stacked icons (🅑 🅖 🅣 — rendered as styled letter badges in coral/mint) above each column; the full Transformer silhouette above "Encoder-Decoder" is complete, while BERT and GPT show half-silhouettes.

**Slide 16 | Key Takeaways**
- Top line: "Attention is all you need — literally the only operation you need to build the best sequence models we have."
- Four bullets, each a one-sentence memorable statement:
  1. **Attention = soft dictionary lookup**: similarity → softmax → weighted sum of values.
  2. **Multi-head**: run attention in parallel with different projections to capture multiple kinds of relationships.
  3. **Architecture is modular**: Positional Encoding + Self-Attn + FFN + LayerNorm + Residual — stack to taste.
  4. **Paradigms**: BERT (understand), GPT (generate), T5 (both) — all are edits of the same core.
- Call to action: "Next step — implement `attention(q, k, v)` from scratch in ~10 lines and watch a tiny Transformer memorize a toy dataset."
- Visual: single centered oversized formula `softmax(QKᵀ/√d_k)·V` in coral, with a subtle radial glow behind it.

---

## Content & Tone Guidelines

- Lead every slide with a bolded single-sentence takeaway; bullets elaborate but never replace it.
- When any DL concept first appears (softmax, embedding, gradient, logits, LayerNorm, etc.) wrap a dotted-border one-line gloss. Use each gloss only on the first occurrence.
- Use math notation where it's tighter than prose (formulas, shape annotations).
- Keep code snippets minimal — only Slide 6 (attention core, ~8 lines) and Slide 10 (multi-head shape reshuffle, ~10 lines). No other code.
- Diagrams should be hand-built with HTML/CSS/SVG, not raster images, so they re-style cleanly.
- Tone is teacherly, not flashy — no hype words, no emoji in body (icon glyphs as decoration only).

---

## Data Sources & References

- `resources/llm-from-scratch/chapter2/第二章 Transformer架构.md` — primary source for attention derivation, self/masked/multi-head attention, positional encoding, FFN/LayerNorm/residual, full Transformer assembly.
- `resources/transformer-architecture/readme.md` — primary source for the PyTorch `nn.Transformer` API walkthrough, mask taxonomy (src_key_padding / tgt_mask / memory_key_padding), Encoder/Decoder layer composition.
- `resources/llm-from-scratch/chapter3/第三章 预训练语言模型.md` — primary source for the three-paradigm comparison (BERT / GPT / T5, MLM vs CLM).
- `resources/transformer-architecture/images/trans-img-*.png` — available as optional visual references for the PyTorch section if we want to embed rather than re-draw.

All source content is in Chinese; deck content must be translated and condensed into presentation-ready English.

---

## Deliverables

- Output: 16 HTML slides in `presentation/slides/` (`slide1.html` … `slide16.html`)
- Viewer: `presentation/index.html` built from `viewer-template.html`
- Each slide: standalone HTML, 1280×720 canvas, 16:9 aspect ratio
- Optional: `PRESENTATION_SUMMARY.md` + `PRESENTATION_SCRIPT.md` (recommended — deck is > 8 slides and audience benefits from speaker notes)
