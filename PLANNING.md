**Task**: Create an English technical sharing deck titled "Understanding Transformer Architecture: Attention Is All You Need"
**Slide count**: 15 ± 2
**Language**: English
**Audience**: Software engineers with limited deep learning background
**Goals**:
- Build intuition for Attention from scratch — dictionary-lookup analogy to Scaled Dot-Product Attention, covering Self-Attention, Masked Self-Attention, Cross-Attention, and Multi-Head Attention (~50% of slides)
- Explain the overall Transformer Encoder-Decoder architecture — Positional Encoding, FFN, LayerNorm, Residual Connection, causal mask, PyTorch nn.Transformer API
- Summarize downstream paradigms — Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5), with key takeaways
**Style**: Restrained, high information density, diagram-heavy, presentation-friendly. Code only for attention formula and multi-head attention core logic. Plain-language jargon explanations on first use. One clear takeaway per slide.

---

## Visual & Layout Guidelines

- **Overall tone**: Warm, technical, generous whitespace — diagrams and formulas over text walls
- **Background**: `#F7F4EF` (warm beige)
- **Primary text**: `#2B2A27` (deep charcoal)
- **Secondary text**: `#5D4E42` (medium brown)
- **Accent (primary)**: `#E07A59` (coral orange — borders, key formulas, highlights)
- **Accent (secondary)**: `#9FD3B8` (mint green — secondary highlights, diagram fills)
- **Typography**: Inter 600-700 for headings (48-64px), Inter 400-500 for body (18-28px), Source Code Pro for code
- **Per-slide rule**: 1 key point + up to 4 supporting bullets; avoid large blocks of text
- **Footer**: page number on content slides (bottom-right)
- **Diagrams**: Use CSS/HTML-rendered diagrams (boxes, arrows, grids) instead of external images. Use color-coded boxes for architectural components. Mathematical formulas rendered as styled HTML/CSS text.
- **Jargon policy**: First occurrence of any DL term (softmax, embedding, gradient, etc.) gets a parenthetical one-sentence plain-language explanation.

---

## Slide-by-Slide Outline

### Slide 1 | Cover
- Title: "Understanding Transformer Architecture"
- Subtitle: "Attention Is All You Need"
- Footer: presenter name placeholder, February 2026
- Visual element: Decorative radial gradient accents (coral + mint), clean centered layout
- Pattern: Cover Slide

---

### Slide 2 | Agenda
- Three-part structure:
  1. **Attention Mechanism** — From dictionary lookup to Multi-Head Attention (Slides 3–9)
  2. **Transformer Architecture** — Positional Encoding, Encoder, Decoder, and how they wire together (Slides 10–13)
  3. **Downstream Paradigms** — BERT, GPT, T5 and beyond (Slide 14)
- Visual element: Numbered list with coral accent bars, proportion indicators showing ~50% / ~35% / ~15% allocation
- Pattern: Agenda / List Slide

---

### Slide 3 | Why Attention? The Sequential Bottleneck
- **Key point**: RNNs process tokens one by one — this kills parallelism and makes long-range dependencies hard to capture. Attention solves both problems.
- Supporting bullets:
  - RNN (Recurrent Neural Network — a network that processes sequences step-by-step, using previous output as input): sequential computation blocks GPU parallelism
  - Long-range dependency decay: token at position 1 struggles to influence token at position 100
  - Attention computes **all-pair relationships in one step** — O(1) path length between any two tokens
- Visual element: Side-by-side diagram — Left: RNN chain (sequential arrows, fading signal); Right: Attention (fully-connected bipartite graph, all tokens connect to all tokens)
- Takeaway: "Attention replaces sequential processing with parallel all-pair computation."
- Source: chapter2/2.1.1 (RNN limitations motivation)

---

### Slide 4 | Attention = Weighted Dictionary Lookup
- **Key point**: Attention is a soft dictionary lookup — given a Query, compute similarity to each Key, then return a weighted sum of Values.
- Supporting bullets:
  - Hard lookup: query "apple" → exact match → value 10
  - Soft lookup: query "fruit" → weighted match → 0.6×apple + 0.4×banana + 0×chair = 8
  - The weights are called **attention scores** — they reflect how much each Key relates to the Query
  - **Softmax** (a function that converts raw scores into probabilities summing to 1): turns raw similarity scores into a probability distribution
- Visual element: Two-panel diagram — Left: traditional dict `{"apple":10, "banana":5, "chair":2}` with exact match arrow; Right: soft attention with weighted arrows and probability labels (0.6, 0.4, 0.0)
- Takeaway: "Attention = soft dictionary lookup: similarity-weighted retrieval."
- Source: chapter2/2.1.2 (dictionary analogy from HappyLLM)

---

### Slide 5 | Scaled Dot-Product Attention
- **Key point**: The core formula — compute Q·K similarity, scale, softmax, then weight V.
- Content:
  - Step 1: Compute similarity — `scores = Q · K^T` (dot product — a way to measure how similar two vectors are)
  - Step 2: Scale — `scores = scores / √d_k` (prevents softmax from producing extreme values when d_k is large)
  - Step 3: Normalize — `weights = softmax(scores)` → probabilities that sum to 1
  - Step 4: Aggregate — `output = weights · V` → weighted combination of values
  - **Formula**: `Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V`
- Visual element: Horizontal flow diagram showing the four steps with matrix shapes annotated: Q(n×d_k) × K^T(d_k×n) → Scale → Softmax → ×V(n×d_v) → Output(n×d_v). Formula displayed prominently in accent color.
- Code snippet (minimal):
  ```python
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  weights = scores.softmax(dim=-1)
  output = torch.matmul(weights, value)
  ```
- Takeaway: "Dot-product + scale + softmax + weighted sum = the entire attention computation."
- Source: chapter2/2.1.2–2.1.3, transformer-architecture/readme (Scaled Dot-Product section)

---

### Slide 6 | Self-Attention: Q = K = V from the Same Sequence
- **Key point**: In self-attention, Q, K, and V all come from the same input sequence — each token learns its relationship to every other token.
- Supporting bullets:
  - Input sequence X passes through three separate linear projections → Q, K, V (same source, different learned transformations)
  - Each token attends to all other tokens (including itself)
  - Result: a context-aware representation where "bank" near "river" gets different attention than "bank" near "money"
  - **Embedding** (a dense vector that represents a word — similar words have similar vectors): input embeddings are enriched with contextual information
- Visual element: Diagram — single input sequence X splits into three paths (W_q, W_k, W_v) → Q, K, V → Attention block → contextual output. Show a concrete example: sentence "The cat sat on the mat" with attention weight heatmap between tokens.
- Code: `attention(x, x, x)  # Q=K=V=same input`
- Takeaway: "Self-attention: every token queries every other token in the same sequence."
- Source: chapter2/2.1.4

---

### Slide 7 | Masked Self-Attention: No Peeking at the Future
- **Key point**: In autoregressive generation, each token can only attend to itself and previous tokens — enforced by a causal mask (upper-triangular matrix of -∞).
- Supporting bullets:
  - Generation is sequential: predict token t based only on tokens 1..t-1
  - Without mask: model sees the answer → trivial identity mapping (cheating)
  - Mask fills future positions with -∞ → after softmax, those weights become 0
  - Enables **parallel training**: all positions can train simultaneously with the same mask applied
- Visual element: Attention matrix visualization (5×5 grid). Lower triangle = colored cells (allowed attention). Upper triangle = crossed-out / grayed cells (masked). Show the -∞ → softmax → 0 pipeline on the side.
- Takeaway: "Causal mask = upper-triangular -∞ matrix → prevents future information leakage."
- Source: chapter2/2.1.5, transformer-architecture/readme (causal mask section)

---

### Slide 8 | Cross-Attention: Bridging Two Sequences
- **Key point**: In cross-attention, Q comes from the decoder while K and V come from the encoder — this is how the decoder "reads" the encoder's understanding.
- Supporting bullets:
  - Self-attention: Q, K, V all from same source → models internal relationships
  - Cross-attention: Q from decoder output, K & V from encoder output (memory)
  - Analogy: decoder asks questions (Q), encoder's representation provides the answers (K, V)
  - Used in the Decoder block between masked self-attention and FFN
- Visual element: Two-stream diagram — Encoder output (green box) → splits into K, V; Decoder intermediate output (orange box) → becomes Q. Both flow into an Attention block. Clear color distinction between encoder stream and decoder stream.
- Takeaway: "Cross-attention: the decoder queries the encoder's memory — Q from decoder, K+V from encoder."
- Source: chapter2/2.2.6, transformer-architecture/readme (cross-attention section, _sa_block vs _mha_block)

---

### Slide 9 | Multi-Head Attention: Many Perspectives at Once
- **Key point**: Instead of one attention computation, run h parallel attention "heads" — each captures different relationship patterns, then concatenate and project.
- Supporting bullets:
  - Single head = single relationship pattern. Multiple heads = richer representation.
  - Split d_model into h heads: each head has dimension d_k = d_model / h (e.g., 512/8 = 64)
  - Each head independently computes attention, then results are concatenated and linearly projected
  - Formula: `MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O` where `head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)`
- Visual element: Diagram showing: Input → split into h parallel Attention blocks → Concat → Linear → Output. Show different heads capturing different patterns (e.g., one head learns syntactic structure, another learns semantic similarity).
- Code snippet (core logic):
  ```python
  # Reshape: (B, T, d_model) → (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
  xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
  # ... same for xk, xv
  scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
  output = torch.matmul(scores.softmax(dim=-1), xv)
  # Merge heads: (B, n_heads, T, head_dim) → (B, T, d_model)
  output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
  ```
- Takeaway: "Multi-head attention = parallel sub-attentions, each specializing in different patterns."
- Source: chapter2/2.1.6, transformer-architecture/readme (multi-head design), code/transformer.py

---

### Slide 10 | Positional Encoding: Teaching Order to the Transformer
- **Key point**: Attention treats all token positions equally — positional encoding injects sequence order via sin/cos functions added to embeddings.
- Supporting bullets:
  - Without PE: "I love you" and "you love I" produce identical attention (no position awareness)
  - Design requirements: (1) deterministic per position, (2) consistent relative distances, (3) generalizes to unseen lengths
  - Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
  - Key property: PE(pos+k) can be expressed as a linear combination of PE(pos) — enables generalization to longer sequences
  - **Residual connections** (adding the input directly to the output of a sublayer) preserve PE information through all layers: x_N = x_0 + Σ F_i(x_{i-1})
- Visual element: Heatmap showing positional encoding values (position × dimension), with sin/cos wave patterns visible. Small formula displayed below.
- Takeaway: "Sin/cos positional encoding: deterministic, captures relative position, generalizes to any length."
- Source: chapter2/2.3.2, transformer-architecture/readme (PE section with three design principles)

---

### Slide 11 | The Encoder: Self-Attention + FFN + Residual + LayerNorm
- **Key point**: Each Encoder layer = Multi-Head Self-Attention → Add & LayerNorm → FFN → Add & LayerNorm. Stack N=6 layers.
- Supporting bullets:
  - **Self-Attention**: captures inter-token relationships (position ↔ position)
  - **FFN** (Feed-Forward Network — two linear layers with activation): processes each position independently, fuses feature dimensions. 512 → 2048 → 512
  - **LayerNorm** (normalizes activations within each sample to stabilize training): applied before each sub-layer (Pre-Norm)
  - **Residual Connection**: `output = x + SubLayer(LayerNorm(x))` — prevents gradient vanishing in deep networks
  - N=6 identical layers, each with shared architecture but independent parameters
- Visual element: Vertical stack diagram of one Encoder layer: Input → [LayerNorm → Multi-Head Self-Attention → + (residual)] → [LayerNorm → FFN → + (residual)] → Output. Side annotation: "×6 layers". Color-code each component.
- Takeaway: "Encoder layer = attention (between positions) + FFN (within positions) + residual shortcuts."
- Source: chapter2/2.2.5, transformer-architecture/readme (EncoderLayer section), code/transformer.py (EncoderLayer class)

---

### Slide 12 | The Decoder: Three-Module Pipeline
- **Key point**: Each Decoder layer has three sub-modules — Masked Self-Attention → Cross-Attention → FFN — each wrapped in Add & LayerNorm.
- Supporting bullets:
  - **Module 1 — Masked Self-Attention**: target sequence attends to itself (with causal mask)
  - **Module 2 — Cross-Attention**: Q = decoder output, K & V = encoder memory → "read" the source
  - **Module 3 — FFN**: same position-wise feed-forward as encoder
  - Decoder is autoregressive: at inference, generates one token at a time, feeding output back as input
- Visual element: Vertical diagram of one Decoder layer (three sub-blocks stacked): Input → [Masked Self-Attn + Add&Norm] → [Cross-Attn + Add&Norm] → [FFN + Add&Norm] → Output. Arrow from Encoder output entering at Cross-Attn. Highlight the decoder's extra module vs encoder.
- Takeaway: "Decoder = masked self-attention + cross-attention (reads encoder) + FFN."
- Source: chapter2/2.2.6, transformer-architecture/readme (Decoder section, two attention mechanisms)

---

### Slide 13 | Full Transformer: Wiring It All Together
- **Key point**: The complete Transformer: source → Embedding + PE → Encoder ×N → memory; target → Embedding + PE → Decoder ×N → Linear → Softmax → output probabilities.
- Supporting bullets:
  - PyTorch: `nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048)`
  - Four core classes: `TransformerEncoderLayer`, `TransformerEncoder`, `TransformerDecoderLayer`, `TransformerDecoder`
  - Forward: encoder(src, src_padding_mask) → memory; decoder(tgt, memory, tgt_mask) → output
  - Masks: `src_padding_mask` (handles variable-length inputs), `tgt_mask` (causal upper-triangular)
- Visual element: Full architecture diagram (classic "Attention Is All You Need" figure recreated in CSS). Left: Encoder stack. Right: Decoder stack. Arrows showing data flow. Annotated with PyTorch class names. Show the final Linear + Softmax output layer.
- Takeaway: "4 classes, 5 key hyperparameters — that's the entire Transformer in PyTorch."
- Source: transformer-architecture/readme (PyTorch API walkthrough, forward function), code/transformer.py

---

### Slide 14 | Transformer's Legacy: Three Paradigms
- **Key point**: The Transformer architecture spawned three dominant paradigms — Encoder-only (BERT), Decoder-only (GPT), and full Encoder-Decoder (T5).
- Supporting content (comparison table):

  | | Encoder-Only (BERT) | Decoder-Only (GPT) | Encoder-Decoder (T5) |
  |---|---|---|---|
  | **Architecture** | Stacked Encoders | Stacked Decoders | Full Transformer |
  | **Attention** | Bidirectional Self-Attn | Masked (Causal) Self-Attn | Self + Cross + Masked |
  | **Pre-training** | MLM (fill in the blank) | CLM (predict next token) | Text-to-text MLM |
  | **Strength** | Understanding (NLU) | Generation (NLG) | Both (Seq2Seq) |
  | **Key Insight** | See all context at once | Natural autoregressive generation | Unified text-to-text format |
  | **Scale Trend** | 110M–340M | 175B → trillions | 220M–11B |

- Visual element: Three-column comparison table with color-coded headers. Architecture mini-diagrams in each column header showing which blocks are used (Encoder blocks highlighted for BERT, Decoder blocks for GPT, both for T5).
- Takeaway: "Same Transformer DNA, three architectures — each optimized for different tasks."
- Source: chapter3/3.1 (BERT), 3.2 (T5), 3.3 (GPT)
- Pattern: Comparison Table

---

### Slide 15 | Key Takeaways & Q&A
- Summary statement (4 core takeaways):
  1. **Attention** is a learnable soft dictionary lookup — it replaces sequential processing with parallel all-pair computation
  2. **Multi-Head Attention** captures diverse relationship patterns by running parallel sub-attentions
  3. **The Transformer** wires attention with FFN, residuals, and LayerNorm into a modular Encoder-Decoder architecture
  4. **Three paradigms** (BERT / GPT / T5) are all specializations of the same Transformer building blocks
- Call to action: "Questions? Discussion?"
- Visual element: Four takeaway cards in a 2×2 grid, each with an accent-colored left border and bold label
- Pattern: Closing / Q&A Slide

---

## Content & Tone Guidelines

- **Plain language first**: Every DL concept gets a one-sentence parenthetical explanation on first use. Examples:
  - Softmax: "a function that converts raw scores into probabilities summing to 1"
  - Embedding: "a dense vector that represents a word — similar words have similar vectors"
  - Gradient: "the signal used to update model weights during training"
  - Residual connection: "adding the input directly to the output of a sublayer"
- **Code sparingly**: Only two slides have code (Slide 5: attention formula, Slide 9: multi-head reshape/compute). Code should be syntax-highlighted with Source Code Pro font.
- **Formulas**: Display key formulas (Attention, Multi-Head, PE) prominently in accent color. Keep mathematical notation clean and large.
- **Diagrams over text**: Every content slide has a primary visual element. Prefer CSS-rendered diagrams (colored boxes + arrows + labels) over text descriptions.
- **Progressive disclosure**: Build from intuition (dictionary lookup) → formal formula → variants (self, masked, cross, multi-head) → full architecture → applications.
- **Restrained tone**: No exclamation marks, no hype language. Let the content speak.

---

## Data Sources & References

| Resource | Used in Slides | Usage |
|----------|---------------|-------|
| `resources/llm-from-scratch/chapter2/第二章 Transformer架构.md` | 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | Core content: attention mechanism analogy, formulas, self/masked/cross/multi-head attention, Encoder-Decoder components, positional encoding |
| `resources/llm-from-scratch/chapter2/code/transformer.py` | 9, 13 | Code snippets: MultiHeadAttention class, Transformer class structure |
| `resources/transformer-architecture/readme.md` | 5, 7, 8, 10, 11, 12, 13 | PyTorch API perspective: nn.Transformer parameters, forward flow, mask design, encoder/decoder layer implementation |
| `resources/transformer-architecture/images/` | Reference only | Architecture diagrams (Chinese annotations — will recreate as English CSS diagrams) |
| `resources/llm-from-scratch/chapter3/第三章 预训练语言模型.md` | 14 | Downstream paradigms: BERT architecture & MLM, GPT architecture & CLM, T5 architecture & text-to-text |

---

## Deliverables

- Output: 15 HTML slides in `presentation/slides/`
- Viewer: `presentation/index.html` (from viewer template)
- Each slide: standalone HTML, 1280×720, 16:9 aspect ratio
- Optional: `PRESENTATION_SUMMARY.md`, `PRESENTATION_SCRIPT.md` (speaker notes)
