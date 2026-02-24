Create an English presentation deck for a technical sharing session titled 'Understanding Transformer Architecture: Attention Is All You Need'. Target length: 15 ± 2 slides. Audience: software engineers with limited deep learning background.

Goals:

1. Build intuition for the Attention mechanism from scratch — from dictionary-lookup analogy to Scaled Dot-Product Attention, covering Self-Attention, Masked Self-Attention, Cross-Attention, and Multi-Head Attention (this is the KEY focus, allocate ~50% of slides);
2. Explain the overall Transformer Encoder-Decoder architecture — Positional Encoding, FFN, LayerNorm, Residual Connection, causal mask, and how these components wire together via PyTorch's nn.Transformer API;
3. Summarize how Transformer spawned downstream paradigms — Encoder-only (BERT), Decoder-only (GPT), and full Encoder-Decoder, with key takeaways.

Style: restrained, high information density, diagram-heavy, presentation-friendly. Use code snippets sparingly (only for attention formula implementation and multi-head attention core logic). Prefer architectural diagrams and formulas over walls of text. Minimize jargon — when a deep learning concept (e.g. softmax, embedding, gradient) first appears, give a one-sentence plain-language explanation. Each slide should have a clear single-point takeaway.
