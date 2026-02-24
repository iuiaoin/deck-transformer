# Speaker Notes

## Slide 1 — Cover
Welcome everyone. Today we'll build intuition for the Transformer architecture from the ground up. The goal is to make attention mechanisms feel intuitive, not magical. By the end, you'll understand every component in the "Attention Is All You Need" paper.

## Slide 2 — Agenda
We have three parts. The first and biggest section covers attention — we'll start from a dictionary lookup analogy and work up to multi-head attention. Then we'll wire these components into the full Transformer architecture. Finally, we'll see how the Transformer spawned BERT, GPT, and T5.

## Slide 3 — Why Attention?
Before Transformers, RNNs dominated NLP. But RNNs have two fundamental problems: they process tokens sequentially, which blocks GPU parallelism, and signals fade over long distances. Attention solves both — it computes all-pair relationships in a single step, with O(1) path length between any two tokens.

## Slide 4 — Dictionary Lookup
Here's the key intuition. A regular dictionary does exact-match lookup. But what if your query is "fruit" and your keys are "apple", "banana", "chair"? Attention does a soft lookup — it computes similarity between the query and each key, normalizes into probabilities via softmax, then returns a weighted sum of values. That's the entire idea.

## Slide 5 — Scaled Dot-Product
Now let's formalize this. The formula has four steps: compute dot-product similarity between Q and K, scale by square root of dimension to prevent softmax saturation, normalize with softmax, and take the weighted sum of V. The implementation is just 3 lines of PyTorch. The scaling factor is crucial — without it, large dimensions cause softmax to produce near-one-hot distributions, killing gradients.

## Slide 6 — Self-Attention
In self-attention, Q, K, and V all come from the same input. The input passes through three separate linear projections to produce Q, K, V — same source, different learned transformations. This lets each token compute its relationship to every other token. Notice the attention heatmap — "cat" attends strongly to itself, "sat", and "mat", capturing semantic relationships.

## Slide 7 — Masked Self-Attention
In generation tasks, the model must predict the next token without seeing the future. We enforce this with a causal mask — an upper-triangular matrix filled with negative infinity. After softmax, negative infinity becomes zero probability. The elegant part: this mask enables parallel training, since all positions can train simultaneously — unlike sequential RNN training.

## Slide 8 — Cross-Attention
Cross-attention bridges two different sequences. The key difference from self-attention: Q comes from the decoder, while K and V come from the encoder output. Think of it as the decoder asking questions, and the encoder providing answers. In machine translation, the encoder understands the source language, and the decoder queries that understanding while generating the target.

## Slide 9 — Multi-Head Attention
A single attention head captures one type of relationship. Multi-head attention runs h parallel attention computations, each specializing in different patterns — syntactic structure, semantic similarity, proximity, etc. We split the d_model dimension into h heads, compute attention independently in each, then concatenate and project back. Crucially, this costs the same total FLOPs as single-head attention.

## Slide 10 — Positional Encoding
Attention is position-agnostic — "I love you" and "you love I" would get identical attention weights. We fix this by adding sinusoidal positional encodings to the input embeddings. The sin/cos design satisfies three key properties: deterministic encoding per position, consistent relative distances, and generalization to unseen sequence lengths. Residual connections ensure this position information survives through all layers.

## Slide 11 — The Encoder
Each encoder layer has two sub-layers: multi-head self-attention followed by a feed-forward network. Each sub-layer has a residual connection and LayerNorm. Self-attention captures relationships between positions; FFN processes each position independently, fusing feature dimensions. We stack 6 of these identical layers. Note the Pre-Norm variant is used in practice — LayerNorm before the sub-layer, not after.

## Slide 12 — The Decoder
The decoder adds a third sub-layer: cross-attention between the masked self-attention and FFN. Module 1 (masked self-attention) models the target sequence's internal relationships. Module 2 (cross-attention) reads the encoder's output. Module 3 (FFN) does the same position-wise processing. At inference, generation is autoregressive — one token at a time.

## Slide 13 — Full Transformer
Here's the complete picture. Source sequence goes through embedding, positional encoding, then N encoder layers to produce memory. Target goes through embedding, PE, then N decoder layers that also receive encoder memory via cross-attention. Finally, a linear layer and softmax produce output probabilities. In PyTorch, this is just 4 classes and 5 hyperparameters.

## Slide 14 — Three Paradigms
The Transformer spawned three dominant architectures. BERT takes only the encoder — bidirectional attention, MLM training, excels at understanding tasks. GPT takes only the decoder — causal attention, CLM training, dominates generation. T5 keeps both — text-to-text format, handles everything. Today's LLMs (GPT-4, Claude, LLaMA) are all decoder-only, following the GPT paradigm at massive scale.

## Slide 15 — Takeaways & Q&A
Four key takeaways: attention is learnable soft dictionary lookup replacing sequential processing; multi-head attention captures diverse patterns at no extra cost; the Transformer wires attention, FFN, residuals, and LayerNorm into a modular architecture; and BERT, GPT, T5 are all specializations of the same building blocks. Happy to take questions.
