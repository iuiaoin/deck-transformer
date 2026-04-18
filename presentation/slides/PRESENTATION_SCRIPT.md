# Presentation Script — Speaker Notes

Two to four sentences per slide. Aim for ~90–120 seconds each; total runtime ≈ 25 minutes.

---

## Slide 1 — Cover

Welcome. Today we're unpacking the Transformer — the 2017 paper that replaced RNNs and CNNs and now underpins every modern LLM. I'll build it piece by piece so that by the end you could code the core in about ten lines. No deep-learning background assumed.

## Slide 2 — Why Transformer? The 2017 inflection point

Before 2017, sequence models were RNNs: recurrent, strictly left-to-right, and slow to train because step N had to wait for step N−1. Long-range dependencies also faded — the model forgot words that were a paragraph back. The Transformer's bet: drop recurrence entirely, let every token see every other token in one shot via attention, and get both parallel training and constant-path distance for free.

## Slide 3 — The mental model: attention ≈ dictionary lookup

Forget neurons for a second — think Python dict. A normal lookup needs an exact key; attention is the *soft* version: you ask "how similar is my query to each key?", turn those similarities into weights, and return a weighted blend of the values. That one idea is the entire attention mechanism. Everything else is how we compute the similarities and what Q, K, V actually are.

## Slide 4 — Q, K, V — three projections from one input

Each input token X gets three learned linear projections: Query (what I'm looking for), Key (what I advertise), and Value (what I actually deliver). For self-attention they all come from the same sequence; for cross-attention Q comes from the decoder and K/V come from the encoder. The shapes are always `(batch, seq_len, d_k)` — they're just three different views of the same tokens.

## Slide 5 — Scaled Dot-Product Attention

Here's the whole paper in one line: `softmax(QKᵀ / √d_k) V`. Read it left to right — compute similarity with a dot product, scale by `√d_k` so the softmax doesn't saturate, softmax to get a probability distribution, then multiply by V to get a weighted blend. The scaling matters: without it, large `d_k` pushes softmax into one-hot territory and gradients die.

## Slide 6 — Attention in ~10 lines of PyTorch

This is literally the code. `matmul` for Q·Kᵀ, divide by `sqrt(d_k)`, optional mask-fill with `-inf` so those positions softmax to zero, `softmax`, then one more `matmul` with V. Notice the only branching is the mask — which is how we'll express causal attention and padding on the next few slides.

## Slide 7 — Self-Attention: tokens look at themselves

Self-attention means Q, K, V all come from the same sequence — every token attends to every other token, itself included. In the sentence "the cat sat on the mat", "cat" ends up attending strongly to "sat" because the query/key similarity is highest there. This is how the model builds contextual representations without any recurrence.

## Slide 8 — Masked Self-Attention: no peeking at the future

In a decoder, step T can't look at step T+1 — that would leak the answer during training. The fix is an upper-triangular mask of `-inf`s; after softmax those positions become exactly zero. This single trick is what lets us train autoregressive models (GPT, Llama, Claude) in parallel across the whole sequence instead of token by token.

## Slide 9 — Cross-Attention: decoder reads encoder

Cross-attention is where translation happens. The decoder's queries ask questions; the encoder's keys and values answer. Same formula, just Q from one stack and K/V from the other. This is how "memory" flows from source to target in seq2seq models like T5 and the original Transformer.

## Slide 10 — Multi-Head Attention: many perspectives in parallel

One head can only capture one kind of relationship. Multi-head splits `d_model` across `h=8` heads, each with its own Q/K/V projection, runs attention in parallel, then concats and projects back. Different heads learn different things — one watches syntax, another tracks coreference, another handles long-range binding. The reshape trick `.view(B, T, h, d_k).transpose(1,2)` is the one line to memorize.

## Slide 11 — Positional Encoding: teaching the model about order

Attention is permutation-equivariant — shuffle the input and you shuffle the output. That means the raw model has no idea whether "dog bites man" or "man bites dog". Positional encoding injects order by adding a fixed sin/cos signal (different frequency per dimension) to each token embedding before the first layer.

## Slide 12 — FFN · LayerNorm · Residual

Between attention layers, three unglamorous pieces keep deep stacks trainable. FFN is a two-layer MLP per position — attention mixes across tokens, FFN adds depth within each token. LayerNorm stabilizes activations across features; residuals let each sublayer learn a delta rather than rewrite the whole signal — also, they're the reason the positional encoding survives to the top of a 24-layer stack.

## Slide 13 — The full picture: encoder-decoder stack

Stack N=6 of each block type. Each encoder layer is self-attn → add&norm → FFN → add&norm. Each decoder layer adds a middle cross-attention sublayer that queries the shared encoder "memory". Top of the decoder stack goes through Linear → Softmax to produce next-token probabilities.

## Slide 14 — Wiring it up with `torch.nn.Transformer`

PyTorch ships the whole architecture as one class — d_model, nhead, num_layers, and you're done. The tricky part is masks: `src_key_padding_mask` hides `<pad>` in the encoder, `tgt_mask` is the causal triangle for the decoder, and `memory_key_padding_mask` hides padding in cross-attention. Getting the shapes right — `(B,S)` vs `(T,T)` — is the #1 source of silently-wrong Transformer bugs.

## Slide 15 — Three paradigms from one architecture

Keep only the encoder and you get BERT: bidirectional, great at understanding, pre-trained with masked-language-modeling. Keep only the decoder and you get GPT: causal, great at generation, pre-trained with next-token prediction — this is what ChatGPT, Claude, and Llama are. Keep both and you get T5: span corruption, everything reframed as text-to-text.

## Slide 16 — Key takeaways

Attention is a soft dictionary lookup. Multi-head gives you parallel perspectives. The architecture is modular — PE + self-attn + FFN + LN + residual, stack to taste. And the same core gives you the three dominant families in NLP. Your homework: implement `attention(q, k, v)` in ten lines and watch a tiny Transformer memorize a toy dataset — it'll make everything click.
