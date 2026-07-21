# epilogue — everything else is efficiency

> The question this essay answers: **what actually separates your 4,928 floats from ChatGPT?**

```
docs -> tokenize -> model -> loss -> backward -> update -> sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
you are here: all of it — this pipeline IS the production pipeline
```

microgpt's header makes a big claim: *"This file is the complete algorithm.
Everything else is just efficiency."* Having now run every line of it, you've
earned the itemized version of that claim. Here is your code, construct by
construct, mapped to what a frontier lab runs:

| you ran | production runs | what changed |
|---|---|---|
| 38 characters, `uchars.index(ch)` | BPE tokenizers, ~100–200k tokens | chunks instead of chars: shorter sequences, more meaning per position. Your `q` → `u` row is their `" the"` |
| scalar `Value`, one number each | tensors: the same graph over million-entry blocks | the 45× overhead you measured at rung 2 is what vanishes when bookkeeping amortizes over blocks — and GPUs then add ~6 more orders of magnitude of arithmetic throughput over interpreted scalar Python |
| `linear()`: two nested `for` loops | fused matmul kernels, FlashAttention | *identical mathematics*; the memory access order is redesigned so the chip never waits. Attention's formula — your rung 3 — is untouched |
| `keys.append(k); values.append(v)` | paged KV caches serving thousands of chats | rung 6 measured why: without the cache, work per token grows with everything said so far |
| `wpe`, a 40-row table | RoPE: positions as rotations, no table | your rung 3 exercise hit the wall (`IndexError` at row 40); rotations don't have rows to run out of — that's "1M-token context" |
| 1 layer, 4 heads, 16 dims, 4,928 params | ~10²-layer, mixture-of-experts, ~10¹¹–10¹² params | the `for li in range(n_layer)` loop, turned up; MoE = most of the MLP asleep per token |
| `docs[step % len(train_docs)]`, one doc | batches of thousands of sequences, thousands of GPUs | your gradient was one document's opinion; theirs averages thousands per step |
| Adam with `m`, `v`, decay | Adam with `m`, `v`, warmup + cosine decay | embarrassingly close to identical. Your rung 5 is the frontier optimizer |
| 488 training docs, ~2 epochs, memorization gauge | trillions of tokens, *less than one* epoch | the regime flips: frontier models barely repeat data. Your overfitting lab is the tabletop model of the field's data-wall debate |
| `temperature`, `random.choices` | the same loop, plus top-p and friends | sampling in every chatbot is your rung 6 ladder, productionized |

Two things on that list deserve a last word.

**The KV cache is the era's workhorse.** The two humble lists you appended to at rung 3 —
and killed/resurrected at rung 6 — are, scaled up, the central object of the
inference industry. Serving systems page them, share them across requests,
quantize them. When a provider quotes you a price per token, the shape of that
price is the shape of your `generate_kv_report(use_cache=False)` measurement.

**What is genuinely *not* in this repo** is what happens after pretraining:
supervised fine-tuning and reinforcement learning. But even those reuse your
machinery — SFT is literally your training loop pointed at curated
conversations (better documents, same loss); RL keeps the same forward pass
and reweights updates by reward instead of by next-token surprise. New data,
new objective. No new engine.

So the honest division is this: **the algorithm** — tokenize, embed,
communicate (attention), compute (MLP), score (loss), blame (backward), step
(Adam), sample — you have now written, run, broken, and repaired at every
layer. **The efficiency** — tensors, kernels, parallelism, serving — is a
decade of magnificent engineering that changes no formulas. And **the wisdom**
— what these models know — is data and scale, not mechanism.

Karpathy closes his post with the Feynman line that has always been the
course's real syllabus: *what I cannot create, I do not understand.* You
created one. It lives in `out/model.json`, it has opinions about idea names,
and every one of its 4,928 numbers is there because a gradient you can derive
by hand put it there.

Where to next, if you want more:

- his [microgpt post](http://karpathy.github.io/2026/02/12/microgpt/) — reread
  it now; it reads differently once you've run every line
- [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — the video
  lineage this course descends from (micrograd → makemore → nanoGPT)
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the same model you built,
  in PyTorch, trainable on real text with real hardware
- or stay here: `train4.py` with `n_layer = 2`, a BPE rung, a batching rung —
  the ladder doesn't end because the gist did

---

Back to [README](../README.md) · the ladder: [0](train0.md) [1](train1.md)
[2](train2.md) [3](train3.md) [4](train4.md) [5](train5.md) [6](train6.md)
[7](train7.md) · [glossary](../GLOSSARY.md)
