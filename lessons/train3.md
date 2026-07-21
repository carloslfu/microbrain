# train3 — attention

> The question this rung answers: **how does a model consult its own past?**

```
docs -> tokenize -> model -> loss -> backward -> update -> sample
                    ^^^^^
        you are here: the box learns to look left
```

Everything so far predicted the next character from *one* character. That
ceiling is structural — no amount of training fixes it. This rung rebuilds the
model box so that every position can consult every position before it, with
learned judgment about where to look. After this file, the architecture is,
in Karpathy's words, *structurally a GPT*. Only three gaps to microgpt remain:
one head instead of four, one hard-wired layer instead of a layer loop, and
SGD instead of Adam.

## Walk the code

**Positions become facts.** A second embedding table, `wpe`, gives each
*position* a vector, added to the token's. "I am a `t`" and "I am seventh"
now both live in `x`. Without the second fact, attention would see a bag of
characters; the break-it exercise at rung 7 measures exactly what that costs.

**Attention is a soft lookup.** Each position computes three vectors from its
`x`: a **query** ("what I'm looking for"), a **key** ("what I contain"), and a
**value** ("what I'll contribute if chosen"). The current query dots against
every stored key; softmax turns those match-scores into weights; the output is
the weighted average of stored values. It's a dictionary lookup where every
entry answers a little, in proportion to how well its key matches. The scores
get divided by √16 first — dot products grow with dimension, and a saturated
softmax learns slowly (a near-one-hot distribution barely moves when its
inputs nudge, so almost no gradient flows back through it; exercise 2 pokes
at exactly this, with a twist).

The whole mechanism, from [train3.py](../train3.py) (the two `attn_trace`
lines are this repo's instrument for drawing the heatmap later — not part of
the algorithm):

```python
    # 1) Single-head attention block
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, state_dict['attn_wq'])
    k = linear(x, state_dict['attn_wk'])
    v = linear(x, state_dict['attn_wv'])
    keys.append(k)
    values.append(v)
    attn_logits = [sum(q[j] * keys[t][j] for j in range(n_embd)) / n_embd**0.5 for t in range(len(keys))]
    attn_weights = softmax(attn_logits)
    if attn_trace is not None:
        attn_trace.append([w.data for w in attn_weights])
    x_attn = [sum(attn_weights[t] * values[t][j] for t in range(len(values))) for j in range(n_embd)]
    x = linear(x_attn, state_dict['attn_wo'])
    x = [a + b for a, b in zip(x, x_residual)]
```

Note the two lists threaded through `gpt(token_id, pos_id, keys, values)`:
every position appends its key and value. During training this looks like mere
bookkeeping. Rung 6 reveals the same two lists are *the* serving optimization
of the LLM era — the KV cache — and measures what they save.

**Residuals: the default is "change nothing."** Both blocks compute
`x = x + block(x)`, an *adjustment* to a stream that flows through untouched.
This is also a gradient highway: through the `+`, blame reaches early layers
undiminished (the local gradient of `+` is 1 — you verified that at rung 2).

**rmsnorm: standard scale.** Before anything sensitive, rescale `x` to unit
root-mean-square. Vectors that drift huge or tiny make training unstable;
normalization is the thermostat. One thing you'll notice in the code: it
normalizes right after the embedding and then *again* at the top of the
attention block, back to back. At layer 1 the second call looks redundant
(normalizing the just-normalized) — it isn't a mistake: residual adds will
un-normalize the stream after every block, so each block re-norms on entry,
and the pattern is kept uniform from layer 1. The original file carries a
comment saying exactly this.

## What the numbers said

```
num params: 4928
step    1 / 1000 | loss 3.7557 | val loss 3.6776 | effective choices 39.6 of 38
step  501 / 1000 | loss 2.7643 | val loss 2.7225 | effective choices 15.2 of 38
step 1000 / 1000 | loss 2.5713 | val loss 2.6886 | effective choices 14.7 of 38
training took 627.8s
```

- **It starts *worse* than the shrug** — 39.6 effective choices out of 38.
  train1 began a whisker from uniform; this model begins *actively confused*,
  because random attention pulls noise from random places into every
  prediction. More machinery, more ways to be wrong before training.
- **Val 2.6886 (14.7 choices): still behind counting's 2.6774 (14.5).** Read
  that again — a transformer, losing to a count table. No trick: SGD is
  underfitting this much richer model, and the fix is two rungs away (Adam).
  Beware the reflex "fancier architecture ⇒ better number"; the optimizer has
  a vote.
- **But the samples stopped being bigram babble:** `malaranti-senge-marin`,
  `ale-stin`, `sinitintheunstin`, `jarn-an-meting`. Suffix machinery (`-ing`,
  `-tion`), hyphen-separated segments with word-like lengths. Loss and taste
  disagree here, and both are honest: average per-character surprise is
  dominated by ordinary characters, while *structure* shows up in samples long
  before it moves the average. Keep both instruments.

And the model drew where it looks. Row = the position being predicted from,
columns = where its attention went:

```
      ^test-time-train
    ^ | @
    t | @@
    e | #@#
    s | @%%#
    t | @@##%
    - | @@%@@@
    ...
    n | @#%**%#@#%%*#%@%
```

The steady dark column under `^` is the find: every position keeps weight
parked on BOS. That's an **attention sink** — a token that carries no content
becomes the place to rest attention when there's nothing useful to fetch.
Production LLMs exhibit exactly this on their first token; serving systems
special-case it. Your 4,928-parameter model rediscovered it in 1000 steps.

## The idea to keep

A transformer block is two phases with a clean division of labor:
**communicate, then compute**. Attention moves information *between* positions
(and is the only place positions touch); the MLP thinks *within* a position.
Both write adjustments onto a shared residual stream. That's the whole
architecture — rung 4 just multiplies the communication channels, and rung 5
tunes the learning. When you read a 96-layer production model, you are reading
this file with the loop counter turned up.

## Exercises

**1. Predict, then run.** train1's step 1 landed a whisker under ln(38)
(3.6369 vs 3.6376). Will this file land as close? Commit to a number and a
reason; check against step 1.

**2. Break it.** Delete the `/ n_embd**0.5` scaling from the attention
logits (rung 4 will spell it `head_dim**0.5`). Predict the *mechanism* of
what goes wrong (what do big dot products do to a softmax? what does a saturated softmax
do to gradients?), then run `--fast` with and without, same seed, and compare
step-301 panels. Separately: predict what removing `wpe` entirely would do —
then hold that prediction; rung 7 runs that surgery under lab conditions.

**3. Extend it.** At inference, force generation past the context window:
change the sampling loop's `range(block_size)` to `range(45)` and remove the
BOS break. Predict the exact failure, line and error type, before running.

<details>
<summary>Solutions</summary>

**1.** No — 3.7557, well above. Near-uniform is what small random logits
alone buy you (train1's 3.6369); attention at random init actively mixes
irrelevant context into the prediction, which is worse than ignorance until
the keys and queries learn to match sensibly.

**2.** The textbook mechanism: unscaled dot products grow with dimension,
softmax saturates toward one-hot, and the gradient through a saturated
softmax goes quiet — attention's *routing* stops learning. Now the measured
truth at *this* dimension: removing the scale changed val loss from 2.8149
to 2.8146 at 300 steps. **Nothing.** Sixteen dimensions of rmsnorm'd
activations against 0.08-scale weights simply never produce logits big
enough to saturate. The 1/√d division is insurance written for large d — at
d = 4,096 it's load-bearing; at d = 16 it's a formality. You just ran your
first ablation whose honest result is "not at this scale" — rung 7 makes a
whole lab out of that sentence.

**3.** `IndexError: list index out of range` at `state_dict['wpe'][pos_id]`,
the moment `pos_id` hits 40. The context window is not a suggestion or a
config value — it is the number of rows in a table. Every "context length"
headline you've ever read is someone making this table longer without making
attention over it unaffordable.

</details>

---

Next: [train4 — multi-head](train4.md). Same parameter count, four places to
look at once.
