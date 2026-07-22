# train4 — multi-head

> The question this rung answers: **why would you split one good lookup into four small ones?**

```
docs -> tokenize -> model -> loss -> backward -> update -> sample
                    ^^^^^
        you are here: same box, four places to look at once
```

This is the quietest rung on the ladder — and that's its lesson. The diff from
train3 does two things: it slices attention into four heads, and it wraps the
block in a `for li in range(n_layer)` loop (set to 1). No new mathematics. And
crucially, as the very first line of output insists:

```
num params: 4928 (same as train3.py — heads are a split, not new capacity)
```

Four heads cost *nothing*. The same 16 query dimensions are cut into four
independent slices of 4; each slice computes its own scores against its own
slice of the keys, takes its own softmax, and averages its own slice of the
values. Four small lookups instead of one big one, concatenated back to 16
numbers, price unchanged. The split itself, from [train4.py](../train4.py)
(the `head_rows` lines are the heatmap instrument):

```python
        x_attn = []
        head_rows = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            if attn_trace is not None and li == 0:
                head_rows.append([w.data for w in attn_weights])
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
```

## Walk the code

Why is a split worth anything? Because **one softmax is one spotlight**. A
single head produces a single weighting over the past — if the position needs
to know "what came right before me" *and* "where did the current word start,"
one distribution must average those needs into mush. Four heads are four
independent spotlights: each can commit to its own criterion, because each has
its own query-key match and its own softmax. Multi-head attention is
portfolio diversification for lookups — never bet all 16 dimensions on one
place to look.

The layer loop is the other half of the diff, and it's shape-only:
state_dict keys become `layer0.attn_wq`, the block body indents one level,
and a transformer becomes "this, stacked." The KV lists nest along with it
(`keys[li].append(k)`): every layer keeps its *own* memory of the sequence —
the right picture for production models too. Set `n_layer = 2` and nothing
else needs to know. When you hear "a 96-layer model," it is this loop.

(The full diff against train3 runs long only because each rung also swaps
its instruments — the single attention triangle becomes four. The blocks
shown above *are* the meaningful changes.)

## What the numbers said

```
step    1 / 1000 | loss 3.5698 | val loss 3.6733 | effective choices 39.4 of 38
step  501 / 1000 | loss 2.7344 | val loss 2.7258 | effective choices 15.3 of 38
step 1000 / 1000 | loss 2.5121 | val loss 2.6926 | effective choices 14.8 of 38
training took 662.2s
```

Versus train3's 2.6886 / 14.7: **statistically the same** (14.8, within noise
on a 55-doc val set), with a slightly lower train loss (2.5121 vs 2.5713 —
a bit more fitting muscle). The scoreboard so far, and it's humbling:

| model | val loss | eff. choices |
|---|---|---|
| counting (train0) | **2.6774** | **14.5** |
| MLP bigram (train1/2) | 2.7301 | 15.3 |
| + attention (train3) | 2.6886 | 14.7 |
| + multi-head (train4) | 2.6926 | 14.8 |

A complete GPT architecture and a count table are neck and neck, table ahead.
If the course stopped here you'd conclude transformers are overrated. The
architecture is finished; the *optimizer* is the bottleneck — which is
precisely why Adam gets its own rung next, and why its result will land the
way it does.

And the four heads, printed over `test-time-training`: at this scale, honesty
compels the observation that they have **not** specialized into crisp,
nameable roles. All four keep the BOS sink — heads 0 and 2 most visibly, head 3
spreading a bit more evenly across recent characters — but they differ in
texture, not in story. Expecting "head 2 tracks hyphens" from 4,928 parameters is romance. What
four heads reliably buy is four *chances* — and redundancy when some of
them waste their slice. In production-scale models,
crisp head roles (previous-token heads, induction heads) do emerge; here you
are seeing their primordial soup.

## The idea to keep

Attention capacity isn't one number. Width-of-lookup and number-of-lookups
trade off inside the same parameter budget, and the field's empirical answer —
many narrow heads beat one wide one — is baked into every model you'll ever
use, at zero marginal cost. When a design choice is free and helps, it becomes
invisible infrastructure. This rung is where you watch one get installed.

## Exercises

**1. Predict, then run.** Before running: can the val loss *get worse* going
from one head to four, given identical parameter counts? What would that tell
you? Then compare your train3 and train4 panels.

**2. Break it.** Rung 7's lab includes two surgeries aimed at this rung's
additions (`no-residual`, `no-rmsnorm`). Write down now which of the two you
expect to hurt more, and by roughly how much in effective choices. You have
skin in the game when the bar chart prints.

**3. Extend it.** Set `n_layer = 2`. Predict all of: the new parameter count
(count it from the state_dict shapes before running), the direction of train
loss, the direction of val loss on 543 docs, and the runtime. Run `--fast` and
grade yourself on all four.

<details>
<summary>Solutions</summary>

**1.** Yes, it can — same capacity, different inductive bias: four narrow
heads can be *worse* if the task really wanted one wide correlation (head_dim
drops 16 → 4, and each softmax sees only a quarter of the signal). Here they
tied (14.7 vs 14.8): at bigram-ish loss levels on 488 docs, the split neither
pays nor charges. The lesson is that it was free, not that it was magic.

**2.** Filed at rung 7 — the lab prints the answer as a bar chart; the notes
there discuss it *after* the reader has committed. No spoilers here.

**3.** Parameter count: each layer adds 4·(16·16) attention + 2·(64·16) MLP =
3,072, so 4,928 → 8,000. The other three you grade from your own run — and
the val-direction question is the interesting one on a corpus this small
(more capacity, same 488 docs: watch the train/val gap, not just val).

</details>

---

Next: [train5 — Adam](train5.md). The architecture stops changing; the
*steps* get smart. This is the rung where the transformer finally earns its
keep — and where you watch the model grow up in the childhood album.
