# train7 — the ablation lab

> The question this rung answers: **does every part of this thing actually earn its place?**

```
docs -> tokenize -> model -> loss -> backward -> update -> sample
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        you are here: six trainings, five wounds, one bar chart
```

You've now *read* justifications for every organ: positions so attention
can see order; residuals as the gradient highway; rmsnorm the thermostat;
relu the nonlinearity; β₂ (the code's `beta2`) Adam's memory of scale. Reading justifications is
how folklore spreads. This rung re-trains the model six times from the same
init — baseline plus five surgeries — and prints what each removal actually
cost. If you ran it as intended, your predicted ranking is written down and
it is about to be graded.

The whole lab mechanism is one dict the model consults, plus the list of
surgeries, from [train7.py](../train7.py):

```python
FLAGS = {'wpe': True, 'residual': True, 'rmsnorm': True, 'relu': True, 'beta2': 0.99}
```

```python
SURGERIES = [
    ('baseline',    {}),
    ('no-wpe',      {'wpe': False}),
    ('no-residual', {'residual': False}),
    ('no-rmsnorm',  {'rmsnorm': False}),
    ('no-relu',     {'relu': False}),
    ('beta2=0.5',   {'beta2': 0.5}),
]
```

Two calibration notes before the reveal. First: the val set is 55 documents,
so differences under ~0.03 nats are coin-flips; differences over ~0.1 are
real wounds. Second: the baseline below reads 15.6 effective choices, not
rung 5's 13.8 — the lab trains 300 steps per run, not 1,000 (six full-budget
trainings would run well over an hour). Same architecture, smaller budget; by now
that distinction should feel load-bearing.

## What the numbers said

```
baseline     | val loss 2.7444 | effective choices  15.6
no-wpe       | val loss 2.7201 | effective choices  15.2   <- ?!
no-residual  | val loss 2.9102 | effective choices  18.4
no-rmsnorm   | val loss 2.8007 | effective choices  16.5
no-relu      | val loss 2.7572 | effective choices  15.8
beta2=0.5    | val loss 2.7684 | effective choices  15.9
```

Two confirmations, two scandals, one shrug:

**Confirmed: residuals are the most load-bearing organ.** +0.17 nats, 15.6 →
18.4 effective choices — the only wound far outside the noise floor. Without
`x = x + block(x)`, every block must *re-derive* its input instead of
adjusting it. And blame reaching the embeddings now filters through every
intervening matrix — rung 2 taught you the `+`'s local gradient is 1; that
was the highway, and you just closed it. This is why "just stack more layers"
only became possible after 2015-era architectures made identity the default.

**Confirmed: normalization is quietly important.** +0.06. Not a
catastrophe at one layer — activations can only drift so far in one block —
but the thermostat is visibly doing work even here, and its value compounds
with depth. (Your `n_layer = 2` exercise from rung 4 is where to see that.)

**Scandal #1: no-wpe *won*.** Lower val loss than baseline (−0.024 — a
coin-flip's width, but the organ you'd have defended to the death just
measured *free to remove*). The explanation is the course's sharpest lesson.
At 300 steps and bigram-ish loss levels, nearly all predictive power is
*which characters tend to follow which* — knowledge that needs no notion of
position. The 640
wpe parameters are, at this budget, noise the model must learn around. Order
information pays rent at the margins (endings, boundary rhythm) that this
training run hasn't reached yet. **An ablation is not a fact about an organ;
it is a fact about an organ at a budget, a scale, and a dataset.** Every
published ablation table you will ever read deserves this same suspicion.

**Scandal #2: the nonlinearity is nearly free to remove.** +0.013 — noise.
Remove the relu and the two stacked linear layers should collapse toward
one linear map; why no damage? Because at this scale the *attention softmax* is already
supplying nonlinearity, and the bigram-ish statistics being learned are
mostly linear-reachable anyway. At production scale the MLP stack is where
most parameters and (by current understanding) most stored knowledge live —
another budget-dependent truth, in the other direction.

**The shrug: β₂ = 0.5** cost +0.024 — twitchier steps, mildly worse, not
fatal at this size. Adam's defaults are robust plumbing, not a knife edge.

## The idea to keep

The lab's method outlives its findings: **same init, same data order, one
change, honest error bars.** That's the entire experimental method of deep
learning, miniaturized. And its two scandals generalize better than its
confirmations. When a component "doesn't matter," the first question is
never "remove it?" — it's "*at what budget would it start mattering?*" When
a paper says "X improves Y," the silent suffix is always "at our scale, on
our data, for our step count." You now own a lab where checking such claims
costs four minutes.

## Exercises

**1. Predict, then run** — you already did; the lab is structured around it.
Grade your ranking now, in writing, including the direction of your no-wpe
prediction. (If you had wpe near the top of the damage chart, you're in good
company — so did the author of this lesson.)

**2. Break it further.** The lab's most important dial is `num_steps`. Rerun
at 600 or 1000 steps: does no-wpe still win? Does no-relu stay free? Commit
to both answers first. This lesson deliberately does not spoil either — the
whole point of owning the lab is that the answer is four minutes away and
*yours*.

**3. Extend it.** Add a seventh surgery of your own to `SURGERIES`. Good
candidates, in ascending ambition:

- `no-attention` — make `x_attn = x`. Is attention even helping at this
  loss level?
- `no-scale` — drop the `/ head_dim**0.5`, rung 3's exercise under lab
  conditions.
- `tied-embeddings` — reuse `wte` as `lm_head`, like production models do.
  What does it change at 4,928 params?

<details>
<summary>Solutions</summary>

**1.** The measured ranking, worst wound first: no-residual (+0.166),
no-rmsnorm (+0.056), beta2=0.5 (+0.024), no-relu (+0.013), no-wpe (−0.024).
Anything within ±0.03 of baseline you should have graded as "tie," not
"win/loss."

**2.** Unspoiled by design. Mechanism to reason with: position information
pays rent on *late-course* structure (endings, boundary spacing), so its
value should grow with budget; the relu question is genuinely open at this
scale — that's why it's an experiment.

**3.** Each is a two-to-four-line change: `no-attention` guards the head loop
the way `residual` is guarded; `tied-embeddings` replaces
`state_dict['lm_head']` with `state_dict['wte']` (and removes it from init —
watch the param count drop by 608). If you add a surgery and its result
surprises you, you have reached the course's actual terminal state:
generating your own curriculum.

</details>

---

You're done with the ladder. Two short reads remain: [train6](train6.md) if
you skipped ahead, and the [epilogue](epilogue.md) — what separates your
4,928 floats from the frontier, itemized. Then go run `python namer.py` and
name something real.
