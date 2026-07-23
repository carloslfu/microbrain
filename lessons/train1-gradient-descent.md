# train1 — gradient descent, by hand

> The question this rung answers: **how can a model improve without being handed the answer?**

```
docs -> tokenize -> model -> loss -> update -> sample
                    ^^^^^           ^^^^^^
        you are here: the table becomes a network,
        and "count it" becomes "descend the loss"
```

*New this rung:* [backward pass](../GLOSSARY.md#backward-pass) · [chain rule](../GLOSSARY.md#chain-rule) · [derivative](../GLOSSARY.md#derivative) · [embedding](../GLOSSARY.md#embedding) · [gradient](../GLOSSARY.md#gradient) · [gradient check](../GLOSSARY.md#gradient-check) · [hidden units](../GLOSSARY.md#hidden-units) · [init](../GLOSSARY.md#init) · [learning rate](../GLOSSARY.md#learning-rate) · [linear layer](../GLOSSARY.md#linear-layer) · [logits](../GLOSSARY.md#logits) · [MLP](../GLOSSARY.md#mlp) · [neural network](../GLOSSARY.md#neural-network) · [one-hot](../GLOSSARY.md#one-hot) · [optimizer](../GLOSSARY.md#optimizer) · [overfitting / underfitting](../GLOSSARY.md#overfitting--underfitting) · [ReLU](../GLOSSARY.md#relu) · [SGD](../GLOSSARY.md#sgd) · [softmax](../GLOSSARY.md#softmax) — every term links to the [glossary](../GLOSSARY.md).

Rung 0 had a formula. This rung throws the formula away and rebuilds the same
bigram model with the only tool that scales: define a loss, compute which
direction each parameter should move to reduce it, take a small step, repeat.
That loop — not attention, not scale — is the load-bearing idea of deep
learning, so it gets a whole rung with nothing else in it. Deliberately: the
model's *information* hasn't changed (still one character of context), only the
*method* has. When one thing changes per rung, you always know what to credit.

## Walk the code

**New words this rung** — defined here once, used freely afterward:

- **embedding** — the 16 numbers the model gets to associate with each token:
  its learned representation, and the only thing the model knows about a
  character. A row of the `wte` table.
- **linear layer** — a matrix multiply: every output is a weighted sum of all
  inputs. `linear()` in the code, four lines.
- **hidden units** — the 64 numbers between the two linear layers; "hidden"
  because they're neither input nor output, just workspace.
- **ReLU** — `max(0, x)`: negatives become 0, positives pass through. The
  cheapest nonlinearity; without one, two stacked linear layers collapse
  into a single linear layer (rung 7 measures what that costs).
- **logits** — the 38 raw scores the model outputs before softmax turns them
  into probabilities.
- **cross-entropy** — the textbook name for the loss you already know: rung
  0's "surprise," `-log P(the true next char)`, `avg_nll` in the code. One
  quantity, four names — they will not multiply further.
- **MLP** — multi-layer perceptron: the sandwich this file builds (linear →
  ReLU → linear). The generic word for it is a **neural network**; this is
  the smallest useful one.
- **SGD** — stochastic gradient descent: the update loop this whole file
  builds. "Stochastic" because each step trusts one document's gradient
  rather than the whole dataset's.

(Every term in the course, with its aliases, lives in the
[glossary](../GLOSSARY.md) — one lookup away whenever a word goes fuzzy.)

**The model.** A row of `wte` (the token's embedding, 16 numbers) → a linear
layer to 64 hidden units → ReLU → a linear layer to 38 logits. `softmax` turns
logits into probabilities: exponentiate (making everything positive, and gaps
multiplicative), then normalize. 4,064 parameters, initialized to small random
noise, `gauss(0, 0.08)` (bell-curve random numbers centered on 0, spread
0.08). That's nearly 3× the count table's 1,444 cells, to do the same
one-character job approximately — the price of the differentiable form. What
the overhead buys: every number is now *nudgeable* by a gradient, and
nudgeable is what scales. The whole network, from [train1.py](../train1.py):

```python
def mlp(token_id):
    x = state_dict['wte'][token_id]
    x = linear(x, state_dict['mlp_fc1'])
    x = [max(0, xi) for xi in x] # relu
    logits = linear(x, state_dict['mlp_fc2'])
    return logits
```

**The two gradients.** This file computes every gradient twice:

- `numerical_gradient`: nudge one parameter by 0.00001, rerun the whole model,
  see how the loss moved. Dumb, unimpeachable, and 4,064× too slow — one full
  forward pass *per parameter*.
- `analytic_gradient`: the chain rule, written out by hand, layer by layer,
  backwards from the loss. One pass total.

The first exists to referee the second:

```
gradient check on 'cscg-george-2021' | loss_n 3.636937 | loss_a 3.636937 | max diff 0.00000000
```

Eight decimal places of agreement between "perturb and measure" and forty
lines of calculus. When you write a backward pass — and in the extend-it
exercise, you will — this check is what stands between you and silently
training garbage.

One line inside the calculus deserves a pause:
`dlogits = probs - one_hot(target)` (where `one_hot(target)` is 38 zeros with
a single 1 at the true character). The gradient of softmax-plus-cross-entropy
is just *what you predicted minus what was true*. The whole apparatus of
backpropagation bottoms out in "the correction is the error." When people say
neural networks learn from their mistakes, this line is the literal mechanism.

## What the numbers said

```
step    1 / 1000 | loss 3.6369 | val loss 3.6403 | effective choices 38.1 of 38
step  501 / 1000 | loss 2.7682 | val loss 2.7960 | effective choices 16.4 of 38
step 1000 / 1000 | loss 2.5796 | val loss 2.7301 | effective choices 15.3 of 38
training took 9.7s
```

- **It starts at the shrug.** 3.6369 ≈ ln(38). Tiny random weights make every
  logit nearly 0, and softmax of all-zeros is uniform. Untrained networks
  don't start wrong, they start maximally noncommittal.
- **It ends where counting ended — almost.** Val loss 2.7301 (15.3 choices)
  versus rung 0's exact answer, 2.6774 (14.5). A thousand steps of calculus
  approximately rediscovered the count table. It did *not* beat it, and it
  can't: for this model class, counting was already optimal. The point was
  never to win here. It was to certify the approximate method against a known
  answer — before rung 3 points it at problems where no known answer exists.
- **Train 2.5796 vs val 2.7301.** The first daylight between the two curves.
  Small corpus, honest split; watch this gap all course.

The rung also draws its own loss landscape — a 1-D slice through a
4,064-dimensional bowl, sweeping one weight while all others hold still:

```
  loss  3.229 |                     W                   *
  loss  3.204 |                     W             **
  loss  3.172 |                     W****
  loss  3.153 | *****************************************
  weight sweeps -3.03 .. +2.97, W = trained value -0.031, min loss 3.1535
```

(Numbers from the full run; `--fast` draws its own, equally valid valley.)

`W` marks where SGD actually left this weight: on the floor of the valley.
One subtlety worth keeping: `W` is *near* the minimum of this curve, not on
it. This curve is the loss on **one** document; training minimized the
**average** over 488. Every parameter lives where hundreds of documents'
tugs cancel out. "The model" is a negotiated settlement.

## The idea to keep

The loss is a surface over parameter space; the gradient is the direction of
steepest *blame*; learning is rolling downhill in 4,064 dimensions at once.
The learning rate starts at 1.0 and decays linearly to zero — big exploratory
strides early, careful settling late. Nothing about this loop will change for
the rest of the course. Rungs 2–5 change how the gradient is *computed* and
how the step is *shaped*. The loop itself is finished, and it fits in four
lines:

```python
    # SGD update
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, (row, j) in enumerate(params):
        row[j] -= lr_t * grad[i]
```

## Exercises

**1. Predict, then run.** Before running: what loss will step 1 print, and
why can you know it without running anything? (Hint: what does softmax do to
near-zero logits?)

**2. Break it.** In `analytic_gradient`, flip the sign:
`dlogits[target_id] += 1.0 / n`. Predict: does anything *detect* the bug? What
does training do? Run it and watch closely.

**3. Extend it.** Add a bias vector `b` to the final layer: after
`logits = linear(x, state_dict['mlp_fc2'])`, add `b[i]` to each logit. You must (a) initialize it, (b) add it to `params`,
(c) use it in *both* forward passes, and (d) derive its gradient by hand. The
gradient check grades your derivation automatically. That's the real lesson.

<details>
<summary>Solutions</summary>

**1.** ≈ 3.6369 ≈ ln(38): init weights are gauss(0, 0.08), so logits are all
nearly zero, so softmax is nearly uniform, so surprise is the shrug price.

**2.** Two-act failure, both acts observed live: first, the referee catches it
immediately — `max diff 0.01862396` where a healthy run prints `0.00000000`.
Ignore the referee, and training runs *uphill*: 3.6369 → 3.6406 → 3.6444...
Five steps later it's so confidently wrong that the true next character's
probability underflows to exact zero and the run dies with
`ValueError: math domain error` at `-math.log(probs[target_id])` — the same
log-of-zero death as rung 0's smoothing exercise. Moral: the gradient check is
not ceremony. It's the difference between a subtle bug and a crash you notice.

**3.** `d(loss)/d(b) = dlogits` — the bias feeds the logits directly, so its
gradient is the error itself, accumulated across positions:
`grad['b'][i] += dlogits[i]`. If your gradcheck prints ~0.00000000, your
calculus is right; if not, it will tell you *which* parameter disagrees, and
that number is a map to your bug.

</details>

---

Next: [train2 — autograd](train2-autograd.md). The forty lines of hand calculus you
just verified become obsolete — replaced by forty lines that derive them
automatically, for any architecture you'll ever write.

[← train0](train0-counting.md) · [home](../README.md) · [glossary](../GLOSSARY.md) · [train2 →](train2-autograd.md)
