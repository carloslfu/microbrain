# train5 — Adam

> The question this rung answers: **the architecture was finished two rungs ago and still lost to a count table — what was missing?**

```
docs -> tokenize -> model -> loss -> backward -> update -> sample
                                                 ^^^^^^
        you are here: the steps get smart. this is microgpt.
```

The diff is five lines of optimizer state and four lines of update rule. It is
also the largest single improvement on the course's scoreboard. Sit with that
asymmetry: rungs 3 and 4 rebuilt the entire model around attention and
*couldn't beat counting*; this rung changes only how the parameters step, and
wins. In deep learning, the optimizer is not plumbing. It's a protagonist.

## Walk the code

SGD moved every parameter by `lr × its gradient`. Adam keeps two running
averages *per parameter* and moves by their ratio:

- **`m` — momentum, a memory of direction.** `m = 0.85·m + 0.15·grad`. Our
  batch size is one document; each gradient is one document's *opinion*, and
  opinions swing wildly (watch the raw loss bounce: 3.65, 3.47, 3.55, 3.71 in
  the first five steps). Momentum averages the last ~7 opinions into a
  consensus direction before moving.
- **`v` — a memory of scale.** `v = 0.99·v + 0.01·grad²`. Dividing the step
  by `√v` gives every parameter its *own* unit system: a rarely-touched
  embedding row (how often does `q` appear?) gets bold steps when its moment
  comes; the constantly-hammered `lm_head` gets careful ones. One global
  learning rate could never serve 4,928 parameters with 4,928 different
  gradient climates.
- **Bias correction.** Both averages start at zero, so early estimates are
  artificially small; dividing by `1 − β^t` un-shrinks them exactly. Without
  it, the first dozen steps would be timid for no reason.

That's the whole optimizer. It shipped in 2014 and is, with cosmetic changes,
what trains the frontier today.

## What the numbers said

```
step    1 / 1000 | loss 3.5698 | val loss 3.6669 | effective choices 39.1 of 38
step  501 / 1000 | loss 2.6792 | val loss 2.6778 | effective choices 14.6 of 38
step 1000 / 1000 | loss 2.4283 | val loss 2.6216 | effective choices 13.8 of 38
```

The completed scoreboard, six files in the making:

| model | optimizer | val loss | eff. choices |
|---|---|---|---|
| uniform shrug | — | 3.6376 | 38.0 |
| count table (train0) | counting | 2.6774 | 14.5 |
| MLP bigram (train1/2) | SGD | 2.7301 | 15.3 |
| + attention (train3) | SGD | 2.6886 | 14.7 |
| + multi-head (train4) | SGD | 2.6926 | 14.8 |
| **same model (train5)** | **Adam** | **2.6216** | **13.8** |

Same 4,928 parameters as train4. Same 1,000 documents in the same order. The
only change is *how the steps were taken*: 14.8 → 13.8, and the count table
finally falls. Note also step 1 is 3.5698 in both train4 and train5's logs —
identical to the fourth decimal — because the first loss is computed before
any optimizer has had a chance to differ. The twin-run trick from rung 2
keeps paying: logs are proofs.

The file draws the race (s = SGD from train4's saved curve, A = Adam):

```
   3.60 | *s
   3.28 |   A
   2.97 |      AAs  s       s
   2.81 |         A*   As**   A A*** *ss**  ss            ss  ss   s
   2.74 |               A             AA  *sAA*s   *ss    AAssA sssAsssss
   2.58 |                                       AsA   AAAA           AA  A
   2.50 |                                        A
```

`A` gets below `s` within the first tenth of training and stays there. And
read the *band*, not just the trend: both curves live inside ±0.1 of jitter,
because every point is a one-document quiz. Learning to see "noisy but
descending" versus "flat" versus "diverging" at a glance is a working skill —
this chart is your first calibration.

And the course's promised centerpiece, the childhood album — four ages of the
same model, straight from the log:

```
step    0 | m129o8dwl-x7nfi1o8kliw13m-uvwax1c7omlpp6, fpx3bbb2kk73p9lfqj3pl, iqz, ...
step   50 | reva-ve, manrema-lalat-g-s, re-la-tonias, jadde-ag-lotinar, nexoa-r, ...
step  250 | menticarerenmiti-landiol-s, arore-agrenengerative-1, tenge-cingentra, ...
step 1000 | mianicov, reat-hivarsian, jang-tin-tining-avingantige, agan-folin, seag, an
```

Step 0 is the uniform shrug as literature — digits and all, 38 effective
choices on display. By step **50** — fifty documents! — the cheap statistics
are in: hyphen rhythm, consonant-vowel alternation, digits mostly gone
(they're rare in the corpus). By 250, morphemes (`-ative`, `-ing` machinery
warming up). By 1000, word-shaped segments with plausible endings. Nobody
scheduled this curriculum. Frequent, high-payoff patterns get learned first
because they dominate the gradient — a scaling law you can watch with the
naked eye, in a minute, on a laptop.

## The idea to keep

"Better model" and "better training" are separable axes, and the second one
is chronically underestimated. A transformer under a weak optimizer *measured
worse than counting* — imagine concluding, two rungs early, that attention
doesn't work. Every capability claim you'll ever evaluate has an optimizer,
a data recipe, and a step budget hiding inside it. This rung is the course's
inoculation against architecture romance.

## Exercises

**1. Predict, then run.** Before running train5: will its step-1 loss match
train4's, given the optimizers differ? Reason it out, then diff the logs.

**2. Break it.** Predict what `beta2 = 0.5` does — a variance estimate that
only remembers the last couple of gradients. Commit to a direction and a
magnitude. Rung 7 runs exactly this surgery under lab conditions; your
prediction is now on file.

**3. Extend it.** Disable the decay: `lr_t = learning_rate`, constant. From
the mechanism (late-stage steps stay large), predict what the tail of the
loss curve and the final panel will look like relative to the decayed run.
Then run `--fast` both ways and compare tails.

<details>
<summary>Solutions</summary>

**1.** Identical (3.5698, both logs): step 1's loss is computed on the same
init with the same document *before* the first update — optimizers can only
diverge from step 2 onward. If your logs disagree at step 1, your seed or
data order changed, not your optimizer.

**2.** On file at rung 7 — compare the bar chart there against your committed
magnitude, not the other way around.

**3.** Expected from mechanism: the curve keeps its jitter to the very end
instead of settling (the basin is being orbited, not entered), and the final
panel typically lands slightly worse. Your `--fast` pair is the evidence;
this solution deliberately states the mechanism, not your numbers.

</details>

---

Next: [train6 — the inference toolkit](train6.md). The gist's ladder ends
here; the model's life doesn't. Save it, resurrect it, tune its temperature,
measure the cache, and quiz your friends.
