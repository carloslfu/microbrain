# The progression — how microgpt was actually built

microbrain follows a published curriculum. This file documents that curriculum
as *evidence*, because both halves of it teach something the final code can't.

## 1. The ladder (six revisions of one gist)

Karpathy's blog post calls the build-up "layers of the onion" and publishes it
as a gist (GitHub's single-file sharing format) —
[build_microgpt.py](https://gist.github.com/karpathy/561ac2de12a47cc06a23691e1be9543a) —
whose **six revisions are the six steps**. Each revision's docstring (the file's opening description block) names its
delta ("Same as trainN−1 / Different from trainN−1"). Vendored snapshots live
in [`reference/`](reference/), fetched revision by revision:

| rung | file | lines | the one new idea |
|---|---|---|---|
| 0 | train0.py | 79  | bigram by **counting** — the closed-form solution |
| 1 | train1.py | 178 | the same bigram as an MLP, **gradients by hand**, SGD |
| 2 | train2.py | 156 | **autograd** — and the file gets *shorter* (−22) |
| 3 | train3.py | 191 | positions + **attention** + rmsnorm + residuals |
| 4 | train4.py | 200 | **multi-head** + the layer loop — structurally complete |
| 5 | train5.py | 203 | **Adam** — "this is train.py" |

Two things to notice in the numbers alone:

- **Rung 2 is the only rung that shrinks.** The autograd abstraction deletes
  the hand-derived backward pass — the hardest 40 lines in the course — and
  replaces them with 40 general ones. Good abstractions are measured in
  deleted difficulty, not added features.
- **The last three rungs cost 47 lines total.** Attention, multi-head, and
  Adam are *small* once autograd exists. The expensive ideas are at the
  bottom of the ladder, which is why this course spends real time there.

## 2. The trim (the main gist's own history)

The finished file — [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) —
has its own revision story, and it's the better lesson:

- **Born Feb 11, 2026 at 243 lines** (+243 −0), after what the blog post calls
  a decade-long obsession with simplifying LLMs to their bare essentials.
- **A dozen revisions follow.** Small renames, docstring tuning, hyperparameter
  nudges — on the 11th, the 12th, again on the 16th.
- **The big one: a single revision removing 78 lines and adding 35** (net
  −43) in the launch-day cluster of edits — *after* the post had already
  declared "I cannot simplify this any further." It turned out he could: that
  cut plus the small nips around it net −44, and the file settled at 243 → 199.
- **Still breathing:** the gist was touched as recently as **July 21, 2026**
  — five months post-launch, single-line edits are still landing.

The meta-lesson, and the reason this file exists: **compression is not an
event, it is a practice.** The most-simplified file by the person most obsessed
with simplification still had 43 removable lines in it on the day he said it
didn't. Hold your own "final" versions with the same suspicion.

## 3. Where microbrain diverges, and why

microbrain ports the six rungs faithfully — same filenames, same delta-
docstring convention, same architecture and optimizer, `diff`-able against
`reference/` — with these deliberate changes:

| dial | microgpt | microbrain | why |
|---|---|---|---|
| corpus | 32,033 human names | 543 idea names from a db.md knowledge store | personal data makes failure legible and success delightful |
| block_size | 16 | 40 | idea names run longer than first names |
| vocab | 27 | 38 | digits and hyphens are load-bearing in slugs |
| val split | none | 10% held out | small corpus ⇒ overfitting is *the* phenomenon to watch |
| instruments | loss prints | + val loss, effective choices, memorization %, self-drawn diagrams | measurement as pedagogy |
| rungs | 0–5 | 0–5 **+ 6 (inference toolkit) + 7 (ablation lab) + namer.py** | the questions readers actually ask next |

And one inheritance kept on purpose: **every rung repeats the full scaffold**
(dataset, tokenizer, Value class) instead of importing it. ~90 lines of
repetition per file buys something rare — the diff between adjacent rungs IS
the lesson, with zero indirection. Repetition here is a feature, the same way
it is in the original.
