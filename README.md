# microbrain

**Build a GPT you can hold in your head — trained on the names of your ideas.**

This is [Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
— a complete GPT (the family of text-predicting models behind ChatGPT) in
~200 lines of dependency-free Python — unrolled into a
course you *walk*: his six published build-steps, ported onto a personal
corpus, instrumented so every claim is a number you printed, illustrated with
diagrams the code draws from its own weights, and extended with the two rungs
readers always ask for next. Pure Python standard library, beginning to end.
No pip install. Ever.

The promise: you know Python but not ML. You walk the rungs in order over a
weekend. You leave holding the deep intuitions — what a loss *is*, why
autograd exists, what attention buys, what an optimizer decides, why models
memorize or don't — plus a 106 KB `model.json` you trained, killed,
resurrected, and shipped as a working name-generator.

(If that paragraph was a wall of unfamiliar words — good, that's the course.
Every one of them is defined, in order, inside the lessons, with a
[glossary](GLOSSARY.md) for lookups; this page is the view from the summit,
not the first step. And if the Python-and-terminal side is new too, read with
an AI agent alongside — it fills gaps on demand.)

## The ladder

Each rung is one runnable file and one short lesson. One new idea per rung —
everything else is frozen scaffold from the rung below, and **every lesson
shows the exact new lines inline**, verbatim from the file, with the full
file one click away. You never need a terminal diff to follow the idea (the
files stay adjacent and diffable for those who like it, or for your agent).
Numbers below are from full runs on this repo's corpus (543 idea names,
38-token vocabulary, 10% held out):

| rung | file | the one new idea | val loss | effective choices¹ |
|---|---|---|---|---|
| — | — | uniform shrug (no model) | 3.6376 | 38.0 |
| 0 | [train0.py](train0.py) · [lesson](lessons/train0.md) | bigram by **counting** | 2.6774 | 14.5 |
| 1 | [train1.py](train1.py) · [lesson](lessons/train1.md) | **gradients, by hand** (SGD) | 2.7301 | 15.3 |
| 2 | [train2.py](train2.py) · [lesson](lessons/train2.md) | **autograd** — same numbers, ~45× slower, Karpathy's file gets *shorter* | 2.7301 | 15.3 |
| 3 | [train3.py](train3.py) · [lesson](lessons/train3.md) | **attention** + positions + residuals + rmsnorm | 2.6886 | 14.7 |
| 4 | [train4.py](train4.py) · [lesson](lessons/train4.md) | **multi-head** — same param count, four spotlights | 2.6926 | 14.8 |
| 5 | [train5.py](train5.py) · [lesson](lessons/train5.md) | **Adam** — the count table finally falls | **2.6216** | **13.8** |
| 6 | [train6.py](train6.py) · [lesson](lessons/train6.md) | *(ours)* save/load, temperature, the KV cache measured, the quiz | 2.6216 | 13.8 |
| 7 | [train7.py](train7.py) · [lesson](lessons/train7.md) | *(ours)* the ablation lab: break it on purpose, organ by organ | — | — |
| end | [namer.py](namer.py) · [epilogue](lessons/epilogue.md) | your model as a tool; the bridge to production | — | — |

¹ `e^(val loss)` — out of 38 possible next characters, how many is the model
still effectively guessing among? Uniform = 38, perfect = 1 — and yes, an
untrained transformer can score *above* 38 by actively mixing noise into its
guesses (rung 3 opens there). It's perplexity, wearing its plain-English
name. Watch it fall down the ladder, 38 → 13.8 — not monotonically: the
bumps at rungs 1 and 4 are lessons, and the lessons own them.

The mid-ladder plot twist is real and deliberate: **a complete GPT loses to a
count table for two straight rungs** (14.7, 14.8 vs 14.5) until the optimizer
rung lands. The architecture was never the bottleneck. Rung 5's lesson is
where that sinks in.

One honesty note: every number in the lessons comes from *this* corpus with
*these* seeds, and the runs are deterministic — walk the course on the same
data and your logs will match the lessons digit for digit. Bring your own
corpus (`--names`, `--from`) and the numbers will differ; the shape of the
story is what transfers.

## Quickstart

```
python data/make_dataset.py     # harvest the corpus (or: --names for Karpathy's 32k names)
python train0.py                # instant
python train1.py                # ~10 s
```

Then read `lessons/train0.md` — the lessons assume you run first, read second.
(`python` here means your Python 3; some systems spell it `python3`.) Not the
author of this particular brain? Step 1 harvests *his* knowledge base, so
bring your own corpus: `--names` downloads Karpathy's 32k human names, or
`--from yourlist.txt` trains on any file of short strings, one per line.
Worth doing from the start: `mkdir -p out` and run each rung as
`python trainN.py | tee out/trainN.log` (`tee` shows the output *and* saves
it to the file) — `compare.py` builds the final
ladder from those logs, and full-run numbers are nicer than the `--fast`
reruns it falls back to.

Then keep climbing. Honest timings on a laptop (pure-Python, one number
at a time — the slowness is the point, not a flaw): train2 ≈ 7 min, train3–train6 ≈ 10–12 min each,
train7 ≈ 20 min for all six runs (five surgeries plus the baseline). Every
rung takes `--fast` (seconds on the early rungs, ~1–3 min on the heavy ones)
when you want the shape without the wait; `python compare.py` prints the
whole ladder from whatever logs you've produced and fast-runs the gaps.

Each lesson ends with three exercises: **predict-then-run** (commit before the
machine answers), **break it** (sabotage with a diagnosis — the crashes are real
and were observed), and **extend it** (the gradcheck grades your calculus).
Do them. The course's actual thesis is that intuition comes from predictions
you got wrong in private.

Concepts arrive in plain words, in order, each defined the first time it's
needed — and everything lives in the [glossary](GLOSSARY.md) with its aliases
("cross-entropy, aka negative log-likelihood, aka rung 0's *surprise*"), so a
term met mid-course is never more than one lookup away.

**Agent-native.** The reading is for you; the terminal work is optional —
any AI agent makes a fine lab partner for it. Have it run the rungs, produce
and explain the diff between any two files, referee your exercise
predictions, or chase a tangent. Everything an agent needs is in-repo:
pinned canon in `reference/`, deterministic seeds, the author's logs in
`runs/`. The lessons show every load-bearing line inline, so *following the
course* never requires running anything at all — though training your own
and hitting the quiz is the whole fun.

## What's in the box

```
train0.py … train5.py    the canon — Karpathy's six build-steps, ported (block 40, vocab 38)
train6.py  train7.py     ours — the inference toolkit and the ablation lab
namer.py                 the payoff: out/model.json as a name-generating tool (try --quiz)
compare.py               the ladder on one screen
data/make_dataset.py     corpus harvester (data.txt stays out of the repository — it's personal)
lessons/                 one lesson per rung + the epilogue
GLOSSARY.md              every term in plain words, with its aliases and the rung
                         where it's earned — for mid-course lookups
runs/                    the author's full logs — every excerpt in the lessons is quoted
                         verbatim from these files, so every claim is checkable
reference/               pinned snapshots of both Karpathy gists, fetched revision by revision
PROGRESSION.md           the published ladder + the gist's own revision story (243 → 199 lines)
```

Requirements: Python 3 (developed and verified on 3.12). Nothing else — no
pip, no venv, ever.

Every train file is self-contained on purpose — dataset, tokenizer, `Value`
class repeated verbatim so adjacent files diff clean. The instruments are the
other constant: train/val loss, effective choices, a memorization gauge
(what fraction of samples are verbatim training docs — it reads 0/20 all
course, and train6 explains why that's the *interesting* outcome), and one
diagram per rung drawn from the live model: the count table as a heatmap, a
loss valley, a computation graph with grades you can check by eye, attention
maps, the SGD-vs-Adam race, and the childhood album:

```
step    0 | m129o8dwl-x7nfi1o8kliw13m-uvwax1c7omlpp6, fpx3bbb2kk73p9lfqj3pl, iqz, ...
step   50 | reva-ve, manrema-lalat-g-s, re-la-tonias, ...
step  250 | menticarerenmiti-landiol-s, eorel-mesellarepe-patinstis-vacpong, ...
step 1000 | mianicov, reat-hivarsian, jang-tin-tining-avingantige, agan-folin, ...
```

That's one model, photographed at four ages, learning the shape of idea names
from pure static (random noise) — in about a minute of real time per photograph.

## Credits

The algorithm, the ladder, and the aesthetic are
[Andrej Karpathy's](http://karpathy.github.io/2026/02/12/microgpt/): microgpt
is the culmination of micrograd → makemore → nanoGPT, and this repo is a
study of it, not a replacement for it. Read
[PROGRESSION.md](PROGRESSION.md) for what his revision history itself
teaches. The corpus is the record slugs of a personal db.md knowledge store (plain-file
databases) — swap in your own list of short strings (`--from`) and the whole
course retrains on *your* world in an hour.

License: MIT for this repo's original material; the `reference/` snapshots
remain Karpathy's, vendored for study with attribution — see
[reference/README.md](reference/README.md) and [LICENSE](LICENSE).
