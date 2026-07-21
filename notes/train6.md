# train6 — the inference toolkit

> The question this rung answers: **training is over — what does it take to actually *use* the thing?**

```
docs -> tokenize -> model -> loss -> backward -> update -> sample
                                                           ^^^^^^
        you are here: past the gist — everything after the last gradient
```

Karpathy's ladder ends at Adam. This rung and the next are microbrain's own,
built from the questions every reader asks the moment train5 finishes: *where
does the model live? what's temperature? why is serving fast?* — questions
about the half of the lifecycle the training gist doesn't cover. The training
in this file is identical to train5's, and the panel agrees; everything
interesting happens after `training took 705.1s — now the toolkit`.

First, note what training itself delivered, because it's the course's
turning point:

```
step 1000 / 1000 | loss 2.4283 | val loss 2.6216 | effective choices 13.8 of 38
```

**13.8. The count table (14.5) has finally been beaten** — by the same
architecture that lost to it under SGD for two straight rungs. Hold that
thought for train5's essay; here, on to the toolkit.

## 1. The file of floats

```
1) saved model to out/model.json (106,388 bytes) | val loss 2.621591
   all 4928 params zeroed. it now babbles: ['02e3x0jbv5bb8-f2l1agx24t-xk9usct6mjtaesi', ...]
   loaded it back. val loss 2.621591 | identical: True
```

The save format is `json.dump` of the state_dict. No serialization framework,
no magic: a GPT *is* a named collection of float lists. To prove it, the file
kills the model — every parameter set to 0.0 — and samples from the corpse:
`02e3x0jbv5bb8-f2l1agx24t...`. Recognize that static? All-zero parameters make
all-zero logits make softmax uniform: you are looking at **38 effective
choices**, the exact shrug the course started from at rung 0, now as text.
Then `json.load`, copy the floats back in, and the val loss returns *identical
to the last decimal*. When you download a "model" from anywhere, this is what
you downloaded — the 106 KB version of a multi-hundred-GB idea.

## 2. Temperature, the one-knob personality

```
   T=0.1 | ant-conting, pran-marating, mare-maratin-sengerion, arin-are-angentining, ...
   T=0.5 | comanes, man, torary-mararan, si-madelint-ferals, mose, ...
   T=1.0 | dhegervssee, visary-mbododittiarl-mastidediag, rktui, yat-wlak-prollinges, ...
```

Mechanically it's one line — divide the logits before softmax. Dividing by
0.1 multiplies every logit gap by 10: the biggest probability swallows the
rest, and the model loops its safest morphemes (`-ing`, `-ating`, over and
over — same seed on every row, so the differences are pure temperature).
Dividing by 1.0 leaves the model's true distribution: braver, weirder,
sometimes `rktui`. As T→0 sampling becomes argmax; as T grows it approaches
the uniform shrug. Every "creativity slider" you've ever seen in an AI
product is this division.

## 3. The KV cache, measured

```
   with cache:    'ceos-fipo-abl-paliting' |  23 model calls | 0.366s
   without cache: 'ceos-fipo-abl-paliting' | 276 model calls | 4.601s
```

Same seed, same name, character for character — the cache changes *nothing*
about the math. It changes the work: with the keys/values lists kept between
characters (as the training loop always did), producing token N costs one
model call; throw the lists away and honesty requires replaying the whole
prefix, so 22 characters cost 276 calls instead of 23. That's 1+2+...+23
versus 23 — quadratic versus linear, a 12.6× wall-clock gap at name length,
and the gap *grows with every character*. Scale the sequence to a chat
history and you understand why the serving industry is, to first
approximation, a KV-cache management industry.

## 4. The novelty filter — and a confession

```
4) novelty filter: 30 samples -> 0 verbatim training docs, 0 empty -> 30 genuinely new
```

The memorization gauge has read 0 all course, and this rung's 30-sample census
confirms it. Time to interrogate that: microgpt's launch thread on Hacker News
caught *his* samples containing verbatim training names — with nearly the same
parameter count. Why does ours not parrot? Because memorization pressure is
about the *density of the space*: 32,033 human names crowd a small space of
short, convergent strings (many "kamon"s are near-inevitable), while 543
hyphenated 20-character slugs rattle around an astronomically larger one. Our
model lacks the capacity to store paths to specific training docs, so it
stores *statistics* — morphemes, hyphen rhythm, endings. Generalization isn't
a virtue here; it's what's left when memorizing is unaffordable. (Rung 4's
`n_layer = 2` exercise is where you can start to buy it back and watch the
gauge move.)

## 5. The quiz

```
   1. gorloni-beal-erfgeltiollan      4. jalien
   2. typed-provenance-for-llm-agents 5. engram-labs-weight-memory-vs-dbmd
   3. ai-driven-drug-discovery        6. onntic-chatthon
```

Three are records in the brain; three came out of 4,928 floats. Commit to
answers before peeking at the log's answer line. If you hesitated even once,
consider what that means: at 106 KB, statistics-of-naming is already halfway
to plausibility. Now recall your last encounter with a fluent paragraph from
a model six orders of magnitude larger, and calibrate accordingly.

## Exercises

**1. Predict, then run.** Before running: for a 22-character name, how many
model calls with the cache? Without? What's the ratio, and is it constant in
name length? Then check the printout.

**2. Break it.** Open `out/model.json` in an editor. Find one weight in
`lm_head` and set it to `1000000.0`. Reload with a small script (steal the
load loop from this file) and sample. Diagnose what you observe from the
mechanism — which single token dominates, and at which temperature does the
damage hide?

**3. Extend it.** Implement *prompting*. Feed the characters of a prefix —
say `agent-` — through `gpt()` to warm the KV lists, then sample the
continuation. ~10 lines around the existing sampling loop. You will have
reimplemented, at 4,928 parameters, the interaction pattern of the entire
chatbot era: conditioning a frozen model with context instead of training it.

<details>
<summary>Solutions</summary>

**1.** 23 with cache (one per generated token incl. the closing BOS draw),
1+2+...+23 = 276 without: the ratio is (N+1)/2 — *linear in length*, not
constant. Observed: 0.366s vs 4.601s.

**2.** One row of `lm_head` is one token's scoring direction; a 10⁶ weight
makes that token's logit explode whenever its input feature is nonzero, so
sampling collapses toward one character. Low temperature *amplifies* the
takeover (gaps multiply); high temperature partially hides it. If you picked
a weight column whose feature is often near zero, the damage is intermittent
— congratulations, you've built a flaky bug.

**3.** Loop the prefix through `gpt(token_id, pos_id, keys, values)` exactly
as the sampler does, but *ignore* the returned logits until the last prefix
character; then continue the normal sampling loop from there. The KV lists
are the only "memory" of the prefix — which is the entire mental model you
need for "context window" in production systems.

</details>

---

Next: [train7 — the ablation lab](train7.md). You've built it and shipped it;
now take it apart, organ by organ, and check your predictions against the
damage report.
