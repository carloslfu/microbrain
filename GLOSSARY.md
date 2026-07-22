# Glossary

Every term the course uses, in plain words, with the rung where it's *earned*
— the lessons introduce each one in place; this page is for looking things up
mid-course. "aka" lists the other names the same idea travels under, so
nothing here ever reads as two concepts when it's one. Aliases also get their
own one-line entries, so a cold lookup by any name lands.

#### ablation

removing one component and retraining, to measure what that
component was worth. A fact about an organ *at a budget*, never in general.
Earned at rung 7.

#### activations

the intermediate numbers flowing through the network for
one input (everything between the embedding and the logits). Parameters are
learned; activations are computed fresh every forward pass.

#### Adam

the optimizer that gives every parameter its own momentum (`m`)
and its own step size (via `v`). aka adaptive moment estimation. Rung 5.

#### argmax

always picking the single highest-scoring option instead of
sampling. Temperature → 0 turns sampling into argmax. Rung 6.

#### attention

the mechanism that lets a position consult earlier
positions: its *query* is dotted against their *keys*, softmax turns the
match scores into weights, and the weighted average of their *values* is
what it reads back. A soft dictionary lookup. Rung 3.

#### attention sink

a token (here BOS) that attention rests weight on when
it has nothing useful to fetch. Visible in our heatmaps; documented in
production models too. Rung 3.

#### autograd

automatic differentiation: every operation records its
inputs and its local derivative during the forward pass, and `backward()`
replays the chain rule (see entry) through that record. aka backpropagation
(backprop), when applied to neural networks. Rung 2.

#### backpropagation / backprop

see **autograd** and **backward pass**.

#### backward pass

the blame-assignment sweep from the loss to every
parameter. Output: a gradient per parameter. Rungs 1 (by hand) and 2
(automated).

#### batch size

how many documents contribute to one gradient step. This
whole course uses batch size 1, which is why the loss curves jitter.

#### bias correction

Adam's early-step fix: its running averages start at
zero, so dividing by `1 − β^t` un-shrinks them. Rung 5.

#### bigram

a model that predicts the next token from the current token
alone: two-token patterns, nothing longer. A *trigram* uses the previous
two; an *n-gram* the previous n−1. Rung 0.

#### BOS

Beginning of Sequence, the one special token. Marks both "start
generating" and, when emitted, "I'm done." Drawn as `^` in diagrams.
Rung 0.

#### BPE

byte-pair encoding: how production tokenizers build word-chunk
tokens ("the", "ing") instead of single characters. Epilogue.

#### chain rule

how a change flows through nested computations: multiply
the step-by-step sensitivities along the path. The whole of backpropagation
is this, applied many times. Rungs 1–2.

#### checkpoint

the model saved to disk. Here: `out/model.json`, a dict of
float lists and nothing else. aka weights file. Rung 6.

#### closed-form

answerable by a direct, exact formula, no trial-and-error
needed. Counting is the closed-form solution to the bigram question.
Rung 0.

#### computation graph

the record autograd builds: every intermediate
number as a node, wired to the nodes it was computed from. One 16-char
document builds ~126,000 nodes here. Rung 2.

#### context window

the maximum sequence length the model can see; here
40, and literally the number of rows in the position-embedding table.
aka block size, context length. Rung 3.

#### corpus

the collection of documents a model trains on. Ours: 543 idea
names. Rung 0.

#### cross-entropy

the textbook name for this course's loss: the average
of `-log P(the true next token)`. aka negative log-likelihood (NLL), aka
rung 0's "surprise," aka `avg_nll` in the code. One quantity, four names.
Rung 0 (concept), rung 1 (name).

#### derivative

how sensitive one quantity is to a nudge in another. The
gradient is the derivative of the loss with respect to each parameter.
Rung 1.

#### diff

a line-by-line comparison of two files showing exactly what
changed; also the terminal command (`diff a.py b.py`) that prints one.
The lessons show the meaningful changes inline so you never need to run it.

#### dot product

multiply matching entries of two equal-length lists and
add the results: the cheapest "how similar are these two vectors?" score.
Attention's match score. Rung 3.

#### effective choices

`e^(loss)`: out of all possible next tokens, how
many the model is still effectively guessing among. Uniform = vocabulary
size, perfect = 1. aka perplexity. This course's favorite instrument.
Rung 0.

#### embedding

the learned vector of numbers (here 16) the model
associates with each token; the only thing it "knows" about a character.
A row of the `wte` table. Rung 1.

#### epoch

one full pass over the training data. Frontier models train
for *less* than one; this course does about two. Rung 5's lesson, en route.

#### gist

GitHub's format for sharing a single file. Both Karpathy
originals live in gists, pinned in `reference/`.

#### GPT

Generative Pre-trained Transformer: the family of
next-token-predicting models behind ChatGPT and friends. This course builds
one, complete, at 4,928 parameters.

#### GPU

the massively parallel processor deep learning runs on; the
epilogue's "everything else is efficiency" is largely about feeding these.

#### gradient

for each parameter, how much the loss would change if that
parameter nudged up: the direction of steepest blame. Computed numerically
(perturb and measure) or analytically (chain rule) at rung 1; automatically
from rung 2 on.

#### gradient check

running both gradient methods and comparing. The
referee that catches silent calculus bugs; ours agrees to eight decimals.
aka gradcheck. Rung 1.

#### head / multi-head

one attention lookup with its own query-key match
and its own softmax. Multi-head splits the embedding into slices (here 4×4
dims) so each slice can look somewhere different, at zero extra parameter
cost. Rung 4.

#### heatmap

a grid of shaded cells, darker = bigger value. Our shade
ramp, light to dark: `.` `:` `-` `=` `+` `*` `#` `%` `@`. Rung 0.

#### hidden units

the intermediate numbers between two linear layers (here
64); workspace, neither input nor output. Rung 1.

#### inference

running a trained model to get outputs; no gradients, no
learning. *Serving* is doing inference for many users at once. Rung 6.

#### init

initialization: the starting values of all parameters, small
random numbers here. Fixed by the seed, so runs reproduce. Rung 1.

#### instruments

this course's word for the measurement code every rung
carries: val loss, effective choices, the memorization gauge, and the
self-drawn diagrams. Instruments never touch the model.

#### KV cache

the `keys` and `values` lists kept while generating, so each
new token costs one model call instead of recomputing the whole prefix.
Measured at rung 6: 23 calls with, 276 without, for one 22-char name.
Rungs 3 (appears) and 6 (earns its name).

#### Laplace smoothing

add one imaginary count to every cell so nothing
has probability exactly zero ("never say never"). aka add-one smoothing.
Rung 0 — and its removal is rung 0's break-it crash.

#### learning rate

the step-size multiplier on every update. Ours starts
big and decays linearly to zero: bold early, careful late. aka lr. Rung 1.

#### linear layer

a matrix multiply: every output is a weighted sum of all
inputs. `linear()` in the code. aka fully-connected layer, dense layer.
Rung 1.

#### lm_head

the final linear layer, turning the 16-number stream back
into 38 token scores: the "language-model head," the model's mouth. Rung 3.

#### logits

the raw scores the model outputs, one per token in the
vocabulary, before softmax turns them into probabilities. Rung 1.

#### loss

the number being minimized; here, average surprise at the true
next character. See cross-entropy. Rung 0.

#### matmul / kernel

matrix multiplication, and the hand-tuned GPU
routines (kernels — e.g. FlashAttention) that compute it at full hardware
speed. Same math as our `linear()`. Epilogue.

#### memorization

emitting training data verbatim instead of
generalizing. Our gauge reads 0/20 all course; rung 6 explains why that's
the interesting outcome on this corpus.

#### MLP

multi-layer perceptron: linear → nonlinearity → linear. The
"compute" half of a transformer block ("communicate, then compute").
Rungs 1 (alone) and 3 (in the block).

#### MoE

mixture of experts: a production MLP split into many "experts"
of which only a few fire per token — most of the network asleep at any
moment. Epilogue.

#### momentum

a running average of recent gradients; a memory of
direction that smooths one-document noise. Adam's `m`. Rung 5.

#### morpheme

a meaningful chunk of a word (`-ing`, `pre-`, `-tion`).
What the samples visibly acquire between steps 50 and 1000.

#### nats

the unit our loss comes in: like bits, but counted with the
natural logarithm instead of base 2. You never need to convert — just
compare numbers. Rung 0.

#### negative log-likelihood / NLL

see **cross-entropy**.

#### neural network

a function built from simple layers (linear +
nonlinearity) whose parameters are set by training rather than by hand.
Rung 1 builds the smallest useful one.

#### normalize

rescale numbers to a standard total or size (probabilities
to sum 1; vectors to standard length). Softmax and rmsnorm both normalize.

#### one-hot

a list of zeros with a single 1 marking the true answer.
`probs - one_hot(target)` is the whole gradient of softmax+cross-entropy.
Rung 1.

#### optimizer

the rule that turns gradients into parameter updates. SGD
(rung 1) and Adam (rung 5) are the two this course runs — and the gap
between them is the course's plot twist.

#### overfitting / underfitting

fitting the training data better than the
val data (memorizing quirks), vs. failing to fit even the training data
(not enough capacity or training). The train/val gap is the gauge. Rung 1
onward.

#### parameters

the numbers training is allowed to change; the model
itself. Here 4,928 of them (bigram rungs: 4,064). aka weights, and stored
in the `state_dict`. Rung 0.

#### perplexity

see **effective choices**.

#### position embeddings

a second table (`wpe`), one learned vector per
*position*, added to the token's so the model knows where it is. Rung 3.

#### query / key / value

the three vectors each position computes for
attention: what I'm looking for / what I contain / what I'll contribute if
chosen. aka q/k/v. Rung 3.

#### ReLU

`max(0, x)`: negatives become zero, positives pass. The cheapest
nonlinearity; without one, stacked linear layers collapse into one. Rung 1.

#### residual stream

the running vector each block *adds* its adjustment
to (`x = x + block(x)`) instead of replacing; also the gradient's highway
back to early layers (the local gradient of `+` is 1). aka skip
connections. Rung 3, measured at rung 7.

#### rmsnorm

rescale a vector so its typical entry size is 1 (unit
root-mean-square) before sensitive operations; the thermostat that keeps
activations at a standard scale. Cousin of layernorm. Rung 3.

#### RoPE

rotary position embeddings: production's replacement for the
`wpe` table — positions as rotations, so there's no table to run out of.
Epilogue.

#### rung

one step of this course's ladder: a runnable file plus its
lesson, adding exactly one idea.

#### sampling

generating: turn logits into probabilities, draw one token,
append, repeat until BOS. Rung 0 onward; temperature added at rung 6.

#### seed

the number that fixes the random generator's choices; same seed
→ same "random" numbers → bit-identical runs. Ours is 42, which is why
your logs can match the lessons exactly.

#### SFT / RL

supervised fine-tuning (the same training loop pointed at
curated conversations) and reinforcement learning (updates weighted by a
reward instead of next-token surprise): what happens to a GPT *after*
pretraining. Same engine, new data and objective. Epilogue.

#### SGD

stochastic gradient descent: step every parameter against its
gradient, where each step trusts one document ("stochastic") rather than
the whole dataset. Rung 1.

#### skip connections

see **residual stream**.

#### slug

a short, hyphenated, identifier-style name, like
`test-time-training`. Our corpus is 543 of them.

#### softmax

turns any list of scores into probabilities: exponentiate
(everything positive, bigger gaps exaggerated), then normalize to sum
to 1. Rung 1.

#### state_dict

the named collection of parameter matrices; save it and
you've saved the model. Rungs 0 (as a count table) through 6 (as JSON).

#### temperature

divide the logits before softmax: below 1 sharpens
toward the safest choice, above 1 flattens toward noise. The one-knob
personality dial. Rung 6.

#### tensor / scalar

a scalar is one number; a tensor is a block of many
(a matrix and beyond). Our `Value` wraps scalars; PyTorch wraps tensors —
same autograd idea, bookkeeping amortized. Rung 2 / epilogue.

#### token

the unit the model reads and writes; here, single characters
(production models use word-chunks via BPE — see the epilogue). Rung 0.

#### top-p

production sampling's extra filter: sample only from the
smallest set of tokens whose probabilities add up to p. Epilogue.

#### training

the loop: forward (predict), loss (measure surprise),
backward (assign blame), update (step parameters), repeat. Every rung is
this loop with a better part swapped in.

#### transformer

the architecture GPTs use: blocks of attention
(communicate between positions) and MLP (compute within one), writing onto
a residual stream, stacked in layers. Built across rungs 3–4.

#### val / validation set

the held-out 10% the model never trains on;
the only numbers worth trusting. The gap between train and val loss is
where overfitting shows. Rung 0.

#### vocabulary

the set of all tokens; here 38 (26 letters, 10 digits,
hyphen, BOS). aka vocab. Rung 0.

#### weights

see **parameters**.
