# Glossary

Every term the course uses, in plain words, with the rung where it's *earned*
— the lessons introduce each one in place; this page is for looking things up
mid-course. "aka" lists the other names the same idea travels under, so
nothing here ever reads as two concepts when it's one.

- **ablation** — removing one component and retraining, to measure what that
  component was worth. A fact about an organ *at a budget*, never in general.
  Earned at rung 7.
- **Adam** — the optimizer that gives every parameter its own momentum (`m`)
  and its own step size (via `v`). aka adaptive moment estimation. Rung 5.
- **attention** — the mechanism that lets a position consult earlier
  positions: its *query* is dotted against their *keys*, softmax turns the
  match scores into weights, and the weighted average of their *values* is
  what it reads back. A soft dictionary lookup. Rung 3.
- **attention sink** — a token (here BOS) that attention rests weight on when
  it has nothing useful to fetch. Visible in our heatmaps; documented in
  production models too. Rung 3.
- **autograd** — automatic differentiation: every operation records its
  inputs and its local derivative during the forward pass, and `backward()`
  replays the chain rule through that record. aka backpropagation (backprop),
  when applied to neural networks. Rung 2.
- **backward pass** — the blame-assignment sweep from the loss to every
  parameter. Output: a gradient per parameter. aka backprop. Rungs 1 (by
  hand) and 2 (automated).
- **batch size** — how many documents contribute to one gradient step. This
  whole course uses batch size 1, which is why the loss curves jitter.
- **bias correction** — Adam's early-step fix: its running averages start at
  zero, so dividing by `1 − β^t` un-shrinks them. Rung 5.
- **BOS** — Beginning of Sequence, the one special token. Marks both "start
  generating" and, when emitted, "I'm done." Drawn as `^` in diagrams.
  Rung 0.
- **checkpoint** — the model saved to disk. Here: `out/model.json`, a dict of
  float lists and nothing else. aka weights file. Rung 6.
- **computation graph** — the record autograd builds: every intermediate
  number as a node, wired to the nodes it was computed from. One 16-char
  document builds ~126,000 nodes here. Rung 2.
- **context window** — the maximum sequence length the model can see; here
  40, and literally the number of rows in the position-embedding table.
  aka block size, context length. Rung 3.
- **cross-entropy** — the textbook name for this course's loss: the average
  of `-log P(the true next token)`. aka negative log-likelihood (NLL), aka
  rung 0's "surprise," aka `avg_nll` in the code. One quantity, four names.
  Rung 0 (concept), rung 1 (name).
- **effective choices** — `e^(loss)`: out of all possible next tokens, how
  many the model is still effectively guessing among. Uniform = vocabulary
  size, perfect = 1. aka perplexity. This course's favorite instrument.
  Rung 0.
- **embedding** — the learned vector of numbers (here 16) the model
  associates with each token; the only thing it "knows" about a character.
  A row of the `wte` table. Rung 1.
- **epoch** — one full pass over the training data. Frontier models train
  for *less* than one; this course does about two. Rung 5's lesson, en route.
- **gradient** — for each parameter, how much the loss would change if that
  parameter nudged up: the direction of steepest blame. Computed numerically
  (perturb and measure) or analytically (chain rule) at rung 1; automatically
  from rung 2 on.
- **gradient check** — running both gradient methods and comparing. The
  referee that catches silent calculus bugs; ours agrees to eight decimals.
  Rung 1.
- **head / multi-head** — one attention lookup with its own query-key match
  and its own softmax. Multi-head splits the embedding into slices (here 4×4
  dims) so each slice can look somewhere different, at zero extra parameter
  cost. Rung 4.
- **hidden units** — the intermediate numbers between two linear layers (here
  64); workspace, neither input nor output. Rung 1.
- **KV cache** — the `keys` and `values` lists kept while generating, so each
  new token costs one model call instead of recomputing the whole prefix.
  Measured at rung 6: 23 calls with, 276 without, for one 22-char name.
  Rungs 3 (appears) and 6 (earns its name).
- **learning rate** — the step-size multiplier on every update. Ours starts
  big and decays linearly to zero: bold early, careful late. aka lr. Rung 1.
- **linear layer** — a matrix multiply: every output is a weighted sum of all
  inputs. `linear()` in the code. aka fully-connected layer, dense layer.
  Rung 1.
- **logits** — the raw scores the model outputs, one per token in the
  vocabulary, before softmax turns them into probabilities. Rung 1.
- **loss** — the number being minimized; here, average surprise at the true
  next character. See cross-entropy. Rung 0.
- **memorization** — emitting training data verbatim instead of
  generalizing. Our gauge reads 0/20 all course; rung 6 explains why that's
  the interesting outcome on this corpus.
- **MLP** — multi-layer perceptron: linear → nonlinearity → linear. The
  "compute" half of a transformer block ("communicate, then compute").
  Rungs 1 (alone) and 3 (in the block).
- **momentum** — a running average of recent gradients; a memory of
  direction that smooths one-document noise. Adam's `m`. Rung 5.
- **nats** — the units of our loss: information measured with the natural
  log, bits' base-e cousin. Rung 0.
- **parameters** — the numbers training is allowed to change; the model
  itself. Here 4,928 of them (bigram rungs: 4,064). aka weights, and stored
  in the `state_dict`. Rung 0.
- **position embeddings** — a second table (`wpe`), one learned vector per
  *position*, added to the token's so the model knows where it is. Rung 3.
- **query / key / value** — the three vectors each position computes for
  attention: what I'm looking for / what I contain / what I'll contribute if
  chosen. aka q/k/v. Rung 3.
- **ReLU** — `max(0, x)`: negatives become zero, positives pass. The cheapest
  nonlinearity; without one, stacked linear layers collapse into one. Rung 1.
- **residual stream** — the running vector each block *adds* its adjustment
  to (`x = x + block(x)`) instead of replacing; also the gradient's highway
  back to early layers (the local gradient of `+` is 1). aka skip
  connections. Rung 3, measured at rung 7.
- **rmsnorm** — rescale a vector to unit root-mean-square before sensitive
  operations; the thermostat that keeps activations at a standard scale.
  Cousin of layernorm. Rung 3.
- **sampling** — generating: turn logits into probabilities, draw one token,
  append, repeat until BOS. Rung 0 onward; temperature added at rung 6.
- **SGD** — stochastic gradient descent: step every parameter against its
  gradient, where each step trusts one document ("stochastic") rather than
  the whole dataset. Rung 1.
- **softmax** — turns any list of scores into probabilities: exponentiate
  (everything positive, gaps multiplicative), then normalize to sum to 1.
  Rung 1.
- **state_dict** — the named collection of parameter matrices; save it and
  you've saved the model. Rungs 0 (as a count table) through 6 (as JSON).
- **temperature** — divide the logits before softmax: below 1 sharpens
  toward the safest choice, above 1 flattens toward noise. The one-knob
  personality dial. Rung 6.
- **token** — the unit the model reads and writes; here, single characters
  (production models use word-chunks via BPE — see the epilogue). Rung 0.
- **training** — the loop: forward (predict), loss (measure surprise),
  backward (assign blame), update (step parameters), repeat. Every rung is
  this loop with a better part swapped in.
- **val / validation set** — the held-out 10% the model never trains on;
  the only numbers worth trusting. The gap between train and val loss is
  where overfitting shows. Rung 0.
- **vocabulary** — the set of all tokens; here 38 (26 letters, 10 digits,
  hyphen, BOS). aka vocab. Rung 0.
