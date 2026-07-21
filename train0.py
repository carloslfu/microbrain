"""
train0.py: Bigram language model trained by counting.

The structure is identical to train5.py: tokenize, forward pass, update, sample.
The only difference is what's inside the "model" box:
- train5.py: gpt(token_id) -> logits, trained by gradient descent
- train0.py: bigram(token_id) -> probs, trained by counting

A bigram model is a special case of a GPT where there is no attention (each token
only looks at itself), no MLP, and the "embedding" is just a row in a lookup table.
Counting is the closed-form solution for this case; gradient descent is what you
need when the model is too expressive for exact solutions.

microbrain deltas (all rungs carry these):
- the corpus is the names of ideas from a db.md knowledge store, not human names
  (543 docs instead of 32,033 — small data is part of this course, run
  data/make_dataset.py first)
- a validation split the model never trains on, and two honest numbers measured
  on it: val loss, and "effective choices" = e^(val loss) — out of 38 possible
  tokens, among how many is the model still really guessing?
- a memorization gauge over the samples
- a diagram drawn from the model itself (here: the count table as a heatmap)

usage: python train0.py        (instant — counting is one pass over the data)
"""

import os       # os.path.exists
import math     # math.log
import random   # random.seed, random.choices
random.seed(42)

# Dataset: the names of ideas (already shuffled by data/make_dataset.py, seed 42)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.txt')
assert os.path.exists(DATA), "no data/data.txt yet — run: python data/make_dataset.py"
docs = [l.strip() for l in open(DATA).read().strip().split('\n') if l.strip()] # list[str] of documents
val_docs = docs[-len(docs) // 10:]    # held out: the model never trains on these
train_docs = docs[:-len(docs) // 10]  # the docs the model learns from
print(f"num docs: {len(train_docs)} train / {len(val_docs)} val")

# Tokenizer: character-level, with a special BOS (Beginning of Sequence) token
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# The floor and the ceiling, before any learning happens:
# a model that knows nothing puts probability 1/38 on everything.
print(f"uniform loss: ln({vocab_size}) = {math.log(vocab_size):.4f} -> effective choices: {vocab_size}")

# Initialize the parameters: a bigram count table. state_dict[i][j] = how many times token j follows token i
state_dict = [[0] * vocab_size for _ in range(vocab_size)]

# The "model": given a token_id, return the probability distribution over the next token
def bigram(token_id):
    row = state_dict[token_id]
    total = sum(row) + vocab_size # add-one (Laplace) smoothing
    return [(c + 1) / total for c in row]

# Instrument: average negative log-likelihood per token over a set of documents
def avg_nll(eval_docs):
    total, count = 0.0, 0
    for doc in eval_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        for pos_id in range(len(tokens) - 1):
            probs = bigram(tokens[pos_id])
            total += -math.log(probs[tokens[pos_id + 1]])
            count += 1
    return total / count

# Train the model: counting needs to see each document exactly once
num_steps = len(train_docs)
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = len(tokens) - 1

    # Forward pass: compute the loss for this document (before counting it — an honest quiz)
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        probs = bigram(token_id)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # Update the model: incorporate this document's bigram counts
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        state_dict[token_id][target_id] += 1

    if step < 3 or (step + 1) % 100 == 0 or step == num_steps - 1:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.4f}")

# Instrument panel: measured on documents the model has never seen
val_loss = avg_nll(val_docs)
train_loss = avg_nll(train_docs)
print(f"\ntrain loss {train_loss:.4f} | val loss {val_loss:.4f} | effective choices {math.exp(val_loss):.1f} of {vocab_size}")

# Diagram: the entire model, drawn as a heatmap. Row = current token, column = next
# token, darker = more probable (each row is scaled by its own max).
print("\n--- the model itself: P(next | current), one row per current token ---")
shades = ' .:-=+*#%@'
labels = uchars + ['^'] # '^' stands for BOS
print('    ' + ''.join(labels))
for i in range(vocab_size):
    row = bigram(i)
    mx = max(row)
    line = ''.join(shades[min(9, int(10 * p / mx))] for p in row)
    print(f'  {labels[i]} {line}')

# A few rows in words: what the counts actually learned
for ch in ['^', '-', 'a', 'q']:
    i = BOS if ch == '^' else uchars.index(ch)
    probs = bigram(i)
    top = sorted(range(vocab_size), key=lambda j: -probs[j])[:3]
    top_str = ', '.join(f"'{uchars[j] if j < BOS else '^'}' {probs[j]:.2f}" for j in top)
    print(f"after '{ch}': {top_str}")

# Inference: sample new idea names from the model
train_set = set(train_docs)
print("\n--- inference (new, hallucinated idea names) ---")
memorized = 0
for sample_idx in range(20):
    token_id = BOS
    sample = []
    for _ in range(40): # maximum sequence length
        token_id = random.choices(range(vocab_size), weights=bigram(token_id))[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    text = ''.join(sample)
    tag = '  <- verbatim training doc' if text in train_set else ''
    memorized += text in train_set
    print(f"sample {sample_idx+1:2d}: {text}{tag}")
print(f"memorization: {memorized}/20 samples are verbatim training docs")
