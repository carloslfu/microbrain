"""
train1.py: Bigram language model with a single-layer MLP, trained by gradient descent.

Same as train0.py:
- Dataset, tokenizer, training loop structure, inference, instruments

Different from train0.py:
- Model is a neural network (MLP) instead of a count table
- Training is gradient descent (SGD) instead of counting
- Introduces: softmax, linear, numerical and analytic gradients

The MLP is effectively a differentiable version of the bigram count table:
token_id -> embedding lookup -> hidden layer -> logits -> softmax -> probs.
The gradient tells us how to nudge each parameter to reduce the loss. We show
two ways to compute it: numerically (perturb and measure) and analytically
(chain rule). They agree, but the analytic version is O(params) faster.

microbrain deltas:
- instrument panel at checkpoints (val loss + effective choices)
- diagram: a 1-D cross-section of the loss landscape, sweeping one trained
  weight while holding the other 4,063 fixed — the valley SGD walked into

usage: python train1.py [--fast]   (--fast: 300 steps instead of 1000)
"""

import os       # os.path.exists
import sys      # sys.argv
import math     # math.log, math.exp
import time     # time.time
import random   # random.seed, random.choices, random.gauss
random.seed(42)
FAST = '--fast' in sys.argv

# Dataset: the names of ideas (already shuffled by data/make_dataset.py, seed 42)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.txt')
assert os.path.exists(DATA), "no data/data.txt yet — run: python data/make_dataset.py"
docs = [l.strip() for l in open(DATA).read().strip().split('\n') if l.strip()] # list[str] of documents
val_docs = docs[-len(docs) // 10:]
train_docs = docs[:-len(docs) // 10]
print(f"num docs: {len(train_docs)} train / {len(val_docs)} val")

# Tokenizer: character-level, with a special BOS (Beginning of Sequence) token
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Initialize the parameters
n_embd = 16     # embedding dimension
matrix = lambda nout, nin: [[random.gauss(0, 0.08) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'mlp_fc1': matrix(4 * n_embd, n_embd),
    'mlp_fc2': matrix(vocab_size, 4 * n_embd),
}
params = [(row, j) for mat in state_dict.values() for row in mat for j in range(len(row))]
print(f"num params: {len(params)}")

# Model: token_id -> logits
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def mlp(token_id):
    x = state_dict['wte'][token_id]
    x = linear(x, state_dict['mlp_fc1'])
    x = [max(0, xi) for xi in x] # relu
    logits = linear(x, state_dict['mlp_fc2'])
    return logits

# Forward pass: run the model on a token sequence, return the average loss
def forward(tokens, n):
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = mlp(token_id)
        probs = softmax(logits)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    return loss

# Instrument: average loss per token over a set of documents
def avg_nll(eval_docs):
    total, count = 0.0, 0
    for doc in eval_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = len(tokens) - 1
        total += forward(tokens, n) * n
        count += n
    return total / count

# Two ways to compute the gradient of the loss w.r.t. all parameters:

def numerical_gradient(tokens, n):
    """Perturb each parameter by eps, measure change in loss."""
    loss = forward(tokens, n)
    eps = 1e-5
    grad = []
    for mat in state_dict.values():
        for row in mat:
            for j in range(len(row)):
                old = row[j]
                row[j] = old + eps
                loss_plus = forward(tokens, n)
                row[j] = old
                grad.append((loss_plus - loss) / eps)
    return loss, grad

def analytic_gradient(tokens, n):
    """Derive the gradient analytically using the chain rule."""
    grad = {k: [[0.0] * len(row) for row in mat] for k, mat in state_dict.items()}
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        # forward pass (saving intermediates for backward)
        x = list(state_dict['wte'][token_id])
        h_pre = linear(x, state_dict['mlp_fc1'])
        h = [max(0, v) for v in h_pre]
        logits = linear(h, state_dict['mlp_fc2'])
        probs = softmax(logits)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)
        # backward pass: chain rule, layer by layer
        # d(loss)/d(logits) for softmax + cross-entropy = probs - one_hot(target)
        dlogits = [p / n for p in probs]
        dlogits[target_id] -= 1.0 / n
        # d(loss)/d(mlp_fc2), d(loss)/d(h): logits = mlp_fc2 @ h
        dh = [0.0] * len(h)
        for i in range(len(dlogits)):
            for j in range(len(h)):
                grad['mlp_fc2'][i][j] += dlogits[i] * h[j]
                dh[j] += state_dict['mlp_fc2'][i][j] * dlogits[i]
        # d(loss)/d(h_pre): relu backward
        dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(len(h_pre))]
        # d(loss)/d(mlp_fc1), d(loss)/d(x): h_pre = mlp_fc1 @ x
        dx = [0.0] * len(x)
        for i in range(len(dh_pre)):
            for j in range(len(x)):
                grad['mlp_fc1'][i][j] += dh_pre[i] * x[j]
                dx[j] += state_dict['mlp_fc1'][i][j] * dh_pre[i]
        # d(loss)/d(wte[token_id]): x = wte[token_id]
        for j in range(len(x)):
            grad['wte'][token_id][j] += dx[j]
    loss = (1 / n) * sum(losses)
    grad_flat = [g for mat in grad.values() for row in mat for g in row]
    return loss, grad_flat

# Gradient check: verify numerical and analytic gradients agree
doc = train_docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = len(tokens) - 1
loss_n, grad_n = numerical_gradient(tokens, n)
loss_a, grad_a = analytic_gradient(tokens, n)
grad_diff = max(abs(gn - ga) for gn, ga in zip(grad_n, grad_a))
print(f"gradient check on '{doc}' | loss_n {loss_n:.6f} | loss_a {loss_a:.6f} | max diff {grad_diff:.8f}")

# Train the model
t0 = time.time()
num_steps = 300 if FAST else 1000
learning_rate = 1.0
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = len(tokens) - 1

    # Forward + backward pass
    loss, grad = analytic_gradient(tokens, n)

    # SGD update
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, (row, j) in enumerate(params):
        row[j] -= lr_t * grad[i]

    # Instrument panel at checkpoints
    if step in (0, num_steps // 2, num_steps - 1):
        vl = avg_nll(val_docs)
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.4f} | val loss {vl:.4f} | effective choices {math.exp(vl):.1f} of {vocab_size}")
    elif step < 5:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.4f}")

print(f"training took {time.time() - t0:.1f}s")

# Diagram: 1-D cross-section of the loss landscape around the trained model.
# One weight (the 'a'-logit's first input) sweeps left-right; all other params stay
# put; the curve is this document's loss. SGD left us near the bottom of the valley.
doc = train_docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = len(tokens) - 1
row, j = state_dict['mlp_fc2'][uchars.index('a')], 0
w_trained = row[j]
xs = [w_trained + (i - 20) * 0.15 for i in range(41)]
ys = []
for w in xs:
    row[j] = w
    ys.append(forward(tokens, n))
row[j] = w_trained
print(f"\n--- the loss valley for one weight (doc: '{doc}') ---")
lo, hi = min(ys), max(ys)
height = 12
for r in range(height + 1):
    level = hi - (hi - lo) * r / height
    line = ''.join('W' if abs(xs[i] - w_trained) < 0.075 and ys[i] <= level else
                   ('*' if ys[i] >= level and (r == height or ys[i] < hi - (hi - lo) * (r - 1) / height) else ' ')
                   for i in range(41))
    print(f"  loss {level:6.3f} | {line}")
print(f"  weight sweeps {xs[0]:+.2f} .. {xs[-1]:+.2f}, W = trained value {w_trained:+.3f}, min loss {lo:.4f}")

# Inference: sample new idea names from the model
temperature = 0.5
train_set = set(train_docs)
print("\n--- inference (new, hallucinated idea names) ---")
memorized = 0
num_samples = 10 if FAST else 20
for sample_idx in range(num_samples):
    token_id = BOS
    sample = []
    for pos_id in range(40):
        logits = mlp(token_id)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    text = ''.join(sample)
    tag = '  <- verbatim training doc' if text in train_set else ''
    memorized += text in train_set
    print(f"sample {sample_idx+1:2d}: {text}{tag}")
print(f"memorization: {memorized}/{num_samples} samples are verbatim training docs")
