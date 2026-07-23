"""
train5.py: Full GPT transformer, trained with Adam. This is microgpt.

Same as train4.py:
- Everything.

Different from train4.py:
- Adam optimizer instead of SGD. Adam tracks per-parameter running averages of
  the gradient (momentum) and squared gradient (adaptive learning rate), so each
  parameter gets its own effective step size. This is the final piece.

This file is the microbrain twin of Karpathy's microgpt.py (reference/microgpt.py):
same architecture, same optimizer, same everything — the only differences are the
corpus (idea names, not human names), block_size 40, and the instruments below.

microbrain deltas:
- diagram 1: SGD vs Adam loss curves overlaid (reads out/train4_losses.txt —
  run train4.py first to see both curves)
- diagram 2: the childhood album — samples taken at steps 0, 50, 250, and the
  end, so you can watch babble become structure

usage: python train5.py [--fast]   (--fast: 300 steps; full run is ~minutes, pure Python)
lesson: lessons/train5-adam.md — run first, read second
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

# Autograd: recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters
n_embd = 16     # embedding dimension
block_size = 40 # maximum sequence length (BOS + up to 39 characters)
n_head = 4      # number of attention heads
n_layer = 1     # number of layers
head_dim = n_embd // n_head # dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# Model: token_id -> logits
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    return probs

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Instrument: average loss per token over a set of documents (forward only)
def avg_nll(eval_docs):
    total, count = 0.0, 0
    for doc in eval_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        for pos_id in range(n):
            probs = softmax(gpt(tokens[pos_id], pos_id, keys, values))
            total += -math.log(probs[tokens[pos_id + 1]].data)
            count += 1
    return total / count

# Sampling (used mid-training for the childhood album, and at the end)
def sample_names(num, temperature=0.5):
    out = []
    for _ in range(num):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        out.append(''.join(sample))
    return out

# Adam optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Train the model
t0 = time.time()
num_steps = 300 if FAST else 1000
album_steps = {0, 50, 250, num_steps} # 0 = before any training
album = [(0, sample_names(6))]
loss_history = []
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward pass
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)
    loss_history.append(loss.data)

    # Backward pass
    loss.backward()

    # Adam update
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    # Childhood album: photograph the model at a few ages
    if (step + 1) in album_steps:
        album.append((step + 1, sample_names(6)))

    # Instrument panel at sampled steps
    if step in (0, num_steps // 2, num_steps - 1):
        vl = avg_nll(val_docs)
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | val loss {vl:.4f} | effective choices {math.exp(vl):.1f} of {vocab_size}")
    elif step < 5:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

print(f"training took {time.time() - t0:.1f}s")

# Leave the loss curve on disk (train7/compare reuse it)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'train5_losses.txt'), 'w') as f:
    f.write('\n'.join(f'{l:.6f}' for l in loss_history) + '\n')

# Diagram 1: SGD (train4) vs Adam (train5), same model, same data, same steps.
# Curves are smoothed with an exponential moving average so the shape is visible.
sgd_path = os.path.join(OUT, 'train4_losses.txt')
if os.path.exists(sgd_path):
    sgd = [float(l) for l in open(sgd_path)]
    def smooth(ys, alpha=0.05):
        out, acc = [], ys[0]
        for y in ys:
            acc = (1 - alpha) * acc + alpha * y
            out.append(acc)
        return out
    a_s, s_s = smooth(loss_history), smooth(sgd)
    steps_shown = min(len(a_s), len(s_s))
    width, height = 64, 14
    cols = [int(i * (steps_shown - 1) / (width - 1)) for i in range(width)]
    lo = min(min(a_s), min(s_s)); hi = max(max(a_s), max(s_s))
    grid = [[' '] * width for _ in range(height + 1)]
    for x, c in enumerate(cols):
        for ys, ch in ((s_s, 's'), (a_s, 'A')):
            r = round((hi - ys[c]) / (hi - lo) * height)
            grid[r][x] = '*' if grid[r][x] not in (' ', ch) else ch
    print(f"\n--- SGD (s, train4) vs Adam (A, train5), smoothed loss over {steps_shown} steps ---")
    for r, row in enumerate(grid):
        label = f"{hi - (hi - lo) * r / height:5.2f}"
        print(f"  {label} | {''.join(row)}")
    print(f"        +{'-' * width}")
    print(f"         step 1 {' ' * (width - 16)} step {steps_shown}")
else:
    print("\n(run train4.py first to see the SGD-vs-Adam overlay diagram)")

# Diagram 2: the childhood album — what the model babbled at each age
print("\n--- the childhood album: samples through training ---")
for step_num, samples in album:
    print(f"step {step_num:4d} | {', '.join(samples)}")

# Inference: sample new idea names from the model
train_set = set(train_docs)
print("\n--- inference (new, hallucinated idea names) ---")
memorized = 0
num_samples = 10 if FAST else 20
samples = sample_names(num_samples)
for sample_idx, text in enumerate(samples):
    tag = '  <- verbatim training doc' if text in train_set else ''
    memorized += text in train_set
    print(f"sample {sample_idx+1:2d}: {text}{tag}")
print(f"memorization: {memorized}/{num_samples} samples are verbatim training docs")
