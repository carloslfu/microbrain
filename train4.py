"""
train4.py: Full GPT transformer, trained with SGD.

Same as train3.py:
- Dataset, tokenizer, autograd, position embeddings, attention, MLP, rmsnorm,
  residual connections, SGD optimizer, inference, instruments

Different from train3.py:
- Multi-head attention (n_head=4) instead of single head
- Configurable n_layer with layer loop and layer-prefixed state_dict keys

The model is now identical to train5.py. The only remaining difference is the
optimizer: SGD here vs Adam there. Note the parameter count printed below is
EXACTLY the same as train3.py's: four heads are not extra capacity, they are
the same 16 dimensions split into 4 independent slices of 4.

microbrain deltas:
- diagram: the four heads' attention over the same document, side by side —
  same weights budget as train3's one head, four different places to look
- writes out/train4_losses.txt so train5.py can overlay SGD vs Adam

usage: python train4.py [--fast]   (--fast: 300 steps; full run is ~minutes, pure Python)
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
print(f"num params: {len(params)} (same as train3.py — heads are a split, not new capacity)")

# Model: token_id -> logits
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values, attn_trace=None):
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
        head_rows = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            if attn_trace is not None and li == 0:
                head_rows.append([w.data for w in attn_weights])
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        if attn_trace is not None:
            attn_trace.append(head_rows)
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

# Train the model
t0 = time.time()
num_steps = 300 if FAST else 1000
learning_rate = 0.1
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

    # SGD update
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        p.data -= lr_t * p.grad
        p.grad = 0

    # Instrument panel at checkpoints
    if step in (0, num_steps // 2, num_steps - 1):
        vl = avg_nll(val_docs)
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | val loss {vl:.4f} | effective choices {math.exp(vl):.1f} of {vocab_size}")
    elif step < 5:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

print(f"training took {time.time() - t0:.1f}s")

# Leave the loss curve on disk so train5.py can overlay SGD vs Adam
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'train4_losses.txt'), 'w') as f:
    f.write('\n'.join(f'{l:.6f}' for l in loss_history) + '\n')

# Diagram: the four heads over the same document, one triangle each. Same
# parameter budget as train3's single head — but four independent places to look.
probe = 'test-time-training'
if probe not in train_docs:
    probe = train_docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in probe]
keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
trace = []
for pos_id, token_id in enumerate(tokens):
    gpt(token_id, pos_id, keys, values, attn_trace=trace)
labels = ['^'] + list(probe)
shades = ' .:-=+*#%@'
print(f"\n--- attention over '{probe}', per head (row = position, cols = looked-at) ---")
for h in range(n_head):
    print(f"head {h}:")
    print('      ' + ''.join(labels))
    for t, head_rows in enumerate(trace):
        row = head_rows[h]
        mx = max(row)
        line = ''.join(shades[min(9, int(10 * w / mx))] for w in row)
        print(f"  {labels[t]:>3} | {line}")

# Inference: sample new idea names from the model
temperature = 0.5
train_set = set(train_docs)
print("\n--- inference (new, hallucinated idea names) ---")
memorized = 0
num_samples = 10 if FAST else 20
for sample_idx in range(num_samples):
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
    text = ''.join(sample)
    tag = '  <- verbatim training doc' if text in train_set else ''
    memorized += text in train_set
    print(f"sample {sample_idx+1:2d}: {text}{tag}")
print(f"memorization: {memorized}/{num_samples} samples are verbatim training docs")
