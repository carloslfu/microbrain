"""
train3.py: Single-head attention + MLP, with position embeddings.

Same as train2.py:
- Dataset, tokenizer, autograd (Value class), SGD optimizer, inference, instruments

Different from train2.py:
- Model now sees the full sequence context, not just the current token
- Introduces: position embeddings (wpe), single-head attention, rmsnorm,
  residual connections, separate lm_head
- Model function takes (token_id, pos_id, keys, values) like train5.py's gpt()

The model is now structurally a GPT: embed -> attention -> MLP -> lm_head.
The only remaining differences from train5.py are: single head (vs multi-head),
single layer (vs configurable), and SGD (vs Adam).

microbrain deltas:
- block_size is 40 (idea names run longer than human names)
- diagram: after training, the attention weights over a real document, printed
  as a triangle — row t shows where position t actually looked
- gpt() takes an optional attn_trace list to capture those weights

usage: python train3.py [--fast]   (--fast: 300 steps; full run is ~minutes, pure Python)
lesson: lessons/train3.md — run first, read second
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
matrix = lambda nout, nin: [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'attn_wq': matrix(n_embd, n_embd),
    'attn_wk': matrix(n_embd, n_embd),
    'attn_wv': matrix(n_embd, n_embd),
    'attn_wo': matrix(n_embd, n_embd),
    'mlp_fc1': matrix(4 * n_embd, n_embd),
    'mlp_fc2': matrix(n_embd, 4 * n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
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

def gpt(token_id, pos_id, keys, values, attn_trace=None):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    # 1) Single-head attention block
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, state_dict['attn_wq'])
    k = linear(x, state_dict['attn_wk'])
    v = linear(x, state_dict['attn_wv'])
    keys.append(k)
    values.append(v)
    attn_logits = [sum(q[j] * keys[t][j] for j in range(n_embd)) / n_embd**0.5 for t in range(len(keys))]
    attn_weights = softmax(attn_logits)
    if attn_trace is not None:
        attn_trace.append([w.data for w in attn_weights])
    x_attn = [sum(attn_weights[t] * values[t][j] for t in range(len(values))) for j in range(n_embd)]
    x = linear(x_attn, state_dict['attn_wo'])
    x = [a + b for a, b in zip(x, x_residual)]
    # 2) MLP block
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, state_dict['mlp_fc1'])
    x = [xi.relu() for xi in x]
    x = linear(x, state_dict['mlp_fc2'])
    x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# Instrument: average loss per token over a set of documents (forward only)
def avg_nll(eval_docs):
    total, count = 0.0, 0
    for doc in eval_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        keys, values = [], []
        for pos_id in range(n):
            probs = softmax(gpt(tokens[pos_id], pos_id, keys, values))
            total += -math.log(probs[tokens[pos_id + 1]].data)
            count += 1
    return total / count

# Train the model
t0 = time.time()
num_steps = 300 if FAST else 1000
learning_rate = 0.1
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward pass
    keys, values = [], []
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

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

# Diagram: where did attention actually look? Run the trained model over one
# real document and print the attention weights as a triangle. Row t = the
# model is at position t (input char on the left), columns = the positions it
# could look back at. Darker = more weight; each row sums to 1.
probe = 'test-time-training'
if probe not in train_docs:
    probe = train_docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in probe]
keys, values, trace = [], [], []
for pos_id, token_id in enumerate(tokens):
    gpt(token_id, pos_id, keys, values, attn_trace=trace)
labels = ['^'] + list(probe)
shades = ' .:-=+*#%@'
print(f"\n--- attention over '{probe}' (row = current position, cols = looked-at positions) ---")
print('      ' + ''.join(labels))
for t, row in enumerate(trace):
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
    keys, values = [], []
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
