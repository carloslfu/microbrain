"""
train6.py: Inference toolkit. The first rung past Karpathy's ladder.

Same as train5.py:
- Everything about training. The model and optimizer do not change again.

Different from train5.py (all of it happens AFTER training):
- save/load: the model is written to out/model.json and resurrected from it.
  To make the point stick, we first ZERO every parameter (the model dies and
  babbles uniform noise), then load the file and verify val loss is restored
  bit-for-bit. The file of floats IS the model.
- the temperature ladder: the same trained model sampled at T=0.1 / 0.5 / 1.0,
  same random seed each time — one knob between timid and unhinged
- the KV-cache reveal: generate the same name with and without the cache and
  count model calls. The cache is why chatbots don't reread the whole chat
  before every character.
- the novelty filter: how many samples are verbatim training docs, and what
  survives the filter
- a quiz: real brain record or hallucination?

usage: python train6.py [--fast]   (--fast: 300 steps; full run is ~minutes, pure Python)
"""

import os       # os.path.exists
import sys      # sys.argv
import math     # math.log, math.exp
import time     # time.time
import json     # json.dump, json.load
import random   # random.seed, random.choices, random.gauss, random.sample, random.shuffle
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

# Train the model (identical to train5.py)
t0 = time.time()
num_steps = 300 if FAST else 1000
for step in range(num_steps):
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
    loss = (1 / n) * sum(losses)
    loss.backward()
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
    if step in (0, num_steps // 2, num_steps - 1):
        vl = avg_nll(val_docs)
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | val loss {vl:.4f} | effective choices {math.exp(vl):.1f} of {vocab_size}")
print(f"training took {time.time() - t0:.1f}s — now the toolkit\n")

# ---------------------------------------------------------------- the toolkit

# 1) Save. The entire model is a dict of float lists. Nothing else.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
model_path = os.path.join(OUT, 'model.json')
checkpoint = {
    'n_embd': n_embd, 'block_size': block_size, 'n_head': n_head, 'n_layer': n_layer,
    'uchars': ''.join(uchars),
    'state_dict': {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()},
}
with open(model_path, 'w') as f:
    json.dump(checkpoint, f)
val_before = avg_nll(val_docs)
print(f"1) saved model to {os.path.relpath(model_path)} ({os.path.getsize(model_path):,} bytes) | val loss {val_before:.6f}")

# 2) Kill the model, then resurrect it from the file.
for p in params:
    p.data = 0.0
random.seed(123)
print(f"   all {len(params)} params zeroed. it now babbles: {sample_names(3, temperature=1.0)}")
loaded = json.load(open(model_path))
for k, mat in state_dict.items():
    for row, loaded_row in zip(mat, loaded['state_dict'][k]):
        for p, w in zip(row, loaded_row):
            p.data = w
val_after = avg_nll(val_docs)
print(f"   loaded it back. val loss {val_after:.6f} | identical: {val_after == val_before}")
print("   the JSON file of floats IS the model. everything else is scaffolding.\n")

# 3) The temperature ladder: same model, same seed, one knob.
print("2) temperature ladder (same random seed each row):")
for T in (0.1, 0.5, 1.0):
    random.seed(777)
    names = sample_names(6, temperature=T)
    print(f"   T={T:>3} | {', '.join(names)}")
print("   low T: argmax-ish, repetitive, safe. high T: adventurous, eventually noise.\n")

# 4) The KV-cache reveal: same name generated two ways.
def generate_kv_report(use_cache):
    random.seed(555)
    calls = 0
    tokens = [BOS]
    sample = []
    t_start = time.time()
    for pos_id in range(block_size):
        if use_cache:
            if pos_id == 0:
                kv = ([[] for _ in range(n_layer)], [[] for _ in range(n_layer)])
            logits = gpt(tokens[-1], pos_id, kv[0], kv[1])
            calls += 1
        else:
            kv = ([[] for _ in range(n_layer)], [[] for _ in range(n_layer)]) # start over every time
            for p_id, t_id in enumerate(tokens):
                logits = gpt(t_id, p_id, kv[0], kv[1])
                calls += 1
        probs = softmax([l / 0.8 for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        tokens.append(token_id)
        sample.append(uchars[token_id])
    return ''.join(sample), calls, time.time() - t_start

name_c, calls_c, secs_c = generate_kv_report(use_cache=True)
name_n, calls_n, secs_n = generate_kv_report(use_cache=False)
print("3) the KV cache, measured on one generation:")
print(f"   with cache:    '{name_c}' | {calls_c:3d} model calls | {secs_c:.3f}s")
print(f"   without cache: '{name_n}' | {calls_n:3d} model calls | {secs_n:.3f}s (recompute the whole prefix per char)")
print(f"   same name, same math — the cache just refuses to redo work. this gap grows with length.\n")

# 5) The novelty filter.
random.seed(42)
train_set = set(train_docs)
raw = sample_names(30, temperature=0.8)
novel = [s for s in raw if s and s not in train_set]
print(f"4) novelty filter: {len(raw)} samples -> {sum(1 for s in raw if s in train_set)} verbatim training docs, {len(raw) - len(novel) - sum(1 for s in raw if s in train_set)} empty -> {len(novel)} genuinely new")
print(f"   survivors: {', '.join(novel[:10])}\n")

# 6) The quiz: which of these are real records in the brain?
random.seed(99)
real = random.sample(train_docs, 3)
fakes = [s for s in novel if 6 <= len(s) <= 30][:3]
quiz = real + fakes
random.shuffle(quiz)
print("5) quiz — real brain record or hallucination?")
for i, q in enumerate(quiz):
    print(f"   {i+1}. {q}")
print("\n   (answers: " + ', '.join(f"{i+1} {'real' if q in train_set else 'FAKE'}" for i, q in enumerate(quiz)) + ")")
