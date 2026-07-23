"""
train7.py: The ablation lab. Break the GPT on purpose, once per organ.

Same as train5.py:
- Architecture, optimizer, data, instruments.

Different from train5.py:
- Trains the model SIX times (shorter runs), each with one piece surgically
  removed, from the same random init. The final val losses go on one bar chart.

The point: every component of a transformer earns its place. You can *read*
that position embeddings matter; here you *measure* what dies without them.

Before the results print, the lab asks you to predict the ranking. Commit to
your answers. Being wrong here is the highest-value moment in the course.

The surgeries:
- baseline:    nothing removed (the control)
- no-wpe:      no position embeddings — the model can't tell character order
- no-residual: blocks replace x instead of adding to it — gradients lose the highway
- no-rmsnorm:  no normalization — activations drift unchecked
- no-relu:     MLP loses its nonlinearity — two matrices collapse into one
- beta2=0.5:   Adam's variance estimate turns twitchy

usage: python train7.py [--fast]   (--fast: 100 steps per surgery instead of 300)
lesson: lessons/train7-ablation-lab.md — but only AFTER the run grades your predictions
"""

import os       # os.path.exists
import sys      # sys.argv
import math     # math.log, math.exp
import time     # time.time
import random   # random.seed, random.choices, random.gauss
FAST = '--fast' in sys.argv

# Dataset: the names of ideas (already shuffled by data/make_dataset.py, seed 42)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.txt')
assert os.path.exists(DATA), "no data/data.txt yet — run: python data/make_dataset.py"
docs = [l.strip() for l in open(DATA).read().strip().split('\n') if l.strip()] # list[str] of documents
val_docs = docs[-len(docs) // 10:]
train_docs = docs[:-len(docs) // 10]

# Tokenizer: character-level, with a special BOS (Beginning of Sequence) token
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"num docs: {len(train_docs)} train / {len(val_docs)} val | vocab size: {vocab_size}")

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

# Model dials (identical to train5.py)
n_embd = 16     # embedding dimension
block_size = 40 # maximum sequence length (BOS + up to 39 characters)
n_head = 4      # number of attention heads
n_layer = 1     # number of layers
head_dim = n_embd // n_head # dimension of each head

# The one difference: a FLAGS dict the model consults, so each run can remove one organ
FLAGS = {'wpe': True, 'residual': True, 'rmsnorm': True, 'relu': True, 'beta2': 0.99}

def init_params():
    random.seed(42) # every surgery starts from the identical random init
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    sd = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        sd[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    return sd

state_dict = init_params()

# Model: token_id -> logits (each optional organ guarded by a FLAG)
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    return probs

def rmsnorm(x):
    if not FLAGS['rmsnorm']:
        return x
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    if FLAGS['wpe']:
        pos_emb = state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
    else:
        x = list(tok_emb) # no idea where in the sequence we are
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
        if FLAGS['residual']:
            x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        if FLAGS['relu']:
            x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        if FLAGS['residual']:
            x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

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

def train_once(num_steps):
    global state_dict
    state_dict = init_params()
    params = [p for mat in state_dict.values() for row in mat for p in row]
    learning_rate, beta1, eps_adam = 0.01, 0.85, 1e-8
    beta2 = FLAGS['beta2']
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    for step in range(num_steps):
        doc = train_docs[step % len(train_docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            probs = softmax(gpt(token_id, pos_id, keys, values))
            losses.append(-probs[target_id].log())
        loss = (1 / n) * sum(losses)
        loss.backward()
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0
    return avg_nll(val_docs)

SURGERIES = [
    ('baseline',    {}),
    ('no-wpe',      {'wpe': False}),
    ('no-residual', {'residual': False}),
    ('no-rmsnorm',  {'rmsnorm': False}),
    ('no-relu',     {'relu': False}),
    ('beta2=0.5',   {'beta2': 0.5}),
]

num_steps = 100 if FAST else 300
print(f"""
Every run: same init, same data order, Adam, {num_steps} steps. Before the
results arrive, commit to predictions — rank the five surgeries by how much
they raise val loss over baseline, and write your ranking down:

   no-wpe, no-residual, no-rmsnorm, no-relu, beta2=0.5

The lab is now running (~minutes per surgery, pure Python)...
""")

results = []
for name, changes in SURGERIES:
    FLAGS.update({'wpe': True, 'residual': True, 'rmsnorm': True, 'relu': True, 'beta2': 0.99})
    FLAGS.update(changes)
    t0 = time.time()
    vl = train_once(num_steps)
    random.seed(1000) # identical sampling randomness for every surgery
    names = sample_names(4)
    results.append((name, vl, names))
    print(f"{name:12s} | val loss {vl:.4f} | effective choices {math.exp(vl):5.1f} | {time.time()-t0:5.1f}s | {', '.join(names)}")

# The bar chart: damage report, sorted
print("\n--- damage report (val loss, lower is better) ---")
worst = max(vl for _, vl, _ in results)
for name, vl, _ in sorted(results, key=lambda r: r[1]):
    bar = '#' * max(1, round(vl / worst * 44))
    print(f"  {name:12s} {vl:.4f} |{bar}")
base = dict((n, v) for n, v, _ in results)['baseline']
print(f"\nbaseline is {base:.4f}; every surgery above it is the price of a missing organ.")
print("interpretations live in lessons/train7-ablation-lab.md — but only after you've compared your predicted ranking.")
