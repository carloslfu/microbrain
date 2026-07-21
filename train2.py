"""
train2.py: Bigram language model with a single-layer MLP, trained with autograd.

Same as train1.py:
- Dataset, tokenizer, model architecture (MLP), SGD optimizer, inference, instruments

Different from train1.py:
- Introduces autograd (Value class) to compute gradients automatically
- No more manual analytic_gradient or numerical_gradient
- The forward pass builds a computation graph, loss.backward() traverses it

The hand-derived chain rule from train1.py is now automated: each operation
records its local gradient, and backward() applies the chain rule recursively.
The training loop becomes: forward pass -> loss.backward() -> SGD update.
Note this file is SHORTER than train1.py: the abstraction deleted the hardest
code (the hand-derived backward pass) and replaced it with 40 general lines.

microbrain deltas:
- diagram: a computation graph small enough to hold in your head, printed with
  every node's data and grad after backward() — chain rule you can check by eye
- the same machinery then counts the nodes in one real document's loss graph
- step 1 prints the same loss as train1.py step 1: same math, now automated

usage: python train2.py [--fast]   (--fast: 300 steps instead of 1000)
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

# Diagram: a computation graph you can hold in your head.
# L = (relu(a*b + 8) + (-5))^2 with a=2, b=-3. Forward computes data top-down,
# backward() fills every grad. Check any grad by eye with the chain rule.
print("\n--- autograd on a graph small enough to check by hand ---")
a = Value(2.0); b = Value(-3.0)
prod = a * b          # -6
s = prod + 8.0        #  2
d = s.relu()          #  2
diff = d + (-5.0)     # -3
L = diff ** 2         #  9
L.backward()
names = {id(L): 'L', id(diff): 'diff', id(d): 'relu', id(s): 'sum', id(prod): 'a*b', id(a): 'a', id(b): 'b'}
def draw(v, prefix=''):
    print(f"{prefix}{names.get(id(v), 'const'):5s} data {v.data:+7.4f} | grad {v.grad:+7.4f}")
    for child in v._children:
        draw(child, prefix + '    ')
draw(L)
print("e.g. dL/da: 2*diff * 1 * 1 * b = 2*(-3) * (-3) = +18 — matches the printout")

# Initialize the parameters
n_embd = 16     # embedding dimension
matrix = lambda nout, nin: [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'mlp_fc1': matrix(4 * n_embd, n_embd),
    'mlp_fc2': matrix(vocab_size, 4 * n_embd),
}
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"\nnum params: {len(params)}")

# Model: token_id -> logits
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    return probs

def mlp(token_id):
    x = state_dict['wte'][token_id]
    x = linear(x, state_dict['mlp_fc1'])
    x = [xi.relu() for xi in x]
    logits = linear(x, state_dict['mlp_fc2'])
    return logits

# Instrument: average loss per token over a set of documents (forward only)
def avg_nll(eval_docs):
    total, count = 0.0, 0
    for doc in eval_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        for pos_id in range(len(tokens) - 1):
            probs = softmax(mlp(tokens[pos_id]))
            total += -math.log(probs[tokens[pos_id + 1]].data)
            count += 1
    return total / count

# The same machinery at real size: count the nodes in one document's loss graph
def count_nodes(v):
    seen, stack = set(), [v]
    while stack:
        node = stack.pop()
        if id(node) not in seen:
            seen.add(id(node))
            stack.extend(node._children)
    return len(seen)

doc = train_docs[0]
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
n = len(tokens) - 1
losses = []
for pos_id in range(n):
    probs = softmax(mlp(tokens[pos_id]))
    losses.append(-probs[tokens[pos_id + 1]].log())
loss = (1 / n) * sum(losses)
print(f"one document ('{doc}') builds a graph of {count_nodes(loss):,} Value nodes")
print("backward() applies the same by-eye chain rule to every one of them")

# Train the model
t0 = time.time()
num_steps = 300 if FAST else 1000
learning_rate = 1.0
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = train_docs[step % len(train_docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = len(tokens) - 1

    # Forward pass
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = mlp(token_id)
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

print(f"training took {time.time() - t0:.1f}s (train1.py did the same math faster — abstraction costs a constant factor)")

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
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    text = ''.join(sample)
    tag = '  <- verbatim training doc' if text in train_set else ''
    memorized += text in train_set
    print(f"sample {sample_idx+1:2d}: {text}{tag}")
print(f"memorization: {memorized}/{num_samples} samples are verbatim training docs")
