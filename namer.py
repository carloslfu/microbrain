"""
namer.py: The payoff. Your trained GPT, shipped as a tool.

Loads the weights train6.py saved to out/model.json and generates genuinely
novel idea names — anything that already exists in the corpus is filtered out.
Notice what is NOT in this file: no Value class, no backward(), no optimizer.
Inference is just the forward pass. Gradients exist only so that training can
happen; the shipped product never needs them. Plain floats, ~10x faster too.

usage: python namer.py [-n 12] [-t 0.8] [--seed N] [--quiz]
  -n      how many novel names to generate (default 12)
  -t      temperature (default 0.8 — braver than training-time 0.5)
  --seed  fix the randomness (default: different every run)
  --quiz  mix hallucinations with real records — can you tell which is which?
"""

import os       # os.path.exists
import sys      # sys.argv
import math     # math.exp
import json     # json.load
import random   # random.seed, random.choices

# CLI args, by hand (the whole course is stdlib; argparse felt like a framework)
def arg(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default
count = arg('-n', 12, int)
temperature = arg('-t', 0.8, float)
seed = arg('--seed', None, int)
if seed is not None:
    random.seed(seed)

HERE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(HERE, 'out', 'model.json')
assert os.path.exists(model_path), "no out/model.json yet — run: python train6.py"
ckpt = json.load(open(model_path))
n_embd, block_size = ckpt['n_embd'], ckpt['block_size']
n_head, n_layer = ckpt['n_head'], ckpt['n_layer']
head_dim = n_embd // n_head
uchars = list(ckpt['uchars'])
BOS = len(uchars)
vocab_size = len(uchars) + 1
state_dict = ckpt['state_dict'] # plain float matrices — that's all a model is

# The corpus, for the novelty filter
DATA = os.path.join(HERE, 'data', 'data.txt')
known = set(l.strip() for l in open(DATA)) if os.path.exists(DATA) else set()

# The forward pass, floats only. Identical math to train6.py's gpt(), minus autograd.
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    x = [t + p for t, p in zip(state_dict['wte'][token_id], state_dict['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
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
            x_attn.extend(sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim))
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [max(0.0, xi) for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    return linear(x, state_dict['lm_head'])

def generate():
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    return ''.join(sample)

def draw_novel(count, min_len=4, max_len=40):
    out, attempts = [], 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        name = generate()
        if min_len <= len(name) <= max_len and name not in known and name not in out:
            out.append(name)
    return out, attempts

if '--quiz' in sys.argv:
    # the payoff, sharpened: three of these are records in the corpus,
    # three came out of 4,928 floats. no peeking.
    assert known, "the quiz needs data/data.txt — run: python data/make_dataset.py"
    real = random.sample(sorted(k for k in known if 6 <= len(k) <= 30), 3)
    fakes, _ = draw_novel(3, min_len=6, max_len=30)
    quiz = real + fakes
    random.shuffle(quiz)
    print("microbrain namer — quiz | which are real records, which did the model invent?")
    for i, q in enumerate(quiz):
        print(f"  {i+1}. {q}")
    print("\n" + " " * 8 + "(answers: " + ', '.join(
        f"{i+1} {'real' if q in known else 'FAKE'}" for i, q in enumerate(quiz)) + ")")
else:
    print(f"microbrain namer | {count} novel idea names | T={temperature}")
    novel, attempts = draw_novel(count)
    for i, name in enumerate(novel):
        print(f"  {i+1:2d}. {name}")
    print(f"({attempts} draws -> {len(novel)} survivors of the novelty filter)")
    print("(now try --quiz)")
