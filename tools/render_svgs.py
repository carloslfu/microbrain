"""
tools/render_svgs.py — draw the course's diagrams as SVG, from the evidence.

Reads the author's logs in runs/ (and the loss curves in out/) and renders the
five images embedded in the README and lessons. Same data as the ASCII
diagrams the code prints — just drawn with rectangles instead of shade
characters. Pure standard library, like everything else here.

usage: python tools/render_svgs.py     (writes assets/*.svg)
"""

import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, 'runs')
OUT = os.path.join(HERE, 'out')
ASSETS = os.path.join(HERE, 'assets')
os.makedirs(ASSETS, exist_ok=True)

INK, MUTED, BLUE, RED, GRID = '#1f2328', '#6a737d', '#0969da', '#cf222e', '#d8dee4'
FONT = 'font-family="SFMono-Regular,Consolas,Menlo,monospace"'
SHADES = ' .:-=+*#%@'

def svg_open(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>']

def text(x, y, s, size=13, fill=INK, anchor='start', bold=False):
    wt = ' font-weight="bold"' if bold else ''
    s = s.replace('&', '&amp;').replace('<', '&lt;')
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{fill}" text-anchor="{anchor}"{wt}>{s}</text>'

def save(name, parts):
    parts.append('</svg>')
    path = os.path.join(ASSETS, name)
    open(path, 'w').write('\n'.join(parts))
    print('wrote', path)

# ---------------------------------------------------------------- ladder hero
def ladder():
    stages = [('uniform', 38.0, MUTED), ('counting', 14.5, BLUE), ('by hand', 15.3, BLUE),
              ('autograd', 15.3, BLUE), ('attention', 14.7, BLUE), ('multi-head', 14.8, BLUE),
              ('Adam', 13.8, RED)]
    W, H, base = 1170, 356, 286
    s = svg_open(W, H)
    s.append(text(56, 34, 'effective choices — among how many characters is the model still guessing?', 15, INK, bold=True))
    s.append(text(56, 54, 'e^(val loss), full runs, 543-doc corpus · lower is better · red: the rung that finally beats the count table', 12, MUTED))

    # left panel — full scale: training collapses 38 -> ~14
    lx, lbw, lgap = 56, 34, 10
    lscale = (base - 96) / 38.0
    for i, (name, v, color) in enumerate(stages):
        x = lx + i * (lbw + lgap)
        h = v * lscale
        s.append(f'<rect x="{x}" y="{base - h:.1f}" width="{lbw}" height="{h:.1f}" fill="{color}" opacity="{0.55 if i == 0 else 0.9}" rx="2"/>')
    s.append(text(lx + lbw / 2, base - 38.0 * lscale - 8, '38.0', 12, INK, 'middle', bold=True))
    lw = 7 * (lbw + lgap) - lgap
    s.append(f'<line x1="{lx-8}" y1="{base}" x2="{lx+lw+8}" y2="{base}" stroke="{GRID}" stroke-width="1"/>')
    s.append(text(lx + lw / 2, base + 20, 'uniform, then the six trained rungs:', 11.5, MUTED, 'middle'))
    s.append(text(lx + lw / 2, base + 36, 'training collapses 38 to ~14', 11.5, MUTED, 'middle'))

    # right panel — the same six trained bars, zoomed so the story is visible
    zx, zbw, zgap, zfloor = 430, 72, 20, 13.0
    zscale = (base - 96) / (15.6 - zfloor)
    s.append(text(zx, 82, 'zoom — same six bars, axis starts at 13 (deliberately truncated)', 11.5, MUTED, bold=True))
    y_count = base - (14.5 - zfloor) * zscale
    zw = 6 * (zbw + zgap) - zgap
    for i, (name, v, color) in enumerate(stages[1:]):
        x = zx + i * (zbw + zgap)
        h = (v - zfloor) * zscale
        s.append(f'<rect x="{x}" y="{base - h:.1f}" width="{zbw}" height="{h:.1f}" fill="{color}" opacity="0.9" rx="3"/>')
        s.append(text(x + zbw / 2, base - h - 8, f'{v:.1f}', 13.5, INK, 'middle', bold=(i == 5)))
        s.append(text(x + zbw / 2, base + 20, name, 12.5, MUTED, 'middle'))
        s.append(text(x + zbw / 2, base + 36, f'rung {i}', 11, MUTED, 'middle'))
    s.append(f'<line x1="{zx-8}" y1="{y_count:.1f}" x2="{zx+zw+8}" y2="{y_count:.1f}" stroke="{INK}" stroke-width="1" stroke-dasharray="5,4" opacity="0.45"/>')
    s.append(text(zx + zw + 18, y_count + 4, 'the count table —', 11, MUTED))
    s.append(text(zx + zw + 18, y_count + 19, 'the bar to beat', 11, MUTED))
    s.append(f'<line x1="{zx-8}" y1="{base}" x2="{zx+zw+8}" y2="{base}" stroke="{GRID}" stroke-width="1"/>')
    s.append(text(zx - 14, base + 4, '13', 11, MUTED, 'end'))
    save('ladder.svg', s)

# ---------------------------------------------------- train0 count-table heatmap
def heatmap():
    lines = open(os.path.join(RUNS, 'train0.log')).read().split('\n')
    i = next(k for k, l in enumerate(lines) if 'the model itself' in l)
    header = lines[i + 1].strip()
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith('  ') or len(l.strip()) < 2: break
        lab, cells = l[2], l[4:4 + len(header)]
        rows.append((lab, cells))
    cell, x0, y0 = 13, 90, 84
    W = max(x0 + len(header) * cell + 40, 700)
    Hh = y0 + len(rows) * cell + 30
    s = svg_open(W, Hh)
    s.append(text(28, 32, "the whole model: P(next char | current char), from the count table", 15, INK, bold=True))
    s.append(text(28, 52, 'row = current char, col = next char, darker = more probable (row-scaled) · runs/train0.log', 12, MUTED))
    for j, ch in enumerate(header):
        s.append(text(x0 + j * cell + cell / 2, y0 - 8, ch, 10, MUTED, 'middle'))
    for r, (lab, cells) in enumerate(rows):
        s.append(text(x0 - 10, y0 + r * cell + cell - 3.5, lab, 10, MUTED, 'end'))
        for j, ch in enumerate(cells):
            v = SHADES.index(ch) / 9 if ch in SHADES else 0
            if v > 0:
                s.append(f'<rect x="{x0 + j * cell}" y="{y0 + r * cell}" width="{cell-1}" height="{cell-1}" fill="{BLUE}" opacity="{0.08 + 0.92 * v:.2f}"/>')
    save('train0-heatmap.svg', s)

# ------------------------------------------------- train3 attention triangle
def attention():
    lines = open(os.path.join(RUNS, 'train3.log')).read().split('\n')
    i = next(k for k, l in enumerate(lines) if 'attention over' in l)
    header = lines[i + 1].strip()
    rows = []
    for l in lines[i + 2:]:
        m = re.match(r'^\s+(\S+) \| (.*)$', l)
        if not m: break
        rows.append((m.group(1), m.group(2)))
    cell, x0, y0 = 24, 96, 96
    W = max(x0 + len(header) * cell + 72, 740)
    Hh = y0 + len(rows) * cell + 30
    s = svg_open(W, Hh)
    s.append(text(28, 32, "where trained attention looked: 'test-time-training'", 15, INK, bold=True))
    s.append(text(28, 52, 'row = position being predicted from, column = position it attends to', 12, MUTED))
    s.append(text(28, 68, 'darker = more weight (row-scaled) · note the ^ (BOS) column: an attention sink · runs/train3.log', 12, MUTED))
    for j, ch in enumerate(header):
        s.append(text(x0 + j * cell + cell / 2, y0 - 8, ch, 12, RED if ch == '^' else MUTED, 'middle', bold=(ch == '^')))
    for r, (lab, cells) in enumerate(rows):
        s.append(text(x0 - 10, y0 + r * cell + cell - 8, lab, 12, MUTED, 'end'))
        for j, ch in enumerate(cells):
            v = SHADES.index(ch) / 9 if ch in SHADES else 0
            if v > 0:
                s.append(f'<rect x="{x0 + j * cell}" y="{y0 + r * cell}" width="{cell-2}" height="{cell-2}" fill="{BLUE}" opacity="{0.06 + 0.94 * v:.2f}" rx="2"/>')
    save('train3-attention.svg', s)

# --------------------------------------------------------- train5 SGD vs Adam
def race():
    def load(p):
        return [float(x) for x in open(os.path.join(OUT, p))]
    def smooth(ys, a=0.05):
        out, acc = [], ys[0]
        for y in ys:
            acc = (1 - a) * acc + a * y
            out.append(acc)
        return out
    sgd, adam = smooth(load('train4_losses.txt')), smooth(load('train5_losses.txt'))
    n = min(len(sgd), len(adam))
    W, Hh, pad_l, pad_r, pad_t, pad_b = 860, 320, 64, 64, 70, 40
    lo = min(min(sgd[:n]), min(adam[:n])); hi = max(max(sgd[:n]), max(adam[:n]))
    def X(i): return pad_l + i * (W - pad_l - pad_r) / (n - 1)
    def Y(v): return pad_t + (hi - v) * (Hh - pad_t - pad_b) / (hi - lo)
    s = svg_open(W, Hh)
    s.append(text(pad_l, 32, 'same model, same data, same steps — only the optimizer differs', 15, INK, bold=True))
    s.append(text(pad_l, 52, 'training loss, smoothed · SGD from train4, Adam from train5 · out/train*_losses.txt', 12, MUTED))
    for v in (2.6, 2.9, 3.2, 3.5):
        if lo <= v <= hi:
            s.append(f'<line x1="{pad_l}" y1="{Y(v):.1f}" x2="{W-pad_r}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
            s.append(text(pad_l - 8, Y(v) + 4, f'{v:.1f}', 11, MUTED, 'end'))
    for k, (ys, color, label) in enumerate(((sgd, MUTED, 'SGD (train4)'), (adam, RED, 'Adam (train5)'))):
        pts = ' '.join(f'{X(i):.1f},{Y(ys[i]):.1f}' for i in range(n))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        lx, ly = W - 260 + k * 130, pad_t + 6
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+22}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        s.append(text(lx + 28, ly + 4, label, 12, color, 'start', bold=True))
    yS, yA = Y(sgd[n - 1]), Y(adam[n - 1])
    if abs(yA - yS) < 14:
        mid = (yS + yA) / 2
        yS, yA = mid - 7, mid + 7
    s.append(text(W - pad_r + 8, yS + 4, f'{sgd[n-1]:.2f}', 11.5, MUTED, bold=True))
    s.append(text(W - pad_r + 8, yA + 4, f'{adam[n-1]:.2f}', 11.5, RED, bold=True))
    s.append(text(pad_l, Hh - 12, 'step 1', 11, MUTED))
    s.append(text(W - pad_r, Hh - 12, f'step {n}', 11, MUTED, 'end'))
    save('train5-race.svg', s)

# ------------------------------------------------------- train7 damage report
def damage():
    results = []
    for l in open(os.path.join(RUNS, 'train7.log')):
        m = re.match(r'^(baseline|no-\S+|beta2=\S+)\s+\| val loss ([\d.]+)', l)
        if m: results.append((m.group(1), float(m.group(2))))
    results.sort(key=lambda r: r[1])
    base = dict(results)['baseline']
    W, Hh, pad_l, pad_t, row_h = 880, 96 + len(results) * 40, 150, 84, 40
    lo, hi = 2.70, 2.93
    def X(v): return pad_l + (v - lo) * (W - pad_l - 40) / (hi - lo)
    s = svg_open(W, Hh)
    s.append(text(28, 32, 'the ablation lab: val loss after each surgery (300 steps, same init)', 15, INK, bold=True))
    s.append(text(28, 52, 'axis starts at 2.70 to make the gaps visible · dashed line = baseline · runs/train7.log', 12, MUTED))
    s.append(f'<line x1="{X(base):.1f}" y1="{pad_t-14}" x2="{X(base):.1f}" y2="{Hh-24}" stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="5,4"/>')
    for i, (name, v) in enumerate(results):
        y = pad_t + i * row_h
        color = RED if v - base > 0.03 else (BLUE if v < base else MUTED)
        s.append(text(pad_l - 12, y + 15, name, 13, INK, 'end'))
        s.append(f'<line x1="{X(lo)}" y1="{y+10}" x2="{X(v):.1f}" y2="{y+10}" stroke="{color}" stroke-width="8" stroke-linecap="round" opacity="0.85"/>')
        s.append(text(X(v) + 10, y + 15, f'{v:.4f}', 12.5, color))
    s.append(text(X(base), Hh - 8, f'baseline {base:.4f}', 11, MUTED, 'middle'))
    save('train7-damage.svg', s)



# ---------------------------------------------------------- pipeline (train0)
def pipeline():
    # rung 0's map: six stages. backward earns its box at rung 2 (this SVG
    # is embedded only in train0's lesson, whose ASCII map matches).
    stages = [('docs', '543 idea names'), ('tokenize', 'chars -> ids'), ('model', 'the box that grows'),
              ('loss', 'surprise'), ('update', 'step params'),
              ('sample', 'babble new names')]
    W, Hh, bw, bh, gap, y = 800, 214, 104, 46, 18, 96
    s = svg_open(W, Hh)
    s.append('<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6a737d"/></marker></defs>')
    s.append(text(24, 32, 'the pipeline — data in, babble out; every rung upgrades one part', 15, INK, bold=True))
    x = 24
    for i, (name, gloss) in enumerate(stages):
        color = BLUE if name == 'model' else INK
        s.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="none" stroke="{color}" stroke-width="{2 if name == "model" else 1.3}" rx="8"/>')
        s.append(text(x + bw / 2, y + 28, name, 13.5, color, 'middle', bold=(name == 'model')))
        s.append(text(x + bw / 2, y + bh + 18, gloss, 10.5, MUTED, 'middle'))
        if i < len(stages) - 1:
            s.append(f'<line x1="{x+bw}" y1="{y+bh/2}" x2="{x+bw+gap-2}" y2="{y+bh/2}" stroke="{MUTED}" stroke-width="1.3" marker-end="url(#a)"/>')
        x += bw + gap
    ux = 24 + 4 * (bw + gap) + bw / 2   # update box center
    mx = 24 + 2 * (bw + gap) + bw / 2   # model box center
    s.append(f'<path d="M {ux} {y-8} C {ux} {y-38}, {mx} {y-38}, {mx} {y-8}" fill="none" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#a)"/>')
    s.append(text((ux + mx) / 2, y - 40, 'next document', 10.5, MUTED, 'middle'))
    save('pipeline.svg', s)

# ----------------------------------------------- computation graph (train2)
def graph():
    #        name      data    grad     x    y
    nodes = [('L',      '+9',  '+1',   320,  64),
             ('diff',   '-3',  '-6',   320, 146),
             ('relu',   '+2',  '-6',   210, 228),
             ('const',  '-5',  '-6',   430, 228),
             ('sum',    '+2',  '-6',   210, 310),
             ('a*b',    '-6',  '-6',   110, 392),
             ('const',  '+8',  '-6',   320, 392),
             ('a',      '+2', '+18',    60, 474),
             ('b',      '-3', '-12',   190, 474)]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6), (5, 7), (5, 8)]
    W, Hh, bw, bh = 584, 540, 106, 46
    s = svg_open(W, Hh)
    s.append(text(20, 26, 'the by-hand graph, drawn: L = (relu(a*b + 8) - 5)^2', 14.5, INK, bold=True))
    s.append(text(20, 44, 'black: value computed forward · red: gradient dL/dnode, filled by one backward()', 11.5, MUTED))
    for i, j in edges:
        x1, y1 = nodes[i][3], nodes[i][4] + bh / 2
        x2, y2 = nodes[j][3], nodes[j][4] - bh / 2 + 2
        s.append(f'<line x1="{x1}" y1="{y1+18}" x2="{x2}" y2="{y2+18}" stroke="{GRID}" stroke-width="1.6"/>')
    for name, d, g, x, y in nodes:
        leaf = name in ('a', 'b') or name == 'const'
        s.append(f'<rect x="{x-bw/2}" y="{y}" width="{bw}" height="{bh}" fill="#ffffff" stroke="{BLUE if name in ("a","b") else INK}" stroke-width="{2 if name in ("a","b","L") else 1.2}" rx="8"/>')
        s.append(text(x - bw / 2 + 10, y + 19, f'{name} {d}', 12.5, INK, bold=(name in ('a', 'b', 'L'))))
        s.append(text(x - bw / 2 + 10, y + 37, f'grad {g}', 12, RED))
    save('graph.svg', s)

# ------------------------------------------------- architecture (train3/4/5)
def architecture():
    W, Hh = 780, 1034
    cx, bw = 300, 300
    s = svg_open(W, Hh)
    s.append('<defs><marker id="m" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6a737d"/></marker></defs>')
    s.append(text(24, 30, 'the whole architecture — gpt() from train3.py to train5.py', 15, INK, bold=True))
    s.append(text(24, 50, "this course's dials: vocab 38 · dims 16 · context 40 · 1 layer · 4 heads from rung 4 · 4,928 params", 12, MUTED))
    def box(y, h, title, sub=None, stroke=INK, wd=1.3, dash=''):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        s.append(f'<rect x="{cx-bw/2}" y="{y}" width="{bw}" height="{h}" fill="#ffffff" stroke="{stroke}" stroke-width="{wd}" rx="9"{d}/>')
        s.append(text(cx, y + 22, title, 13, stroke if stroke != GRID else INK, 'middle', bold=True))
        if sub: s.append(text(cx, y + 40, sub, 11, MUTED, 'middle'))
    def arrow(y1, y2, x=None):
        x = cx if x is None else x
        s.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2-3}" stroke="{MUTED}" stroke-width="1.4" marker-end="url(#m)"/>')
    def plus(y):
        s.append(f'<circle cx="{cx}" cy="{y}" r="11" fill="#ffffff" stroke="{INK}" stroke-width="1.4"/>')
        s.append(text(cx, y + 4.5, '+', 14, INK, 'middle', bold=True))
    def note(y, t):
        s.append(text(cx + bw / 2 + 62, y, t, 11.5, MUTED))
    # inputs
    s.append(f'<rect x="{cx-bw/2}" y="76" width="140" height="40" fill="#ffffff" stroke="{INK}" stroke-width="1.2" rx="8"/>')
    s.append(text(cx - bw / 2 + 70, 100, 'token id', 12.5, INK, 'middle'))
    s.append(f'<rect x="{cx+bw/2-140}" y="76" width="140" height="40" fill="#ffffff" stroke="{INK}" stroke-width="1.2" rx="8"/>')
    s.append(text(cx + bw / 2 - 70, 100, 'position id', 12.5, INK, 'middle'))
    arrow(116, 142, cx - bw / 2 + 70); arrow(116, 142, cx + bw / 2 - 70)
    s.append(f'<rect x="{cx-bw/2}" y="144" width="140" height="40" fill="#ffffff" stroke="{INK}" stroke-width="1.2" rx="8"/>')
    s.append(text(cx - bw / 2 + 70, 168, 'wte row (16)', 12, INK, 'middle'))
    s.append(f'<rect x="{cx+bw/2-140}" y="144" width="140" height="40" fill="#ffffff" stroke="{INK}" stroke-width="1.2" rx="8"/>')
    s.append(text(cx + bw / 2 - 70, 168, 'wpe row (16)', 12, INK, 'middle'))
    note(168, 'wte 608 · wpe 640')
    s.append(f'<line x1="{cx-bw/2+70}" y1="184" x2="{cx-4}" y2="216" stroke="{MUTED}" stroke-width="1.3" marker-end="url(#m)"/>')
    s.append(f'<line x1="{cx+bw/2-70}" y1="184" x2="{cx+4}" y2="216" stroke="{MUTED}" stroke-width="1.3" marker-end="url(#m)"/>')
    plus(228); arrow(239, 260)
    box(262, 34, 'rmsnorm'); arrow(296, 318)
    # transformer block container
    s.append(f'<rect x="{cx-bw/2-40}" y="320" width="{bw+80}" height="420" fill="none" stroke="{MUTED}" stroke-width="1.1" stroke-dasharray="6,4" rx="12"/>')
    s.append(text(cx, 344, 'transformer block — the for-li layer loop (x1 here, x96 in production)', 11, MUTED, 'middle'))
    bypass_x = cx + bw / 2 + 20
    # attention half
    box(360, 34, 'rmsnorm'); arrow(394, 416)
    box(418, 74, 'attention — communicate', 'q-k match -> softmax -> weighted v')
    s.append(text(cx, 476, '4 heads · keys/values lists = the KV cache', 10.5, BLUE, 'middle'))
    note(452, 'wq wk wv wo: 1,024 params')
    arrow(492, 514); plus(526)
    s.append(f'<path d="M {cx} 356 C {bypass_x} 356, {bypass_x} 526, {cx+12} 526" fill="none" stroke="{BLUE}" stroke-width="1.4" marker-end="url(#m)"/>')
    s.append(text(bypass_x + 6, 530, 'residual', 10.5, BLUE))
    arrow(537, 558)
    # mlp half
    box(560, 34, 'rmsnorm'); arrow(594, 616)
    box(618, 74, 'MLP — compute', 'linear 16->64 -> ReLU -> linear 64->16')
    note(652, 'fc1 + fc2: 2,048 params')
    arrow(692, 712); plus(724)
    s.append(f'<path d="M {cx} 556 C {bypass_x} 556, {bypass_x} 724, {cx+12} 724" fill="none" stroke="{BLUE}" stroke-width="1.4" marker-end="url(#m)"/>')
    s.append(text(bypass_x + 6, 728, 'residual', 10.5, BLUE))
    arrow(735, 762)
    box(764, 56, 'lm_head', '16 -> 38 scores, one per token')
    note(792, 'lm_head: 608 params')
    arrow(820, 842)
    box(844, 40, 'softmax -> probabilities')
    arrow(884, 906)
    box(908, 56, 'sample the next token', 'append it, move to the next position')
    s.append(f'<path d="M {cx+bw/2} 936 C {W-36} 936, {W-36} 96, {cx+bw/2+2} 96" fill="none" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#m)"/>')
    s.append(text(W - 30, 520, 'repeat', 11, MUTED, 'end'))
    s.append(text(cx, 1004, 'total: 4,928 parameters — every box above is a list of plain Python floats', 12, INK, 'middle', bold=True))
    save('architecture.svg', s)

if __name__ == '__main__':
    ladder(); heatmap(); attention(); race(); damage(); pipeline(); graph(); architecture()
