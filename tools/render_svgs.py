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
    W, H, pad, base = 860, 340, 56, 280
    bw, gap = 88, 26
    s = svg_open(W, H)
    s.append(text(pad, 34, 'effective choices — among how many characters is the model still guessing?', 15, INK, bold=True))
    s.append(text(pad, 54, 'e^(val loss), full runs, 543-doc corpus · lower is better · red: the rung that finally beats the count table', 12, MUTED))
    scale = (base - 90) / 38.0
    for i, (name, v, color) in enumerate(stages):
        x = pad + i * (bw + gap)
        h = v * scale
        s.append(f'<rect x="{x}" y="{base - h:.1f}" width="{bw}" height="{h:.1f}" fill="{color}" opacity="{0.55 if i == 0 else 0.9}" rx="3"/>')
        s.append(text(x + bw / 2, base - h - 8, f'{v:.1f}', 14, INK, 'middle', bold=(i in (0, 6))))
        s.append(text(x + bw / 2, base + 20, name, 12.5, MUTED, 'middle'))
        s.append(text(x + bw / 2, base + 36, '—' if i == 0 else f'rung {i - 1}', 11, MUTED, 'middle'))
    s.append(f'<line x1="{pad-10}" y1="{base}" x2="{W-pad+10}" y2="{base}" stroke="{GRID}" stroke-width="1"/>')
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
    W = x0 + len(header) * cell + 40
    Hh = y0 + len(rows) * cell + 30
    s = svg_open(W, Hh)
    s.append(text(28, 32, "the whole model: P(next char | current char), from the count table", 15, INK, bold=True))
    s.append(text(28, 52, 'row = current char, column = next char, darker = more probable (row-scaled) · runs/train0.log', 12, MUTED))
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
    W = x0 + len(header) * cell + 40
    Hh = y0 + len(rows) * cell + 30
    s = svg_open(W, Hh)
    s.append(text(28, 32, "where trained attention looked: 'test-time-training'", 15, INK, bold=True))
    s.append(text(28, 52, 'row = position being predicted from, column = position it attends to', 12, MUTED))
    s.append(text(28, 68, 'darker = more weight (row-scaled) · note the ^ (BOS) column: an attention sink · runs/train3.log', 12, MUTED))
    for j, ch in enumerate(header):
        s.append(text(x0 + j * cell + cell / 2, y0 - 8, ch, 12, MUTED, 'middle'))
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
    W, Hh, pad_l, pad_r, pad_t, pad_b = 860, 320, 64, 24, 70, 40
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

if __name__ == '__main__':
    ladder(); heatmap(); attention(); race(); damage()
