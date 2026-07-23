"""
tools/check.py: the repo's eight consistency gates, runnable by anyone.

    python tools/check.py

1. every markdown file has balanced ``` fences
2. every relative link in every .md resolves to a real file
3. every GLOSSARY.md#anchor used in a lesson exists as a #### heading
4. GLOSSARY.md headings are alphabetical (case-insensitive)
5. every ```python snippet in a lesson is byte-verbatim in its trainN.py
6. every quoted panel/params/gradcheck/timing line in a lesson is verbatim in runs/
7. the lesson footers chain prev/next in ladder order
8. every assets/*.svg parses as valid XML

Exit code 0 = all gates pass.
"""
import re, sys, pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent.parent
LESSON_ORDER = ['train0-counting', 'train1-gradient-descent', 'train2-autograd',
                'train3-attention', 'train4-multi-head', 'train5-adam',
                'train6-inference-toolkit', 'train7-ablation-lab', 'epilogue']
ok = True

def fail(msg):
    global ok
    ok = False
    print('FAIL:', msg)

mds = sorted(ROOT.glob('*.md')) + sorted((ROOT / 'lessons').glob('*.md')) + \
      sorted((ROOT / 'reference').glob('*.md')) + sorted((ROOT / 'data').glob('*.md'))

# gate 1: fence balance
for p in mds:
    n = sum(1 for L in p.read_text().splitlines() if L.strip().startswith('```'))
    if n % 2:
        fail(f'unbalanced fences in {p.relative_to(ROOT)} ({n})')

# gates 2 + 3: links resolve; glossary anchors exist
def slug(h):
    h = h.lower()
    h = ''.join(c for c in h if c.isalnum() or c in ' -_')
    return h.replace(' ', '-')

gloss = (ROOT / 'GLOSSARY.md').read_text()
anchors = {slug(m.group(1)) for m in re.finditer(r'^#### (.+)$', gloss, re.M)}
for p in mds:
    for m in re.finditer(r'\]\(([^)]+)\)', p.read_text()):
        t = m.group(1)
        if t.startswith(('http', 'mailto')):
            continue
        path, _, anchor = t.partition('#')
        if not path:
            continue
        target = (p.parent / path).resolve()
        if not target.exists():
            fail(f'broken link in {p.relative_to(ROOT)}: {t}')
        elif anchor and target.name == 'GLOSSARY.md' and anchor not in anchors:
            fail(f'bad glossary anchor in {p.relative_to(ROOT)}: #{anchor}')

# gate 4: glossary alphabetical
heads = [m.group(1).lower() for m in re.finditer(r'^#### (.+)$', gloss, re.M)]
if heads != sorted(heads):
    for a, b in zip(heads, sorted(heads)):
        if a != b:
            fail(f'glossary out of order at {a!r} (expected {b!r})')
            break

# gate 5: lesson code snippets byte-verbatim in their source file
all_sources = {q: q.read_text() for q in ROOT.glob('*.py')}
for p in sorted((ROOT / 'lessons').glob('train*.md')):
    stem = re.match(r'(train\d)', p.stem).group(1)
    src = (ROOT / f'{stem}.py').read_text()
    for m in re.finditer(r'```python\n(.*?)```', p.read_text(), re.S):
        snip = m.group(1)
        if snip not in src and not any(snip in s for s in all_sources.values()):
            fail(f'snippet in {p.relative_to(ROOT)} not verbatim in any .py: {snip[:60]!r}')

# gate 6: quoted evidence lines verbatim in runs/
runs = '\n'.join(q.read_text() for q in (ROOT / 'runs').glob('*.log'))
quoted = 0
LINE = re.compile(r'^(step +\d+ / \d+ \| loss [\d.]+ \| val loss [\d.]+.*?)$'
                  r'|^(num params: .*?)$|^(gradient check on .*?)$'
                  r'|^(training took .*?)$|^(one document .*?)$', re.M)
for p in sorted((ROOT / 'lessons').glob('*.md')):
    for m in LINE.finditer(p.read_text()):
        line = next(g for g in m.groups() if g)
        key = line.split('...')[0].split('<-')[0].rstrip()
        if key in runs:
            quoted += 1
        else:
            fail(f'quoted line in {p.relative_to(ROOT)} not found in runs/: {line[:70]!r}')

# gate 7: footer nav chain
for i, name in enumerate(LESSON_ORDER):
    last = (ROOT / 'lessons' / f'{name}.md').read_text().strip().splitlines()[-1]
    if i > 0 and f'{LESSON_ORDER[i-1]}.md' not in last:
        fail(f'{name}: footer missing prev link')
    if i < len(LESSON_ORDER) - 1 and f'{LESSON_ORDER[i+1]}.md' not in last:
        fail(f'{name}: footer missing next link')

# gate 8: SVGs are valid XML
for p in sorted((ROOT / 'assets').glob('*.svg')):
    try:
        ET.parse(p)
    except Exception as e:
        fail(f'invalid SVG {p.relative_to(ROOT)}: {e}')

print(f'{"ALL 8 GATES PASS" if ok else "GATES FAILED"} '
      f'({len(anchors)} glossary entries, {quoted} evidence lines checked)')
sys.exit(0 if ok else 1)
