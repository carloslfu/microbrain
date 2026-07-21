"""
compare.py: The whole ladder on one screen.

Collects the instrument-panel numbers from each rung's saved log (out/trainN.log)
and prints the course's summary table: what each idea bought, in effective
choices. Rungs you haven't run yet are run here with --fast (shorter budgets),
marked with * in the table.

usage: python compare.py           # use existing full logs, fast-run the missing
       python compare.py --run     # fast-run everything fresh (cold ladder)
"""

import os         # os.path.exists
import re         # re.findall
import sys        # sys.argv, sys.executable
import subprocess # subprocess.run

HERE = os.path.dirname(os.path.abspath(__file__))
assert os.path.exists(os.path.join(HERE, 'data', 'data.txt')), \
    "no data/data.txt yet — run: python data/make_dataset.py"
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)
FORCE = '--run' in sys.argv

RUNGS = [
    (0, 'counting'),
    (1, 'gradient descent, by hand'),
    (2, 'autograd'),
    (3, 'attention'),
    (4, 'multi-head'),
    (5, 'Adam  (= microgpt)'),
]

def parse(log_text):
    panels = re.findall(r'val loss ([\d.]+) \| effective choices ([\d.]+)', log_text)
    mems = re.findall(r'memorization: (\d+)/(\d+)', log_text)
    if not panels:
        return None
    vl, eff = panels[-1]
    mem = f"{mems[-1][0]}/{mems[-1][1]}" if mems else '-'
    return float(vl), float(eff), mem

rows = []
for n, adds in RUNGS:
    log_path = os.path.join(OUT, f'train{n}.log')
    fast_path = os.path.join(OUT, f'train{n}.fast.log')
    fast = False
    text = open(log_path).read() if os.path.exists(log_path) else ''
    if FORCE or parse(text) is None: # no full log (or an unfinished one): fast-run it here
        print(f"running train{n}.py --fast ...")
        r = subprocess.run([sys.executable, os.path.join(HERE, f'train{n}.py'), '--fast'],
                           capture_output=True, text=True)
        open(fast_path, 'w').write(r.stdout + r.stderr)
        text, fast = r.stdout, True
    parsed = parse(text)
    rows.append((n, adds, parsed, fast))

print()
print("rung  file        adds                        val loss   eff. choices   memorized")
print("      (no model)  uniform guessing              3.6376      38.0        -")
for n, adds, parsed, fast in rows:
    star = '*' if fast else ' '
    if parsed is None:
        print(f"{n}{star}    train{n}.py   {adds:26s}  (no panel found in log)")
        continue
    vl, eff, mem = parsed
    print(f"{n}{star}    train{n}.py   {adds:26s}  {vl:.4f}      {eff:4.1f}        {mem}")
if any(fast for *_, fast in rows):
    print("*  ran here with --fast (shorter budget) — numbers are a bit worse than full runs")

# The childhood album, if train5 has been run in full
t5 = os.path.join(OUT, 'train5.log')
if os.path.exists(t5):
    text = open(t5).read()
    if 'childhood album' in text:
        album = text.split('childhood album: samples through training ---')[1].split('\n\n')[0]
        print("\n--- watch it learn (from train5.py) ---" + album)

print("\nthe ladder ends here; the toolkit does not: train6.py (save/load, temperature,")
print("KV cache), train7.py (the ablation lab), namer.py (your model as a tool).")
