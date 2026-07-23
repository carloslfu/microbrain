"""
Harvest the training corpus for microbrain: the names of ideas.

The dataset is a list of short strings, one per line — same shape as the
32,033 human names Karpathy trains microgpt on. Ours are the slugs of a
personal db.md knowledge store: concept names, entity names, source-note
titles. Things like:

    test-time-training
    geva-2021-ffn-kv-memory
    representation-quality
    thin-harness-bet-on-model-curve

The model's whole job will be: given the start of a slug, predict the next
character. That objective, pushed hard enough, forces it to learn what idea
names *look like*.

Filters (each drops a real failure mode, printed at the end — no silent caps):
  - date prefixes stripped (2026-05-18-foo -> foo): calendar noise, not naming
  - slugs with 6+ consecutive digits dropped: tweet ids and booking codes,
    machine-generated — there is no pattern to learn there
  - anything outside [a-z0-9-] dropped: keeps the vocabulary tiny (38 tokens)
  - slugs longer than 39 chars dropped: our context window is 40 (BOS + 39)

Usage:
    python data/make_dataset.py                 # harvest the brain (default)
    python data/make_dataset.py --names         # Karpathy's names dataset instead
    python data/make_dataset.py --from list.txt # any file of short strings, one per line

Output: data/data.txt (one doc per line, shuffled with a fixed seed so every
train script sees the same order; the last 10% will serve as validation data).
"""

import os
import re
import sys
import random
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data.txt")
DB = os.path.expanduser("~/Projects/command-center/db/records")
DIRS = ["concepts", "entities", "source-notes", "synthesis"]
MAX_LEN = 39  # block_size 40 = BOS + 39 characters

def harvest_brain():
    slugs, drops = set(), {"machine-id": 0, "charset": 0, "too-long": 0}
    raw = 0
    for d in DIRS:
        for root, _, files in os.walk(os.path.join(DB, d)):
            for f in sorted(files):
                if not f.endswith(".md") or f.startswith("index"):
                    continue
                raw += 1
                s = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", f[:-3])
                if re.search(r"\d{6,}", s):
                    drops["machine-id"] += 1
                elif not re.fullmatch(r"[a-z0-9-]+", s):
                    drops["charset"] += 1
                elif len(s) > MAX_LEN:
                    drops["too-long"] += 1
                else:
                    slugs.add(s)
    return sorted(slugs), raw, drops

def harvest_file(path):
    """Any newline-separated list of short strings becomes a corpus."""
    raw_docs = [l.strip().lower() for l in open(path) if l.strip()]
    drops = {"charset": 0, "too-long": 0}
    docs = set()
    for d in raw_docs:
        if not re.fullmatch(r"[a-z0-9-]+", d):
            drops["charset"] += 1
        elif len(d) > MAX_LEN:
            drops["too-long"] += 1
        else:
            docs.add(d)
    return sorted(docs), len(raw_docs), drops

def harvest_names():
    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    path = os.path.join(HERE, "names.txt")
    if not os.path.exists(path):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return harvest_file(path)

if __name__ == "__main__":
    if "--from" in sys.argv:
        docs, raw, drops = harvest_file(sys.argv[sys.argv.index("--from") + 1])
    elif "--names" in sys.argv:
        docs, raw, drops = harvest_names()
    else:
        assert os.path.isdir(DB), f"brain not found at {DB}; try --names, or --from <your list of strings>"
        docs, raw, drops = harvest_brain()

    random.seed(42)
    random.shuffle(docs)
    with open(OUT, "w") as f:
        f.write("\n".join(docs) + "\n")

    lens = [len(d) for d in docs]
    charset = "".join(sorted(set("".join(docs))))
    print(f"scanned {raw} inputs -> kept {len(docs)} unique docs -> {OUT}")
    if drops:
        print(f"dropped: {drops}")
    print(f"doc length min/median/max: {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    print(f"charset ({len(charset)} chars): {charset}")
    print(f"vocab will be {len(charset) + 1} tokens (chars + BOS)")
    print("sample:", ", ".join(docs[:4]))

    # The committed runs/ logs pin one frozen snapshot of this corpus
    # (data/MANIFEST.txt). A live brain grows, so a fresh harvest can drift —
    # say so, instead of letting the lesson numbers silently stop matching.
    manifest = os.path.join(os.path.dirname(OUT), "MANIFEST.txt")
    if os.path.exists(manifest):
        import hashlib
        sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
        pinned = next((l.split()[1] for l in open(manifest) if l.startswith("sha256:")), None)
        if pinned and sha != pinned:
            print(f"note: this harvest differs from the canonical snapshot in {manifest}")
            print("      the committed logs and lesson numbers pin that snapshot; yours will differ")
