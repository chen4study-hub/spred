"""AUROC of SPRED and of the supervised linear probe on correctness for training sizes up to 10,000 items
(scores/matched-sizes/*.csv; Qwen2.5-VL-3B and Qwen2.5-VL-7B).

usage (from the top directory of the package):
  python code/matched_sizes.py table     AUROC for all errors per model, list and size; SPRED minus the probe
  python code/matched_sizes.py check     every number of code/expected_matched.csv against the recomputed value

All-error AUROC: items with correct = 0 against items with correct = 1, ties counted as one half. SPRED was fitted on
the training items without their labels, the probe on the same items with their labels (feature, layer and penalty
searched at each size). The intervals of the paper add ten refits on redrawn training items, which are not part of
this package; the point estimates follow from these files.
"""
import csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ('Qwen2.5-VL-3B', 'Qwen2.5-VL-7B'); SETS = ('replication', 'heldout')


def auroc(s, y):
    s = np.asarray(s, float); pos = y == 1; neg = y == 0
    o = np.argsort(s, kind='stable'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1: r[m] = r[m].mean()
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def load(sname, model):
    with open(os.path.join(ROOT, 'scores', 'matched-sizes', f'{sname}_{model}.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    sizes = [c[len('spred_'):] for c in rows[0] if c.startswith('spred_')]
    y = np.array([1 - int(r['correct']) for r in rows])
    return sizes, {(k, n): auroc([float(r[f'{k}_{n}']) for r in rows], y) for k in ('spred', 'probe') for n in sizes}


def values():
    out = {}
    for sname in SETS:
        d = []
        for m in MODELS:
            sizes, A = load(sname, m)
            for n in sizes:
                out[f'{sname}|{m}|spred_{n}'] = A['spred', n]; out[f'{sname}|{m}|probe_{n}'] = A['probe', n]
                out[f'{sname}|{m}|diff_{n}'] = A['spred', n] - A['probe', n]
            d.append(A['spred', sizes[-1]] - A['probe', sizes[-1]])
        out[f'{sname}|mean|diff_largest'] = float(np.mean(d))
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'table'
    V = values()
    if cmd == 'table':
        for k, v in V.items():
            print(f'{k:40s} {v:+.4f}')
        return
    bad = n = 0
    with open(os.path.join(ROOT, 'code', 'expected_matched.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['quantity'] not in V:
                n += 1; bad += 1
                print(f"MISSING {r['place']:28s} {r['quantity']}")
                continue
            n += 1; got = round(V[r['quantity']], int(r['decimals'])); want = float(r['value'])
            ok = abs(got - want) < 1e-9
            bad += not ok
            print(f"{'ok ' if ok else 'DIFF'} {r['place']:28s} {r['quantity']:40s} recomputed {got:+.{r['decimals']}f}  paper {want:+.{r['decimals']}f}")
    print(f'{bad} of the {n} listed numbers differ after rounding')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
