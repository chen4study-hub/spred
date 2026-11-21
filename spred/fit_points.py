"""Fit the point read-out and the probes on the point answers of the training items and score item sets (CPU).

An answer is an error when the written point lies outside the annotated box. Distances are in relative image units
(fractions of the width and height), capped at 0.5 for the teachers (also when a view has no point).
  two_teacher_probe        ridge on h_last or h_coord distilled from the mean of two teachers: the distance between the
                           greedy point and the point of the shifted view, and the distance to one sampled point;
                           feature, layer and penalty by 5-fold out-of-fold AUROC against the top 15% of the teacher
  sampling_teacher_probe   the same with the sampling teacher only
  self_prediction_residual ridge read-out of the written point from the hidden state of the last prompt token (layer
                           and penalty by the out-of-fold distance); score = distance between read-out and written point
  linear_probe_correctness, saplma   supervised linear probe and perceptron on correctness labels
  coordinate_logprob       minus the mean log-probability of the coordinate tokens
z(score) uses the mean and (population) standard deviation of the out-of-fold training values; the perceptron has
none and is standardized with its values on the scored items.

  python -m spred.fit_points --model-name M --train work/points/training_M.jsonl work/points/training_M.pt \
      --train-samples work/points/sampled_training_M.jsonl \
      --test replication work/points/replication_M.jsonl work/points/replication_M.pt --out-dir work/points/fit0

--refit K (K >= 1) draws the training items with replacement (seed 3000 + K; copies of an item stay in one fold, folds
from seed K) and fits everything again. Writes <out-dir>/points_<set>_<model>.csv and standardization_points_<model>.csv.
"""
import argparse
import csv
import math
import os

import numpy as np
import torch

from spred import geometry as G
from spred import probes as P

CAP = 0.5
COLUMNS = ['two_teacher_probe', 'self_prediction_residual', 'sampling_teacher_probe', 'linear_probe_correctness', 'saplma', 'coordinate_logprob']


def rel(p, r):
    return None if p is None else [p[0] / r['W'], p[1] / r['H']]


def dist(p, q):
    return CAP if p is None or q is None else min(float(np.hypot(p[0] - q[0], p[1] - q[1])), CAP)


def load(answers, features):
    """answers and features of one list; the first record of an item is used"""
    g = {}
    for r in G.load_jsonl(answers):
        g.setdefault(r['id'], r)
    F = torch.load(features, weights_only=False); keep, seen = [], set()
    for k, i in enumerate(F['ids']):
        if i in g and i not in seen:
            seen.add(i); keep.append(k)
    rs = [g[F['ids'][k]] for k in keep]
    return dict(rs=rs, ids=[r['id'] for r in rs], X={'h_last': F['h_last'][keep].float(), 'h_coord': F['h_coord'][keep].float()},
                H=F['h_last'][keep].double(), pt=torch.tensor([rel(r['pt'], r) for r in rs], dtype=torch.float64),
                y=torch.tensor([0 if r['hit'] else 1 for r in rs]), logp=-F['logp'][keep].double(),
                shift=torch.tensor([dist(rel(r['pt'], r), rel(r.get('pt_shifted_back'), r)) for r in rs], dtype=torch.float64))


def top15(t):
    return (t >= t.quantile(0.85)).long()


def fit(tr0, samples, tests, k):
    n0 = len(tr0['rs'])
    if k == 0:
        rows = torch.arange(n0); folds = list(torch.randperm(n0, generator=torch.Generator().manual_seed(0)).chunk(5))
    else:
        rows = torch.randint(0, n0, (n0,), generator=torch.Generator().manual_seed(3000 + k)); fold = torch.empty(n0, dtype=torch.long)
        for j, p in enumerate(torch.randperm(n0, generator=torch.Generator().manual_seed(k)).chunk(5)):
            fold[p] = j
        folds = [(fold[rows] == j).nonzero().squeeze(1) for j in range(5)]
    e_samp0 = torch.tensor([dist(rel(r['pt'], r), rel(samples[r['id']], r) if r['id'] in samples else None) for r in tr0['rs']], dtype=torch.float64)
    n = len(rows); X = {q: v[rows] for q, v in tr0['X'].items()}; H = tr0['H'][rows]; pt = tr0['pt'][rows]; yt = tr0['y'][rows]
    e_samp = e_samp0[rows]; t_two = (tr0['shift'][rows] + e_samp) / 2
    S, TRAIN, CFG = {}, {}, {}
    for name, target, sel in (('two_teacher_probe', t_two, top15(t_two)), ('sampling_teacher_probe', e_samp, top15(e_samp)), ('linear_probe_correctness', yt.double(), yt)):
        mdl, c = P.select_and_fit(X, target, sel, folds); CFG[name] = c
        S[name] = [mdl(T['X'][c['feat']][:, c['layer_idx']]).double().numpy() for T in tests]
        TRAIN[name] = P.out_of_fold(X[c['feat']][:, c['layer_idx']], target, c['lam'], folds).numpy()
    cg = CFG['linear_probe_correctness']; mlp = P.fit_mlp(X[cg['feat']][:, cg['layer_idx']], yt)
    S['saplma'] = [np.asarray(mlp(T['X'][cg['feat']][:, cg['layer_idx']]), float) for T in tests]
    d2 = lambda p, q: (p - q).norm(dim=1)
    best = None
    for li in range(H.shape[1]):
        for lam in P.LAMS:
            cv = 0.0
            for p in folds:
                m = torch.ones(n, dtype=torch.bool); m[p] = False; cv += d2(P.BoxRidge(H[m][:, li], pt[m], lam)(H[p][:, li]), pt[p]).sum().item()
            if best is None or cv < best[0]:
                best = (cv, li, lam)
    _, li, lam = best; CFG['self_prediction_residual'] = {'feat': 'h_last', 'layer_idx': li, 'lam': lam}
    ro = P.BoxRidge(H[:, li], pt, lam); S['self_prediction_residual'] = [d2(ro(T['H'][:, li]), T['pt']).numpy() for T in tests]
    o = torch.zeros(n, dtype=torch.float64)
    for p in folds:
        m = torch.ones(n, dtype=torch.bool); m[p] = False; o[p] = d2(P.BoxRidge(H[m][:, li], pt[m], lam)(H[p][:, li]), pt[p])
    TRAIN['self_prediction_residual'] = o.numpy()
    S['coordinate_logprob'] = [T['logp'].numpy() for T in tests]; TRAIN['coordinate_logprob'] = tr0['logp'][rows].numpy()
    return S, {q: (float(np.mean(v)), float(np.std(v))) for q, v in TRAIN.items()}, CFG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--train', nargs=2, required=True, metavar=('ANSWERS', 'FEATURES'))
    ap.add_argument('--train-samples', required=True)
    ap.add_argument('--test', nargs=3, action='append', required=True, metavar=('SET', 'ANSWERS', 'FEATURES'))
    ap.add_argument('--refit', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--digits', type=int, default=8)
    a = ap.parse_args()
    tr0 = load(*a.train)
    samples = {}
    for r in G.load_jsonl(a.train_samples):
        samples.setdefault(r['id'], r['pt'])
    tests = [load(ans, feat) for _, ans, feat in a.test]
    S, std, cfg = fit(tr0, samples, tests, a.refit)
    os.makedirs(a.out_dir, exist_ok=True)
    fmt = lambda x: '' if x is None or math.isnan(x) else f'{x:.{a.digits}g}'
    with open(os.path.join(a.out_dir, f'standardization_points_{a.model_name}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'score', 'training_mean', 'training_sd'])
        for c in COLUMNS:
            if c in std:
                w.writerow([a.model_name, c, f'{std[c][0]:.10g}', f'{std[c][1]:.10g}'])
    for j, (name, _, _) in enumerate(a.test):
        T = tests[j]
        with open(os.path.join(a.out_dir, f'points_{name}_{a.model_name}.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['item_id', 'point_x', 'point_y', 'correct'] + COLUMNS)
            for i, r in enumerate(T['rs']):
                w.writerow([r['id'], fmt(r['pt'][0]), fmt(r['pt'][1]), 1 - int(T['y'][i])] + [fmt(float(S[c][j][i])) for c in COLUMNS])
        print(f'[fit] {name}: {len(T["rs"])} items')
    print('[fit] ' + '; '.join(f"{q} {c['feat']}#{c['layer_idx']} lam {c['lam']:g}" for q, c in cfg.items()))


if __name__ == '__main__':
    main()
