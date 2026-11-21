"""Fit the read-out, the probes and the reference scores on the training items of one model and score item sets.

Inputs are the files written by spred.run_model: the answers (with the shifted view) and features of the 2,000
training items, their sampled answers, and the answers and features of each item set to be scored. Nothing is
fitted on the scored items, and only the probes marked "supervised" below use the labels of the training items.

Fitted on the training items (5 folds; the folds decide the selected feature, layer and ridge strength, and the
out-of-fold values give the standardization constants):
  self_prediction_residual   ridge read-out of the written box (relative coordinates) from the hidden state of the
                             last prompt token, layer and strength chosen by out-of-fold IoU; score 1 - IoU(read-out,
                             written box), 1 for an answer without a box
  two_teacher_probe          ridge on h_last or h_coord distilled from the mean of the shift teacher and the sampling
                             teacher; selected against the top 15% of the teacher
  shift_teacher_probe, sampling_teacher_probe   the same with one teacher, selected against teacher > 0.5
  linear_probe_correctness, linear_probe_one_minus_iou, pregeneration_probe (h_last only), linear_probe_wrong_object_all,
  linear_probe_wrong_object_vs_correct (wrong-object and correct items only), saplma (perceptron on the configuration
  of the correctness probe), box_geometry_supervised                                             (supervised)
  box_geometry_teacher       ridge on eleven features of the written box fitted to the two-teacher target
  mahalanobis                distance to the training items in the top-256 principal subspace of the shift-teacher
                             probe's feature
Not fitted: coordinate_logprob, in_box_attention, whole_image_attention (Qwen models), content_shift_check.
Sums: detector = z(two_teacher_probe) + 0.5 z(self_prediction_residual), and the same with the sampling-teacher
probe and with the linear correctness probe, z with the training constants.

--refit K (K >= 1) refits every fitted score on training items drawn with replacement (seed 3000 + K; the copies of
an item stay in one fold, folds from seed K); with --train-samples-2 the sampled answer of each training item is
also drawn from the two stored draws (seed 2000 + K). --refit 0 (default) is the single fit reported in the paper.

  python -m spred.fit_scores --model-name Qwen2.5-VL-3B \
      --train work/answers/training_qwen25vl3b.jsonl work/features/training_qwen25vl3b.pt \
      --train-samples work/answers/sampled_training_qwen25vl3b.jsonl \
      --test new-items work/answers/new-items_refcoco-testB_qwen25vl3b.jsonl work/features/new-items_refcoco-testB_qwen25vl3b.pt \
      --test new-items work/answers/new-items_refcocog-val_qwen25vl3b.jsonl work/features/new-items_refcocog-val_qwen25vl3b.pt \
      --out-dir work/scores/fit0

Writes <out-dir>/<set>_<model>.csv (item_id, iou, correct, error_type, written and read-out box in relative image
coordinates, every score; larger = more likely wrong), <out-dir>/standardization_<model>.csv and
<out-dir>/config_<model>.json (selected feature, layer and ridge strength of every fitted score).
"""
import argparse
import csv
import json
import math
import os
import statistics

import torch

from spred import geometry as G
from spred import probes as P

W_RESIDUAL = 0.5
COLUMNS = ['detector', 'two_teacher_probe', 'self_prediction_residual', 'sampling_teacher_probe_plus_residual',
           'linear_probe_correctness_plus_residual', 'shift_teacher_probe', 'sampling_teacher_probe', 'mahalanobis',
           'box_geometry_teacher', 'coordinate_logprob', 'in_box_attention', 'whole_image_attention', 'linear_probe_correctness',
           'linear_probe_one_minus_iou', 'saplma', 'pregeneration_probe', 'linear_probe_wrong_object_all',
           'linear_probe_wrong_object_vs_correct', 'box_geometry_supervised', 'content_shift_check']
SUMS = {'detector': 'two_teacher_probe', 'sampling_teacher_probe_plus_residual': 'sampling_teacher_probe',
        'linear_probe_correctness_plus_residual': 'linear_probe_correctness'}


def relative(r, key='pred'):
    b = r.get(key)
    return None if not b else [b[0] / r['W'], b[1] / r['H'], b[2] / r['W'], b[3] / r['H']]


def load(answers, features):
    A = {r['id']: r for r in G.load_jsonl(answers)}
    F = torch.load(features, weights_only=False)
    rs = [A[i] for i in F['ids']]
    nan4 = [float('nan')] * 4
    return dict(ids=list(F['ids']), rs=rs, X={'h_last': F['h_last'].float(), 'h_coord': F['h_coord'].float()},
                logp=(-F['logp']).double(), in_box=(-F['att_in'].float().flatten(1).mean(1)).double(),
                whole=(-F['att_img'].float().flatten(1).mean(1)).double(),
                box=torch.tensor([relative(r) or nan4 for r in rs], dtype=torch.float64),
                y=torch.tensor([int((r['iou'] or 0) < .5) for r in rs]), iou=torch.tensor([float(r['iou'] or 0) for r in rs]),
                shift=torch.tensor([G.disagreement(r['pred'], r.get('pred_shifted_back')) for r in rs], dtype=torch.float64))


def samples(path):
    return {r['id']: r.get('box') for r in G.load_jsonl(path)}


def top15(t):
    return (t >= t.quantile(0.85)).long()


def fit(tr0, samp1, samp2, k):
    """All fitted scores on the training items (refit k; 0 = the reported fit). Returns the appliers, the training
    values that standardize each score, and the configurations."""
    n0 = len(tr0['y'])
    boot = k > 0
    rows = torch.randint(0, n0, (n0,), generator=torch.Generator().manual_seed(3000 + k)) if boot else torch.arange(n0)
    tr = {'rs': [tr0['rs'][i] for i in rows.tolist()], 'X': {f: v[rows] for f, v in tr0['X'].items()}, 'box': tr0['box'][rows],
          'y': tr0['y'][rows], 'iou': tr0['iou'][rows], 'shift': tr0['shift'][rows]}
    n = n0
    ok = ~tr['box'][:, 0].isnan(); B = tr['box'][ok]; nok = int(ok.sum()); idx_ok = ok.nonzero().squeeze(1)
    if samp2 is not None and boot:
        pick = (torch.rand(n0, generator=torch.Generator().manual_seed(2000 + k)) < .5)[rows]
        samp = [samp2.get(r['id']) if p else samp1.get(r['id']) for r, p in zip(tr['rs'], pick.tolist())]
    else:
        samp = [samp1.get(r['id']) for r in tr['rs']]
    e_samp = torch.tensor([G.disagreement(r['pred'], b) for r, b in zip(tr['rs'], samp)], dtype=torch.float64)
    t_two = (tr['shift'] + e_samp) / 2
    wo = tr['iou'] < .1; sub = wo | (tr['iou'] >= .5); ns = int(sub.sum())
    if boot:
        fold = torch.empty(n0, dtype=torch.long)
        for j, p in enumerate(torch.randperm(n0, generator=torch.Generator().manual_seed(k)).chunk(5)):
            fold[p] = j
        fr = fold[rows]
        folds = [(fr == j).nonzero().squeeze(1) for j in range(5)]
        folds_ok = [(fr[ok] == j).nonzero().squeeze(1) for j in range(5)]
        folds_sub = [(fr[sub] == j).nonzero().squeeze(1) for j in range(5)]
    else:
        folds = torch.randperm(n, generator=torch.Generator().manual_seed(0)).chunk(5)
        folds_ok = torch.randperm(nok, generator=torch.Generator().manual_seed(0)).chunk(5)
        folds_sub = list(torch.randperm(ns, generator=torch.Generator().manual_seed(0)).chunk(5))

    apply, train, config = {}, {}, {}

    def probe(name, feats, target, sel, fl=folds, where=None):
        mdl, c = P.select_and_fit(feats, target, sel, fl)
        o = P.out_of_fold(feats[c['feat']][:, c['layer_idx']], target, c['lam'], fl)
        if where is not None:
            full = torch.full((n,), float('nan'), dtype=torch.float64); full[where] = o; o = full
        apply[name] = lambda T, mdl=mdl, c=c: mdl(T['X'][c['feat']][:, c['layer_idx']]).double()
        train[name] = o; config[name] = c

    yd = tr['y'].double()
    probe('two_teacher_probe', tr['X'], t_two, top15(t_two))
    probe('sampling_teacher_probe', tr['X'], e_samp, (e_samp > .5).long())
    probe('linear_probe_correctness', tr['X'], yd, tr['y'])
    probe('shift_teacher_probe', tr['X'], tr['shift'], (tr['shift'] > .5).long())
    probe('linear_probe_one_minus_iou', tr['X'], (1 - tr['iou']).double(), tr['y'])
    probe('pregeneration_probe', {'h_last': tr['X']['h_last']}, yd, tr['y'])
    probe('linear_probe_wrong_object_all', tr['X'], wo.double(), wo.long())
    probe('linear_probe_wrong_object_vs_correct', {f: v[sub] for f, v in tr['X'].items()}, wo[sub].double(), wo[sub].long(), folds_sub, where=sub)

    cg = config['linear_probe_correctness']; Xg = tr['X'][cg['feat']][:, cg['layer_idx']]
    mlp = P.fit_mlp(Xg, tr['y']); o = torch.zeros(n, dtype=torch.float64)
    for p in folds:
        m = torch.ones(n, dtype=torch.bool); m[p] = False
        o[p] = torch.as_tensor(P.fit_mlp(Xg[m], tr['y'][m])(Xg[p])).double()
    train['saplma'] = o
    apply['saplma'] = lambda T, mlp=mlp, cg=cg: torch.as_tensor(mlp(T['X'][cg['feat']][:, cg['layer_idx']])).double()

    ce = config['shift_teacher_probe']; Xe = tr['X'][ce['feat']][:, ce['layer_idx']].double(); mu = Xe.mean(0)
    _, Sg, Vt = torch.linalg.svd(Xe - mu, full_matrices=False); kk = min(256, len(Sg) - 1); Pm = Vt[:kk].T; var = (Sg[:kk] ** 2) / (len(Xe) - 1)
    mah = lambda X, mu=mu, Pm=Pm, var=var: ((((X.double() - mu) @ Pm) ** 2) / var).sum(1).sqrt()
    train['mahalanobis'] = mah(Xe)
    apply['mahalanobis'] = lambda T, mah=mah, ce=ce: mah(T['X'][ce['feat']][:, ce['layer_idx']])

    Gtr = P.geometry_features(tr['box'])
    for name, tgt, lab in (('box_geometry_teacher', t_two, top15(t_two)), ('box_geometry_supervised', yd, tr['y'])):
        lam = max((P.auroc(P.out_of_fold(Gtr, tgt, l, folds), lab), l) for l in P.GEOMETRY_LAMS)[1]
        gm = P.Ridge(Gtr, tgt, lam)
        train[name] = P.out_of_fold(Gtr, tgt, lam, folds); config[name] = {'feat': 'box geometry', 'layer_idx': 0, 'lam': lam}
        apply[name] = lambda T, gm=gm: gm(P.geometry_features(T['box'])).double()

    best = None
    for li in range(tr['X']['h_last'].shape[1]):
        Xl = tr['X']['h_last'][ok][:, li].double()
        for lam in P.LAMS:
            cv = 0.0
            for p in folds_ok:
                m = torch.ones(nok, dtype=torch.bool); m[p] = False
                cv += P.box_iou(P.BoxRidge(Xl[m], B[m], lam)(Xl[p]), B[p]).sum().item()
            if best is None or cv > best[0]:
                best = (cv, li, lam)
    _, rli, rlam = best
    Xr = tr['X']['h_last'][ok][:, rli].double(); readout = P.BoxRidge(Xr, B, rlam); o = torch.ones(n, dtype=torch.float64)
    ro = torch.empty(nok, 4, dtype=torch.float64)
    for p in folds_ok:
        m = torch.ones(nok, dtype=torch.bool); m[p] = False
        ro[p] = P.BoxRidge(Xr[m], B[m], rlam)(Xr[p])
        o[idx_ok[p]] = 1 - P.box_iou(ro[p], B[p])
    train['self_prediction_residual'] = o
    # read-out error sigma: median distance between the centres of the out-of-fold read-out and the written box
    cen = lambda b: (b[:, :2] + b[:, 2:]) / 2
    sigma = statistics.median((cen(ro) - cen(B)).norm(dim=1).tolist())
    config['self_prediction_residual'] = {'feat': 'h_last', 'layer_idx': rli, 'lam': rlam, 'cv_iou': best[0] / nok, 'sigma': sigma}

    def residual(T):
        r = readout(T['X']['h_last'][:, rli].double())
        s = 1 - P.box_iou(r, torch.nan_to_num(T['box'], nan=0.0)); s[T['box'][:, 0].isnan()] = 1.0
        return s, r
    apply['self_prediction_residual'] = residual
    for name, key in (('coordinate_logprob', 'logp'), ('in_box_attention', 'in_box'), ('whole_image_attention', 'whole'), ('content_shift_check', 'shift')):
        train[name] = tr[key] if key == 'shift' else tr0[key][rows]
        apply[name] = lambda T, key=key: T[key]
    return apply, train, config


def constants(v):
    """Mean and population standard deviation of the non-missing training values."""
    x = [float(t) for t in v.tolist() if not math.isnan(t)]
    m = sum(x) / len(x)
    return m, math.sqrt(sum((t - m) ** 2 for t in x) / len(x)), len(v.tolist()) - len(x)


def error_type(u):
    return 'correct' if u >= 0.5 else 'wrong_object' if u < 0.1 else 'partial_overlap' if u < 0.3 else 'wrong_extent'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--train', nargs=2, required=True, metavar=('ANSWERS', 'FEATURES'))
    ap.add_argument('--train-samples', required=True)
    ap.add_argument('--train-samples-2', help='a second draw of the sampled answers (run_model sample --seed-offset 1), used by the refits')
    ap.add_argument('--test', nargs=3, action='append', required=True, metavar=('SET', 'ANSWERS', 'FEATURES'))
    ap.add_argument('--refit', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--digits', type=int, default=8)
    a = ap.parse_args()
    tr0 = load(*a.train)
    s1 = samples(a.train_samples)
    s2 = samples(a.train_samples_2) if a.train_samples_2 else None
    apply, train, config = fit(tr0, s1, s2, a.refit)
    attention = float(train['in_box_attention'].std()) > 0
    cols = [c for c in COLUMNS if attention or c not in ('in_box_attention', 'whole_image_attention')]
    std = {k: constants(v) for k, v in train.items()}
    os.makedirs(a.out_dir, exist_ok=True)
    fmt = lambda x: '' if x is None or math.isnan(x) else f'{x:.{a.digits}g}'
    with open(os.path.join(a.out_dir, f'standardization_{a.model_name}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'score', 'training_rows', 'missing', 'training_mean', 'training_sd'])
        for c in cols:
            if c in std:
                m, sd, miss = std[c]; w.writerow([a.model_name, c, len(train[c]), miss, f'{m:.10g}', f'{sd:.10g}'])
    with open(os.path.join(a.out_dir, f'config_{a.model_name}.json'), 'w') as f:
        json.dump({'refit': a.refit, 'config': config}, f, indent=1)
    sets = {}
    for name, ans, feat in a.test:
        sets.setdefault(name, []).append((ans, feat))
    for name, parts in sets.items():
        path = os.path.join(a.out_dir, f'{name}_{a.model_name}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['item_id', 'iou', 'correct', 'error_type', 'written_x1', 'written_y1', 'written_x2', 'written_y2',
                        'readout_x1', 'readout_y1', 'readout_x2', 'readout_y2'] + cols)
            for ans, feat in parts:
                T = load(ans, feat)
                S = {}
                for c, fn in apply.items():
                    if c == 'self_prediction_residual':
                        S[c], R = fn(T)
                    else:
                        S[c] = fn(T)
                for c, base in SUMS.items():
                    (mb, sb, _), (mr, sr, _) = std[base], std['self_prediction_residual']
                    S[c] = (S[base] - mb) / (sb + 1e-12) + W_RESIDUAL * (S['self_prediction_residual'] - mr) / (sr + 1e-12)
                for i, item in enumerate(T['ids']):
                    u = float(T['iou'][i])
                    w.writerow([item, f'{u:.6f}', int(u >= 0.5), error_type(u)] + [fmt(float(v)) for v in T['box'][i]]
                               + [fmt(float(v)) for v in R[i]] + [fmt(float(S[c][i])) for c in cols])
        print(f'[fit] {name}: {sum(len(load(x, y)["ids"]) for x, y in parts)} items -> {path}')
    print('[fit] ' + '; '.join(f"{k} {c['feat']}#{c['layer_idx']} lam {c['lam']:g}" for k, c in config.items()))


if __name__ == '__main__':
    main()
