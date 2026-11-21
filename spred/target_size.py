"""Gain from the residual by target size in units of the read-out error (CPU, label-free size measure).

sigma is the median distance between the centres of the out-of-fold read-out and the written box over the training
items of the fit (written by spred.fit_scores into config_<model>.json). r = sqrt(area of the written box) / sigma,
both in coordinates relative to the image size; an answer without a box has r = 0. Strata r < 2 and r >= 2.
Within a stratum the increment of a base score is AUROC(z(base) + 0.5 z(residual)) - AUROC(z(base)), z with the
training constants of the fit (standardization_<model>.csv), for wrong-object errors (IoU < 0.1) and for all errors
(IoU < 0.5) against correct answers (IoU >= 0.5); a cell with fewer than 5 errors of the kind or 5 correct answers is
undefined. Rows: the two-teacher probe (SPRED minus the probe), the supervised linear probe and the mean over seven
hidden-state probes. Intervals: image-cluster bootstrap, the images of all models resampled together, seed 0.

  python -m spred.target_size --fit-dir work/scores/objects365_fitted-on-RefCOCO --set objects365 \
      --list work/lists/objects365_test.csv --models Qwen2.5-VL-3B Qwen2.5-VL-7B Qwen3-VL-4B

reads <fit-dir>/<set>_<model>.csv, standardization_<model>.csv and config_<model>.json; --list gives the image of each
item (column coco_image_id or image_id).
"""
import argparse
import csv
import json
import os

import numpy as np

SEVEN = ('two_teacher_probe', 'sampling_teacher_probe', 'shift_teacher_probe', 'linear_probe_correctness',
         'linear_probe_one_minus_iou', 'pregeneration_probe', 'saplma')
ROWS = (('two-teacher probe (SPRED minus the probe)', ('two_teacher_probe',)), ('supervised linear probe', ('linear_probe_correctness',)),
        ('seven hidden-state probes', SEVEN))
DEFS = (('wrong-object errors', .1), ('all errors', .5))
STRATA = (('r < 2', 0., 2.), ('r >= 2', 2., np.inf))


def wauc(s, pos, neg, W):
    """AUROC of s for the positives, one value per row of the weights W (nan where a class has no weight)"""
    o = np.argsort(s, kind='stable'); st = np.r_[0, np.where(np.diff(s[o]) != 0)[0] + 1]
    Gp = np.add.reduceat(W[:, o] * pos[o], st, axis=1); Gn = np.add.reduceat(W[:, o] * neg[o], st, axis=1); den = Gp.sum(1) * Gn.sum(1)
    return np.where(den > 0, (Gp * (np.cumsum(Gn, axis=1) - Gn + .5 * Gn)).sum(1) / np.maximum(den, 1e-9), np.nan)


def ci(d):
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))] if np.isfinite(d).mean() >= .8 else [float('nan')] * 2


def load(fit_dir, name, model, image_of):
    rows = list(csv.DictReader(open(os.path.join(fit_dir, f'{name}_{model}.csv'))))
    std = {r['score']: (float(r['training_mean']), float(r['training_sd'])) for r in csv.DictReader(open(os.path.join(fit_dir, f'standardization_{model}.csv')))}
    sigma = json.load(open(os.path.join(fit_dir, f'config_{model}.json')))['config']['self_prediction_residual']['sigma']
    num = lambda c: np.array([float(r[c]) if r[c] != '' else np.nan for r in rows])
    B = np.stack([num(f'written_{k}') for k in ('x1', 'y1', 'x2', 'y2')], 1)
    area = np.nan_to_num(np.clip(B[:, 2] - B[:, 0], 0, None) * np.clip(B[:, 3] - B[:, 1], 0, None), nan=0.)
    Z = {}
    for q in SEVEN + ('self_prediction_residual',):
        v = num(q); v = np.where(np.isnan(v), np.nanmax(v), v)
        Z[q] = (v - std[q][0]) / std[q][1]
    return {'sigma': sigma, 'r': np.sqrt(area) / sigma, 'iou': np.nan_to_num(num('iou')), 'img': np.array([image_of[r['item_id']] for r in rows]), 'Z': Z}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit-dir', required=True); ap.add_argument('--set', required=True); ap.add_argument('--list', required=True)
    ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--draws', type=int, default=10000)
    a = ap.parse_args()
    image_of = {r['item_id']: str(r.get('coco_image_id') or r['image_id']) for r in csv.DictReader(open(a.list))}
    M = {t: load(a.fit_dir, a.set, t, image_of) for t in a.models}; tags = a.models
    imgs = sorted(set().union(*[set(M[t]['img']) for t in tags])); pos = {i: j for j, i in enumerate(imgs)}; rng = np.random.default_rng(0)
    CNT = np.stack([np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)) for _ in range(a.draws)]).astype(np.float32)
    g3 = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    print(f"{a.set}: {len(tags)} models, {len(imgs)} images, {a.draws} image resamples")
    print("  sigma per model: " + ', '.join(f"{t} {M[t]['sigma']:.3f}" for t in tags))
    PT, BT = {}, {}
    for t in tags:
        d = M[t]; W = CNT[:, np.array([pos[i] for i in d['img']])]; W0 = np.ones((1, len(d['iou'])), np.float32); zr = d['Z']['self_prediction_residual']
        n = [int((d['r'] < 2).sum()), len(d['r'])]
        print(f"  {t}: items with r < 2: {n[0]} of {n[1]} ({n[0] / n[1]:.3f}); errors with r < 2: "
              f"{int(((d['r'] < 2) & (d['iou'] < .5)).sum())}, wrong-object {int(((d['r'] < 2) & (d['iou'] < .1)).sum())}")
        for sname, lo, hi in STRATA:
            m = (d['r'] >= lo) & (d['r'] < hi)
            for dname, thr in DEFS:
                pf, nf = (m & (d['iou'] < thr)).astype(np.float32), (m & (d['iou'] >= .5)).astype(np.float32)
                if pf.sum() < 5 or nf.sum() < 5:
                    continue
                for b in SEVEN:
                    zb = d['Z'][b]
                    PT[sname, dname, b, t] = float(wauc(zb + .5 * zr, pf, nf, W0)[0] - wauc(zb, pf, nf, W0)[0])
                    BT[sname, dname, b, t] = (wauc(zb + .5 * zr, pf, nf, W) - wauc(zb, pf, nf, W)).astype(np.float32)
        del W
    for dname, _ in DEFS:
        print(f"  ## {dname}")
        for rname, bases in ROWS:
            for sname, _, _ in STRATA:
                per = {t: [float(np.mean([PT[k] for k in ks])), *ci(np.mean([BT[k] for k in ks], 0))]
                       for t in tags for ks in [[(sname, dname, b, t) for b in bases if (sname, dname, b, t) in PT]] if ks}
                ks = [(sname, dname, b, t) for b in bases for t in tags if (sname, dname, b, t) in PT]
                mean = f"; mean over {len(per)} models {g3([float(np.mean([PT[k] for k in ks])), *ci(np.mean([BT[k] for k in ks], 0))])}" if ks else ''
                print(f"    {rname} | {sname}: per model " + ' / '.join(g3(per[t]) if t in per else 'n/a' for t in tags) + mean)


if __name__ == '__main__':
    main()
