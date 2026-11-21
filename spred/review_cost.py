"""Review with unequal costs: flagging a share of the answers when a wrong-object error costs c times another error.

The share q (default 0.2) of the answers with the highest score is flagged; a flagged error counts as caught. A
wrong-object error (IoU < 0.1) costs c, another error (0.1 <= IoU < 0.5) costs 1, and the cost-weighted recall is
the cost of the caught errors over the cost of all errors. Scores: SPRED = z(two-teacher probe) + 0.5 z(residual),
the two-teacher probe, the coordinate log-probability and the supervised linear probe on correctness (uses labels),
z with the training constants of each fit. Items are taken in the order of their ids. Reported: the recall for each
cost in --costs with the differences SPRED minus each other score, the cost ratio at which SPRED and the probe break
even (stored fit, steps of 0.01), and the first c of --grid at which the interval of SPRED minus the probe lies above
zero. Means over the models; intervals: stored value +- 1.96 sqrt(image SE^2 + refit SD^2), the images of all models
resampled together (seed 0, --draws), the refits being fits on training items drawn with replacement.

  python -m spred.review_cost --set replication --list data/lists/replication.csv --models M1 M2 ... \
      --fits work/scores/fit0 work/scores/refit1 ... work/scores/refit10

reads <dir>/<set>_<model>.csv and <dir>/standardization_<model>.csv of each fit written by spred.fit_scores (the
first directory is the stored fit, the others its refits, --refit 1 ... 10).
"""
import argparse
import csv
import os

import numpy as np

SCORES = ('SPRED', 'two-teacher probe', 'coordinate log-probability', 'supervised linear probe')
CONTRASTS = (('SPRED minus the probe', 0, 1), ('SPRED minus the log-probability', 0, 2), ('SPRED minus the supervised probe', 0, 3))
COLUMNS = ('two_teacher_probe', 'self_prediction_residual', 'coordinate_logprob', 'linear_probe_correctness')


def caught(s, wo, ot, W, share):
    """caught wrong-object errors, caught other errors, flagged answers and answers, one value per row of the weights W"""
    o = np.argsort(s, kind='stable')[::-1]; w = W[:, o].astype(np.float64); N = w.sum(1); k = N - np.floor((1 - share) * N)
    before = np.cumsum(w, 1) - w; f = np.clip(k[:, None] - before, 0, w)
    return np.stack([(f * wo[o]).sum(1), (f * ot[o]).sum(1), k, N])


def combined(st, B0, pts):
    tot = float(np.hypot(np.std(pts, ddof=1) if len(pts) > 1 else 0., np.std(B0)))
    return [float(st), float(st - 1.96 * tot), float(st + 1.96 * tot)]


def load(d, name, model):
    rows = sorted(csv.DictReader(open(os.path.join(d, f'{name}_{model}.csv'))), key=lambda r: r['item_id'])
    std = {r['score']: (float(r['training_mean']), float(r['training_sd'])) for r in csv.DictReader(open(os.path.join(d, f'standardization_{model}.csv')))}

    def z(q):
        v = np.array([float(r[q]) if r[q] != '' else np.nan for r in rows]); v = np.where(np.isnan(v), np.nanmax(v), v)
        return (v - std[q][0]) / std[q][1]
    sc = (z('two_teacher_probe') + .5 * z('self_prediction_residual'), z('two_teacher_probe'), z('coordinate_logprob'), z('linear_probe_correctness'))
    return [r['item_id'] for r in rows], np.array([float(r['iou'] or 0) for r in rows]), sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', required=True); ap.add_argument('--list', required=True); ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--fits', nargs='+', required=True); ap.add_argument('--share', type=float, default=.2)
    ap.add_argument('--costs', type=float, nargs='+', default=[1, 2, 3, 5])
    ap.add_argument('--grid', type=float, nargs='+', default=[round(1 + .1 * i, 1) for i in range(41)]); ap.add_argument('--draws', type=int, default=2000)
    a = ap.parse_args()
    image_of = {r['item_id']: str(r.get('coco_image_id') or r['image_id']) for r in csv.DictReader(open(a.list))}
    tags = a.models; L = {(f, t): load(d, a.set, t) for f, d in enumerate(a.fits) for t in tags}
    ids = {t: L[0, t][0] for t in tags}
    for (f, t), v in L.items():
        assert v[0] == ids[t], f'item ids of {a.fits[f]} differ from the stored fit ({t})'
    imgs = sorted(set().union(*[{image_of[i] for i in ids[t]} for t in tags])); pos = {i: j for j, i in enumerate(imgs)}; rng = np.random.default_rng(0)
    CNT = np.stack([np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)) for _ in range(a.draws)])
    C, TOT, errs, n = {}, {}, {}, {}
    for t in tags:
        iou = L[0, t][1]; wo, ot = (iou < .1).astype(np.float64), ((iou >= .1) & (iou < .5)).astype(np.float64)
        W0 = np.ones((1, len(iou))); W = np.vstack([W0, CNT[:, np.array([pos[image_of[i]] for i in ids[t]])]])
        TOT[t] = np.stack([(W * wo).sum(1), (W * ot).sum(1)]); errs[t] = (int(TOT[t][0, 0]), int(TOT[t][1, 0])); n[t] = len(iou)
        for f in range(len(a.fits)):
            C.setdefault(f, {})[t] = np.stack([caught(x, wo, ot, W if f == 0 else W0, a.share) for x in L[f, t][2]])
    refits = list(range(1, len(a.fits)))
    print(f"{a.set}: {len(tags)} models, {len(imgs)} images, {len(refits)} refits, {a.draws} image resamples; flagged share {a.share:g}")
    print("  answers / wrong-object errors / other errors per model: " + '; '.join(f"{n[t]} / {errs[t][0]} / {errs[t][1]}" for t in tags))

    def recall(c):
        fn = lambda cw, co, k, N, Wt, Ot: (c * cw + co) / np.maximum(c * Wt + Ot, 1e-9)
        st = np.mean([fn(*C[0][t][:, :, 0].T, TOT[t][0, 0], TOT[t][1, 0]) for t in tags], 0)
        B0 = np.mean([fn(*[C[0][t][:, j, 1:] for j in range(4)], TOT[t][0, 1:], TOT[t][1, 1:]) for t in tags], 0)
        pts = np.array([np.mean([fn(*C[f][t][:, :, 0].T, TOT[t][0, 0], TOT[t][1, 0]) for t in tags], 0) for f in refits]) if refits else np.zeros((0, len(SCORES)))
        return st, B0, pts
    first = None
    for c in sorted(set(a.costs) | set(a.grid)):
        st, B0, pts = recall(c); con = {name: combined(st[i] - st[j], B0[i] - B0[j], pts[:, i] - pts[:, j]) for name, i, j in CONTRASTS}
        if first is None and c in a.grid and con['SPRED minus the probe'][1] > 0:
            first = c
        if c in a.costs:
            print(f"  c = {c:g}: cost-weighted recall " + ', '.join(f"{k} {v:.3f}" for k, v in zip(SCORES, st)) + '; '
                  + '; '.join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in con.items()))
    fine = np.round(np.arange(.1, 10.001, .01), 2); d = np.array([np.subtract(*recall(c)[0][:2]) for c in fine])
    be = '< 0.1' if (d > 0).all() else '> 10' if (d <= 0).all() else f"{fine[np.where(np.diff(np.sign(d)) != 0)[0][0] + 1]:.2f}"
    print(f"  SPRED minus the probe: break-even cost ratio {be}; first c of the grid with the interval above zero: {first}")


if __name__ == '__main__':
    main()
