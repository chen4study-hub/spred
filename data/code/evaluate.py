#!/usr/bin/env python3
"""Recompute the reported numbers from the per-item scores of this package.

Python 3 with numpy; no other dependency, no model, no hidden states. Run from the directory of the package
(the one that holds lists/, scores/, refits/ and standardization.csv) or pass it with --root.

  python code/evaluate.py counts  --set replication
  python code/evaluate.py table4  --set replication            per-model AUROC rows and the bound against the supervised probe
  python code/evaluate.py gain    --set replication --errors wrong_object     gain of every score when the residual is added
  python code/evaluate.py review  --set replication --share 0.2               errors caught when a share of the answers is reviewed
  python code/evaluate.py types   --set replication            among wrong answers: wrong-object errors against the other errors;
                                                          the residual added to the classifier on box geometry
  python code/evaluate.py pull    --set replication            where the read-out lies on wrong-object errors
  python code/evaluate.py pairs   [--set replication]          new items unless --set is given: which of two models erred (any error;
                                                          wrong-object errors only)
  python code/evaluate.py others  --set replication            own residual against the residuals of the other models (both lists are printed)
  python code/evaluate.py views                           new items: the detector against the disagreement over seven views
  python code/evaluate.py points                          point answers on the replication and held-out lists: gain when the residual is added
  python code/evaluate.py check   [--expected code/expected.csv]         all of the above against the numbers of the paper

Item sets: replication, heldout, new-items. Error types: all, wrong_object, partial_overlap, wrong_extent.
Conventions (as in the paper). An answer is an error when IoU < 0.5; an error type is scored against the correct
answers. z(score) = (value - training mean) / training sd with the constants of standardization.csv; a missing
value is first set to the largest value of that score in the file. Adding the residual to a score b means
z(b) + 0.5 z(residual). Means over models resample the images of all models together; the combined interval of a
gain is the stored gain +- 1.96 sqrt(SE^2 + SD^2), SE from the images and SD from the ten refits in
refits/plugin_gain_refits.csv (for the classifier on box geometry: refits/geometry_classifier_gain_refits.csv).
"""
import argparse, csv, itertools, os, sys
import numpy as np

MODELS = ['Qwen2.5-VL-3B', 'Qwen2.5-VL-7B', 'Qwen3-VL-4B', 'Qwen3-VL-8B', 'InternVL2.5-8B-MPO', 'LLaVA-1.5-7B']
BASES = ['two_teacher_probe', 'sampling_teacher_probe', 'shift_teacher_probe', 'linear_probe_correctness', 'linear_probe_one_minus_iou', 'pregeneration_probe', 'saplma', 'linear_probe_wrong_object_all',
         'linear_probe_wrong_object_vs_correct', 'coordinate_logprob', 'in_box_attention', 'whole_image_attention', 'mahalanobis', 'box_geometry_supervised', 'box_geometry_teacher']
LISTS = {'replication': ['replication'], 'heldout': ['heldout'], 'new-items': ['new-items_refcoco-testB', 'new-items_refcocog-val']}
TYPES = {'all': None, 'wrong_object': 'wrong_object', 'partial_overlap': 'partial_overlap', 'wrong_extent': 'wrong_extent'}
RES = 'self_prediction_residual'


def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(rows, col):
    return np.array([float(r[col]) if r.get(col, '') != '' else np.nan for r in rows])


class Package:
    def __init__(self, root):
        self.root = root; self.std = {(r['model'], r['score']): (float(r['training_mean']), float(r['training_sd'])) for r in read(os.path.join(root, 'standardization.csv'))}; self._lists, self._scores = {}, {}

    def items(self, name):
        """item_id -> row of the item list(s) of an item set"""
        if name not in self._lists:
            self._lists[name] = {r['item_id']: r for f in LISTS[name] for r in read(os.path.join(self.root, 'lists', f + '.csv'))}
        return self._lists[name]

    def scores(self, name, model):
        """per-item rows of one model on one item set, or None when the file is absent"""
        key = (name, model)
        if key not in self._scores:
            p = os.path.join(self.root, 'scores', f'{name}_{model}.csv'); self._scores[key] = read(p) if os.path.exists(p) else None
        return self._scores[key]

    def models(self, name):
        return [m for m in MODELS if self.scores(name, m) is not None]

    def z(self, name, model, col):
        """standardized score, or None when the model has no such score"""
        rows = self.scores(name, model)
        if col not in rows[0] or (model, col) not in self.std: return None
        v = num(rows, col)
        if np.isnan(v).all() or np.nanstd(v) == 0: return None
        mu, sd = self.std[model, col]; return (np.where(np.isnan(v), np.nanmax(v), v) - mu) / (sd + 1e-12)

    def labels(self, name, model, errors):
        """positives (errors of the type) and negatives (correct answers) as 0/1 arrays"""
        rows = self.scores(name, model); et = np.array([r['error_type'] for r in rows]); neg = (et == 'correct').astype(float)
        return ((et != 'correct') if errors == 'all' else (et == TYPES[errors])).astype(float), neg

    def images(self, name, model):
        L = self.items(name); return [L[r['item_id']]['coco_image_id'] for r in self.scores(name, model)]


def wauc(s, pos, neg, W):
    """AUROC of s (positives against negatives; ties count one half) for every row of item weights W"""
    o = np.argsort(s, kind='stable'); st = np.r_[0, np.where(np.diff(s[o]) != 0)[0] + 1]
    gp = np.add.reduceat(W[:, o] * pos[o], st, axis=1); gn = np.add.reduceat(W[:, o] * neg[o], st, axis=1); den = gp.sum(1) * gn.sum(1)
    return np.where(den > 0, (gp * (np.cumsum(gn, axis=1) - gn + .5 * gn)).sum(1) / np.maximum(den, 1e-12), np.nan)


def auc(s, pos, neg):
    return float(wauc(s, pos, neg, np.ones((1, len(s))))[0])


def joint_counts(pkg, name, models, draws, seed=0):
    """image resampling shared by the models: (draws, images) counts and, per model, the image index of every item"""
    per = {m: pkg.images(name, m) for m in models}; imgs = sorted(set().union(*[set(v) for v in per.values()])); pos = {i: j for j, i in enumerate(imgs)}; rng = np.random.default_rng(seed)
    cnt = np.stack([np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)) for _ in range(draws)]).astype(np.float32)
    return cnt, {m: np.array([pos[i] for i in v]) for m, v in per.items()}, len(imgs)


def cmd_counts(pkg, a):
    out = {}
    for m in pkg.models(a.set):
        rows = pkg.scores(a.set, m); et = [r['error_type'] for r in rows]; n, e, w = len(rows), sum(x != 'correct' for x in et), et.count('wrong_object'); out[m] = (n, e, w)
        print(f"{m}: items {n}, errors {e}, wrong-object errors {w} ({100 * w / e:.1f}% of the errors), images {len(set(pkg.images(a.set, m)))}")
    L = pkg.items(a.set); print(f"list: {len(L)} items on {len({r['coco_image_id'] for r in L.values()})} images"); return out


def cmd_table4(pkg, a):
    """per model: AUROC of the detector, its parts, the coordinate log-probability and the supervised linear probe; one-sided bound of detector minus supervised"""
    out = {}
    for m in pkg.models(a.set):
        rows = pkg.scores(a.set, m); raw = lambda c: (lambda v: np.where(np.isnan(v), np.nanmax(v), v))(num(rows, c)); det = pkg.z(a.set, m, 'two_teacher_probe') + .5 * pkg.z(a.set, m, RES)
        S = {'detector': det, 'probe alone': raw('two_teacher_probe'), 'residual alone': raw(RES), 'coordinate log-prob': raw('coordinate_logprob'), 'supervised probe': raw('linear_probe_correctness')}; r = {}
        for errors in ('all', 'wrong_object'):
            pos, neg = pkg.labels(a.set, m, errors)
            for k, s in S.items(): r[errors, k] = auc(s, pos, neg)
        pos, neg = pkg.labels(a.set, m, 'all'); img = np.array(pkg.images(a.set, m)); uniq, inv = np.unique(img, return_inverse=True); rng = np.random.default_rng(0)      # image ids ordered as strings
        W = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(a.boot)])[:, inv].astype(np.float64)
        d = wauc(S['detector'], pos, neg, W) - wauc(S['supervised probe'], pos, neg, W); r['bound'] = float(np.nanpercentile(d, 5)); out[m] = r
        print(f"{m}: all errors " + ', '.join(f"{k} {r['all', k]:.3f}" for k in S) + f"; detector minus supervised {r['all', 'detector'] - r['all', 'supervised probe']:+.3f}, one-sided 95% lower bound {r['bound']:+.3f}; wrong object: detector {r['wrong_object', 'detector']:.3f}, probe alone {r['wrong_object', 'probe alone']:.3f}, residual alone {r['wrong_object', 'residual alone']:.3f}, "
              f"detector minus supervised {r['wrong_object', 'detector'] - r['wrong_object', 'supervised probe']:+.3f}")
    return out


def refit_gains(pkg):
    out = {}
    for r in read(os.path.join(pkg.root, 'refits', 'plugin_gain_refits.csv')):
        out[r['item_set'], r['error_type'], r['score'], r['model']] = (float(r['stored_gain']), np.array([float(r[f'refit_{k}']) for k in range(1, 11)]))
    return out


def cmd_gain(pkg, a):
    """gain of every score when the residual is added: per model, mean over the models, combined interval"""
    models = pkg.models(a.set); cnt, idx, nimg = joint_counts(pkg, a.set, models, a.draws); RF = refit_gains(pkg); et = 'all_errors' if a.errors == 'all' else a.errors; out = {}
    print(f"# {a.set}, {a.errors}: gain in AUROC when 0.5 z(residual) is added; {len(models)} models, {nimg} images, {a.draws} image resamples")
    for b in BASES:
        st, B, ms = [], [], []
        for m in models:
            zb, zr = pkg.z(a.set, m, b), pkg.z(a.set, m, RES)
            if zb is None: continue
            pos, neg = pkg.labels(a.set, m, a.errors); W = cnt[:, idx[m]]; W0 = np.ones((1, len(zb)), np.float32)
            st.append(float(wauc(zb + .5 * zr, pos, neg, W0)[0] - wauc(zb, pos, neg, W0)[0])); B.append(wauc(zb + .5 * zr, pos, neg, W) - wauc(zb, pos, neg, W)); ms.append(m)
        mean = float(np.mean(st)); se = float(np.std(np.mean(B, 0))); key = (a.set, et, b, 'mean'); r = {'models': ms, 'per_model': st, 'mean': mean, 'image_se': se}
        line = f"  {b}: per model " + ' / '.join(f"{v:+.3f}" for v in st) + f"; mean {mean:+.3f}"
        if key in RF:
            sd = float(np.std(RF[key][1], ddof=1)); tot = float(np.hypot(se, sd)); r.update({'refit_sd': sd, 'combined': [mean - 1.96 * tot, mean + 1.96 * tot], 'stored_in_refits_file': RF[key][0]}); line += f", combined interval [{r['combined'][0]:+.3f}, {r['combined'][1]:+.3f}] (image SE {se:.4f}, refit SD {sd:.4f})"
        out[b] = r; print(line)
    return out


def caught(s, target, W, share):
    """the share of the answers with the highest score is flagged; returns the share of the target errors that are flagged and the number of flagged answers per flagged target error"""
    o = np.argsort(s, kind='stable')[::-1]; w = W[:, o].astype(np.float64); n = w.sum(1); k = n - np.floor((1 - share) * n); f = np.clip(k[:, None] - (np.cumsum(w, 1) - w), 0, w)
    return (f * target[o]).sum(1) / np.maximum((w * target[o]).sum(1), 1e-12), k / np.maximum((f * target[o]).sum(1), 1e-12)


def cmd_review(pkg, a):
    """wrong-object errors caught when a share of the answers is reviewed: four scores, and the mean over the fifteen scores with and without the residual;
    for the detector and its probe alone also the share of all errors caught and, with a wrong-object error costing c and any other
    error 1, the cost-weighted recall (cost of the errors caught over the cost of all errors): detector minus probe at c = 1, and the
    break-even c, where that difference changes sign on a grid from 0.1 to 10 in steps of 0.01 (means over the models)"""
    models = pkg.models(a.set); R = {k: [] for k in ('detector', 'probe alone', 'coordinate log-prob', 'supervised probe')}; P = {k: [] for k in R}; base, withres = [], []
    RA = {'detector': [], 'probe alone': []}; CW = {'detector': [], 'probe alone': []}; NW = []
    for m in models:
        rows = pkg.scores(a.set, m); order = np.argsort([r['item_id'] for r in rows], kind='stable'); wo = pkg.labels(a.set, m, 'wrong_object')[0][order]; W0 = np.ones((1, len(rows))); z = lambda c: pkg.z(a.set, m, c)
        S = {'detector': z('two_teacher_probe') + .5 * z(RES), 'probe alone': z('two_teacher_probe'), 'coordinate log-prob': z('coordinate_logprob'), 'supervised probe': z('linear_probe_correctness')}
        for k, s in S.items(): rec, per = caught(s[order], wo, W0, a.share); R[k].append(float(rec[0])); P[k].append(float(per[0]))
        al = pkg.labels(a.set, m, 'all')[0][order].astype(bool); ot = (al & ~wo.astype(bool)).astype(np.float64)
        for k in ('detector', 'probe alone'):
            s = S[k][order]; RA[k].append(float(caught(s, al, W0, a.share)[0][0])); CW[k].append((float(caught(s, wo, W0, a.share)[0][0]), float(caught(s, ot, W0, a.share)[0][0])))
        NW.append((float(np.sum(wo)), float(np.sum(ot))))
        for b in BASES:
            if z(b) is None: continue
            base.append(float(caught(z(b)[order], wo, W0, a.share)[0][0])); withres.append(float(caught((z(b) + .5 * z(RES))[order], wo, W0, a.share)[0][0]))
    out = {'recall': {k: float(np.mean(v)) for k, v in R.items()}, 'reviewed_per_caught': {k: float(np.mean(v)) for k, v in P.items()}, 'fifteen_scores': [float(np.mean(base)), float(np.mean(withres))], 'cells': len(base)}
    cwr = lambda c, k: float(np.mean([(c * w * n[0] + o * n[1]) / max(c * n[0] + n[1], 1e-9) for (w, o), n in zip(CW[k], NW)]))
    fine = np.round(np.arange(.1, 10.001, .01), 2); d = np.array([cwr(c, 'detector') - cwr(c, 'probe alone') for c in fine]); sign = np.sign(d)
    be = float(fine[np.where(np.diff(sign) != 0)[0][0] + 1]) if (d > 0).any() and (d <= 0).any() else float('nan')
    out['all_errors'] = {k: float(np.mean(v)) for k, v in RA.items()}; out['cost'] = {'c1_difference': cwr(1, 'detector') - cwr(1, 'probe alone'), 'break_even': be}
    print(f"# {a.set}, share reviewed {a.share:.0%}, wrong-object errors, mean over {len(models)} models")
    for k in R: print(f"  {k}: share caught {out['recall'][k]:.3f}; answers reviewed per caught error {out['reviewed_per_caught'][k]:.2f}")
    print(f"  mean over the fifteen scores and the models ({len(base)} cells): share caught {out['fifteen_scores'][0]:.3f} -> {out['fifteen_scores'][1]:.3f} with the residual added")
    print(f"  all errors: share caught, probe alone {out['all_errors']['probe alone']:.3f} -> detector {out['all_errors']['detector']:.3f}; "
          f"cost-weighted recall, detector minus probe at c = 1 {out['cost']['c1_difference']:+.3f}; break-even c {out['cost']['break_even']:.2f}"); return out


def cmd_types(pkg, a):
    """among wrong answers only: AUROC of wrong-object errors against the other errors; the residual added to the classifier on box geometry;
    the rows of the table of error types (mean over the models, 95% interval over images, the images of all models resampled together)"""
    out = {}; cols = [RES, 'coordinate_logprob', 'two_teacher_probe', 'linear_probe_correctness', 'pregeneration_probe']
    for c in cols:
        v = []
        for m in pkg.models(a.set):
            rows = pkg.scores(a.set, m); et = np.array([r['error_type'] for r in rows]); e = et != 'correct'; s = num(rows, c); s = np.where(np.isnan(s), np.nanmax(s), s)
            v.append(auc(s[e], (et[e] == 'wrong_object').astype(float), (et[e] != 'wrong_object').astype(float)))
        out[c] = (float(np.mean(v)), v); print(f"  {c}: per model " + ' / '.join(f"{x:.3f}" for x in v) + f"; mean {np.mean(v):.3f}")
    models = pkg.models(a.set); G, GR = 'box_geometry_classifier', 'box_geometry_plus_residual_classifier'
    # the residual added to the classifier on box geometry
    cnt, idx, _ = joint_counts(pkg, a.set, models, a.boot); st, B, base = [], [], []
    for m in models:
        rows = pkg.scores(a.set, m); et = np.array([r['error_type'] for r in rows]); pos, neg = (et == 'wrong_object').astype(float), ((et != 'wrong_object') & (et != 'correct')).astype(float); g, gr = num(rows, G), num(rows, GR); W = cnt[:, idx[m]]
        base.append(auc(g, pos, neg)); st.append(auc(gr, pos, neg) - base[-1]); B.append(wauc(gr, pos, neg, W) - wauc(g, pos, neg, W))
    mean, se = float(np.mean(st)), float(np.nanstd(np.mean(B, 0))); r = {'geometry': float(np.mean(base)), 'per_model': st, 'mean': mean, 'image_se': se}; line = f"  residual added to the classifier on box geometry ({np.mean(base):.3f}): gain per model " + ' / '.join(f"{v:+.3f}" for v in st) + f"; mean {mean:+.3f}"
    f = os.path.join(pkg.root, 'refits', 'geometry_classifier_gain_refits.csv'); ref = [x for x in read(f) if x['item_set'] == a.set and x['model'] == 'mean']
    sd = float(np.std([float(ref[0][f'refit_{k}']) for k in range(1, 11)], ddof=1)); tot = float(np.hypot(se, sd)); r['combined'] = [mean - 1.96 * tot, mean + 1.96 * tot]; line += f", combined interval [{r['combined'][0]:+.3f}, {r['combined'][1]:+.3f}] (image SE {se:.4f}, refit SD {sd:.4f})"
    out['geometry_gain'] = r; print(line)
    # rows of the table of error types: AUROC among wrong answers, mean over the models with its interval over images
    T = {'residual': RES, 'detector': None, 'two_teacher_probe': 'two_teacher_probe', 'shift_teacher_probe': 'shift_teacher_probe', 'linear_probe_correctness': 'linear_probe_correctness',
         'pregeneration_probe': 'pregeneration_probe', 'linear_probe_wrong_object_all': 'linear_probe_wrong_object_all', 'coordinate_logprob': 'coordinate_logprob', 'minus_box_area': None,
         'box_geometry_classifier': G, 'box_geometry_plus_residual_classifier': GR}
    L = pkg.items(a.set); PTS = {k: [] for k in T}; BS = {k: [] for k in T}
    for m in models:
        rows = pkg.scores(a.set, m); et = np.array([r['error_type'] for r in rows]); pos, neg = (et == 'wrong_object').astype(float), ((et != 'wrong_object') & (et != 'correct')).astype(float); W = cnt[:, idx[m]]; W1 = np.ones((1, len(rows)))
        size = np.array([[float(L[r['item_id']]['image_width']), float(L[r['item_id']]['image_height'])] for r in rows])
        bw = np.clip((num(rows, 'written_x2') - num(rows, 'written_x1')) / size[:, 0], 1e-4, None); bh = np.clip((num(rows, 'written_y2') - num(rows, 'written_y1')) / size[:, 1], 1e-4, None)
        for k, c in T.items():
            if k == 'detector': s = pkg.z(a.set, m, 'two_teacher_probe') + .5 * pkg.z(a.set, m, RES)
            elif k == 'minus_box_area': s = -(bw * bh)                                    # larger = smaller written box (relative area)
            else: s = num(rows, c); s = np.where(np.isnan(s), np.nanmax(s), s)
            PTS[k].append(float(wauc(s, pos, neg, W1)[0])); BS[k].append(wauc(s, pos, neg, W))
    out['table'] = {k: [float(np.mean(PTS[k])), float(np.nanpercentile(np.mean(BS[k], 0), 2.5)), float(np.nanpercentile(np.mean(BS[k], 0), 97.5))] for k in T}
    print('  table of error types, mean over the models [95% interval over images]: ' + '; '.join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in out['table'].items()))
    return out


def cmd_views(pkg, a):
    """new items, Qwen models: AUROC of the detector (one pass) minus that of the mean disagreement over seven further views, all errors"""
    out = {}
    for m in pkg.models('new-items'):
        rows = pkg.scores('new-items', m)
        if 'seven_view_disagreement' not in rows[0]: continue
        v = num(rows, 'seven_view_disagreement'); ok = ~np.isnan(v)
        if not ok.any(): continue
        pos, neg = pkg.labels('new-items', m, 'all'); det = pkg.z('new-items', m, 'two_teacher_probe') + .5 * pkg.z('new-items', m, RES); d, s7 = auc(det[ok], pos[ok], neg[ok]), auc(v[ok], pos[ok], neg[ok]); out[m] = d - s7
        print(f"  {m}: {int(ok.sum())} items; detector {d:.3f}, seven views {s7:.3f}, detector minus seven views {d - s7:+.3f}")
    return out


def cmd_pull(pkg, a):
    """wrong-object errors, all models pooled: least squares (no intercept) of the offset of the read-out centre from the written centre on the offsets of the annotated centre and of the mean written box of the training items; relative image coordinates"""
    L = pkg.items(a.set); T = {r['item_id']: r for r in read(os.path.join(pkg.root, 'lists', 'training.csv'))}; Y, G, Mn = [], [], []
    ctr = lambda rows, pre, size: np.c_[(num(rows, pre + '_x1') + num(rows, pre + '_x2')) / 2 / size[:, 0], (num(rows, pre + '_y1') + num(rows, pre + '_y2')) / 2 / size[:, 1]]
    for m in pkg.models(a.set):
        rows = pkg.scores(a.set, m)
        size = np.array([[float(L[r['item_id']]['image_width']), float(L[r['item_id']]['image_height'])] for r in rows]); lst = [L[r['item_id']] for r in rows]
        tr = read(os.path.join(pkg.root, 'answers', f'training_{m}.csv')); ts = np.array([[float(T[r['item_id']]['image_width']), float(T[r['item_id']]['image_height'])] for r in tr]); mean_c = ctr(tr, 'written', ts).mean(0)
        p = ctr(rows, 'written', size); wo = np.array([r['error_type'] == 'wrong_object' for r in rows]); Y.append((ctr(rows, 'readout', size) - p)[wo]); G.append((ctr(lst, 'box', size) - p)[wo]); Mn.append((mean_c - p)[wo])
    Y, G, Mn = map(np.concatenate, (Y, G, Mn)); A = np.array([[(G * G).sum(), (G * Mn).sum()], [(G * Mn).sum(), (Mn * Mn).sum()]]); b = np.array([(G * Y).sum(), (Mn * Y).sum()]); c = np.linalg.solve(A, b)
    print(f"# {a.set}: {len(Y)} wrong-object errors; coefficient of the annotated object {c[0]:+.3f}, of the mean written box {c[1]:+.3f}"); return {'towards_annotated': float(c[0]), 'towards_mean_box': float(c[1]), 'errors': int(len(Y))}


def cmd_pairs(pkg, a):
    """items of an item set (new items unless --set is given) answered by the five models with shared items: among pairs of models where exactly one erred, how often the score ranks the erring model higher (rank within each model); for errors of any type and for wrong-object errors only.
    Intervals: on the new items combined with the ten refits in refits/pair_test_refits.csv, on the other item sets over images only"""
    models = [m for m in pkg.models(a.set) if m != 'LLaVA-1.5-7B']; rows = {m: {r['item_id']: r for r in pkg.scores(a.set, m)} for m in models}; L = pkg.items(a.set); last = {}
    for i in sorted(set.intersection(*[set(v) for v in rows.values()])): last[L[i]['dataset'], L[i]['split'], L[i]['coco_annotation_id']] = i      # an item listed twice (same annotation, same expression) is counted once
    ids = sorted(last.values()); out = {}; n = len(ids)
    img = np.array([pkg.items(a.set)[i]['coco_image_id'] for i in ids]); ui, inv = np.unique(img, return_inverse=True); rng = np.random.default_rng(0); Wt = np.stack([np.bincount(rng.integers(0, len(ui), len(ui)), minlength=len(ui))[inv] for _ in range(a.boot)]).astype(float)
    f = os.path.join(pkg.root, 'refits', 'pair_test_refits.csv'); RF = {(r['errors_counted'], r['score']): [float(r[f'refit_{k}']) for k in range(1, 11)] for r in read(f)}
    iou = {m: np.array([float(rows[m][i]['iou']) for i in ids]) for m in models}; rk = {}
    for col, name in ((RES, 'residual'), ('detector', 'detector')):
        for m in models:
            full = pkg.scores(a.set, m); v = num(full, col); s = pkg.z(a.set, m, 'two_teacher_probe') + .5 * pkg.z(a.set, m, RES) if col == 'detector' else np.where(np.isnan(v), np.nanmax(v), v)
            pos = {r['item_id']: k for k, r in enumerate(full)}; s = s[[pos[i] for i in ids]]; o = np.argsort(s, kind='stable'); r_ = np.empty(n); r_[o] = np.arange(n, dtype=float)
            for u in np.unique(s):                                             # mid-ranks for ties
                t = s == u
                if t.sum() > 1: r_[t] = r_[t].mean()
            rk[m, name] = r_
    print(f"# {a.set}: {n} items answered by all {len(models)} models, {len(ui)} images, {a.boot} image resamples")
    for label, lim in (('any_error', .5), ('wrong_object', .1)):
        for name in ('residual', 'detector'):
            w = np.zeros(n); c = np.zeros(n)
            for x, y in itertools.combinations(models, 2):
                ex, ey = iou[x] < .5, iou[y] < .5; d = (ex != ey) & np.where(ex, iou[x] < lim, iou[y] < lim); hi = np.where(ex, rk[x, name] - rk[y, name], rk[y, name] - rk[x, name]); w += d * ((hi > 0) + .5 * (hi == 0)); c += d
            st = float(w.sum() / c.sum()); se = float(np.std((Wt @ w) / (Wt @ c))); r = {'rate': st, 'pairs': int(c.sum())}; line = f"  {label}, {name}: picks the model that erred in {st:.3f} of {int(c.sum())} discordant item pairs"
            key = (label, RES if name == 'residual' else 'detector')
            if a.set == 'new-items' and key in RF: tot = float(np.hypot(se, np.std(RF[key], ddof=1))); r['combined'] = [st - 1.96 * tot, st + 1.96 * tot]; line += f", combined interval [{r['combined'][0]:.3f}, {r['combined'][1]:.3f}]"
            else: r['image'] = [st - 1.96 * se, st + 1.96 * se]; line += f", interval over images [{r['image'][0]:.3f}, {r['image'][1]:.3f}]"
            out[label, name] = r; print(line)
    return out


def cmd_others(pkg, a):
    """the term added to a score is the model's own residual, the mean residual of the other models on the same item, or one other model's residual: gains on wrong-object errors, replication and held-out list"""
    const = {r['model']: (float(r['training_mean']), float(r['training_sd'])) for r in read(os.path.join(pkg.root, 'standardization_others_residual.csv'))}
    RF = {(r['item_set'], r['score'], r['term']): np.array([float(r[f'refit_{k}']) for k in range(1, 11)]) for r in read(os.path.join(pkg.root, 'refits', 'others_residual_gain_refits.csv'))}; out = {}
    for name in ('replication', 'heldout'):
        models = pkg.models(name); cnt, idx, nimg = joint_counts(pkg, name, models, a.draws); Z = {m: dict(zip([r['item_id'] for r in pkg.scores(name, m)], pkg.z(name, m, RES))) for m in models}; G, PM = {}, {}
        for m in models:
            rows = pkg.scores(name, m); ids = [r['item_id'] for r in rows]; pos, neg = pkg.labels(name, m, 'wrong_object'); W = cnt[:, idx[m]]; W0 = np.ones((1, len(rows)), np.float32); own = pkg.z(name, m, RES)
            te = np.array([np.mean([Z[u][i] for u in models if u != m and i in Z[u]]) if any(i in Z[u] for u in models if u != m) else np.nan for i in ids]); te = np.where(np.isnan(te), np.nanmean(te), te); oth = (te - const[m][0]) / (const[m][1] + 1e-12)
            for b in BASES:
                zb = pkg.z(name, m, b)
                if zb is None: continue
                f = lambda x, Wm: wauc(zb + .5 * x, pos, neg, Wm) - wauc(zb, pos, neg, Wm); G[b, m] = (np.array([float(f(own, W0)[0]), float(f(oth, W0)[0])]), np.stack([f(own, W), f(oth, W)]))
            zb = pkg.z(name, m, 'two_teacher_probe'); base = float(wauc(zb, pos, neg, W0)[0]); one = [float(wauc(zb + .5 * np.array([Z[u].get(i, 0.) for i in ids]), pos, neg, W0)[0]) - base for u in models if u != m]
            PM[m] = {'own': float(G['two_teacher_probe', m][0][0]), 'others': float(G['two_teacher_probe', m][0][1]), 'one': float(np.mean(one))}
        def agg(group):
            cells = [[G[b, m] for b in group if (b, m) in G] for m in models]; cells = [c for c in cells if c]; st = np.mean([np.mean([e[0] for e in c], 0) for c in cells], 0); B = np.mean([np.mean([e[1] for e in c], 0) for c in cells], 0)
            ratio = B[1] / np.where(np.abs(B[0]) < 1e-9, np.nan, B[0]); return {'own': float(st[0]), 'others': float(st[1]), 'ratio': float(st[1] / st[0]), 'ratio_interval': [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))], 'B': B}
        R = {b: agg([b]) for b in BASES}; R['seven probes'] = agg(BASES[:7]); R['fifteen scores'] = agg(BASES); n = {'own': 0, 'others': 0, 'own_minus_others': 0}
        for b in BASES:                                                       # combined intervals of the gains, per score
            r = R[b]; own_r, oth_r = RF[name, b, 'own'], RF[name, b, 'others']
            for k, st, B, pts in (('own', r['own'], r['B'][0], own_r), ('others', r['others'], r['B'][1], oth_r), ('own_minus_others', r['own'] - r['others'], r['B'][0] - r['B'][1], own_r - oth_r)): n[k] += st - 1.96 * float(np.hypot(np.std(pts, ddof=1), np.std(B))) > 0
        out[name] = {'rows': R, 'per_model': PM, 'counts': n, 'one_mean': float(np.mean([v['one'] for v in PM.values()]))}
        print(f"# {name}, wrong-object errors: gain with the own residual / with the mean residual of the other models / ratio; {len(models)} models, {nimg} images, {a.draws} image resamples")
        for k in ('two_teacher_probe', 'linear_probe_correctness', 'coordinate_logprob', 'seven probes', 'fifteen scores'): r = R[k]; print(f"  {k}: own {r['own']:+.3f}, others {r['others']:+.3f}, ratio {r['ratio']:.2f} [{r['ratio_interval'][0]:.2f}, {r['ratio_interval'][1]:.2f}]")
        print("  two-teacher probe, per model (own / others / one other model alone): " + '; '.join(f"{m} {v['own']:+.3f} / {v['others']:+.3f} / {v['one']:+.3f}" for m, v in PM.items()) + f"; mean of one other model alone {out[name]['one_mean']:+.3f}")
        print(f"  scores of 15 whose combined interval is above zero: own {n['own']}, others {n['others']}, own minus others {n['own_minus_others']}")
    return out


POINT_BASES = ['two_teacher_probe', 'sampling_teacher_probe', 'linear_probe_correctness', 'saplma', 'coordinate_logprob']
POINT_MODELS = ['Qwen2.5-VL-3B', 'Qwen2.5-VL-7B', 'Qwen3-VL-4B']


def cmd_points(pkg, a):
    """point answers on the replication and held-out lists (an answer is an error when the point lies outside the annotated box): gain in AUROC over all errors when 0.5 z(residual) is added to a score, for both lists together and for each list"""
    std = {(r['model'], r['score']): (float(r['training_mean']), float(r['training_sd'])) for r in read(os.path.join(pkg.root, 'standardization_points.csv'))}
    RF = {(r['item_set'], r['score'], r['model']): np.array([float(r[f'refit_{k}']) for k in range(1, 11)]) for r in read(os.path.join(pkg.root, 'refits', 'point_gain_refits.csv'))}; out = {}
    for name, lists in (('both', ('replication', 'heldout')), ('replication', ('replication',)), ('heldout', ('heldout',))):
        data = {}
        for m in POINT_MODELS:
            rows = [r for l in lists for r in read(os.path.join(pkg.root, 'scores', f'points_{l}_{m}.csv'))]; L = {}
            for l in lists: L.update(pkg.items(l))
            data[m] = (rows, [L[r['item_id']]['coco_image_id'] for r in rows])
        imgs = sorted(set().union(*[set(v[1]) for v in data.values()])); pos = {i: j for j, i in enumerate(imgs)}; rng = np.random.default_rng(0)
        cnt = np.stack([np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)) for _ in range(a.boot)]).astype(np.float32); out[name] = {}
        print(f"# point answers, {name}: {len(POINT_MODELS)} models, {len(imgs)} images; items / errors per model " + '; '.join(f"{len(data[m][0])} / {sum(r['correct'] == '0' for r in data[m][0])}" for m in POINT_MODELS))
        for b in POINT_BASES:
            st, B = [], []
            for m in POINT_MODELS:
                rows, im = data[m]; y = np.array([r['correct'] == '0' for r in rows], float); W = cnt[:, np.array([pos[i] for i in im])]; W0 = np.ones((1, len(rows)), np.float32)
                def z(c):
                    v = num(rows, c); mu, sd = std.get((m, c), (v.mean(), v.std())); return (v - mu) / (sd + 1e-12)      # a score without training values is standardized on the scored items
                zb, zr = z(b), z(RES); st.append(float(wauc(zb + .5 * zr, y, 1 - y, W0)[0] - wauc(zb, y, 1 - y, W0)[0])); B.append(wauc(zb + .5 * zr, y, 1 - y, W) - wauc(zb, y, 1 - y, W))
            mean, se = float(np.mean(st)), float(np.std(np.mean(B, 0))); sd = float(np.std(RF[name, b, 'mean'], ddof=1)); tot = float(np.hypot(se, sd)); out[name][b] = {'per_model': st, 'mean': mean, 'combined': [mean - 1.96 * tot, mean + 1.96 * tot]}
            print(f"  {b}: per model " + ' / '.join(f"{v:+.3f}" for v in st) + f"; mean {mean:+.3f}, combined interval [{mean - 1.96 * tot:+.3f}, {mean + 1.96 * tot:+.3f}] (image SE {se:.4f}, refit SD {sd:.4f})")
    return out


def cmd_check(pkg, a):
    """every number listed in the expected file: where it stands in the paper, the recomputed value, the value in the paper"""
    ns = lambda **k: argparse.Namespace(**{'set': 'replication', 'errors': 'wrong_object', 'share': .2, 'draws': a.draws, 'boot': a.boot, **k}); quiet = open(os.devnull, 'w'); keep = sys.stdout; sys.stdout = quiet
    try:
        C = cmd_counts(pkg, ns()); T4 = cmd_table4(pkg, ns()); GW = {s: cmd_gain(pkg, ns(set=s)) for s in ('replication', 'heldout')}; GA = cmd_gain(pkg, ns(errors='all')); RV = cmd_review(pkg, ns()); TY = cmd_types(pkg, ns()); PR = cmd_pairs(pkg, ns(set='new-items')); PRL = cmd_pairs(pkg, ns()); VW = cmd_views(pkg, ns()); PT = cmd_points(pkg, ns()); OT = cmd_others(pkg, ns())
        PL = cmd_pull(pkg, ns())
    finally:
        sys.stdout = keep
    def value(q):
        k = q.split('|')
        if k[0] == 'count': return {'items': C[k[1]][0], 'errors': C[k[1]][1], 'wo_share_pct': 100 * C[k[1]][2] / C[k[1]][1]}[k[2]]
        if k[0] == 'list_items': return len(pkg.items('replication'))
        if k[0] == 'table4': return T4[k[1]]['bound'] if k[2] == 'bound' else (T4[k[1]][k[2], 'detector'] - T4[k[1]][k[2], 'supervised probe'] if k[3] == 'detector minus supervised' else T4[k[1]][k[2], k[3]])
        if k[0] == 'gain': r = (GA if k[2] == 'all' else GW[k[1]])[k[3]]; return {'mean': r['mean'], 'lo': r['combined'][0], 'hi': r['combined'][1]}[k[4]] if k[4] in ('mean', 'lo', 'hi') else r['per_model'][r['models'].index(k[4])]
        if k[0] == 'review' and k[1] == 'all': return RV['all_errors'][k[2]]
        if k[0] == 'review' and k[1] == 'cost': return RV['cost'][k[2]]
        if k[0] == 'review': return RV['fifteen_scores'][int(k[1])] * 100 if k[1] in '01' else RV['recall'][k[1]]
        if k[0] == 'types': return TY[k[1]][0]
        if k[0] == 'types_table': return TY['table'][k[1]][{'auroc': 0, 'lo': 1, 'hi': 2}[k[2]]]
        if k[0] == 'geometry': r = TY['geometry_gain']; return {'classifier': r['geometry'], 'mean': r['mean'], 'lo': r['combined'][0], 'hi': r['combined'][1]}[k[1]]
        if k[0] == 'views': return VW[k[1]]
        if k[0] == 'points': r = PT[k[1]][k[2]]; return {'mean': r['mean'], 'lo': r['combined'][0], 'hi': r['combined'][1]}[k[3]]
        if k[0] == 'pairs': return PR['any_error', k[1]]['rate']
        if k[0] == 'pairs_list': r = PRL[k[1], k[2]]; return r['rate'] if k[3] == 'rate' else r['image'][0 if k[3] == 'lo' else 1]
        if k[0] == 'pairs_wo': r = PR['wrong_object', k[1]]; return r['rate'] if k[2] == 'rate' else (r['combined'][0 if k[2] == 'lo' else 1] if 'combined' in r else None)
        if k[0] == 'others':
            o = OT[k[1]]
            if k[2] == 'model': return o['per_model'][k[3]][k[4]]
            if k[2] == 'count': return o['counts'][k[3]]
            if k[2] == 'one_mean': return o['one_mean']
            r = o['rows'][k[2]]; return {'own': r['own'], 'others': r['others'], 'ratio': r['ratio'], 'ratio_lo': r['ratio_interval'][0], 'ratio_hi': r['ratio_interval'][1]}[k[3]]
        if k[0] == 'pull': return PL['towards_annotated']
        raise KeyError(q)
    bad = 0; print(f"{'where in the paper':34s} {'quantity':64s} {'recomputed':>10s} {'paper':>8s}")
    for r in read(a.expected):
        v = value(r['quantity']); d = int(r['decimals']); exp = float(r['paper'])
        ok = abs(round(v, d) - exp) < 0.5 * 10 ** -d + 1e-12; bad += not ok; print(f"{r['where']:34s} {r['quantity']:64s} {v:10.{d}f} {exp:8.{d}f}{'' if ok else '   <-- differs'}")
    print(f"{bad} of the listed numbers differ after rounding"); return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument('cmd', choices=('counts', 'table4', 'gain', 'review', 'types', 'pull', 'pairs', 'views', 'points', 'others', 'check')); ap.add_argument('--root', default='.')
    ap.add_argument('--set', default=None, choices=tuple(LISTS), help='item set (default: replication; for pairs: new-items)'); ap.add_argument('--errors', default='wrong_object', choices=tuple(TYPES)); ap.add_argument('--share', type=float, default=.2)
    ap.add_argument('--draws', type=int, default=2000, help='image resamples for the means over models'); ap.add_argument('--boot', type=int, default=10000, help='image resamples for the per-model bound'); ap.add_argument('--expected', default=None)
    a = ap.parse_args(); pkg = Package(a.root); a.set = a.set or ('new-items' if a.cmd == 'pairs' else 'replication')
    if a.cmd == 'check': a.expected = a.expected or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expected.csv'); sys.exit(1 if cmd_check(pkg, a) else 0)
    {'counts': cmd_counts, 'table4': cmd_table4, 'gain': cmd_gain, 'review': cmd_review, 'types': cmd_types, 'pull': cmd_pull, 'pairs': cmd_pairs, 'views': cmd_views, 'points': cmd_points, 'others': cmd_others}[a.cmd](pkg, a)


if __name__ == '__main__':
    main()
