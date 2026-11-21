#!/usr/bin/env python3
"""Numbers of the main-text tables and figures that code/evaluate.py does not print (numpy only).

  python code/reproduce_main.py lists          Table 2: items per list
  python code/reproduce_main.py ablation       Table 5: probe alone, residual alone, SPRED and their differences
  python code/reproduce_main.py table4         Table 4 rows not printed by evaluate.py: BICR, and per model the share of
                                               wrong-object errors caught in the top-scoring fifth and answers read per catch;
                                               the BICR rows of Table S9.2 (held-out list)
  python code/reproduce_main.py second-answer  Figure 3: slope of the second answer on the read-out, per item set and model
  python code/reproduce_main.py pull           Figure 4: displacement of the read-out towards the annotated object and towards
                                               control boxes
  python code/reproduce_main.py margin         Section 5.2: models on which the probe alone and SPRED come within 0.03 of
                                               the supervised linear probe
  python code/reproduce_main.py narrow         Section 3.6: read-out error per model and fit; share of written boxes
                                               narrower than twice the read-out error, per item set and model
  python code/reproduce_main.py instances      Section 6.3 and Introduction: wrong-object errors on another annotated instance
  python code/reproduce_main.py readout-accuracy  Section 3.3: acc@0.5 of the read-out box and of the written box
  python code/reproduce_main.py domains        Section 7 (limitations), Section 3.6: SPRED on Objects365 and VRSBench; gain by target width
  python code/reproduce_main.py check          all of the above against results/expected_main.csv

Inputs: the item lists, per-item scores and standardization of this package, and results/ (boxes in relative image
coordinates, the mean written box of the training items, COCO instances of the images, the development sets of
Qwen2.5-VL-3B, the BICR scores on the replication and held-out lists, the training constants of BICR and the read-out
error of each fit). Resampling is fixed (seed 0); the order of the items and of the models is part of the
computation and follows the files.
"""
import argparse, csv, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE); EXTRA = os.path.join(DATA, 'results')
spec = importlib.util.spec_from_file_location('evaluate', os.path.join(DATA, 'code', 'evaluate.py')); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
MODELS = E.MODELS


def read(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fl(x):
    return np.nan if x == '' else float(x)


def auroc(s, y):
    """AUROC with average ranks for ties; y boolean (positives)."""
    s = np.asarray(s, float); y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    o = np.argsort(s, kind='mergesort'); ss = s[o]; r = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ci(d):
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def iou_rows(p, q):
    ix = np.clip(np.minimum(p[:, 2], q[:, 2]) - np.maximum(p[:, 0], q[:, 0]), 0, None); iy = np.clip(np.minimum(p[:, 3], q[:, 3]) - np.maximum(p[:, 1], q[:, 1]), 0, None)
    ar = lambda t: np.clip(t[:, 2] - t[:, 0], 0, None) * np.clip(t[:, 3] - t[:, 1], 0, None); u = ar(p) + ar(q) - ix * iy
    return np.where(u > 0, ix * iy / np.where(u > 0, u, 1), 0.)


def ctr(b):
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def lstsq(y, *xs):
    return np.linalg.lstsq(np.stack([x.ravel() for x in xs], 1), y.ravel(), rcond=None)[0]


STD = {(r['model'], r['score']): (float(r['training_mean']), float(r['training_sd'])) for r in read(os.path.join(DATA, 'standardization.csv'))}
MEAN_BOX = {r['model']: np.array([float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')]) for r in read(os.path.join(EXTRA, 'mean_training_box.csv'))}


def z(model, col, v):
    m, sd = STD[(model, col)]; return (v - m) / (sd + 1e-12)


def boxes(name):
    """results/boxes_<name>.csv -> {model: dict of arrays in file order}"""
    out = {}
    for r in read(os.path.join(EXTRA, f'boxes_{name}.csv')):
        out.setdefault(r['model'], []).append(r)
    res = {}
    for m, rows in out.items():
        B = lambda p: np.array([[fl(r[f'{p}_{c}']) for c in ('x1', 'y1', 'x2', 'y2')] for r in rows], float)
        res[m] = {'item_id': [r['item_id'] for r in rows], 'order': np.array([int(r['order']) for r in rows]), 'img': np.array([r['image_id'] for r in rows]),
                  'iou': np.array([float(r['iou']) for r in rows]), 'pred': B('written'), 'readout': B('readout'), 'gold': B('annotated'), 'second': B('shifted')}
    return res


def sorted_view(d):
    o = np.argsort(d['order'], kind='stable')
    return {k: (v[o] if isinstance(v, np.ndarray) else [v[i] for i in o]) for k, v in d.items()}


# ---------------------------------------------------------------- Table 2
def cmd_lists(a):
    L = lambda n: read(os.path.join(DATA, 'lists', f'{n}.csv'))
    out = {'training items': len(L('training')), 'replication items': len(L('replication')), 'replication images': len({r['coco_image_id'] for r in L('replication')}),
           'held-out items': len(L('heldout')), 'held-out images': len({r['coco_image_id'] for r in L('heldout')}),
           'new items, RefCOCO testB': len(L('new-items_refcoco-testB')), 'new items, RefCOCOg val': len(L('new-items_refcocog-val')),
           'RefCOCO+ items': len(L('refcocoplus_testA')) + len(L('refcocoplus_val'))}
    for dom, models in (('objects365', MODELS[:3]), ('vrsbench', [MODELS[0], MODELS[2]])):
        n = [len(read(os.path.join(DATA, 'scores', f'{dom}_fitted-on-RefCOCO_{m}.csv'))) for m in models]
        out[f'{dom} items scored, smallest'] = min(n); out[f'{dom} items scored, largest'] = max(n)
    for k, v in out.items():
        print(f'  {k}: {v}')
    return out


# ---------------------------------------------------------------- ablation
def ablation_cells(model, iou, S):
    allerr = iou < .5; keep = (iou < .1) | (iou >= .5); wo = (iou < .1)[keep]
    sc = {'probe': S['two_teacher_probe'], 'residual': S['self_prediction_residual'],
          'SPRED': z(model, 'two_teacher_probe', S['two_teacher_probe']) + .5 * z(model, 'self_prediction_residual', S['self_prediction_residual']),
          'shift': S['shift_teacher_probe'], 'sampling': S['sampling_teacher_probe']}
    A = {k: auroc(v, allerr) for k, v in sc.items()}; W = {k: auroc(v[keep], wo) for k, v in sc.items()}
    res = {}
    for lab, (x, y) in {'SPRED - probe only': ('SPRED', 'probe'), 'SPRED - residual only': ('SPRED', 'residual'),
                        'probe: two - shift teacher': ('probe', 'shift'), 'probe: two - sampling teacher': ('probe', 'sampling')}.items():
        res[lab] = (A[x] - A[y], W[x] - W[y])
    return {'probe only': (A['probe'], W['probe']), 'residual only': (A['residual'], W['residual']), 'SPRED': (A['SPRED'], W['SPRED']), **res}


def cmd_ablation(a):
    cols = ('two_teacher_probe', 'self_prediction_residual', 'shift_teacher_probe', 'sampling_teacher_probe')
    out = {}
    for model, label in ((MODELS[0], 'six sets'), (MODELS[1], 'new items'), (MODELS[2], 'new items')):
        rows = read(os.path.join(DATA, 'scores', f'new-items_{model}.csv'))
        if model == MODELS[0]:
            rows = rows + read(os.path.join(EXTRA, f'development_{model}.csv'))
        iou = np.array([float(r['iou']) for r in rows]); S = {c: np.array([float(r[c]) for r in rows]) for c in cols}
        out[model] = ablation_cells(model, iou, S)
        print(f'## {model}, {label}: AUROC all errors / wrong-object errors')
        for k, (x, y) in out[model].items():
            print(f'  {k:32s} {x:.3f}  {y:.3f}')
    return out


# ---------------------------------------------------------------- Table 4
def cmd_table4(a):
    pkg = E.Package(DATA); out = {'bicr': {}, 'top fifth': {}}
    bicr = {}
    for r in read(os.path.join(EXTRA, 'bicr_replication.csv')):
        bicr.setdefault(r['model'], {})[r['item_id']] = float(r['bicr'])
    print('# replication list: BICR (our reimplementation), AUROC all errors / wrong-object errors')
    for m in MODELS:
        rows = pkg.scores('replication', m); iou = np.array([float(r['iou']) for r in rows]); s = np.array([bicr[m][r['item_id']] for r in rows]); keep = (iou < .1) | (iou >= .5)
        out['bicr'][m] = (auroc(s, iou < .5), auroc(s[keep], (iou < .1)[keep])); print(f'  {m:20s} {out["bicr"][m][0]:.3f}  {out["bicr"][m][1]:.3f}')
    bstd = {r['model']: (float(r['training_mean']), float(r['training_sd'])) for r in read(os.path.join(EXTRA, 'bicr_standardization.csv'))}
    bh = {}
    for r in read(os.path.join(EXTRA, 'bicr_heldout.csv')):
        bh.setdefault(r['model'], {})[r['item_id']] = float(r['bicr'])
    print('# held-out list: BICR, AUROC all errors / wrong-object errors; detector minus BICR (all errors); BICR + residual minus BICR (wrong-object errors)')
    out['bicr heldout'] = {}
    for m in MODELS:
        rows = pkg.scores('heldout', m); iou = np.array([float(r['iou']) for r in rows]); s = np.array([bh[m][r['item_id']] for r in rows]); keep = (iou < .1) | (iou >= .5)
        zr = pkg.z('heldout', m, E.RES); det = pkg.z('heldout', m, 'two_teacher_probe') + .5 * zr; zb = (s - bstd[m][0]) / (bstd[m][1] + 1e-12)
        a_all, a_wo = auroc(s, iou < .5), auroc(s[keep], (iou < .1)[keep])
        rec = {'all errors': a_all, 'wrong object': a_wo, 'detector - bicr': auroc(det, iou < .5) - a_all,
               'bicr + residual - bicr, wrong object': auroc((zb + .5 * zr)[keep], (iou < .1)[keep]) - a_wo}
        out['bicr heldout'][m] = rec; print(f'  {m:20s} ' + '  '.join(f'{v:+.3f}' for v in rec.values()))
    print('# replication list, top-scoring fifth of the answers: share of the wrong-object errors caught; answers read per catch')
    rnd = []
    for m in MODELS:
        rows = pkg.scores('replication', m); order = np.argsort([r['item_id'] for r in rows], kind='stable'); wo = pkg.labels('replication', m, 'wrong_object')[0][order]; W0 = np.ones((1, len(rows)))
        res = {}
        for k, s in (('SPRED', pkg.z('replication', m, 'two_teacher_probe') + .5 * pkg.z('replication', m, E.RES)), ('probe alone', pkg.z('replication', m, 'two_teacher_probe'))):
            rec, per = E.caught(s[order], wo, W0, 0.2); res[k] = (float(rec[0]), float(per[0]))
        rnd.append(len(rows) / wo.sum())                    # random review: answers read per wrong-object error caught
        out['top fifth'][m] = res; print(f'  {m:20s} SPRED {res["SPRED"][0]:.3f}, {res["SPRED"][1]:.2f}; probe alone {res["probe alone"][0]:.3f}, {res["probe alone"][1]:.2f}')
    for k in ('SPRED', 'probe alone'):
        out[f'mean {k}'] = (float(np.mean([out['top fifth'][m][k][0] for m in MODELS])), float(np.mean([out['top fifth'][m][k][1] for m in MODELS])))
    out['mean random'] = (0.2, float(np.mean(rnd)))
    print('  means over the models: ' + '; '.join(f'{k[5:]} {v[0]:.3f} and {v[1]:.2f}' for k, v in out.items() if k.startswith('mean')))
    return out


# ---------------------------------------------------------------- Figure 3
def second_answer_cell(d, mean_box, rng, draws=1000, perms=200):
    first, ro, gold, second, img = d['pred'], d['readout'], d['gold'], d['second'], d['img']
    have = ~np.isnan(first[:, 0]) & ~np.isnan(ro[:, 0]) & ~np.isnan(second[:, 0]) & ~np.isnan(gold[:, 0])
    ch = np.where(have & (iou_rows(np.nan_to_num(first), np.nan_to_num(second)) < .5))[0]
    if len(ch) < 30:
        return None
    p, q, r, g = ctr(first[ch]), ctr(second[ch]), ctr(ro[ch]), ctr(gold[ch]); m = np.repeat(ctr(mean_box[None]), len(ch), 0)
    al = lstsq(q - p, r - p, m - p); ui, inv = np.unique(img[ch], return_inverse=True); mem = [np.where(inv == j)[0] for j in range(len(ui))]
    dr = [np.concatenate([mem[j] for j in rng.integers(0, len(ui), len(ui))]) for _ in range(draws)]
    B = np.array([lstsq((q - p)[x], (r - p)[x], (m - p)[x]) for x in dr]); pm = np.array([lstsq(q - p, (r - p)[rng.permutation(len(ch))], m - p)[0] for _ in range(perms)])
    rec = {'changed': int(len(ch)), 'slope': [float(al[0]), *ci(B[:, 0])], 'permuted': [float(pm.mean()), *ci(pm)]}
    a3 = lstsq(q - p, r - p, m - p, g - p); B3 = np.array([lstsq((q - p)[x], (r - p)[x], (m - p)[x], (g - p)[x]) for x in dr])
    rec['with the annotated box'] = [float(a3[0]), *ci(B3[:, 0])]
    wrong2 = iou_rows(second[ch], gold[ch]) < .5
    if wrong2.sum() >= 30:      # items whose second answer is also wrong
        w = np.where(wrong2)[0]; aw = lstsq((q - p)[w], (r - p)[w], (m - p)[w]); uw, iw = np.unique(img[ch][w], return_inverse=True); mw = [np.where(iw == j)[0] for j in range(len(uw))]
        Bw = np.array([lstsq((q - p)[w][x], (r - p)[w][x], (m - p)[w][x]) for x in (np.concatenate([mw[j] for j in rng.integers(0, len(uw), len(uw))]) for _ in range(draws))])
        pw = np.array([lstsq((q - p)[w], (r - p)[w][rng.permutation(len(w))], (m - p)[w])[0] for _ in range(perms)])
        rec['second answer also wrong'] = [float(aw[0]), *ci(Bw[:, 0]), float(pw.mean())]
    return rec


def cmd_second_answer(a):
    out = {}
    rng = np.random.default_rng(0)                              # one stream over the four sets and the models, in this order
    for name in ('new-items', 'refcocoplus', 'heldout', 'objects365'):
        bx = boxes(name)
        for m in [m for m in MODELS if m in bx]:
            rec = second_answer_cell(bx[m], MEAN_BOX[m], rng)
            if rec:
                out[f'{name}|{m}'] = rec
    bx = boxes('replication'); rng = np.random.default_rng(0)       # the replication list on its own stream
    for m in MODELS:
        rec = second_answer_cell(bx[m], MEAN_BOX[m], rng)
        if rec:
            out[f'replication|{m}'] = rec
    print('# items whose answer changes under the content shift: slope of the second box on the read-out [95% interval]; read-outs permuted: mean [central 95%]')
    for k, r in out.items():
        print(f'  {k:32s} n={r["changed"]:4d}  slope {r["slope"][0]:+.3f} [{r["slope"][1]:+.3f}, {r["slope"][2]:+.3f}]  permuted {r["permuted"][0]:+.3f} [{r["permuted"][1]:+.3f}, {r["permuted"][2]:+.3f}]')
    return out


# ---------------------------------------------------------------- Figure 4
def pull_set(bx, draws=2000, perms=200):
    rng = np.random.default_rng(0); out = {}
    def coef(r, p, g, m):
        return np.linalg.lstsq(np.stack([(g - p).ravel(), (m - p).ravel()], 1), (r - p).ravel(), rcond=None)[0]
    for mname in MODELS:
        d = sorted_view(bx[mname])
        k = np.where((d['iou'] < .1) & ~np.isnan(d['pred'][:, 0]) & ~np.isnan(d['readout'][:, 0]) & ~np.isnan(d['gold'][:, 0]))[0]
        r, p, g = ctr(d['readout'][k]), ctr(d['pred'][k]), ctr(d['gold'][k]); m = np.repeat(ctr(MEAN_BOX[mname][None]), len(k), 0)
        ab = coef(r, p, g, m); ui, inv = np.unique(d['img'][k], return_inverse=True); mem = [np.where(inv == j)[0] for j in range(len(ui))]; Bt = []
        for _ in range(draws):
            ix = np.concatenate([mem[j] for j in rng.integers(0, len(ui), len(ui))]); Bt.append(coef(r[ix], p[ix], g[ix], m[ix]))
        Bt = np.array(Bt); pm = np.array([coef(r, p, g[rng.permutation(len(k))], m) for _ in range(perms)])
        out[mname] = {'errors': int(len(k)), 'towards the annotated object': [float(ab[0]), *ci(Bt[:, 0])], 'towards the annotated box of another wrong-object item': [float(pm[:, 0].mean()), *ci(pm[:, 0])]}
    return out


def pull_same_image(sets, draws=2000):
    rows = []
    for name in sets:
        bx = boxes(name)
        for mname in MODELS:
            d = sorted_view(bx[mname]); byimg = {}
            for j, im in enumerate(d['img']):
                byimg.setdefault(im, []).append(j)
            for i in np.where((d['iou'] < .1) & ~np.isnan(d['pred'][:, 0]) & ~np.isnan(d['readout'][:, 0]) & ~np.isnan(d['gold'][:, 0]))[0]:
                cand = [j for j in byimg[d['img'][i]] if j != i and not np.isnan(d['gold'][j, 0]) and iou_rows(d['gold'][[j]], d['gold'][[i]])[0] < .1 and iou_rows(d['gold'][[j]], d['pred'][[i]])[0] < .1]
                if cand:
                    rows.append((f'{name}/{d["img"][i]}', ctr(d['readout'][[i]])[0], ctr(d['pred'][[i]])[0], ctr(d['gold'][[i]])[0], ctr(d['gold'][[cand[0]]])[0], ctr(MEAN_BOX[mname][None])[0]))
    img = np.array([x[0] for x in rows]); r, p, g, o, m = (np.array([x[k] for x in rows]) for k in range(1, 6)); rng = np.random.default_rng(0)
    def coef3(ix):
        X = np.stack([(g[ix] - p[ix]).ravel(), (o[ix] - p[ix]).ravel(), (m[ix] - p[ix]).ravel()], 1); return np.linalg.lstsq(X, (r[ix] - p[ix]).ravel(), rcond=None)[0]
    full = coef3(np.arange(len(rows))); ui, inv = np.unique(img, return_inverse=True); mem = [np.where(inv == j)[0] for j in range(len(ui))]
    Bt = np.array([coef3(np.concatenate([mem[j] for j in rng.integers(0, len(ui), len(ui))])) for _ in range(draws)])
    return {'errors': len(rows), 'towards the annotated object': [float(full[0]), *ci(Bt[:, 0])], 'towards another annotated object of the image': [float(full[1]), *ci(Bt[:, 1])]}


def iou1(p, q):
    if any(x != x for x in p) or any(x != x for x in q):
        return 0.
    ix = max(0., min(p[2], q[2]) - max(p[0], q[0])); iy = max(0., min(p[3], q[3]) - max(p[1], q[1])); u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.


def same_class(coco, image_id, gold, pred):
    """centre of the same-class object nearest to the written box that is neither the target instance nor the object the written box lies on"""
    im = coco.get(str(image_id))
    if im is None:
        return None
    Wd, Ht = float(im['width']), float(im['height'])
    inst = [(x, [x['bbox'][0] / Wd, x['bbox'][1] / Ht, (x['bbox'][0] + x['bbox'][2]) / Wd, (x['bbox'][1] + x['bbox'][3]) / Ht]) for x in im['anns']]
    if not inst:
        return None
    tg = max(inst, key=lambda x: iou1(gold, x[1]))
    if iou1(gold, tg[1]) < .5:
        return None
    same = [x for x in inst if x[0]['category_id'] == tg[0]['category_id'] and x[0]['id'] != tg[0]['id'] and not x[0].get('iscrowd')]
    free = [x for x in same if iou1(pred, x[1]) < .5]; pc = [(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2] if pred[0] == pred[0] else None
    c = min(free, key=lambda x: ((x[1][0] + x[1][2]) / 2 - pc[0]) ** 2 + ((x[1][1] + x[1][3]) / 2 - pc[1]) ** 2)[1] if (free and pc) else None
    return [(c[0] + c[2]) / 2, (c[1] + c[3]) / 2] if c else [np.nan, np.nan]


def wls(Y, X, W):
    """least squares without intercept, one fit per row of the weight matrix W"""
    k = len(X); G = np.empty((W.shape[0], k, k)); b = np.empty((W.shape[0], k))
    for i in range(k):
        b[:, i] = W @ (X[i] * Y).sum(1)
        for j in range(i, k):
            G[:, i, j] = G[:, j, i] = W @ (X[i] * X[j]).sum(1)
    out = np.full((W.shape[0], k), np.nan); ok = np.abs(np.linalg.det(G)) > 1e-14; out[ok] = np.linalg.solve(G[ok], b[ok][..., None])[..., 0]
    return out


def same_class_set(name, coco, draws=10000):
    bx = boxes(name); M = {}
    for mname in [m for m in MODELS if m in bx]:
        d = sorted_view(bx[mname]); info = [same_class(coco, d['img'][i], d['gold'][i], d['pred'][i]) for i in range(len(d['iou']))]
        d['have'] = np.array([x is not None for x in info]); d['c'] = np.array([x if x else [np.nan, np.nan] for x in info], float); M[mname] = d
    imgs = sorted(set().union(*[set(d['img']) for d in M.values()])); pos = {i: j for j, i in enumerate(imgs)}; rng = np.random.default_rng(0)
    CNT = np.stack([np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)) for _ in range(draws)]).astype(np.float32)
    V = {k: [] for k in ('r', 'g', 'c', 'm')}; IM = []
    for mname, d in M.items():
        sel = (d['iou'] < .1) & d['have'] & ~np.isnan(d['pred']).any(1) & ~np.isnan(d['readout']).any(1); p = ctr(d['pred'])
        v = {'r': ctr(d['readout']) - p, 'g': ctr(d['gold']) - p, 'c': d['c'] - p, 'm': np.tile(ctr(MEAN_BOX[mname][None]), (len(p), 1)) - p}
        sel &= ~np.any([np.isnan(v[k]).any(1) for k in v], 0)
        for k in V:
            V[k].append(v[k][sel])
        IM.append(np.array([pos[i] for i in d['img'][sel]]))
    V = {k: np.concatenate(x) for k, x in V.items()}; IM = np.concatenate(IM)
    X = [V['g'], V['c'], V['m']]; st = wls(V['r'], X, np.ones((1, len(IM))))[0]; B = wls(V['r'], X, CNT[:, IM])
    return {'errors': int(len(IM)), 'towards the annotated object': [float(st[0]), *ci(B[:, 0])], 'towards the nearest same-class object': [float(st[1]), *ci(B[:, 1])]}


def cmd_pull(a):
    out = {}
    for name in ('new-items', 'heldout', 'replication'):
        for m, rec in pull_set(boxes(name)).items():
            out[f'{name}|{m}'] = rec
    out['same image|new-items+heldout'] = pull_same_image(('new-items', 'heldout'))
    coco = json.load(open(os.path.join(EXTRA, 'coco_instances.json')))['images']
    for name in ('new-items', 'heldout', 'replication'):
        out[f'same class|{name}'] = same_class_set(name, coco)
    print('# wrong-object errors: coefficient of the displacement of the read-out towards a box [95% interval]')
    for k, r in out.items():
        print(f'  {k:40s} n={r["errors"]:4d}  ' + '; '.join(f'{lab} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]' for lab, v in r.items() if lab != 'errors'))
    return out


# ---------------------------------------------------------------- Section 5.2
def cmd_margin(a):
    """models on which the probe alone and SPRED come within 0.03 of the supervised linear probe (one-sided 95% lower
    bound of the difference in AUROC over all errors, 10,000 resamples of the images of the model), replication and held-out
    lists"""
    pkg = E.Package(DATA); out = {}
    for name in ('replication', 'heldout'):
        b = {'probe alone': [], 'SPRED': []}
        for m in MODELS:
            rows = pkg.scores(name, m); raw = lambda c: (lambda v: np.where(np.isnan(v), np.nanmax(v), v))(E.num(rows, c))
            S = {'SPRED': pkg.z(name, m, 'two_teacher_probe') + .5 * pkg.z(name, m, E.RES), 'probe alone': raw('two_teacher_probe')}; sup = raw('linear_probe_correctness')
            pos, neg = pkg.labels(name, m, 'all'); img = np.array(pkg.images(name, m)); uniq, inv = np.unique(img, return_inverse=True); rng = np.random.default_rng(0)
            W = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(10000)])[:, inv].astype(np.float64)
            ws = E.wauc(sup, pos, neg, W)
            for k in b:
                b[k].append(float(np.nanpercentile(E.wauc(S[k], pos, neg, W) - ws, 5)))
        out[name] = {k: {'bounds': v, 'models within 0.03': int(sum(x > -0.03 for x in v))} for k, v in b.items()}
        for k, v in out[name].items():
            print(f'# {name}: {k} minus the supervised linear probe, one-sided bounds ' + ' / '.join(f'{x:+.3f}' for x in v['bounds']) + f'; within 0.03 on {v["models within 0.03"]} of 6')
    return out


# ---------------------------------------------------------------- Section 3.6
NARROW_SETS = (('heldout', 'RefCOCO', ['heldout'], 'heldout_{m}'),
               ('new-items', 'RefCOCO', ['new-items_refcoco-testB', 'new-items_refcocog-val'], 'new-items_{m}'),
               ('objects365', 'Objects365 (fitted in domain)', ['objects365_test'], 'objects365_fitted-in-domain_{m}'),
               ('vrsbench', 'VRSBench (fitted in domain)', ['vrsbench_test'], 'vrsbench_fitted-in-domain_{m}'))
SIGMA_FITS = (('RefCOCO', 'RefCOCO'), ('Objects365 (fitted in domain)', 'Objects365'), ('VRSBench (fitted in domain)', 'VRSBench'))


def cmd_narrow(a):
    """read-out error sigma per model and fit (results/readout_error.csv: median distance between the centres of the
    out-of-fold read-out and the written box over the training items of the fit, relative image coordinates), and per
    item set and model the share of written boxes whose width, the square root of their area in coordinates divided by
    the image width and height, is below 2 sigma of the fit that produced the scores"""
    sig = {(r['model'], r['fit']): float(r['sigma']) for r in read(os.path.join(EXTRA, 'readout_error.csv'))}
    out = {'sigma': {}, 'share': {}}
    for fit, lab in SIGMA_FITS:
        v = {m: s for (m, f), s in sig.items() if f == fit}
        out['sigma'][lab] = {**v, 'min': min(v.values()), 'max': max(v.values())}
        print(f'# read-out error sigma, fit {fit}: ' + ', '.join(f'{m} {s:.3f}' for m, s in v.items()))
    for name, fit, lists, stem in NARROW_SETS:
        size = {r['item_id']: (float(r['image_width']), float(r['image_height'])) for l in lists for r in read(os.path.join(DATA, 'lists', f'{l}.csv'))}
        per = {}
        for m in MODELS:
            p = os.path.join(DATA, 'scores', stem.format(m=m) + '.csv')
            if (m, fit) not in sig or not os.path.exists(p):
                continue
            rows = read(p); W = np.array([size[r['item_id']][0] for r in rows]); H = np.array([size[r['item_id']][1] for r in rows])
            w = np.clip(np.array([float(r['written_x2']) - float(r['written_x1']) for r in rows]), 0, None) / W
            h = np.clip(np.array([float(r['written_y2']) - float(r['written_y1']) for r in rows]), 0, None) / H
            per[m] = float(np.mean(np.sqrt(w * h) < 2 * sig[(m, fit)]))
        out['share'][name] = {**per, 'min': min(per.values()), 'max': max(per.values())}
        print(f'# {name}, read-out fitted on {fit}: share of written boxes narrower than 2 sigma: ' + ', '.join(f'{m} {v:.3f}' for m, v in per.items()))
    return out


# ---------------------------------------------------------------- Section 6.3 (instances)
def cmd_instances(a):
    """wrong-object errors on the replication list whose written box lies on another annotated MS-COCO instance: IoU >= 0.5
    with a non-crowd instance other than the target instance (the instance with the highest IoU to the annotated box,
    which must reach 0.5); their share over the six models, and SPRED minus its probe in AUROC for these errors against
    the correct answers, mean over the models"""
    pkg = E.Package(DATA); coco = json.load(open(os.path.join(EXTRA, 'coco_instances.json')))['images']; bx = boxes('replication'); tw = tc = 0; d = {}
    for m in MODELS:
        b = bx[m]; at = {i: k for k, i in enumerate(b['item_id'])}; rows = pkg.scores('replication', m); iou = np.array([float(r['iou']) for r in rows]); wo = iou < .1; on = np.zeros(len(rows), bool)
        for j, r in enumerate(rows):
            k = at[r['item_id']]; im = coco.get(str(b['img'][k]))
            if not wo[j] or im is None or not im['anns']:
                continue
            Wd, Ht = float(im['width']), float(im['height'])
            inst = [(x, [x['bbox'][0] / Wd, x['bbox'][1] / Ht, (x['bbox'][0] + x['bbox'][2]) / Wd, (x['bbox'][1] + x['bbox'][3]) / Ht]) for x in im['anns']]
            gold, pred = list(b['gold'][k]), list(b['pred'][k]); tg = max(inst, key=lambda x: iou1(gold, x[1]))
            if iou1(gold, tg[1]) >= .5:
                on[j] = any(x[0]['id'] != tg[0]['id'] and not x[0].get('iscrowd') and iou1(pred, x[1]) >= .5 for x in inst)
        tw += int(wo.sum()); tc += int(on.sum()); keep = on | (iou >= .5); zp = pkg.z('replication', m, 'two_teacher_probe'); zd = zp + .5 * pkg.z('replication', m, E.RES)
        d[m] = float(auroc(zd[keep], on[keep]) - auroc(zp[keep], on[keep]))
    out = {'share percent': 100 * tc / tw, 'SPRED minus probe': float(np.mean(list(d.values())))}
    print(f'# replication list: {tc} of {tw} wrong-object errors lie on another annotated instance ({out["share percent"]:.1f}%); on them SPRED minus its probe '
          + ' / '.join(f'{v:+.3f}' for v in d.values()) + f'; mean over the models {out["SPRED minus probe"]:+.3f}')
    return out


# ---------------------------------------------------------------- Section 3.3
def cmd_readout_accuracy(a):
    """replication list, first three models: share of read-out boxes and of written boxes with IoU >= 0.5 with the annotated
    box; lowest and highest over the three models"""
    pkg = E.Package(DATA); gold = {r['item_id']: [float(r[f'box_{c}']) for c in ('x1', 'y1', 'x2', 'y2')] for r in read(os.path.join(DATA, 'lists', 'replication.csv'))}; per = {}
    for m in MODELS[:3]:
        rows = pkg.scores('replication', m); g = np.array([gold[r['item_id']] for r in rows])
        bx = lambda p: np.array([[float(r[f'{p}_{c}']) for c in ('x1', 'y1', 'x2', 'y2')] for r in rows])
        per[m] = {'read-out': float(np.mean(iou_rows(bx('readout'), g) >= .5)), 'written': float(np.mean(iou_rows(bx('written'), g) >= .5))}
    out = {k: {'min': min(v[k] for v in per.values()), 'max': max(v[k] for v in per.values())} for k in ('read-out', 'written')}
    print('# replication list, acc@0.5 of the read-out box / of the written box: ' + '; '.join(f"{m} {v['read-out']:.3f} / {v['written']:.3f}" for m, v in per.items()))
    return out


# ---------------------------------------------------------------- Section 7 (limitations) and Section 3.6 (other image domains)
SEVEN = ('two_teacher_probe', 'sampling_teacher_probe', 'shift_teacher_probe', 'linear_probe_correctness', 'linear_probe_one_minus_iou', 'pregeneration_probe', 'saplma')


def width_strata(models, draws=10000):
    """Objects365, read-out and probes fitted on RefCOCO: gain of 0.5 z(residual) on wrong-object errors (against correct
    answers) within r < 2 and r >= 2; mean over the seven hidden-state probes and the models with its 95% interval"""
    sig = {(r['model'], r['fit']): float(r['sigma']) for r in read(os.path.join(EXTRA, 'readout_error.csv'))}
    lst = {r['item_id']: r for r in read(os.path.join(DATA, 'lists', 'objects365_test.csv'))}; M = {}
    for m in models:
        rows = read(os.path.join(DATA, 'scores', f'objects365_fitted-on-RefCOCO_{m}.csv'))
        W = np.array([float(lst[r['item_id']]['image_width']) for r in rows]); H = np.array([float(lst[r['item_id']]['image_height']) for r in rows])
        w = np.clip(np.array([float(r['written_x2']) - float(r['written_x1']) for r in rows]), 0, None) / W
        h = np.clip(np.array([float(r['written_y2']) - float(r['written_y1']) for r in rows]), 0, None) / H
        Z = {}
        for q in SEVEN + (E.RES,):
            v = np.array([fl(r[q]) for r in rows]); Z[q] = z(m, q, np.where(np.isnan(v), np.nanmax(v), v))
        M[m] = {'r': np.sqrt(w * h) / sig[(m, 'RefCOCO')], 'iou': np.array([float(r['iou']) for r in rows]), 'img': np.array([lst[r['item_id']]['image_id'] for r in rows]), 'Z': Z}
    imgs = sorted(set().union(*[set(d['img']) for d in M.values()])); pos = {i: j for j, i in enumerate(imgs)}; rng = np.random.default_rng(0)
    CNT = np.stack([np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)) for _ in range(draws)]).astype(np.float32); out = {}
    for name, lo, hi in (('r < 2', 0., 2.), ('r >= 2', 2., np.inf)):
        pts, B = [], []
        for m, d in M.items():
            s = (d['r'] >= lo) & (d['r'] < hi); pf, nf = (s & (d['iou'] < .1)).astype(np.float32), (s & (d['iou'] >= .5)).astype(np.float32)
            if pf.sum() < 5 or nf.sum() < 5:
                continue
            W = CNT[:, np.array([pos[i] for i in d['img']])]; W0 = np.ones((1, len(d['iou'])), np.float32); zr = d['Z'][E.RES]
            for q in SEVEN:
                zb = d['Z'][q]; pts.append(float(E.wauc(zb + .5 * zr, pf, nf, W0)[0] - E.wauc(zb, pf, nf, W0)[0])); B.append(E.wauc(zb + .5 * zr, pf, nf, W) - E.wauc(zb, pf, nf, W))
        out[name] = [float(np.mean(pts)), *ci(np.mean(B, 0))]
    return out


def cmd_domains(a):
    """AUROC of SPRED for all errors on the Objects365 and VRSBench test items, read-out and probe fitted on unlabelled
    training items of the domain; and, with the read-out and the probes fitted on RefCOCO and applied unchanged to
    Objects365, the gain of the residual on wrong-object errors by the width r of the written box (square root of its
    area in coordinates divided by the image size) in units of the read-out error sigma of the fit, mean over the seven
    hidden-state probes and the three models (images of all models resampled together, 10,000 times, seed 0; a stratum of
    a model with fewer than 5 errors or 5 correct answers is left out)"""
    out = {'auroc': {}}
    for dom, models in (('objects365', MODELS[:3]), ('vrsbench', [MODELS[0], MODELS[2]])):
        per = {}
        for m in models:
            rows = read(os.path.join(DATA, 'scores', f'{dom}_fitted-in-domain_{m}.csv')); per[m] = float(auroc([float(r['detector']) for r in rows], [r['correct'] == '0' for r in rows]))
        out['auroc'][dom] = {**per, 'min': min(per.values()), 'max': max(per.values())}
        print(f'# {dom}, fitted on the domain: AUROC of SPRED for all errors ' + ', '.join(f'{m} {v:.3f}' for m, v in per.items()))
    out['width strata'] = width_strata(MODELS[:3])
    for k, v in out['width strata'].items():
        print(f'# Objects365, fitted on RefCOCO: gain on wrong-object errors, {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] (seven probes, three models)')
    return out


# ---------------------------------------------------------------- check
def flatten(prefix, o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            flatten(f'{prefix}|{k}' if prefix else str(k), v, out)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            flatten(f'{prefix}|{i}', v, out)
    else:
        out[prefix] = o


def cmd_check(a):
    quiet = lambda f: f(a)
    import contextlib, io
    got = {}
    for name, f in (('lists', cmd_lists), ('ablation', cmd_ablation), ('table4', cmd_table4), ('second-answer', cmd_second_answer), ('pull', cmd_pull), ('margin', cmd_margin), ('narrow', cmd_narrow), ('instances', cmd_instances), ('readout-accuracy', cmd_readout_accuracy), ('domains', cmd_domains)):
        with contextlib.redirect_stdout(io.StringIO()):
            flatten(name, quiet(f), got)
    bad = 0; n = 0
    for r in read(a.expected):
        n += 1; v = got.get(r['quantity']); d = int(r['decimals'])
        mine = None if v is None else f'{float(v):.{d}f}'
        ok = mine == f'{float(r["paper"]):.{d}f}'
        bad += not ok
        print(f'{r["where"]:28s} {r["quantity"]:70s} {mine if mine is not None else "missing":>8s} {float(r["paper"]):>8.{d}f}' + ('' if ok else '   DIFFERS'))
    print(f'{bad} of the {n} listed numbers differ after rounding')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('lists', 'ablation', 'table4', 'second-answer', 'pull', 'margin', 'narrow', 'instances', 'readout-accuracy', 'domains', 'check'))
    ap.add_argument('--expected', default=os.path.join(EXTRA, 'expected_main.csv'))
    a = ap.parse_args()
    f = {'lists': cmd_lists, 'ablation': cmd_ablation, 'table4': cmd_table4, 'second-answer': cmd_second_answer, 'pull': cmd_pull, 'margin': cmd_margin, 'narrow': cmd_narrow, 'instances': cmd_instances, 'readout-accuracy': cmd_readout_accuracy, 'domains': cmd_domains, 'check': cmd_check}[a.cmd]
    r = f(a)
    if a.cmd == 'check':
        sys.exit(1 if r else 0)


if __name__ == '__main__':
    main()
