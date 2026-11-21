"""BICR baseline: a correctness probe with a blind-image ranking term (arXiv:2605.10893), our reimplementation.

  blank-states  (GPU) for every training item with a written box: one forward pass with the image replaced by a solid
                black image of the same size, over the prompt followed by the written answer tokens; the hidden state at
                the last prompt token is stored for the last two of the saved layers
  fit           (CPU) MLP probe f on the standardized final-layer state of the last prompt token (from the feature
                files of spred.run_model), trained on the training items with
                    BCE(f(h), y; positive weight n_wrong / n_correct) + beta * mean((p - y)^2)
                    + lambda * sum_i relu(gamma - (p_i - p_blank_i)) y_i / (sum_i y_i + 1e-8),   p = sigmoid(f(h)),
                y = 1 for a correct answer; 40 random configurations of the published search space (hidden sizes,
                dropout, learning rate, weight decay, beta, lambda, gamma), each scored by the out-of-fold AUROC of 5
                folds; the best is refitted on all training items. Adam, batch 64, 40 epochs. Score = -f(h), larger =
                more likely wrong; one forward pass at test time. --trial K refits configuration K without the search.

  python -m spred.bicr blank-states --model M --items work/items/training.jsonl --answers work/answers/training_M.jsonl \
      --out work/features/blank_training_M.pt
  python -m spred.bicr fit --train work/answers/training_M.jsonl work/features/training_M.pt --blank work/features/blank_training_M.pt \
      --test replication work/features/replication_M.pt --out-dir work/scores/bicr
"""
import argparse
import csv
import math
import os
import time

import torch

from spred import geometry as G

SIZES = ((), (256,), (512,), (128, 64), (256, 128), (512, 256), (1024, 512), (1024, 512, 256))
DROPOUT = (0.0, 0.1, 0.3, 0.5)


def cmd_blank_states(a):
    from PIL import Image
    from spred import models
    run = models.load(a.model, attn="eager")
    saved = sorted(set(range(0, run.n_layers + 1, a.layer_step)) | {run.n_layers})
    layers = saved[-2:]
    items = {it["id"]: it for it in G.load_jsonl(a.items)}
    answers = [g for g in G.load_jsonl(a.answers) if g["pred"] is not None and g.get("gen_ids")][: a.limit or None]
    Hs, ids, t0 = [], [], time.time()
    with torch.inference_mode():
        for k, g in enumerate(answers):
            it = items[g["id"]]
            img, _ = run.image(it)
            inp = run.inputs(Image.new("RGB", img.size, (0, 0, 0)), it["ref"])
            P = inp["input_ids"].shape[1]
            full = torch.cat([inp["input_ids"], torch.tensor([g["gen_ids"]], device=inp["input_ids"].device)], 1)
            kw = dict(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=inp["pixel_values"], output_hidden_states=True)
            if models.family_of(a.model) == "qwen":
                run.model.model.rope_deltas = None
                kw["image_grid_thw"] = inp["image_grid_thw"]
            out = run.model(**kw)
            Hs.append(torch.stack([out.hidden_states[l][0, P - 1] for l in layers]).half().cpu())
            ids.append(g["id"])
            if (k + 1) % 100 == 0:
                print(f"[blank-states] {k + 1}/{len(answers)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"ids": ids, "layers": layers, "h_blank": torch.stack(Hs)}, a.out)
    print(f"[blank-states] {len(ids)} items in {time.time() - t0:.0f}s -> {a.out}; layers {layers}")


def auroc(s, y):
    """AUROC of score s for the positives y == 1 (ties count one half)"""
    s, y = s.double(), y.bool()
    o = torch.argsort(s); r = torch.empty_like(s); r[o] = torch.arange(1, len(s) + 1, dtype=torch.double)
    for v in torch.unique(s[torch.cat([torch.tensor([False]), s[o][1:] == s[o][:-1]])]) if len(s) > 1 else []:
        m = s == v; r[m] = r[m].mean()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float("nan") if n1 == 0 or n0 == 0 else float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sample_configs(n, seed):
    g = torch.Generator().manual_seed(seed); u = lambda: torch.rand(1, generator=g).item(); out = []
    for _ in range(n):
        out.append({"sizes": SIZES[int(u() * len(SIZES))], "dropout": DROPOUT[int(u() * len(DROPOUT))], "lr": 10 ** (-5 + 2 * u()), "wd": 10 ** (-6 + 3 * u()),
                    "beta": 0.5 * u(), "lam": 0.01 + 0.29 * u(), "gamma": 0.05 + 0.20 * u()})
    return out


def train(Xb, Xk, y, c, epochs, seed):
    """trains the probe on standardized real-image and black-image states; y = 1 for a correct answer; returns f"""
    torch.manual_seed(seed)
    dims, layers = [Xb.shape[1], *c["sizes"]], []
    for i in range(len(c["sizes"])):
        layers += [torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ReLU(), torch.nn.Dropout(c["dropout"])]
    net = torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], 1))
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"], weight_decay=c["wd"])
    yt = y.float(); w = torch.tensor(float((1 - yt).sum() / yt.sum().clamp(min=1)))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(Xb), generator=g).split(64):
            opt.zero_grad()
            zb = net(Xb[ix]).squeeze(1); pb = torch.sigmoid(zb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(zb, yt[ix], pos_weight=w)
            loss = loss + c["beta"] * ((pb - yt[ix]) ** 2).mean()
            pk = torch.sigmoid(net(Xk[ix]).squeeze(1))
            loss = loss + c["lam"] * (torch.relu(c["gamma"] - (pb - pk)) * yt[ix]).sum() / (yt[ix].sum() + 1e-8)
            loss.backward(); opt.step()
    net.eval()

    def f(X):
        with torch.no_grad():
            return net(X).squeeze(1)
    return f


def cmd_fit(a):
    F = torch.load(a.train[1], weights_only=False); B = torch.load(a.blank, weights_only=False)
    iou = {r["id"]: float(r["iou"] or 0) for r in G.load_jsonl(a.train[0])}
    li, lb = F["layers"].index(B["layers"][-1]), len(B["layers"]) - 1
    bpos = {i: j for j, i in enumerate(B["ids"])}; keep = [j for j, i in enumerate(F["ids"]) if i in bpos]; ids = [F["ids"][j] for j in keep]
    Xb = F["h_last"][keep][:, li].float(); Xk = torch.stack([B["h_blank"][bpos[i]][lb] for i in ids]).float(); y = torch.tensor([int(iou[i] >= .5) for i in ids])
    mu, sd = Xb.mean(0), Xb.std(0).clamp(min=1e-6); Zb, Zk = (Xb - mu) / sd, (Xk - mu) / sd
    folds = torch.randperm(len(ids), generator=torch.Generator().manual_seed(a.seed)).chunk(5); configs = sample_configs(a.trials, a.seed)

    def oof(c):
        o = torch.zeros(len(ids))
        for p in folds:
            m = torch.ones(len(ids), dtype=torch.bool); m[p] = False
            o[p] = -train(Zb[m], Zk[m], y[m], c, a.epochs, a.seed)(Zb[p])
        return o
    if a.trial is None:
        best = None
        for k, c in enumerate(configs):
            o = oof(c); au = auroc(o, 1 - y)
            if not math.isnan(au) and (best is None or au > best[0]):
                best = (au, k, o)
        au, k, o = best
    else:
        k = a.trial; o = oof(configs[k]); au = auroc(o, 1 - y)
    c = configs[k]; f = train(Zb, Zk, y, c, a.epochs, a.seed)
    print(f"[fit] configuration {k} of {len(configs)}, out-of-fold AUROC {au:.3f}: sizes {list(c['sizes'])}, dropout {c['dropout']}, lr {c['lr']:.2e}, "
          f"wd {c['wd']:.2e}, beta {c['beta']:.2f}, lambda {c['lam']:.2f}, gamma {c['gamma']:.2f}")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "bicr_training.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["item_id", "bicr_out_of_fold"]); w.writerows([[i, f"{float(v):.{a.digits}g}"] for i, v in zip(ids, o)])
    for name, path in a.test:
        T = torch.load(path, weights_only=False); Z = (T["h_last"][:, T["layers"].index(F["layers"][li])].float() - mu) / sd
        with open(os.path.join(a.out_dir, f"bicr_{name}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["item_id", "bicr"]); w.writerows([[i, f"{float(v):.{a.digits}g}"] for i, v in zip(T["ids"], -f(Z))])
        print(f"[fit] {name}: {len(T['ids'])} items")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("blank-states")
    p.add_argument("--model", required=True); p.add_argument("--items", required=True); p.add_argument("--answers", required=True)
    p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0); p.add_argument("--layer-step", type=int, default=4)
    p = sp.add_parser("fit")
    p.add_argument("--train", nargs=2, required=True, metavar=("ANSWERS", "FEATURES")); p.add_argument("--blank", required=True)
    p.add_argument("--test", nargs=2, action="append", default=[], metavar=("SET", "FEATURES")); p.add_argument("--out-dir", required=True)
    p.add_argument("--trials", type=int, default=40); p.add_argument("--epochs", type=int, default=40); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial", type=int); p.add_argument("--digits", type=int, default=8)
    a = ap.parse_args()
    {"blank-states": cmd_blank_states, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
