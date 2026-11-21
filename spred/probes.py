"""Ridge probes, configuration selection, the perceptron probe and AUROC (torch, CPU)."""
import math

import torch

LAMS = (1e2, 1e3, 1e4, 1e5)
GEOMETRY_LAMS = (1e-2, 1e-1, 1e0, 1e1, 1e2)


def auroc(s, y):
    """AUROC with label 1 as the positive class; ties count one half."""
    s = torch.as_tensor(s, dtype=torch.float64)
    y = torch.as_tensor(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return ((d > 0).double().mean() + 0.5 * (d == 0).double().mean()).item()


class Ridge:
    """Ridge regression on standardized features (dual form, fast when N < D); one target."""

    def __init__(self, X, y, lam):
        X = X.double()
        self.mu, self.sd = X.mean(0), X.std(0).clamp_min(1e-6)
        Z = (X - self.mu) / self.sd
        self.b = y.double().mean()
        K = Z @ Z.T + lam * torch.eye(len(Z), dtype=torch.float64)
        self.w = Z.T @ torch.linalg.solve(K, y.double() - self.b)

    def __call__(self, X):
        return ((X.double() - self.mu) / self.sd) @ self.w + self.b


class BoxRidge:
    """Ridge regression on standardized features with several targets (the four box coordinates)."""

    def __init__(self, X, Y, lam):
        self.mu, self.sd = X.mean(0), X.std(0).clamp(min=1e-6)
        Z = (X - self.mu) / self.sd
        self.ym = Y.mean(0)
        self.w = Z.T @ torch.linalg.solve(Z @ Z.T + lam * torch.eye(len(Z), dtype=X.dtype), Y - self.ym)

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.ym


def out_of_fold(X, target, lam, folds):
    """Out-of-fold ridge predictions for the given folds (index tensors)."""
    pred = torch.zeros(len(target), dtype=torch.float64)
    for f in folds:
        m = torch.ones(len(target), dtype=torch.bool)
        m[f] = False
        pred[f] = Ridge(X[m], target[m], lam)(X[f])
    return pred


def select_and_fit(feats, target, sel_label, folds, lams=LAMS):
    """feats: {name: (N, layers, D)}. Picks the (feature, layer, ridge strength) whose out-of-fold predictions of
    `target` have the highest AUROC against the binary `sel_label`, then refits on all rows."""
    best = None
    for name, X in feats.items():
        for li in range(X.shape[1]):
            for lam in lams:
                a = auroc(out_of_fold(X[:, li], target, lam, folds), sel_label)
                if not math.isnan(a) and (best is None or a > best[0]):
                    best = (a, name, li, lam)
    if best is None:
        raise ValueError("the selection label has a single class")
    a, name, li, lam = best
    return Ridge(feats[name][:, li], target, lam), {"feat": name, "layer_idx": li, "lam": lam, "cv_auroc": a}


def fit_mlp(X, y, hidden=(256, 128, 64), epochs=50, lr=1e-3, wd=1e-4, seed=0):
    """Three-layer perceptron on correctness labels (SAPLMA), fixed hyperparameters; returns a scoring function
    (larger = more likely wrong). Single precision: the result depends slightly on the number of threads."""
    torch.manual_seed(seed)
    X = X.float()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Z, yt = (X - mu) / sd, torch.as_tensor(y, dtype=torch.float32)
    dims, layers = [X.shape[1], *hidden], []
    for i in range(len(hidden)):
        layers += [torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.ReLU()]
    net = torch.nn.Sequential(*layers, torch.nn.Linear(dims[-1], 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for ix in torch.randperm(len(Z), generator=g).split(64):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Z[ix]).squeeze(1), yt[ix]).backward()
            opt.step()
    net.eval()

    def score(Xn):
        with torch.no_grad():
            return net((Xn.float() - mu) / sd).squeeze(1).double()
    return score


def box_iou(p, q):
    """Row-wise IoU of two (N, 4) box tensors."""
    x1 = torch.maximum(p[:, 0], q[:, 0]); y1 = torch.maximum(p[:, 1], q[:, 1])
    x2 = torch.minimum(p[:, 2], q[:, 2]); y2 = torch.minimum(p[:, 3], q[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = lambda t: (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    u = area(p) + area(q) - inter
    return torch.where(u > 0, inter / u, torch.zeros_like(u))


def geometry_features(box):
    """Eleven label-free features of the written box (relative coordinates): area, its square root, width, height,
    centre, distances of the centre from the image centre, log aspect ratio and its absolute value."""
    b = torch.nan_to_num(box, nan=0.0)
    w = (b[:, 2] - b[:, 0]).clamp(min=1e-4); h = (b[:, 3] - b[:, 1]).clamp(min=1e-4)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return torch.stack([w * h, (w * h).sqrt(), w, h, cx, cy, (cx - .5).abs(), (cy - .5).abs(), ((cx - .5) ** 2 + (cy - .5) ** 2).sqrt(),
                        (w / h).log(), (w / h).log().abs()], 1).float()
