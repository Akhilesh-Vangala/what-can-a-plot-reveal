"""Rational-benchmark estimators for B0 (full state), B1 (mark table) and B2 (image).

All levels use the same splits; B0 and B1 use the same model families and hyperparameter grids (gap 2.4).
The best model is chosen on validation and reported on test (estimates = lower bounds on the true optimum, gap 2.5).
Saturation curves (test AUC vs training fraction) check the benchmark is not under-trained (gap 2.2).
"""
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# PyTorch is imported lazily (only the image benchmark needs it): loading it alongside scikit-learn's OpenMP runtime
# can crash gradient boosting on macOS.
torch = nn = None


def _torch():
    global torch, nn
    if torch is None:
        import torch as _t
        from torch import nn as _n
        torch, nn = _t, _n
        torch.set_num_threads(os.cpu_count())
    return torch, nn

GRIDS = {
    "logreg": [dict(C=c) for c in (0.01, 0.1, 1.0, 10.0)],
    "hgb": [dict(learning_rate=lr, max_leaf_nodes=m) for lr in (0.05, 0.1) for m in (15, 31)],
    "rf": [dict(min_samples_leaf=m) for m in (1, 5)],
    "mlp": [dict(alpha=a) for a in (1e-4, 1e-2)],
}
SAT_FRACS = (0.25, 0.5, 1.0)


def _make(name, params, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **params))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, random_state=seed, **params)
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed, **params)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=1000, early_stopping=True,
                                                             random_state=seed, **params))
    raise ValueError(name)


def make_split(y, seed, frac=(0.6, 0.2, 0.2)):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=frac[0], stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=frac[1] / (frac[1] + frac[2]), stratify=y[rest], random_state=seed)
    return dict(train=tr, val=va, test=te)


def _subsample(tr, y, frac, seed):
    if frac >= 1:
        return tr
    keep, _ = train_test_split(tr, train_size=frac, stratify=y[tr], random_state=seed)
    return keep


def auc_ci(y, s, seed=0, n=1000):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                           np.r_[s[rng.choice(pos, len(pos))], s[rng.choice(neg, len(neg))]]) for _ in range(n)]
    return float(roc_auc_score(y, s)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def best_of_k(X, y, split, seed=0, saturation=True):
    """Tabular benchmark (B0 / B1): best of K families × small grids, selected on validation."""
    tr, va, te = split["train"], split["val"], split["test"]
    cands = []
    for name, grid in GRIDS.items():
        for params in grid:
            m = _make(name, params, seed).fit(X[tr], y[tr])
            cands.append((roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]), name, params, m))
    val_auc, name, params, model = max(cands, key=lambda c: c[0])
    scores = model.predict_proba(X[te])[:, 1]
    auc, ci = auc_ci(y[te], scores, seed)
    out = dict(best=name, params=params, val_auc=float(val_auc), test_auc=auc, ci=ci,
               test_scores=scores, family_val_auc={n: float(max(c[0] for c in cands if c[1] == n)) for n in GRIDS})
    if saturation:
        out["saturation"] = [
            (f, float(roc_auc_score(y[te], _make(name, params, seed).fit(X[_subsample(tr, y, f, seed)],
                                                                         y[_subsample(tr, y, f, seed)])
                                     .predict_proba(X[te])[:, 1]))) for f in SAT_FRACS]
    return out


def SmallCNN():
    _, nn = _torch()

    class _CNN(nn.Module):
        def __init__(self):
            super().__init__()

            def block(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
            self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64),
                                          nn.AdaptiveAvgPool2d(1))
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            return self.head(self.features(x).flatten(1)).squeeze(1)
    return _CNN()


def _cnn_fit(imgs, y, tr, va, seed, epochs=40, patience=8):
    torch, nn = _torch()
    torch.manual_seed(seed)
    X = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    X = (X - X.mean()) / (X.std() + 1e-6)
    Y = torch.tensor(y, dtype=torch.float32)
    net, best, best_auc, wait = SmallCNN(), None, -1, 0
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for b in torch.tensor(tr)[torch.randperm(len(tr), generator=g)].split(32):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(X[b]), Y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(y[va], net(X[va]).numpy())
        if auc > best_auc:
            best_auc, best, wait = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best)
    net.eval()
    return net, X, best_auc


def image_benchmark(imgs, y, split, seed=0, saturation=True):
    """B2: best of {small CNN, logistic regression on coarse grayscale pixels}, selected on validation."""
    torch, _ = _torch()
    tr, va, te = split["train"], split["val"], split["test"]
    net, X, cnn_val = _cnn_fit(imgs, y, tr, va, seed)
    with torch.no_grad():
        cnn_scores = net(X[te]).numpy()
    small = imgs[:, ::4, ::4].mean(-1).reshape(len(imgs), -1)
    px = max((( roc_auc_score(y[va], m.predict_proba(small[va])[:, 1]), m) for m in
              (_make("logreg", p, seed).fit(small[tr], y[tr]) for p in GRIDS["logreg"])), key=lambda c: c[0])
    if cnn_val >= px[0]:
        name, val_auc, scores = "cnn", cnn_val, cnn_scores
    else:
        name, val_auc, scores = "pixel_logreg", px[0], px[1].predict_proba(small[te])[:, 1]
    auc, ci = auc_ci(y[te], scores, seed)
    out = dict(best=name, val_auc=float(val_auc), test_auc=auc, ci=ci, test_scores=scores,
               family_val_auc=dict(cnn=float(cnn_val), pixel_logreg=float(px[0])))
    if saturation and name == "cnn":
        sat = []
        for f in SAT_FRACS[:-1]:
            sub = _subsample(tr, y, f, seed)
            n2, X2, _ = _cnn_fit(imgs, y, sub, va, seed)
            with torch.no_grad():
                sat.append((f, float(roc_auc_score(y[te], n2(X2[te]).numpy()))))
        out["saturation"] = sat + [(1.0, auc)]
    return out


def lineup_accuracy(fail_scores, healthy_scores, seed=0, n=5000, size=20):
    """P(the real panel scores highest among 1 failed + 19 healthy test plots): benchmark lineup detection."""
    rng = np.random.default_rng(seed)
    hits = 0.0
    for _ in range(n):
        f = rng.choice(fail_scores)
        h = rng.choice(healthy_scores, size - 1, replace=False)
        hits += 1.0 if f > h.max() else (1.0 / (1 + (h == f).sum()) if f == h.max() else 0.0)
    return hits / n


def cv_auc(X, y, seed=0, folds=5):
    """Calibration-grade detection AUC: choose the model family/params on one split, then report the mean 5-fold
    cross-validated AUC of that configuration (less noisy than a single held-out split)."""
    from sklearn.model_selection import StratifiedKFold
    r = best_of_k(X, y, make_split(y, seed), seed=seed, saturation=False)
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = _make(r["best"], r["params"], seed).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs) / np.sqrt(folds)), r["best"]


def best_of_k_multi(X, y, split, seed=0, classes=None):
    """Diagnosis benchmark: which state (healthy or one failure) produced this plot? Best of K families on validation
    macro one-vs-rest AUC; reports test accuracy, macro AUC and the confusion matrix (rows = truth, cols = predicted)."""
    from sklearn.metrics import accuracy_score, confusion_matrix
    tr, va, te = split["train"], split["val"], split["test"]
    cands = []
    for name, grid in GRIDS.items():
        for params in grid:
            m = _make(name, params, seed).fit(X[tr], y[tr])
            cands.append((roc_auc_score(y[va], m.predict_proba(X[va]), multi_class="ovr"), name, params, m))
    val_auc, name, params, model = max(cands, key=lambda c: c[0])
    P = model.predict_proba(X[te])
    prior = np.bincount(y[tr], minlength=P.shape[1]) / len(tr)
    Pb = P / np.maximum(prior, 1e-9)  # Bayes decision for balanced loss (classes are imbalanced by design)
    pred = Pb.argmax(1)
    cm = confusion_matrix(y[te], pred, labels=list(range(P.shape[1])))
    return dict(best=name, params=params, val_auc=float(val_auc),
                test_auc=float(roc_auc_score(y[te], P, multi_class="ovr")),
                test_acc=float(accuracy_score(y[te], pred)),
                balanced_acc=float(np.mean(np.diag(cm) / np.maximum(cm.sum(1), 1))),
                confusion=cm.tolist(), classes=classes, test_proba=P, prior=prior.tolist())
