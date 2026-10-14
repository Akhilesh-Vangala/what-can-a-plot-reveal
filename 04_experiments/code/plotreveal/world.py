"""Multi-failure world (v3): healthy models + 4 planted failures, with held-out predictions kept for diagnostic plots.

Failures (severity s in [0, 1]; 0 = healthy):
  overfitting      capacity up, data down, regularisation off
  miscalibration   overconfident outputs: logits × (1 + 4s) — rank-preserving, accuracy unchanged
  label_noise      a fraction 0.4·s of labels flipped (all splits: the data pipeline is noisy)
  class_imbalance  positive class subsampled: positive rate falls from ~0.5 towards ~0.04 (all splits)
  data_leakage     an extra feature encodes the label with strength 3·s in train/val/test (random-split leakage);
                   a clean deployment set (leak feature = noise) gives the true performance, kept in the state only
  distribution_shift  test features shifted by 3·s standard deviations along a random direction (train/val unchanged)
Healthy configs vary naturally (data difficulty, size, width, regularisation, learning rate, base positive rate ±5%).
The v2 generator (generator.py) is left untouched so the overfitting prototype stays exactly reproducible.
"""
import os
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from .seeds import rng_for

N_TRAIN_MAX, N_VAL, N_TEST = 2500, 500, 1000
EPOCHS = 100
FAILURES_V3 = ("overfitting", "miscalibration", "label_noise", "class_imbalance")  # the Oct 7 cluster run
FAILURES = FAILURES_V3 + ("data_leakage", "distribution_shift")


_REAL = {}


def real_data(name):
    """Real tabular data (OpenML), cached in memory and on disk (SCIKIT_LEARN_DATA or ~/scikit_learn_data)."""
    if name not in _REAL:
        from sklearn.datasets import fetch_openml
        home = os.environ.get("SCIKIT_LEARN_DATA")
        try:
            d = fetch_openml(name="MagicTelescope", version=1, as_frame=False, parser="liac-arff", data_home=home)
        except (ValueError, TypeError, ImportError):  # newer scikit-learn: let it choose the parser
            d = fetch_openml(name="MagicTelescope", version=1, as_frame=False, data_home=home)
        X = d.data.astype(np.float64)
        y = (np.asarray(d.target) == "h").astype(int)  # hadron (background) = positive class, 35% of rows
        _REAL[name] = (X, y)
    return _REAL[name]


def healthy_config(*seed_parts, epochs=EPOCHS, source="synthetic"):
    r = rng_for("world-config", *seed_parts) if source == "synthetic" else rng_for("world-config", source, *seed_parts)
    if source != "synthetic":  # real data: natural variation = row sample, feature subset, size, model settings
        return dict(data_seed=int(r.integers(2**31)), split_seed=int(r.integers(2**31)),
                    model_seed=int(r.integers(2**31)), noise_seed=int(r.integers(2**31)), source=source,
                    n_drop=int(r.integers(0, 3)), pos_rate=float(r.uniform(0.45, 0.55)),
                    n_train=int(r.integers(1500, N_TRAIN_MAX + 1)), hidden=int(r.choice([16, 24, 32])),
                    alpha=float(10 ** r.uniform(-1, 0)), lr=float(10 ** r.uniform(-3.3, -2.7)), epochs=int(epochs),
                    logit_scale=1.0, label_noise=0.0, minority_keep=1.0, leak=0.0, shift=0.0,
                    failure="healthy", severity=0.0)
    return dict(
        data_seed=int(r.integers(2**31)), split_seed=int(r.integers(2**31)), model_seed=int(r.integers(2**31)),
        noise_seed=int(r.integers(2**31)),
        n_features=20, n_informative=int(r.integers(5, 11)), class_sep=float(r.uniform(0.6, 1.2)),
        flip_y=float(r.uniform(0.01, 0.05)), pos_rate=float(r.uniform(0.45, 0.55)),
        n_train=int(r.integers(1500, N_TRAIN_MAX + 1)), hidden=int(r.choice([16, 24, 32])),
        alpha=float(10 ** r.uniform(-1, 0)), lr=float(10 ** r.uniform(-3.3, -2.7)), epochs=int(epochs),
        logit_scale=1.0, label_noise=0.0, minority_keep=1.0, leak=0.0, shift=0.0, failure="healthy", severity=0.0)


def apply_failure(cfg, failure, s):
    cfg = dict(cfg, failure=failure, severity=float(s))
    if failure == "healthy" or s == 0:
        return cfg
    if failure == "overfitting":
        cfg["n_train"] = max(100, int(round(cfg["n_train"] * (1 - 0.85 * s))))
        cfg["hidden"] = int(round(cfg["hidden"] * (1 + 3 * s)))
        cfg["alpha"] = cfg["alpha"] * 10 ** (-4 * s)
    elif failure == "miscalibration":
        cfg["logit_scale"] = 1 + 4 * s
    elif failure == "label_noise":
        cfg["label_noise"] = 0.4 * s
    elif failure == "class_imbalance":
        cfg["minority_keep"] = 1 - 0.95 * s
    elif failure == "data_leakage":
        cfg["leak"] = 3 * s
    elif failure == "distribution_shift":
        cfg["shift"] = 3 * s
    else:
        raise NotImplementedError(failure)
    return cfg


def _sample_split(idx_pos, idx_neg, n, q, rng):
    k = int(round(n * q))
    return np.r_[rng.choice(idx_pos, k, replace=False), rng.choice(idx_neg, n - k, replace=False)]


def _ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def train_run(cfg):
    """Train one model; return its full state: training logs + held-out test predictions."""
    total = N_TRAIN_MAX + N_VAL + N_TEST
    if cfg.get("source", "synthetic") == "synthetic":
        X, y = make_classification(n_samples=3 * total, n_features=cfg["n_features"],
                                   n_informative=cfg["n_informative"], n_redundant=2, n_clusters_per_class=2,
                                   class_sep=cfg["class_sep"], flip_y=cfg["flip_y"], random_state=cfg["data_seed"])
    else:
        Xa, ya = real_data(cfg["source"])
        dr = np.random.default_rng(cfg["data_seed"])
        keep = np.sort(dr.permutation(Xa.shape[1])[:Xa.shape[1] - cfg["n_drop"]])  # random feature subset
        X, y = Xa[:, keep], ya
    rng = np.random.default_rng(cfg["split_seed"])
    pos, neg = rng.permutation(np.where(y == 1)[0]), rng.permutation(np.where(y == 0)[0])
    q0, keep = cfg["pos_rate"], cfg["minority_keep"]
    q = q0 * keep / (q0 * keep + (1 - q0))
    n_pos_te, n_pos_va = int(round(N_TEST * q)), int(round(N_VAL * q))
    te = np.r_[pos[:n_pos_te], neg[:N_TEST - n_pos_te]]
    va = np.r_[pos[n_pos_te:n_pos_te + n_pos_va], neg[N_TEST - n_pos_te:N_TEST - n_pos_te + N_VAL - n_pos_va]]
    tr = _sample_split(pos[n_pos_te + n_pos_va:], neg[N_TEST + N_VAL - n_pos_te - n_pos_va:], cfg["n_train"], q, rng)
    y = y.copy()
    if cfg["label_noise"] > 0:
        flip = np.random.default_rng(cfg["noise_seed"]).random(len(y)) < cfg["label_noise"]
        y[flip] = 1 - y[flip]
    used = np.zeros(len(y), bool)
    used[np.r_[tr, va, te]] = True
    dep = np.random.default_rng(cfg["noise_seed"] + 1).choice(np.where(~used)[0], N_TEST, replace=False)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xva, Xte, Xdep = (X[tr] - mu) / sd, (X[va] - mu) / sd, (X[te] - mu) / sd, (X[dep] - mu) / sd
    lr = np.random.default_rng(cfg["noise_seed"] + 2)
    if cfg.get("leak", 0) > 0:  # label-encoding feature everywhere except the clean deployment set
        col = lambda idx: (2 * y[idx] - 1) * cfg["leak"] + lr.standard_normal(len(idx))  # noqa: E731
        Xtr, Xva, Xte = (np.c_[Xtr, col(tr)], np.c_[Xva, col(va)], np.c_[Xte, col(te)])
        Xdep = np.c_[Xdep, lr.standard_normal(len(dep))]
    if cfg.get("shift", 0) > 0:
        u = lr.standard_normal(Xte.shape[1])
        Xte = Xte + cfg["shift"] * u / np.linalg.norm(u)
    ytr, yva, yte = y[tr], y[va], y[te]

    def proba(clf, Xs):  # the model's reported probability (logit scale = miscalibration)
        p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-7, 1 - 1e-7)
        if cfg["logit_scale"] != 1:
            p = 1 / (1 + np.exp(-np.log(p / (1 - p)) * cfg["logit_scale"]))
        return np.clip(p, 1e-7, 1 - 1e-7)

    clf = MLPClassifier(hidden_layer_sizes=(cfg["hidden"], cfg["hidden"]), alpha=cfg["alpha"],
                        learning_rate_init=cfg["lr"], batch_size=min(128, len(tr)), random_state=cfg["model_seed"])
    logs = {k: np.empty(cfg["epochs"], dtype=np.float32) for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for e in range(cfg["epochs"]):
            clf.partial_fit(Xtr, ytr, classes=[0, 1])
            ptr, pva = proba(clf, Xtr), proba(clf, Xva)
            logs["train_loss"][e] = log_loss(ytr, ptr, labels=[0, 1])
            logs["val_loss"][e] = log_loss(yva, pva, labels=[0, 1])
            logs["train_acc"][e] = ((ptr >= 0.5) == ytr).mean()
            logs["val_acc"][e] = ((pva >= 0.5) == yva).mean()
    pte = proba(clf, Xte)
    pdep = proba(clf, Xdep)
    return dict(**logs, test_p=pte.astype(np.float32), test_y=yte.astype(np.int8), val_p=pva.astype(np.float32),
                deploy_acc=float(((pdep >= 0.5) == y[dep]).mean()),
                test_loss=float(log_loss(yte, pte, labels=[0, 1])), test_acc=float(((pte >= 0.5) == yte).mean()),
                test_pos_rate=float(yte.mean()), test_ece=_ece(pte, yte), val_conf=float(np.maximum(pva, 1 - pva).mean()))
