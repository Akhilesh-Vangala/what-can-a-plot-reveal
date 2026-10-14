"""Controlled failure world: small models on synthetic tabular data, healthy or with a planted failure.

A run = config -> trained model -> "full state" (B0 input): per-epoch training logs plus validation/test predictions.
Healthy configs vary naturally (data difficulty, size, width, regularisation, learning rate), so null plots are
not copies of each other (gap 3.1). Failures are applied on top of a healthy config with a severity s in [0, 1].
"""
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from .seeds import rng_for

N_POOL_TRAIN, N_VAL, N_TEST = 2500, 500, 1000
EPOCHS = 100


def healthy_config(*seed_parts, epochs=EPOCHS):
    r = rng_for("config", *seed_parts)
    return dict(
        data_seed=int(r.integers(2**31)),
        split_seed=int(r.integers(2**31)),
        model_seed=int(r.integers(2**31)),
        n_features=20,
        n_informative=int(r.integers(5, 11)),
        class_sep=float(r.uniform(0.6, 1.2)),
        flip_y=float(r.uniform(0.01, 0.05)),
        n_train=int(r.integers(1500, N_POOL_TRAIN + 1)),
        hidden=int(r.choice([16, 24, 32])),
        alpha=float(10 ** r.uniform(-1, 0)),  # regularised enough that healthy runs do not overfit
        lr=float(10 ** r.uniform(-3.3, -2.7)),
        epochs=int(epochs),
        failure="healthy",
        severity=0.0,
    )


def apply_failure(cfg, failure, s):
    """Return a copy of a healthy config with `failure` planted at severity s (0 = healthy, 1 = strongest)."""
    cfg = dict(cfg, failure=failure, severity=float(s))
    if failure == "healthy" or s == 0:
        return cfg
    if failure == "overfitting":
        # capacity up, data down, regularisation off — one knob for "capacity relative to data"
        cfg["n_train"] = max(100, int(round(cfg["n_train"] * (1 - 0.85 * s))))
        cfg["hidden"] = int(round(cfg["hidden"] * (1 + 3 * s)))
        cfg["alpha"] = cfg["alpha"] * 10 ** (-4 * s)
        return cfg
    raise NotImplementedError(failure)


def _ece(p, y, bins=10):
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = (p >= 0.5) == y
    idx = np.minimum((conf - 0.5) * 2 * bins, bins - 1).astype(int)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def train_run(cfg):
    """Train one model and return its full state (training logs + held-out predictions)."""
    X, y = make_classification(
        n_samples=N_POOL_TRAIN + N_VAL + N_TEST, n_features=cfg["n_features"],
        n_informative=cfg["n_informative"], n_redundant=2, n_clusters_per_class=2,
        class_sep=cfg["class_sep"], flip_y=cfg["flip_y"], random_state=cfg["data_seed"])
    perm = np.random.default_rng(cfg["split_seed"]).permutation(len(y))
    te, va = perm[:N_TEST], perm[N_TEST:N_TEST + N_VAL]
    tr = perm[N_TEST + N_VAL:][:cfg["n_train"]]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xva, Xte = (X[tr] - mu) / sd, (X[va] - mu) / sd, (X[te] - mu) / sd
    ytr, yva, yte = y[tr], y[va], y[te]

    clf = MLPClassifier(hidden_layer_sizes=(cfg["hidden"], cfg["hidden"]), alpha=cfg["alpha"],
                        learning_rate_init=cfg["lr"], batch_size=min(128, len(tr)),
                        random_state=cfg["model_seed"])
    logs = {k: np.empty(cfg["epochs"]) for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for e in range(cfg["epochs"]):
            clf.partial_fit(Xtr, ytr, classes=[0, 1])
            ptr, pva = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xva)[:, 1]
            logs["train_loss"][e] = log_loss(ytr, ptr, labels=[0, 1])
            logs["val_loss"][e] = log_loss(yva, pva, labels=[0, 1])
            logs["train_acc"][e] = ((ptr >= 0.5) == ytr).mean()
            logs["val_acc"][e] = ((pva >= 0.5) == yva).mean()
    pte = clf.predict_proba(Xte)[:, 1]
    return dict(
        **logs,
        test_loss=float(log_loss(yte, pte, labels=[0, 1])),
        test_acc=float(((pte >= 0.5) == yte).mean()),
        val_conf=float(np.maximum(pva, 1 - pva).mean()),
        val_ece=_ece(pva, yva),
    )


def b0_features(state):
    """Full-state features for the B0 benchmark: the complete training logs (a superset of anything a
    learning-curve plot draws) plus held-out summaries the plots never show."""
    curves = [state[c] for c in ("train_loss", "val_loss", "train_acc", "val_acc")]
    scalars = [state["test_loss"], state["test_acc"], state["val_conf"], state["val_ece"],
               state["val_loss"][-1] - state["train_loss"][-1], state["val_loss"][-1] - state["val_loss"].min(),
               state["train_acc"][-1] - state["val_acc"][-1]]
    return np.concatenate(curves + [np.array(scalars)])
