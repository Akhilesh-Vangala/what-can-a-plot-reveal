"""B3: a stronger and different image benchmark (instructor review, point 4).

  python b3_strong.py <run_dir>     -> <run_dir>/b3_results.json, b3_item_scores.json

Same items and splits as B0/B1/B2, strongest severity only. Differences from B2 (ResNet-18 on 224x224 resized
images): ConvNeXt-Tiny (ImageNet-pretrained, about 2.5x the compute of ResNet-18), the plot at its native 320x240
resolution (no resizing, so no detail is lost), label smoothing, a one-cycle learning-rate schedule, up to 20 epochs
and 3 seeds with the validation-best kept. If a cell stays below AUC 0.70 here too, the "weak/blind"
label does not depend on the choice of image estimator.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

RUN = pathlib.Path(sys.argv[1])
META = json.load(open(RUN / "meta.json"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = int(os.environ.get("B3_LIMIT", "0"))  # >0: tiny local test only
EPOCHS, SEEDS = (2, 1) if LIMIT else (20, 3)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TOP = len(META["targets"]) - 1
T0 = time.time()


def groups():
    keys = ["healthy|None"] + [f"{f}|{TOP}" for f in META["failures"]]
    g = {}
    for k in keys:
        ids = list(META["groups"][k])
        if LIMIT:
            ids = ids[:: max(1, len(ids) // LIMIT)]
        g[k] = (ids, [META["groups"][k][i] for i in ids])
    return g


def load(ids, spec):
    arr = [np.asarray(Image.open(RUN / "img" / f"{i}_{spec}.png").convert("RGB")) for i in ids]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def prep(x):
    return ((x.float() / 255.0) - MEAN) / STD


def predict(net, X, idx):
    out = []
    net.eval()
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
        for b in torch.tensor(idx).split(256):
            out.append(torch.softmax(net(prep(X[b]).to(DEV)).float(), 1).cpu())
    return torch.cat(out).numpy()


def score(y, P):
    return roc_auc_score(y, P[:, 1]) if P.shape[1] == 2 else roc_auc_score(y, P, multi_class="ovr")


def fit(X, y, tr, va, n_cls, seed, patience=5):
    torch.manual_seed(seed)
    net = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
    net.classifier[2] = torch.nn.Linear(net.classifier[2].in_features, n_cls)
    net = net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.05)
    steps = EPOCHS * ((len(tr) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.1)
    Y = torch.tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    best, best_auc, wait = None, -1.0, 0
    for _ in range(EPOCHS):
        net.train()
        for b in torch.tensor(tr)[torch.randperm(len(tr), generator=g)].split(64):
            opt.zero_grad()
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = torch.nn.functional.cross_entropy(net(prep(X[b]).to(DEV)).float(), Y[b].to(DEV), label_smoothing=0.1)
            loss.backward()
            opt.step()
            sched.step()
        auc = score(y[va], predict(net, X, va))
        if auc > best_auc:
            best_auc, best, wait = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best)
    return net, best_auc


def run_task(keys, G, cache, seed):
    ids = sum((G[k][0] for k in keys), [])
    split = sum((G[k][1] for k in keys), [])
    y = np.concatenate([np.full(len(G[k][0]), c) for c, k in enumerate(keys)])
    X = torch.cat([cache[k] for k in keys])
    idx = {s: np.array([i for i, t in enumerate(split) if t == s], dtype=int) for s in ("train", "val", "test")}
    best = None
    for k in range(SEEDS):
        net, val_auc = fit(X, y, idx["train"], idx["val"], len(keys), seed + 1000 * k)
        if best is None or val_auc > best[1]:
            best = (net, val_auc)
    net, val_auc = best
    P = predict(net, X, idx["test"])
    yt = y[idx["test"]]
    prior = np.bincount(y[idx["train"]], minlength=len(keys)) / len(idx["train"])
    pred = (P / np.maximum(prior, 1e-9)).argmax(1)
    out = dict(val_auc=float(val_auc), test_auc=float(score(yt, P)), test_acc=float(accuracy_score(yt, pred)),
               n_test=int(len(yt)), test_ids=[ids[i] for i in idx["test"]], test_proba=np.round(P, 5).tolist(),
               prior=prior.tolist())
    if len(keys) > 2:
        cm = confusion_matrix(yt, pred, labels=list(range(len(keys))))
        out["confusion"] = cm.tolist()
        out["balanced_acc"] = float(np.mean(np.diag(cm) / np.maximum(cm.sum(1), 1)))
    return out


def main():
    G = groups()
    res, items = dict(model="convnext_tiny@native", level=TOP, detection=[], diagnosis=[]), {}
    strip = lambda r: {k: v for k, v in r.items() if k not in ("test_ids", "test_proba")}  # noqa: E731
    for spec in META["specs"]:
        cache = {k: load(G[k][0], spec) for k in G}
        for f in META["failures"]:
            r = run_task(["healthy|None", f"{f}|{TOP}"], G, cache, seed=7)
            items[f"det|{f}|{TOP}|{spec}"] = dict(ids=r["test_ids"], score=[p[1] for p in r["test_proba"]])
            res["detection"].append(dict(spec=spec, failure=f, level=TOP, **strip(r)))
            print(f"[{time.time() - T0:5.0f}s] detect {spec:7s} {f:18s}: AUC {r['test_auc']:.3f}", flush=True)
        r = run_task(["healthy|None"] + [f"{f}|{TOP}" for f in META["failures"]], G, cache, seed=107)
        items[f"diag|{TOP}|{spec}"] = dict(ids=r["test_ids"], proba=r["test_proba"], prior=r["prior"])
        res["diagnosis"].append(dict(spec=spec, level=TOP, classes=["healthy"] + META["failures"], **strip(r)))
        print(f"[{time.time() - T0:5.0f}s] diagnose {spec:7s}: balanced acc {r['balanced_acc']:.3f}", flush=True)
        json.dump(res, open(RUN / "b3_results.json", "w"), indent=1)
        json.dump(items, open(RUN / "b3_item_scores.json", "w"))
        del cache
    print("saved", RUN / "b3_results.json")


if __name__ == "__main__":
    main()
