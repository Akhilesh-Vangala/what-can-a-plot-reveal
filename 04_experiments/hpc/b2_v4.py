"""B2 (image-level) benchmark, v4: ImageNet-pretrained ResNet-18 fine-tuned on the rendered plots, same items and
splits as the B0/B1 run (meta.json). Saves per-item test scores (b2_item_scores.json) for the over-claiming measure.

  python b2_v4.py <run_dir>     -> <run_dir>/b2_results.json, b2_item_scores.json
Detection: healthy vs failure per failure × level × plot (1 seed). Diagnosis: (1 + #failures)-way per level × plot,
2 seeds with the validation-best kept, + 50%-data saturation point; balanced accuracy with prior-corrected decisions.
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
LIMIT = int(os.environ.get("B2_LIMIT", "0"))  # >0: tiny local test only
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
T0 = time.time()


def groups():
    g = {}
    for key, members in META["groups"].items():
        ids = list(members)
        if LIMIT:
            ids = ids[:: max(1, len(ids) // LIMIT)]
        g[key] = (ids, [members[i] for i in ids])
    return g


def load(ids, spec):
    arr = [np.asarray(Image.open(RUN / "img" / f"{i}_{spec}.png").convert("RGB").resize((224, 224), Image.BILINEAR))
           for i in ids]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def prep(x):
    return ((x.float() / 255.0) - MEAN) / STD


def predict(net, X, idx):
    out = []
    net.eval()
    with torch.no_grad():
        for b in torch.tensor(idx).split(256):
            out.append(torch.softmax(net(prep(X[b]).to(DEV)).float(), 1).cpu())
    return torch.cat(out).numpy()


def score(y, P):
    return roc_auc_score(y, P[:, 1]) if P.shape[1] == 2 else roc_auc_score(y, P, multi_class="ovr")


def fit(X, y, tr, va, n_cls, seed, epochs=2 if LIMIT else 15, patience=4):
    torch.manual_seed(seed)
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Linear(net.fc.in_features, n_cls)
    net = net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    Y = torch.tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    best, best_auc, wait = None, -1.0, 0
    for _ in range(epochs):
        net.train()
        for b in torch.tensor(tr)[torch.randperm(len(tr), generator=g)].split(64):
            opt.zero_grad()
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = torch.nn.functional.cross_entropy(net(prep(X[b]).to(DEV)).float(), Y[b].to(DEV))
            loss.backward()
            opt.step()
        auc = score(y[va], predict(net, X, va))
        if auc > best_auc:
            best_auc, best, wait = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best)
    return net, best_auc


def run_task(keys, G, cache, spec, seed, sat=False, seeds=1):
    ids = sum((G[k][0] for k in keys), [])
    split = sum((G[k][1] for k in keys), [])
    y = np.concatenate([np.full(len(G[k][0]), c) for c, k in enumerate(keys)])
    X = torch.cat([cache[k] for k in keys])
    idx = {s: np.array([i for i, t in enumerate(split) if t == s], dtype=int) for s in ("train", "val", "test")}
    best = None
    for k in range(seeds):
        net, val_auc = fit(X, y, idx["train"], idx["val"], len(keys), seed + 1000 * k)
        if best is None or val_auc > best[1]:
            best = (net, val_auc, seed + 1000 * k)
    net, val_auc, used_seed = best
    P = predict(net, X, idx["test"])
    yt = y[idx["test"]]
    test_ids = [ids[i] for i in idx["test"]]
    prior = np.bincount(y[idx["train"]], minlength=len(keys)) / len(idx["train"])
    pred = (P / np.maximum(prior, 1e-9)).argmax(1)
    out = dict(val_auc=float(val_auc), test_auc=float(score(yt, P)), test_acc=float(accuracy_score(yt, pred)),
               n_test=int(len(yt)), seed=int(used_seed), test_ids=test_ids, test_proba=np.round(P, 5).tolist(),
               prior=prior.tolist())
    if len(keys) > 2:
        cm = confusion_matrix(yt, pred, labels=list(range(len(keys))))
        out["confusion"] = cm.tolist()
        out["balanced_acc"] = float(np.mean(np.diag(cm) / np.maximum(cm.sum(1), 1)))
    if sat:
        r = np.random.default_rng(seed)
        half = np.concatenate([r.choice(idx["train"][y[idx["train"]] == c], (y[idx["train"]] == c).sum() // 2,
                                        replace=False) for c in range(len(keys))])
        n2, _ = fit(X, y, half, idx["val"], len(keys), seed)
        out["saturation"] = [(0.5, float(score(yt, predict(n2, X, idx["test"])))), (1.0, out["test_auc"])]
    return out


def strip(r):
    return {k: v for k, v in r.items() if k not in ("test_ids", "test_proba")}


def main():
    G = groups()
    res, items = dict(detection=[], diagnosis=[]), {}
    for spec in META["specs"]:
        cache = {k: load(G[k][0], spec) for k in G}
        for f in META["failures"]:
            for j in range(len(META["targets"])):
                r = run_task(["healthy|None", f"{f}|{j}"], G, cache, spec, seed=j)
                items[f"det|{f}|{j}|{spec}"] = dict(ids=r["test_ids"], score=[p[1] for p in r["test_proba"]])
                res["detection"].append(dict(spec=spec, failure=f, level=j, target=META["targets"][j], **strip(r)))
                print(f"[{time.time() - T0:5.0f}s] detect {spec:7s} {f:16s} L{j}: AUC {r['test_auc']:.3f}", flush=True)
        for j in range(len(META["targets"])):
            keys = ["healthy|None"] + [f"{f}|{j}" for f in META["failures"]]
            r = run_task(keys, G, cache, spec, seed=100 + j, sat=True, seeds=2)
            items[f"diag|{j}|{spec}"] = dict(ids=r["test_ids"], proba=r["test_proba"], prior=r["prior"])
            res["diagnosis"].append(dict(spec=spec, level=j, target=META["targets"][j],
                                         classes=["healthy"] + META["failures"], **strip(r)))
            print(f"[{time.time() - T0:5.0f}s] diagnose {spec:7s} L{j}: balanced acc {r['balanced_acc']:.3f} "
                  f"AUC {r['test_auc']:.3f}", flush=True)
        json.dump(res, open(RUN / "b2_results.json", "w"), indent=1)
        json.dump(items, open(RUN / "b2_item_scores.json", "w"))
    print("saved", RUN / "b2_results.json")


if __name__ == "__main__":
    main()
