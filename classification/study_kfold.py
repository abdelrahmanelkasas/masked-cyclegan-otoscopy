"""
Per-dataset, multi-classifier, leakage-free K-fold study (ENGLISH outputs).

For ONE dataset (chile=Dataset 1, ohio=Dataset 2) and SEVERAL classifiers
(CNNs: ResNet50, EfficientNet-B0 ; Transformer: ViT-B/16), runs grouped
(near-duplicate-safe) K-fold and compares Original vs Masked-deglared vs
Ensemble for each classifier. Saves JSON + an English comparison figure.

Prereqomg: dataset downloaded; per-dataset masked set produced
(cyclegan/apply_deglare.py --mask --src <raw> --dst <masked> --ckpt <G_AB_x.pth>).

Usage:
    python classification/study_kfold.py --dataset chile --models resnet50,efficientnet_b0,vit_b_16 --folds 5 --epochs 30
"""
import os, sys, argparse, json
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torchvision.models as M
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ds_paths, RESULTS, SEED
from classification.dataset import scan, EarDataset, make_tf
from classification.grouped_split import grouped_kfold

def build_model(name, n_cls, dev):
    name = name.lower()
    if name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
    elif name == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    elif name == "vit_b_16":
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_cls)
    else:
        raise ValueError(f"Bilinmeyen model: {name}")
    return m.to(dev)

def map_masked(items, raw_dir, masked_dir):
    out = []
    for p, lab, t in items:
        rel = os.path.relpath(p, raw_dir)
        mp = os.path.join(masked_dir, os.path.splitext(rel)[0] + ".png")
        if os.path.exists(mp):
            out.append((mp, lab, t))
        else:
            out.append((p, lab, t))   # maskeli yoksa orijinali kullan (guvenli)
    return out

def train_one(model_name, items, n_cls, dev, epochs, batch, lr, workers):
    model = build_model(model_name, n_cls, dev)
    counts = np.bincount([it[1] for it in items], minlength=n_cls)
    w = torch.tensor((counts.sum() / (counts + 1e-6)) / n_cls, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    dl = DataLoader(EarDataset(items, True), batch, shuffle=True, num_workers=workers)
    model.train()
    for _ in range(epochs):
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
    model.eval()
    return model

@torch.no_grad()
def probs(model, items, dev):
    tf = make_tf(False)
    out = []
    for p, _, _ in items:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        pr = torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, [3])), 1)
        out.append((pr / 2)[0].cpu().numpy())
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["chile", "ohio"], required=True)
    ap.add_argument("--models", default="resnet50,efficientnet_b0,vit_b_16")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dp = ds_paths(a.dataset); classes = dp["classes"]; n_cls = len(classes)
    print(f"[device] {dev} | dataset={a.dataset} ({dp['label']}) | classes={classes}")

    items = scan(dp["raw"], classes)
    if not items:
        print("[ERROR] No images found at", dp["raw"]); sys.exit(1)
    if not os.path.isdir(dp["masked"]):
        print("[WARN] Masked set not found:", dp["masked"], "-> run apply_deglare --mask first.")
    folds = grouped_kfold(items, k=a.folds, seed=SEED)
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    results = {}
    for mdl in models:
        acc = {"original": [], "masked": [], "ensemble": []}
        f1 = {"original": [], "masked": [], "ensemble": []}
        for fi, (tr, te) in enumerate(folds):
            tr_m = map_masked(tr, dp["raw"], dp["masked"])
            te_m = map_masked(te, dp["raw"], dp["masked"])
            print(f"\n[{mdl}] Fold {fi+1}/{a.folds}  train={len(tr)} test={len(te)}")
            m_o = train_one(mdl, tr, n_cls, dev, a.epochs, a.batch, a.lr, a.workers)
            m_m = train_one(mdl, tr_m, n_cls, dev, a.epochs, a.batch, a.lr, a.workers)
            y = np.array([it[1] for it in te])
            po = probs(m_o, te, dev); pm = probs(m_m, te_m, dev); pe = (po + pm) / 2
            for key, pred in [("original", po.argmax(1)), ("masked", pm.argmax(1)),
                              ("ensemble", pe.argmax(1))]:
                acc[key].append(accuracy_score(y, pred))
                f1[key].append(f1_score(y, pred, average="macro"))
            print(f"   acc  orig={acc['original'][-1]:.4f}  masked={acc['masked'][-1]:.4f}  ens={acc['ensemble'][-1]:.4f}")
            del m_o, m_m
            if dev == "cuda":
                torch.cuda.empty_cache()
        results[mdl] = {k: {"acc_mean": float(np.mean(acc[k])), "acc_std": float(np.std(acc[k])),
                            "f1_mean": float(np.mean(f1[k])), "f1_std": float(np.std(f1[k])),
                            "acc_folds": acc[k]} for k in acc}
        print(f"\n=== {mdl} ({a.dataset}) ===")
        for k in ["original", "masked", "ensemble"]:
            r = results[mdl][k]
            print(f"  {k:9s}: acc {r['acc_mean']:.4f} +/- {r['acc_std']:.4f} | f1 {r['f1_mean']:.4f} +/- {r['f1_std']:.4f}")

    out_json = os.path.join(RESULTS, f"study_{a.dataset}.json")
    with open(out_json, "w") as f:
        json.dump({"dataset": a.dataset, "label": dp["label"], "models": models, "results": results}, f, indent=2)

    # ENGLISH grouped bar chart
    conds = ["original", "masked", "ensemble"]
    colors = {"original": "888888", "masked": "2CA58D", "ensemble": "2E86C1"}
    x = np.arange(len(models)); w = 0.25
    plt.figure(figsize=(max(7, 2.2*len(models)+3), 5))
    for j, c in enumerate(conds):
        means = [results[m][c]["acc_mean"] for m in models]
        stds = [results[m][c]["acc_std"] for m in models]
        b = plt.bar(x + (j-1)*w, means, w, yerr=stds, capsize=5,
                    label=c.capitalize(), color="#"+colors[c])
        for bi, mv in zip(b, means):
            plt.text(bi.get_x()+bi.get_width()/2, mv+0.004, f"{mv:.3f}", ha="center", fontsize=8)
    plt.xticks(x, [m.replace("_", "-") for m in models])
    plt.ylim(0.80, 1.01); plt.ylabel("Accuracy (mean ± std)")
    plt.title(f"{dp['label']} — {a.folds}-Fold Grouped CV: Original vs Masked vs Ensemble")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    out_png = os.path.join(RESULTS, f"study_{a.dataset}.png")
    plt.savefig(out_png, dpi=600)
    print("\n[saved]", out_json, "and", out_png)

if __name__ == "__main__":
    main()
