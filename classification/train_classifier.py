"""
Kulak zari hastaligi siniflandirmasi (ResNet50, transfer learning).
Ayni kodu hem orijinal hem parlama-giderilmis veriyle calistirip karsilastirin.

Kullanim:
    python classification/train_classifier.py --data original --tag orig
    python classification/train_classifier.py --data deglared --tag deglared
"""
import os, sys, argparse, json, random
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torchvision.models as M
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (CHILE_DIR, DEGLARED_DIR, DEGLARED_MASKED_DIR,
                            RESULTS, CHECKPOINTS, CHILE_CLASSES, SEED)
from classification.dataset import scan, EarDataset

def build_model(n_cls, dev):
    m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m.to(dev)

@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); ys, ps = [], []
    for x, y in dl:
        out = model(x.to(dev)).argmax(1).cpu().numpy()
        ps += out.tolist(); ys += y.numpy().tolist()
    return ys, ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["original", "deglared", "masked"], default="original")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--split", choices=["predefined", "grouped"], default="predefined",
                    help="predefined=repo Testing klasoru + rastgele val; "
                         "grouped=near-duplicate'a duyarli, sizintisiz bolme")
    a = ap.parse_args()
    tag = a.tag or a.data
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[cihaz]", dev)

    root = {"original": CHILE_DIR, "deglared": DEGLARED_DIR,
            "masked": DEGLARED_MASKED_DIR}[a.data]
    items = scan(root)
    if not items:
        print("[HATA] Veri bulunamadi:", root); sys.exit(1)
    if a.split == "grouped":
        from classification.grouped_split import grouped_split
        tr, val, test = grouped_split(items, seed=SEED)
        print(f"[bolum-GRUP] train={len(tr)} val={len(val)} test={len(test)} "
              f"(near-duplicate kumeleri bolunmedi -> sizinti yok)")
    else:
        test = [it for it in items if it[2]]
        pool = [it for it in items if not it[2]]
        if not test:   # onceden bolunmus test yoksa kendimiz ayiralim
            labs = [it[1] for it in pool]
            pool, test = train_test_split(pool, test_size=0.2, stratify=labs, random_state=SEED)
        tr_labs = [it[1] for it in pool]
        tr, val = train_test_split(pool, test_size=0.15, stratify=tr_labs, random_state=SEED)
        print(f"[bolum] train={len(tr)} val={len(val)} test={len(test)}")

    dl_tr = DataLoader(EarDataset(tr, True), a.batch, shuffle=True, num_workers=a.workers)
    dl_va = DataLoader(EarDataset(val, False), a.batch, num_workers=a.workers)
    dl_te = DataLoader(EarDataset(test, False), a.batch, num_workers=a.workers)

    model = build_model(len(CHILE_CLASSES), dev)
    # sinif dengesizligi icin agirlik
    counts = np.bincount([it[1] for it in tr], minlength=len(CHILE_CLASSES))
    w = torch.tensor((counts.sum() / (counts + 1e-6)) / len(counts), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    best_f1, best = -1, None
    curve_csv = os.path.join(RESULTS, f"clsf_{tag}_curve.csv")
    with open(curve_csv, "w") as f:
        f.write("epoch,train_loss,val_acc,val_f1\n")
    for ep in range(a.epochs):
        model.train()
        run_loss, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item(); nb += 1
        sch.step()
        ys, ps = evaluate(model, dl_va, dev)
        acc = accuracy_score(ys, ps); f1 = f1_score(ys, ps, average="macro")
        print(f"ep{ep} val_acc={acc:.3f} val_f1={f1:.3f}")
        with open(curve_csv, "a") as f:
            f.write(f"{ep},{run_loss/max(nb,1):.5f},{acc:.5f},{f1:.5f}\n")
        if f1 > best_f1:
            best_f1 = f1; best = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    torch.save(best, os.path.join(CHECKPOINTS, f"classifier_{tag}.pth"))

    ys, ps = evaluate(model, dl_te, dev)
    res = {
        "tag": tag, "data": a.data,
        "test_accuracy": accuracy_score(ys, ps),
        "test_f1_macro": f1_score(ys, ps, average="macro"),
        "confusion_matrix": confusion_matrix(ys, ps).tolist(),
        "report": classification_report(ys, ps, target_names=CHILE_CLASSES, zero_division=0, output_dict=True),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"metrics_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n[TEST] acc={res['test_accuracy']:.4f}  f1_macro={res['test_f1_macro']:.4f}")
    print("[kaydedildi]", os.path.join(RESULTS, f"metrics_{tag}.json"))

if __name__ == "__main__":
    main()
