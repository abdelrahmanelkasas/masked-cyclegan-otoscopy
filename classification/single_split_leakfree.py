"""
LEAKAGE-FREE SINGLE GROUPED SPLIT  +  HELD-OUT GRAD-CAM   (Reviewer 1, points 1, 2, 6)
======================================================================================
Fixes two remaining weaknesses of the single-split analyses:

  (a) In the old single-split runs, grouped_split() was called separately on each
      image set (original / whole-image / masked), so the test sets differed
      (177 / 182 / 181 images). Here the split is computed ONCE on the original
      images and the SAME train / val / test images are used for every condition.
  (b) The old dataset-level CycleGAN had seen the test images (without labels).
      Here a fresh CycleGAN is trained on the TRAINING images of this split only;
      validation and test images never touch CycleGAN training.

Then, on the identical 177 test images:
  * ResNet50 is trained for Original, Whole-image CycleGAN and Masked conditions
    (40 epochs, checkpoint chosen by validation macro-F1, same recipe as before);
  * accuracy, macro-F1, per-class sensitivity/specificity, confusion matrices;
  * paired exact McNemar tests (Masked vs Original, Whole vs Original);
  * Grad-CAM glare-attention ratio for the Original-trained model on the original
    vs the leakage-free deglared test images (all test images + top-12 by glare).
Per-image predictions are saved so that cluster-bootstrap CIs can be computed.

USAGE (GPU, roughly 2-3 h on an RTX 3060):
    python classification/single_split_leakfree.py
    # quick smoke test:
    python classification/single_split_leakfree.py --gan_epochs 2 --cls_epochs 1 --max_per_domain 20

OUTPUT: results/single_split_leakfree.json  and  results/single_split_leakfree_perimage.csv
Send these two files back to update Tables 3-4, Figure 5 and the Grad-CAM section.
"""
import os, sys, argparse, json, csv, random, shutil, time
import numpy as np, cv2, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHILE_DIR, RESULTS, CHILE_CLASSES, SEED, IMG_SIZE
from classification.dataset import scan, EarDataset
from classification.grouped_split import grouped_split
from classification.run_perfold_cyclegan import build_fold_domains, train_cyclegan_fold, deglare_items
from classification.gradcam_v2 import cam_of
from classification.train_classifier import build_model, evaluate
from preprocessing.glare_utils import glare_ratio, glare_mask


@torch.no_grad()
def whole_items(items, G_AB, dev, img_size, out_dir):
    """Whole-image CycleGAN translation (no mask), resized back to original size."""
    os.makedirs(out_dir, exist_ok=True)
    pre = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    out = []
    for k, (p, lab, t) in enumerate(items):
        img = Image.open(p).convert("RGB"); W, H = img.size
        y = G_AB(pre(img).unsqueeze(0).to(dev))[0].cpu() * 0.5 + 0.5
        g = T.ToPILImage()(y.clamp(0, 1)).resize((W, H), Image.BICUBIC)
        q = os.path.join(out_dir, f"w_{k:05d}.png"); g.save(q); out.append((q, lab, t))
    return out


def train_select(tr, va, dev, epochs, batch, lr, workers):
    model = build_model(len(CHILE_CLASSES), dev)
    counts = np.bincount([it[1] for it in tr], minlength=len(CHILE_CLASSES))
    w = torch.tensor((counts.sum() / (counts + 1e-6)) / len(counts), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    dl_tr = DataLoader(EarDataset(tr, True), batch, shuffle=True, num_workers=workers)
    dl_va = DataLoader(EarDataset(va, False), batch, num_workers=workers)
    best_f1, best = -1, None
    for ep in range(epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        ys, ps = evaluate(model, dl_va, dev)
        f1 = f1_score(ys, ps, average="macro")
        if f1 > best_f1:
            best_f1, best = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    return model


def per_class(cm):
    cm = np.array(cm); n = cm.sum(); out = []
    for i in range(len(cm)):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = n - tp - fn - fp
        out.append({"class": CHILE_CLASSES[i], "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp)})
    return out


def mcnemar_exact(y, a, b):
    ca, cb = (np.array(a) == y), (np.array(b) == y)
    n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    p = 1.0 if n01 + n10 == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n01 + n10, 0.5)))
    return {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "p_exact": p}


def gradcam_ratios(model, items_o, items_d, dev):
    pre = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    rows = []
    for (po, _, _), (pd, _, _) in zip(items_o, items_d):
        o = Image.open(po).convert("RGB"); d = Image.open(pd).convert("RGB")
        gm = glare_mask(cv2.resize(cv2.cvtColor(np.array(o), cv2.COLOR_RGB2BGR), (224, 224))); ins = gm > 0
        if ins.sum() < 20:
            continue
        co = cam_of(model, pre(o).unsqueeze(0).to(dev), dev); cd = cam_of(model, pre(d).unsqueeze(0).to(dev), dev)
        rows.append({"image": os.path.relpath(po, CHILE_DIR), "glare": glare_ratio(cv2.imread(po)),
                     "ratio_orig": float(co[ins].mean() / (co[~ins].mean() + 1e-8)),
                     "ratio_deg": float(cd[ins].mean() / (cd[~ins].mean() + 1e-8))})
    return rows


def summ(rows):
    ro = np.array([r["ratio_orig"] for r in rows]); rd = np.array([r["ratio_deg"] for r in rows])
    return {"n": len(rows), "orig_median": float(np.median(ro)), "orig_iqr": np.percentile(ro, [25, 75]).tolist(),
            "deg_median": float(np.median(rd)), "deg_iqr": np.percentile(rd, [25, 75]).tolist(),
            "orig_mean": float(ro.mean()), "deg_mean": float(rd.mean()),
            "n_decreased": int((rd < ro).sum()), "n_orig_gt1": int((ro > 1).sum()), "n_deg_gt1": int((rd > 1).sum()),
            "wilcoxon_p": float(stats.wilcoxon(ro, rd).pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan_epochs", type=int, default=60)
    ap.add_argument("--gan_decay_start", type=int, default=30)
    ap.add_argument("--cls_epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_per_domain", type=int, default=None)
    a = ap.parse_args()
    a.gan_epochs = max(a.gan_epochs, 2)                       # LR-decay schedule needs >= 2 epochs
    a.gan_decay_start = min(a.gan_decay_start, a.gan_epochs - 1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    work = os.path.join(RESULTS, "single_split_work"); os.makedirs(work, exist_ok=True)

    items = scan(CHILE_DIR)
    tr, va, te = grouped_split(items, seed=SEED)            # ONE split, original images
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)}  (expected 567/136/177)")

    # CycleGAN on TRAINING images only
    dom = os.path.join(work, "domains")
    nA, nB = build_fold_domains(tr, dom, IMG_SIZE, max_per_domain=a.max_per_domain, seed=SEED)
    print(f"[cyclegan] domains A={nA} B={nB} (train images only)")
    G = train_cyclegan_fold(dom, dev, IMG_SIZE, a.gan_epochs, a.gan_decay_start, workers=a.workers,
                            log_prefix="[single]", loss_csv=os.path.join(RESULTS, "single_split_cyclegan_loss.csv"))
    torch.save(G.state_dict(), os.path.join(work, "G_AB_single_split_trainonly.pth"))

    sets = {"original": (tr, va, te)}
    sets["masked"] = tuple(deglare_items(s, G, dev, IMG_SIZE, os.path.join(work, f"masked_{n}"))
                           for s, n in zip((tr, va, te), ("tr", "va", "te")))
    sets["whole"] = tuple(whole_items(s, G, dev, IMG_SIZE, os.path.join(work, f"whole_{n}"))
                          for s, n in zip((tr, va, te), ("tr", "va", "te")))
    del G
    y = np.array([it[1] for it in te]); res = {"n_test": len(te), "conditions": {}}; preds = {}; models = {}
    for cond, (s_tr, s_va, s_te) in sets.items():
        torch.manual_seed(SEED)
        m = train_select(s_tr, s_va, dev, a.cls_epochs, a.batch, a.lr, a.workers)
        ys, ps = evaluate(m, DataLoader(EarDataset(s_te, False), a.batch, num_workers=a.workers), dev)
        assert np.array_equal(np.array(ys), y)
        cm = confusion_matrix(y, ps, labels=list(range(len(CHILE_CLASSES))))
        res["conditions"][cond] = {"accuracy": accuracy_score(y, ps), "macro_f1": f1_score(y, ps, average="macro"),
                                   "confusion_matrix": cm.tolist(), "per_class": per_class(cm)}
        preds[cond] = list(map(int, ps)); print(f"[{cond}] acc={res['conditions'][cond]['accuracy']:.4f}")
        if cond == "original":
            models["original"] = m
        else:
            del m
    res["mcnemar_masked_vs_original"] = mcnemar_exact(y, preds["masked"], preds["original"])
    res["mcnemar_whole_vs_original"] = mcnemar_exact(y, preds["whole"], preds["original"])
    res["mcnemar_masked_vs_whole"] = mcnemar_exact(y, preds["masked"], preds["whole"])

    # Grad-CAM: original-trained model (this split), original vs leakage-free masked test images
    rows = gradcam_ratios(models["original"], te, sets["masked"][2], dev)
    rows_sorted = sorted(rows, key=lambda r: -r["glare"])
    res["gradcam_all_test"] = summ(rows); res["gradcam_top12"] = summ(rows_sorted[:12])
    res["config"] = vars(a)
    json.dump(res, open(os.path.join(RESULTS, "single_split_leakfree.json"), "w"), indent=2, default=float)
    with open(os.path.join(RESULTS, "single_split_leakfree_perimage.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "label"] + [f"pred_{c}" for c in preds])
        for i, (p, lab, _) in enumerate(te):
            w.writerow([os.path.relpath(p, CHILE_DIR), lab] + [preds[c][i] for c in preds])
    with open(os.path.join(RESULTS, "single_split_leakfree_gradcam.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: v for k, v in res.items() if k != "conditions"}, indent=2, default=float))
    print("[saved] results/single_split_leakfree.json")


if __name__ == "__main__":
    main()
