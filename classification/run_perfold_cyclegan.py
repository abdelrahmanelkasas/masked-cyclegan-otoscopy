"""
LEAKAGE-FREE, PER-FOLD CycleGAN pipeline  (Reviewer 1, Point 2)
================================================================
This addresses the reviewer's concern that, in the original study, the CycleGAN
(the learnable pre-processing) was trained ONCE on ALL images of a dataset, so
test-fold images could in principle influence the pre-processing stage.

Here, EVERYTHING that touches the pixels is rebuilt inside each fold using ONLY
the training-fold images:

  For every grouped, near-duplicate-safe fold k:
    1. Build Domain A (high glare) / Domain B (low glare) from TRAIN-fold images ONLY.
    2. Train a FRESH CycleGAN (G_AB) on those domains.
    3. Apply masked/targeted deglaring with THIS fold's G_AB to all fold images
       (train + test) -> per-fold "masked" set.
    4. Train the classifier on Original(train) and on Masked(train);
       evaluate on the held-out test fold (Original / Masked / Ensemble).
  Aggregate over folds -> mean +/- SD, logit 95% CI, Wilcoxon (Original vs Masked).

The test fold never contributes to domain construction, CycleGAN training, or
classifier training, so the whole pipeline is leakage-free end to end.

USAGE (on your GPU):
    python classification/run_perfold_cyclegan.py --dataset chile \
        --models resnet50,efficientnet_b0,vit_b_16 \
        --folds 5 --gan_epochs 60 --cls_epochs 30

    # quick smoke test (tiny, just to check it runs end-to-end):
    python classification/run_perfold_cyclegan.py --dataset chile \
        --models resnet50 --folds 2 --gan_epochs 2 --cls_epochs 2 --max_per_domain 40

Output:
    results/study_<dataset>_perfold.json   (same schema as study_kfold.py + CIs)

Notes
-----
* Runtime is dominated by CycleGAN training x folds. With gan_epochs=60 and 5
  folds, budget a few hours on an RTX 3060. Lower --gan_epochs for a faster
  (slightly noisier) estimate; the point is the leakage-free protocol, not a
  new SOTA number.
* Everything is written under a temporary working dir (--work_dir, default
  results/perfold_work/<dataset>) and can be deleted afterwards.
"""
import os, sys, argparse, json, shutil, time, itertools, random
import numpy as np
import cv2
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ds_paths, RESULTS, SEED, IMG_SIZE
from classification.dataset import scan, make_tf, EarDataset
from classification.grouped_split import grouped_kfold
# reuse the SAME classifier training/eval used in the published study
from classification.study_kfold import build_model, train_one, probs
from cyclegan.models import Generator, Discriminator, init_weights
from cyclegan.datasets import UnpairedDataset
from cyclegan.utils import ReplayBuffer, LambdaLR
from preprocessing.glare_utils import glare_ratio, blend_deglare

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
#  1. Build per-fold CycleGAN domains from TRAIN-fold images only
# --------------------------------------------------------------------------- #
def build_fold_domains(train_items, out_dir, img_size, high_pct=0.40, low_pct=0.40,
                       max_per_domain=None, seed=SEED):
    """Split TRAIN-fold images into Domain A (most glare) / B (least glare) and
    write them to out_dir/trainA, out_dir/trainB (resized). Returns (nA, nB)."""
    rng = random.Random(seed)
    scored = []
    for p, _, _ in train_items:
        im = cv2.imread(p)
        if im is None:
            continue
        scored.append((glare_ratio(im), p))
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    low = [p for _, p in scored[:int(n * low_pct)]]                 # least glare  -> B
    high = [p for _, p in scored[int(n * (1 - high_pct)):]]         # most glare   -> A
    rng.shuffle(low); rng.shuffle(high)
    if max_per_domain:
        low = low[:max_per_domain]; high = high[:max_per_domain]

    for sub in ("trainA", "trainB"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    def dump(files, name):
        for k, src in enumerate(files):
            im = cv2.imread(src)
            if im is None:
                continue
            im = cv2.resize(im, (img_size, img_size))
            cv2.imwrite(os.path.join(out_dir, "train" + name, f"{name}_{k:05d}.png"), im)

    dump(high, "A"); dump(low, "B")
    return len(high), len(low)


# --------------------------------------------------------------------------- #
#  2. Train a FRESH CycleGAN on this fold's domains (compact version of
#     cyclegan/train.py; returns G_AB in eval mode)
# --------------------------------------------------------------------------- #
def train_cyclegan_fold(domain_dir, dev, img_size, epochs, decay_start,
                        lr=2e-4, batch=1, lambda_cyc=10.0, lambda_id=5.0,
                        workers=2, log_prefix="", loss_csv=None):
    G_AB = init_weights(Generator()).to(dev)   # glare -> clean
    G_BA = init_weights(Generator()).to(dev)   # clean -> glare
    D_A = init_weights(Discriminator()).to(dev)
    D_B = init_weights(Discriminator()).to(dev)
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                             lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                             lr=lr, betas=(0.5, 0.999))
    sched = LambdaLR(epochs, min(decay_start, max(1, epochs - 1)))
    sG = torch.optim.lr_scheduler.LambdaLR(opt_G, sched.step)
    sD = torch.optim.lr_scheduler.LambdaLR(opt_D, sched.step)

    ds = UnpairedDataset(domain_dir, "train", img_size)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True)
    bufA, bufB = ReplayBuffer(), ReplayBuffer()
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    if loss_csv:
        with open(loss_csv, "w") as f:
            f.write("epoch,G,D,cyc,identity,gan,lr,seconds\n")

    for ep in range(epochs):
        t0 = time.time()
        acc = {"G": 0.0, "D": 0.0, "cyc": 0.0, "id": 0.0, "gan": 0.0, "n": 0}
        for batch_data in dl:
            real_A = batch_data["A"].to(dev); real_B = batch_data["B"].to(dev)
            valid = torch.ones((real_A.size(0), 1, 30, 30), device=dev)
            fake = torch.zeros_like(valid)
            # Generators
            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss_id = (l1(G_BA(real_A), real_A) + l1(G_AB(real_B), real_B)) * lambda_id
                fake_B = G_AB(real_A); fake_A = G_BA(real_B)
                loss_gan = mse(D_B(fake_B), valid) + mse(D_A(fake_A), valid)
                loss_cyc = (l1(G_BA(fake_B), real_A) + l1(G_AB(fake_A), real_B)) * lambda_cyc
                loss_G = loss_gan + loss_cyc + loss_id
            scaler.scale(loss_G).backward(); scaler.step(opt_G)
            # Discriminators
            opt_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                fA = bufA.push_pop(fake_A.detach()); fB = bufB.push_pop(fake_B.detach())
                loss_D = (mse(D_A(real_A), valid) + mse(D_A(fA), fake) +
                          mse(D_B(real_B), valid) + mse(D_B(fB), fake)) * 0.5
            scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()
            acc["G"] += loss_G.item(); acc["D"] += loss_D.item(); acc["cyc"] += loss_cyc.item()
            acc["id"] += loss_id.item(); acc["gan"] += loss_gan.item(); acc["n"] += 1
        sG.step(); sD.step()
        nb = max(acc["n"], 1); dt = time.time() - t0
        if loss_csv:
            with open(loss_csv, "a") as f:
                f.write(f"{ep},{acc['G']/nb:.5f},{acc['D']/nb:.5f},{acc['cyc']/nb:.5f},"
                        f"{acc['id']/nb:.5f},{acc['gan']/nb:.5f},"
                        f"{opt_G.param_groups[0]['lr']:.6f},{dt:.0f}\n")
        print(f"   {log_prefix} cyclegan ep {ep+1}/{epochs}  "
              f"G={acc['G']/nb:.3f} D={acc['D']/nb:.3f} cyc={acc['cyc']/nb:.3f}  ({dt:.0f}s)")

    G_AB.eval()
    for m in (G_BA, D_A, D_B):
        del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return G_AB


# --------------------------------------------------------------------------- #
#  3. Apply THIS fold's G_AB (masked/targeted) to a list of items
# --------------------------------------------------------------------------- #
@torch.no_grad()
def deglare_items(items, G_AB, dev, img_size, out_dir, skip_thresh=0.005):
    """Masked deglaring (blend_deglare): GAN output only inside glare region,
    tissue kept from original. Returns new item list pointing to masked pngs."""
    os.makedirs(out_dir, exist_ok=True)
    pre = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(),
                     T.Normalize([0.5] * 3, [0.5] * 3)])
    to_img = T.ToPILImage()
    out_items = []
    for k, (p, lab, is_test) in enumerate(items):
        img = Image.open(p).convert("RGB"); W, H = img.size
        orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        outpng = os.path.join(out_dir, f"m_{k:05d}.png")
        if glare_ratio(orig_bgr) < skip_thresh:                 # adaptive bypass
            cv2.imwrite(outpng, orig_bgr)
        else:
            x = pre(img).unsqueeze(0).to(dev)
            y = G_AB(x)[0].cpu() * 0.5 + 0.5
            gan_rgb = to_img(y.clamp(0, 1)).resize((W, H), Image.BICUBIC)
            gan_bgr = cv2.cvtColor(np.array(gan_rgb), cv2.COLOR_RGB2BGR)
            cv2.imwrite(outpng, blend_deglare(orig_bgr, gan_bgr))
        out_items.append((outpng, lab, is_test))
    return out_items


# --------------------------------------------------------------------------- #
#  Classifier training with per-epoch logging (same recipe as study_kfold.train_one)
# --------------------------------------------------------------------------- #
def train_classifier_logged(model_name, items, n_cls, dev, epochs, batch, lr,
                            workers, log_csv=None):
    model = build_model(model_name, n_cls, dev)
    counts = np.bincount([it[1] for it in items], minlength=n_cls)
    w = torch.tensor((counts.sum() / (counts + 1e-6)) / n_cls, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    dl = DataLoader(EarDataset(items, True), batch, shuffle=True, num_workers=workers)
    if log_csv:
        with open(log_csv, "w") as f:
            f.write("epoch,train_loss,train_acc,lr\n")
    for ep in range(epochs):
        model.train(); tot, correct, lsum, nb = 0, 0, 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            lsum += loss.item(); nb += 1
            correct += (out.argmax(1) == y).sum().item(); tot += y.size(0)
        sch.step()
        if log_csv:
            with open(log_csv, "a") as f:
                f.write(f"{ep},{lsum/max(nb,1):.5f},{correct/max(tot,1):.5f},"
                        f"{opt.param_groups[0]['lr']:.6f}\n")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  Stats: logit 95% CI (bounded 0..1) + Wilcoxon
# --------------------------------------------------------------------------- #
def logit_ci(vals):
    v = np.clip(np.array(vals, dtype=float), 1e-4, 1 - 1e-4)
    m = float(np.mean(v))
    lg = np.log(v / (1 - v))
    if len(v) > 1 and np.std(lg, ddof=1) > 0:
        from scipy import stats
        se = np.std(lg, ddof=1) / np.sqrt(len(lg))
        t = stats.t.ppf(0.975, len(lg) - 1)
        lo, hi = np.mean(lg) - t * se, np.mean(lg) + t * se
        return m, float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi)))
    return m, m, m


def wilcoxon_p(a, b):
    try:
        from scipy import stats
        d = np.array(a) - np.array(b)
        if np.allclose(d, 0):
            return 1.0
        return float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["chile", "ohio"], required=True)
    ap.add_argument("--models", default="resnet50,efficientnet_b0,vit_b_16")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gan_epochs", type=int, default=60)
    ap.add_argument("--gan_decay_start", type=int, default=30)
    ap.add_argument("--cls_epochs", type=int, default=30)
    ap.add_argument("--cls_batch", type=int, default=16)
    ap.add_argument("--cls_lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip_thresh", type=float, default=0.005)
    ap.add_argument("--max_per_domain", type=int, default=None,
                    help="cap Domain A/B size (smoke test / speed)")
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--keep_work", action="store_true", help="do not delete temp files")
    ap.add_argument("--out", default=None,
                    help="explicit output JSON path (default: auto, never overwrites)")
    a = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dp = ds_paths(a.dataset); classes = dp["classes"]; n_cls = len(classes)
    work = a.work_dir or os.path.join(RESULTS, "perfold_work", a.dataset)
    os.makedirs(work, exist_ok=True)
    models_tag = "-".join(m.strip() for m in a.models.split(",") if m.strip())
    logdir = os.path.join(RESULTS, "perfold_logs",
                          f"{a.dataset}_{models_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(logdir, exist_ok=True)
    print(f"[device] {dev} | dataset={a.dataset} ({dp['label']}) | classes={classes}")
    print(f"[work]   {work}")
    print(f"[logs]   {logdir}   (per-epoch CycleGAN + classifier CSVs)")

    items = scan(dp["raw"], classes)
    if not items:
        print("[ERROR] No images at", dp["raw"]); sys.exit(1)
    folds = grouped_kfold(items, k=a.folds, seed=SEED)
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    # G_AB is shared across classifiers within a fold (trained once per fold)
    results = {m: {c: {"acc": [], "f1": []} for c in ("original", "masked", "ensemble")}
               for m in models}

    for fi, (tr, te) in enumerate(folds):
        fold_dir = os.path.join(work, f"fold{fi}")
        dom_dir = os.path.join(fold_dir, "domains")
        os.makedirs(dom_dir, exist_ok=True)
        print(f"\n================  FOLD {fi+1}/{a.folds}  "
              f"train={len(tr)} test={len(te)}  ================")

        # 1) domains from TRAIN only
        nA, nB = build_fold_domains(tr, dom_dir, a.img_size,
                                    max_per_domain=a.max_per_domain, seed=SEED + fi)
        print(f"   domains: A(glare)={nA}  B(clean)={nB}")

        # 2) fresh CycleGAN on this fold
        G_AB = train_cyclegan_fold(dom_dir, dev, a.img_size, a.gan_epochs,
                                   a.gan_decay_start, workers=a.workers,
                                   log_prefix=f"[fold{fi+1}]",
                                   loss_csv=os.path.join(logdir, f"cyclegan_fold{fi}.csv"))

        # 3) masked deglaring with THIS fold's G_AB (train + test)
        tr_m = deglare_items(tr, G_AB, dev, a.img_size,
                             os.path.join(fold_dir, "masked_train"), a.skip_thresh)
        te_m = deglare_items(te, G_AB, dev, a.img_size,
                             os.path.join(fold_dir, "masked_test"), a.skip_thresh)
        del G_AB
        if dev == "cuda":
            torch.cuda.empty_cache()

        # 4) classifiers: Original vs Masked, evaluate on held-out test
        y = np.array([it[1] for it in te])
        for mdl in models:
            m_o = train_classifier_logged(mdl, tr, n_cls, dev, a.cls_epochs, a.cls_batch,
                                          a.cls_lr, a.workers,
                                          log_csv=os.path.join(logdir, f"cls_{mdl}_original_fold{fi}.csv"))
            m_m = train_classifier_logged(mdl, tr_m, n_cls, dev, a.cls_epochs, a.cls_batch,
                                          a.cls_lr, a.workers,
                                          log_csv=os.path.join(logdir, f"cls_{mdl}_masked_fold{fi}.csv"))
            po = probs(m_o, te, dev); pm = probs(m_m, te_m, dev); pe = (po + pm) / 2
            for key, pred in [("original", po.argmax(1)), ("masked", pm.argmax(1)),
                              ("ensemble", pe.argmax(1))]:
                results[mdl][key]["acc"].append(float(accuracy_score(y, pred)))
                results[mdl][key]["f1"].append(float(f1_score(y, pred, average="macro")))
            print(f"   [{mdl}] fold{fi+1}: "
                  f"orig={results[mdl]['original']['acc'][-1]:.4f} "
                  f"masked={results[mdl]['masked']['acc'][-1]:.4f} "
                  f"ens={results[mdl]['ensemble']['acc'][-1]:.4f}")
            del m_o, m_m
            if dev == "cuda":
                torch.cuda.empty_cache()

        if not a.keep_work:
            shutil.rmtree(fold_dir, ignore_errors=True)

    # ------- aggregate + stats -------
    summary = {}
    for mdl in models:
        summary[mdl] = {}
        for c in ("original", "masked", "ensemble"):
            acc = results[mdl][c]["acc"]; f1 = results[mdl][c]["f1"]
            m, lo, hi = logit_ci(acc)
            summary[mdl][c] = {
                "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)),
                "acc_ci95_logit": [round(lo * 100, 1), round(hi * 100, 1)],
                "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1)),
                "acc_folds": acc,
            }
        summary[mdl]["wilcoxon_orig_vs_masked_p"] = wilcoxon_p(
            results[mdl]["original"]["acc"], results[mdl]["masked"]["acc"])

    out = {"dataset": a.dataset, "label": dp["label"], "protocol": "per-fold CycleGAN (leakage-free)",
           "folds": a.folds, "gan_epochs": a.gan_epochs, "cls_epochs": a.cls_epochs,
           "models": models, "log_dir": logdir, "results": summary}
    # ---- unique output filename: never overwrite an existing result ----
    if a.out:
        out_json = a.out
    else:
        mtag = "-".join(models)
        out_json = os.path.join(RESULTS, f"study_{a.dataset}_perfold_{mtag}.json")
        base, ext = os.path.splitext(out_json)
        i = 2
        while os.path.exists(out_json):
            out_json = f"{base}_{i}{ext}"; i += 1
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("\n================  SUMMARY (leakage-free per-fold CycleGAN)  ================")
    for mdl in models:
        print(f"\n{mdl} ({a.dataset}):")
        for c in ("original", "masked", "ensemble"):
            r = summary[mdl][c]
            print(f"  {c:9s}: {r['acc_mean']*100:.1f} +/- {r['acc_std']*100:.1f}%  "
                  f"[{r['acc_ci95_logit'][0]}-{r['acc_ci95_logit'][1]}]  "
                  f"F1={r['f1_mean']*100:.1f}")
        print(f"  Wilcoxon original vs masked p = {summary[mdl]['wilcoxon_orig_vs_masked_p']}")
    print("\n[saved]", out_json)
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
