"""
Grad-CAM on HELD-OUT test images (reviewer-proof version of gradcam_v2.py).

Differences from gradcam_v2.py:
  * Images are selected ONLY from the TEST subset of the single grouped split
    (the same split used to train classifier_orig_grouped.pth), so the model
    has never seen them during training or checkpoint selection.
  * Per-image ratios are saved; mean, median, IQR and an exact Wilcoxon test
    (original vs deglared ratio, paired by image) are reported.

Usage:  python classification/gradcam_heldout.py          (all 177 test images, primary)
        python classification/gradcam_heldout.py --n 12  (top-12 by glare, secondary)
Expected (computed for the manuscript): all 177 -> median ratio 1.45 -> 1.43;
          top-12 -> mean 1.53 -> 1.54, p = 0.91
Output: results/gradcam_heldout_summary.json, results/gradcam_heldout_perimage.csv
"""
import os, sys, argparse, json, csv
import numpy as np, cv2, torch
from PIL import Image
import torchvision.transforms as T
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHILE_DIR, DEGLARED_MASKED_DIR, CHECKPOINTS, RESULTS
from preprocessing.glare_utils import glare_ratio, glare_mask
from classification.dataset import scan
from classification.grouped_split import grouped_split
from classification.gradcam_v2 import build, cam_of

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=0, help="0 = all test images (primary analysis); 12 = top-12 by glare (secondary)"); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pre = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = build(os.path.join(CHECKPOINTS, "classifier_orig_grouped.pth"), dev)
    items = scan(CHILE_DIR)
    _, _, test = grouped_split(items, seed=1234)
    print(f"[split] test images = {len(test)} (expected 177)")
    scored = sorted(((glare_ratio(cv2.imread(p)), p) for p, _, _ in test), reverse=True)
    if a.n > 0:
        scored = scored[:a.n]
    rows = []
    for g, p in scored:
        oimg = Image.open(p).convert("RGB")
        rel = os.path.relpath(p, CHILE_DIR)
        mp = os.path.join(DEGLARED_MASKED_DIR, os.path.splitext(rel)[0] + ".png")
        dimg = Image.open(mp).convert("RGB")
        gm = glare_mask(cv2.resize(cv2.cvtColor(np.array(oimg), cv2.COLOR_RGB2BGR), (224, 224)))
        inside = gm > 0
        if inside.sum() < 20:
            continue
        co = cam_of(model, pre(oimg).unsqueeze(0).to(dev), dev)
        cd = cam_of(model, pre(dimg).unsqueeze(0).to(dev), dev)
        r = lambda c: (float(c[inside].mean() / (c[~inside].mean() + 1e-8)), float(c[inside].mean()))
        (ro, io), (rd, idg) = r(co), r(cd)
        rows.append(dict(image=rel, glare_ratio=round(g, 4), ratio_orig=ro, ratio_deg=rd, cam_in_orig=io, cam_in_deg=idg))
    ro = np.array([x["ratio_orig"] for x in rows]); rd = np.array([x["ratio_deg"] for x in rows])
    from scipy import stats
    p = float(stats.wilcoxon(ro, rd).pvalue)
    q = lambda v: [float(np.percentile(v, 25)), float(np.percentile(v, 75))]
    res = {"n": len(rows), "selection": "top-n glare ratio within grouped-split TEST set (held-out)",
           "ratio_orig_mean": float(ro.mean()), "ratio_orig_median": float(np.median(ro)), "ratio_orig_iqr": q(ro),
           "ratio_deg_mean": float(rd.mean()), "ratio_deg_median": float(np.median(rd)), "ratio_deg_iqr": q(rd),
           "n_images_ratio_decreased": int((rd < ro).sum()),
           "cam_in_glare_orig_mean": float(np.mean([x["cam_in_orig"] for x in rows])),
           "cam_in_glare_deg_mean": float(np.mean([x["cam_in_deg"] for x in rows])),
           "wilcoxon_exact_p": p}
    json.dump(res, open(os.path.join(RESULTS, f"gradcam_heldout_summary_n{a.n or len(rows)}.json"), "w"), indent=2)
    with open(os.path.join(RESULTS, f"gradcam_heldout_perimage_n{a.n or len(rows)}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
