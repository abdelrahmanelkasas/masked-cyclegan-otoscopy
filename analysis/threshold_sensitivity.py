"""
Glare-threshold sensitivity analysis (Supplementary Table S1), Dataset 1, all images.
The dataset-level generator (checkpoints/G_AB_chile.pth) is held fixed; only the HSV
mask thresholds change (V >= 210..230, S <= 50..70, step 5; 25 settings).

Metrics per setting:
  glare_area_pct   mean fraction of pixels flagged as glare (that setting's detector)
  ssim             global greyscale SSIM (scikit-image defaults), original vs processed
  psnr_nonglare    per-image RGB PSNR over pixels outside that setting's binary mask,
                   averaged over images (unchanged images, PSNR undefined, excluded)
  ident_pct        per-image % of non-glare pixels with all 3 channels unchanged, averaged
  glare_red_ref    relative decrease of mean glare ratio, measured with the reference
                   detector (V >= 220, S <= 60)
  n_bypass         images with glare ratio < 0.5% (returned unchanged)

Usage:  python analysis/threshold_sensitivity.py
Output: results/threshold_sensitivity_full.json
"""
import os, sys, json
import numpy as np, cv2, torch
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import structural_similarity
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config.config import CHILE_DIR, CHECKPOINTS, RESULTS, IMG_SIZE
from cyclegan.models import Generator
from preprocessing.glare_utils import glare_mask, feather_mask

V = [210, 215, 220, 225, 230]; S = [70, 65, 60, 55, 50]


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator().to(dev).eval()
    G.load_state_dict(torch.load(os.path.join(CHECKPOINTS, "G_AB_chile.pth"), map_location=dev))
    pre = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    paths = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(CHILE_DIR) for f in fs
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    acc = {(v, s): {"ss": [], "ps": [], "idp": [], "area": [], "g_out": 0.0, "g_in": 0.0, "byp": 0} for v in V for s in S}
    for k, p in enumerate(paths):
        img = Image.open(p).convert("RGB"); W, H = img.size
        o = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        y = G(pre(img).unsqueeze(0).to(dev))[0].cpu() * 0.5 + 0.5
        g = cv2.cvtColor(np.array(T.ToPILImage()(y.clamp(0, 1)).resize((W, H), Image.BICUBIC)), cv2.COLOR_RGB2BGR)
        og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY); ref_in = (glare_mask(o) > 0).mean()
        for v in V:
            for s in S:
                A = acc[(v, s)]; m = glare_mask(o, v_thresh=v, s_thresh=s); r = (m > 0).mean(); A["area"].append(r)
                if r < 0.005:
                    out = o; A["byp"] += 1
                else:
                    M = feather_mask(m)[:, :, None]
                    out = (o.astype(np.float32) * (1 - M) + g.astype(np.float32) * M).clip(0, 255).astype(np.uint8)
                A["ss"].append(structural_similarity(og, cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)))
                ng = m == 0
                mse = ((o.astype(np.float64) - out.astype(np.float64))[ng] ** 2).mean()
                A["ps"].append(10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf)
                A["idp"].append(((o == out).all(2) & ng).sum() / ng.sum())
                A["g_in"] += ref_in; A["g_out"] += (glare_mask(out) > 0).mean()
        if k % 100 == 0:
            print(f"{k}/{len(paths)}", flush=True)
    grid = []
    for (v, s), A in acc.items():
        ps = np.array(A["ps"])
        grid.append({"V": v, "S": s, "glare_area_pct": 100 * float(np.mean(A["area"])), "ssim": float(np.mean(A["ss"])),
                     "psnr_nonglare": float(ps[np.isfinite(ps)].mean()), "ident_pct": 100 * float(np.mean(A["idp"])),
                     "glare_red_ref": 100 * (1 - A["g_out"] / A["g_in"]), "n_bypass": A["byp"]})
        print(grid[-1])
    json.dump({"n": len(paths), "grid": grid}, open(os.path.join(RESULTS, "threshold_sensitivity_full.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
