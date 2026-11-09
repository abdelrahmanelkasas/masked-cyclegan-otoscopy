"""
GAN kalite degerlendirmesi (parlama giderme):
- Parlama orani azaldi mi? (orijinal vs deglared)
- Doku korundu mu? SSIM ve PSNR (yuksek = doku bozulmadi)
Cikti: konsol + results/gan_quality.json + results/gan_glare_hist.png

Kullanim: python cyclegan/eval_quality.py
"""
import os, sys, argparse, json
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHILE_DIR, DEGLARED_DIR, RESULTS
from preprocessing.glare_utils import glare_ratio

EXT = (".jpg", ".jpeg", ".png", ".bmp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=CHILE_DIR)
    ap.add_argument("--deg", default=DEGLARED_DIR)
    ap.add_argument("--tag", default="", help="cikti adlarina ek (orn: chile, ohio)")
    a = ap.parse_args()
    SFX = f"_{a.tag}" if a.tag else ""
    pairs, gr_o, gr_d, ssims, psnrs = 0, [], [], [], []
    for dp, _, fs in os.walk(a.orig):
        for f in fs:
            if not f.lower().endswith(EXT):
                continue
            op = os.path.join(dp, f)
            rel = os.path.relpath(dp, a.orig)
            dpth = os.path.join(a.deg, rel, os.path.splitext(f)[0] + ".png")
            if not os.path.exists(dpth):
                continue
            o = cv2.imread(op); d = cv2.imread(dpth)
            if o is None or d is None:
                continue
            d = cv2.resize(d, (o.shape[1], o.shape[0]))
            gr_o.append(glare_ratio(o)); gr_d.append(glare_ratio(d))
            og = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY); dg = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            ssims.append(float(ssim(og, dg)))
            psnrs.append(float(psnr(og, dg, data_range=255)))
            pairs += 1
    if pairs == 0:
        print("[HATA] Eslesen goruntu yok. Once apply_deglare.py calistirin."); return
    res = {
        "pairs": pairs,
        "glare_ratio_original_mean": float(np.mean(gr_o)),
        "glare_ratio_deglared_mean": float(np.mean(gr_d)),
        "glare_reduction_percent": float((1 - np.mean(gr_d) / max(np.mean(gr_o), 1e-9)) * 100),
        "ssim_mean": float(np.mean(ssims)),
        "psnr_mean": float(np.mean(psnrs)),
    }
    print(f"Eslesen goruntu: {pairs}")
    print(f"Parlama orani  orijinal={res['glare_ratio_original_mean']:.4f}  "
          f"deglared={res['glare_ratio_deglared_mean']:.4f}")
    print(f"Parlama azalmasi: {res['glare_reduction_percent']:.1f}%")
    print(f"SSIM (doku korunumu): {res['ssim_mean']:.4f}  (1.0'a yakin = iyi)")
    print(f"PSNR: {res['psnr_mean']:.2f} dB")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"gan_quality{SFX}.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(8, 5))
    plt.hist(gr_o, bins=30, alpha=0.6, label="Original")
    plt.hist(gr_d, bins=30, alpha=0.6, label="Deglared (masked)")
    plt.xlabel("Glare ratio"); plt.ylabel("Number of images")
    plt.title("Glare Ratio Distribution (before vs after)")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(RESULTS, f"gan_glare_hist{SFX}.png")
    plt.tight_layout(); plt.savefig(out, dpi=600)
    print("[saved]", os.path.join(RESULTS, f"gan_quality{SFX}.json"), "and", out)

if __name__ == "__main__":
    main()
