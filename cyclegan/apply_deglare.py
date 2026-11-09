"""
Egitilmis G_AB ile bir klasor agacindaki TUM goruntulerin parlamasini giderir,
ayni klasor yapisini koruyarak cikti yazar.

Iki mod:
  (varsayilan) tam   : GAN ciktisi tum goruntuye uygulanir   -> chile_deglared
  --mask            : GAN ciktisi SADECE parlama bolgesine uygulanir, kalan doku
                      orijinal kalir (hedefli/maskeli)        -> chile_deglared_masked

Kullanim:
    python cyclegan/apply_deglare.py                 # tam
    python cyclegan/apply_deglare.py --mask          # maskeli (doku korumali)
"""
import os, sys, argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHILE_DIR, DEGLARED_DIR, DEGLARED_MASKED_DIR, CHECKPOINTS, IMG_SIZE
from cyclegan.models import Generator
from preprocessing.glare_utils import blend_deglare, glare_ratio

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=CHILE_DIR)
    ap.add_argument("--dst", default=None, help="bos birakirsan moda gore otomatik secilir")
    ap.add_argument("--ckpt", default=os.path.join(CHECKPOINTS, "G_AB.pth"))
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--mask", action="store_true",
                    help="maskeli/hedefli: sadece parlama bolgesini degistir, dokuyu koru")
    ap.add_argument("--skip_thresh", type=float, default=0.005,
                    help="parlama orani bu esigin altindaysa goruntuye HIC dokunma (sadece --mask modunda)")
    a = ap.parse_args()
    dst = a.dst or (DEGLARED_MASKED_DIR if a.mask else DEGLARED_DIR)
    n_skipped = 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator().to(dev).eval()
    G.load_state_dict(torch.load(a.ckpt, map_location=dev))
    pre = T.Compose([T.Resize((a.img_size, a.img_size)), T.ToTensor(),
                     T.Normalize([0.5] * 3, [0.5] * 3)])
    to_img = T.ToPILImage()
    print(f"[mod] {'MASKELI (doku korumali)' if a.mask else 'TAM goruntu'}  -> {dst}")

    n = 0
    for dp, _, fs in os.walk(a.src):
        for f in fs:
            if not f.lower().endswith(EXT):
                continue
            src = os.path.join(dp, f)
            rel = os.path.relpath(dp, a.src)
            outdir = os.path.join(dst, rel); os.makedirs(outdir, exist_ok=True)
            try:
                img = Image.open(src).convert("RGB"); W, H = img.size
                outpng = os.path.join(outdir, os.path.splitext(f)[0] + ".png")
                # --- ACIK ADAPTIF KURAL ---
                # Parlama orani esigin altinda olan goruntulere HIC dokunma:
                # orijinali oldugu gibi yaz (GAN bile calistirilmaz). Boylece
                # parlamasi az goruntuler korunur, parlamasi cok olanlara odaklanilir.
                if a.mask:
                    orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    if glare_ratio(orig_bgr) < a.skip_thresh:
                        cv2.imwrite(outpng, orig_bgr); n += 1; n_skipped += 1
                        if n % 100 == 0:
                            print(f"  {n} goruntu islendi ({n_skipped} dokunulmadi)")
                        continue
                with torch.no_grad():
                    x = pre(img).unsqueeze(0).to(dev)
                    y = G(x)[0].cpu() * 0.5 + 0.5
                gan_rgb = to_img(y.clamp(0, 1)).resize((W, H), Image.BICUBIC)
                if a.mask:
                    gan_bgr = cv2.cvtColor(np.array(gan_rgb), cv2.COLOR_RGB2BGR)
                    blended = blend_deglare(orig_bgr, gan_bgr)
                    cv2.imwrite(outpng, blended)
                else:
                    gan_rgb.save(os.path.join(outdir, os.path.splitext(f)[0] + ".png"))
                n += 1
                if n % 100 == 0:
                    print(f"  {n} goruntu islendi")
            except Exception as e:
                print("[atlandi]", src, e)
    print(f"[tamam] {n} goruntu islendi -> {dst}")
    if a.mask:
        print(f"[bilgi] {n_skipped} goruntu parlamasi esik altinda oldugu icin hic degistirilmedi (esik={a.skip_thresh}).")

if __name__ == "__main__":
    main()
