"""
Goruntuleri parlama oranina gore iki domaine ayirir (A=parlamali, B=temiz)
ve CycleGAN icin trainA/trainB/testA/testB klasorlerini kurar.

Kullanim:
    python preprocessing/build_cyclegan_dataset.py                 # Chile+Ohio (havuz)
    python preprocessing/build_cyclegan_dataset.py --only chile    # yalnizca Dataset 1
    python preprocessing/build_cyclegan_dataset.py --only ohio     # yalnizca Dataset 2
"""
import os, sys, random, shutil, argparse
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHILE_DIR, OHIO_DIR, CYCLEGAN_DATA, SEED, ds_paths
from preprocessing.glare_utils import glare_ratio

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def collect_images(root):
    if not os.path.isdir(root):
        return []
    out = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(dp, f))
    return out

def main(only=None, out=None, high_pct=0.40, low_pct=0.40, test_frac=0.1):
    random.seed(SEED)
    if only in ("chile", "ohio"):
        srcs = [ds_paths(only)["raw"]]
        out = out or ds_paths(only)["cyclegan"]
        print(f"[bilgi] Tek veri seti: {only}")
    else:
        srcs = [CHILE_DIR, OHIO_DIR]
        out = out or CYCLEGAN_DATA
        print("[bilgi] Havuz: Chile + Ohio")
    imgs = []
    for r in srcs:
        imgs += collect_images(r)
    if not imgs:
        print("[HATA] Goruntu bulunamadi. Once download_data.py calistirin."); sys.exit(1)
    print(f"[bilgi] {len(imgs)} goruntu | parlama orani hesaplaniyor...")
    scored = []
    for i, p in enumerate(imgs):
        im = cv2.imread(p)
        if im is None:
            continue
        scored.append((glare_ratio(im), p))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(imgs)}")
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    low = [p for _, p in scored[:int(n * low_pct)]]
    high = [p for _, p in scored[int(n * (1 - high_pct)):]]
    random.shuffle(low); random.shuffle(high)

    def split_copy(files, name):
        ntest = max(1, int(len(files) * test_frac))
        for sub, fs in [("test" + name, files[:ntest]), ("train" + name, files[ntest:])]:
            d = os.path.join(out, sub)
            os.makedirs(d, exist_ok=True)
            for k, src in enumerate(fs):
                shutil.copy(src, os.path.join(d, f"{name}_{k:05d}.png"))
            print(f"  {sub}: {len(fs)} goruntu")

    if os.path.isdir(out):
        shutil.rmtree(out)
    split_copy(high, "A")
    split_copy(low, "B")
    print("[tamam] CycleGAN veri seti ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["chile", "ohio"], default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(only=a.only, out=a.out)
