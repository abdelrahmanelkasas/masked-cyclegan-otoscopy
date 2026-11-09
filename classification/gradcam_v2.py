"""
Grad-CAM v2 — sayisal ve adil tasarim.

Tasarim: AYNI model (orijinal-egitilmis) hem ORIJINAL hem TEMIZLENMIS goruntude
calistirilir. Boylece tek degisken parlamadir (model sabit). Ayrica parlama maskesi
kullanilarak "isi haritasi enerjisinin parlama bolgesine dusen orani" sayisal olcuLur.

Olcut (glare-attention oran): r = ort(CAM | parlama icinde) / ort(CAM | parlama disinda)
  r > 1  => model orantisiz olarak parlamaya odaklaniyor (confounding bias kaniti)
Deglaring sonrasi bu oranin (ve parlama bolgesindeki CAM'in) dusmesi beklenir.

Kullanim: python classification/gradcam_v2.py --n 12
Cikti: results/gradcam_v2/ overlay'ler + konsolda ortalama oranlar + sayisal ozet JSON.
"""
import os, sys, argparse, json
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as M, torchvision.transforms as T
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHILE_DIR, DEGLARED_MASKED_DIR, CHECKPOINTS, RESULTS, CHILE_CLASSES
from preprocessing.glare_utils import glare_ratio, glare_mask

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def build(ckpt, dev):
    m = M.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, len(CHILE_CLASSES))
    m.load_state_dict(torch.load(ckpt, map_location=dev)); return m.to(dev).eval()

def cam_of(model, x, dev):
    feats, grads = {}, {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("v", o.detach()))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))
    out = model(x); cls = out.argmax(1); model.zero_grad(); out[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = os.path.join(RESULTS, "gradcam_v2"); os.makedirs(outdir, exist_ok=True)
    pre = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = build(os.path.join(CHECKPOINTS, "classifier_orig_grouped.pth"), dev)  # AYNI model

    imgs = []
    for dp, _, fs in os.walk(CHILE_DIR):
        for f in fs:
            if f.lower().endswith(EXT):
                p = os.path.join(dp, f); im = cv2.imread(p)
                if im is not None: imgs.append((glare_ratio(im), p))
    imgs.sort(reverse=True); sel = [p for _, p in imgs[:a.n]]

    ratio_orig, ratio_deg, camglare_orig, camglare_deg = [], [], [], []
    examples = []  # (ratio_o, plain, ov_orig, ov_deg)
    SEP = np.full((224, 6, 3), 255, np.uint8)
    for i, p in enumerate(sel):
        oimg = Image.open(p).convert("RGB")
        rel = os.path.relpath(p, CHILE_DIR)
        mp = os.path.join(DEGLARED_MASKED_DIR, os.path.splitext(rel)[0] + ".png")
        dimg = Image.open(mp).convert("RGB") if os.path.exists(mp) else oimg
        gm = glare_mask(cv2.resize(cv2.cvtColor(np.array(oimg), cv2.COLOR_RGB2BGR), (224, 224)))
        inside = gm > 0
        if inside.sum() < 20:
            continue
        cam_o = cam_of(model, pre(oimg).unsqueeze(0).to(dev), dev)
        cam_d = cam_of(model, pre(dimg).unsqueeze(0).to(dev), dev)
        def metric(cam):
            mi = cam[inside].mean(); mo = cam[~inside].mean()
            return float(mi / (mo + 1e-8)), float(mi)
        r_o, ci_o = metric(cam_o); r_d, ci_d = metric(cam_d)
        ratio_orig.append(r_o); ratio_deg.append(r_d); camglare_orig.append(ci_o); camglare_deg.append(ci_d)
        plain = cv2.cvtColor(np.array(oimg.resize((224, 224))), cv2.COLOR_RGB2BGR)  # ISI YOK (orijinal)
        def ov(pil, cam):
            base = cv2.cvtColor(np.array(pil.resize((224, 224))), cv2.COLOR_RGB2BGR)
            hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            o = cv2.addWeighted(base, 0.55, hm, 0.45, 0)
            cont, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(o, cont, -1, (255, 255, 255), 1)
            return o
        ovo, ovd = ov(oimg, cam_o), ov(dimg, cam_d)
        # 3 sutun: Orijinal (isisiz) | Grad-CAM orijinal | Grad-CAM temizlenmis
        trip = np.hstack([plain, SEP, ovo, SEP, ovd])
        # panel ustune basliklar yaz
        hdr = np.full((30, trip.shape[1], 3), 255, np.uint8)
        for c, ttl in enumerate(["Original", "Grad-CAM (original)", "Grad-CAM (deglared)"]):
            cv2.putText(hdr, ttl, (c*230+6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(outdir, f"gc_{i:02d}.png"), np.vstack([hdr, trip]))
        examples.append((r_o, plain, ovo, ovd))

    # === Yayina hazir Figure_8: en ikna edici 3 ornek, sutun basliklariyla ===
    if examples:
        examples.sort(key=lambda e: e[0], reverse=True)
        top = examples[:3]
        titles = ["Original image", "Grad-CAM (original)", "Grad-CAM (deglared)"]
        col_w = 224; hgap = 12; header = 34
        grid_w = col_w * 3 + hgap * 2
        rows_imgs = []
        for _, plain, ovo, ovd in top:
            row = np.hstack([plain, np.full((224, hgap, 3), 255, np.uint8), ovo,
                             np.full((224, hgap, 3), 255, np.uint8), ovd])
            rows_imgs.append(row); rows_imgs.append(np.full((10, grid_w, 3), 255, np.uint8))
        body = np.vstack(rows_imgs)
        head = np.full((header, grid_w, 3), 255, np.uint8)
        for c, ttl in enumerate(titles):
            x = c * (col_w + hgap) + 8
            cv2.putText(head, ttl, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        fig8 = np.vstack([head, body])
        fig8 = cv2.resize(fig8, (fig8.shape[1]*2, fig8.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        fig8_path = os.path.join(os.path.dirname(outdir), "..", "paper", "figures", "Figure_8.png")
        fig8_path = os.path.normpath(fig8_path)
        os.makedirs(os.path.dirname(fig8_path), exist_ok=True)
        cv2.imwrite(fig8_path, fig8)
        print("[Figure_8] ->", fig8_path, "(white outline = glare region)")

    res = {
        "n": len(ratio_orig),
        "glare_attention_ratio_original_mean": float(np.mean(ratio_orig)) if ratio_orig else None,
        "glare_attention_ratio_deglared_mean": float(np.mean(ratio_deg)) if ratio_deg else None,
        "cam_inside_glare_original_mean": float(np.mean(camglare_orig)) if camglare_orig else None,
        "cam_inside_glare_deglared_mean": float(np.mean(camglare_deg)) if camglare_deg else None,
    }
    with open(os.path.join(RESULTS, "gradcam_v2_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("\nYorum: glare_attention_ratio_original > 1 ise orijinal model parlamaya orantisiz odaklaniyor.")
    print("Deglared'da bu oran/CAM-in-glare dusuyorsa, deglaring dikkatı parlamadan uzaklastirmistir.")
    print("Overlay'ler:", outdir, "(beyaz kontur = parlama bolgesi)")

if __name__ == "__main__":
    main()
