"""Merkezi konfigürasyon. Tum scriptler buradan okur."""
import os

# Proje kok dizini (bu dosyanin bulundugu yerin ust klasoru)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ham veri
DATA_MAIN_DIR = os.path.join(ROOT, "data", "eardrum_public_data")
CHILE_DIR = os.path.join(DATA_MAIN_DIR, "Chile")
OHIO_DIR  = os.path.join(DATA_MAIN_DIR, "Ohio")

# CycleGAN icin domain veri seti (A = parlamali, B = temiz)
CYCLEGAN_DATA = os.path.join(ROOT, "data", "cyclegan_glare")

# Parlamasi giderilmis (deglared) tam veri seti (siniflandirma icin)
DEGLARED_DIR = os.path.join(ROOT, "data", "chile_deglared")
# Maskeli/hedefli parlama giderme (sadece parlama bolgesi degisir, doku korunur)
DEGLARED_MASKED_DIR = os.path.join(ROOT, "data", "chile_deglared_masked")

# Sinifladirma icin orijinal (referans) veri seti
CLASSIFY_ORIG = os.path.join(ROOT, "data", "chile_classify_original")

CHECKPOINTS = os.path.join(ROOT, "checkpoints")
RESULTS     = os.path.join(ROOT, "results")

# Goruntu / egitim ayarlari (RTX 3060 icin)
IMG_SIZE = 256          # CycleGAN egitim cozunurlugu
CLS_IMG_SIZE = 224      # Siniflandirma cozunurlugu
SEED = 1234

CHILE_CLASSES = ["Chronic otitis media", "Earwax plug", "Myringosclerosis", "Normal"]
OHIO_CLASSES  = ["Effusion", "Normal", "Tube"]

# Her veri setini ayri ayri islemek icin kayit defteri (Dataset 1 = chile, Dataset 2 = ohio)
DATASETS = {
    "chile": {"raw": CHILE_DIR, "classes": CHILE_CLASSES, "label": "Dataset 1 (Chile)"},
    "ohio":  {"raw": OHIO_DIR,  "classes": OHIO_CLASSES,  "label": "Dataset 2 (Ohio)"},
}

def ds_paths(name):
    """Bir veri setine ait tum yollar (ayri pipeline icin)."""
    return {
        "name": name,
        "raw": DATASETS[name]["raw"],
        "classes": DATASETS[name]["classes"],
        "label": DATASETS[name]["label"],
        "cyclegan": os.path.join(ROOT, "data", f"cyclegan_{name}"),     # A/B domain seti
        "full":     os.path.join(ROOT, "data", f"{name}_deglared"),     # tam deglared
        "masked":   os.path.join(ROOT, "data", f"{name}_deglared_masked"),
        "ckpt":     os.path.join(CHECKPOINTS, f"G_AB_{name}.pth"),
        "tag":      name,
    }

for d in [CHECKPOINTS, RESULTS]:
    os.makedirs(d, exist_ok=True)
