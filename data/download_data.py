"""
Chile (figshare) ve Ohio (Zenodo) kulak zari veri setlerini indirir.

Akilli davranis:
- Once veri zaten inmis mi diye BAKAR. Inmisse tekrar indirmez, hata vermez, devam eder.
- Inmemisse indirir. Ohio indirilemezse (Zenodo 403) uyarip gecer; Chile yeterli.
- Sonunda HER SETTEN kac goruntu bulundugunu yazar.

Kullanim:
    python data/download_data.py
    python data/download_data.py --only chile
    python data/download_data.py --only ohio
"""
import os, sys, argparse, zipfile, subprocess, urllib.request, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_MAIN = os.path.join(HERE, "eardrum_public_data")
CHILE_DIR = os.path.join(DATA_MAIN, "Chile")
OHIO_DIR  = os.path.join(DATA_MAIN, "Ohio")
CHILE_URL = "https://ndownloader.figshare.com/files/21793293"   # Datos.rar
CHILE_MD5 = "4631b558a716bbbdbaadf97e5fcd7413"
OHIO_URLS = [
    "https://zenodo.org/records/4558155/files-archive",
    "https://zenodo.org/api/records/4558155/files-archive",
]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def count_images(root):
    n = 0
    if os.path.isdir(root):
        for _, _, fs in os.walk(root):
            n += sum(1 for f in fs if f.lower().endswith(IMG_EXT))
    return n

def _download(url, dst):
    print(f"[indiriliyor] {url}\n           -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r           {done*100//total:3d}%  ({done//1024} KB)", end="")
    print("\n[tamam]", dst)

def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()

def download_chile():
    # 1) Zaten inmis mi?
    if count_images(CHILE_DIR) > 0:
        print(f"[atlandi] Chile zaten mevcut ({count_images(CHILE_DIR)} goruntu).")
        return
    os.makedirs(DATA_MAIN, exist_ok=True)
    rar = os.path.join(DATA_MAIN, "Datos.rar")
    if not os.path.exists(rar):
        _download(CHILE_URL, rar)
    if _md5(rar) != CHILE_MD5:
        print("[UYARI] Chile MD5 eslesmedi, dosya bozuk olabilir.")
    os.makedirs(CHILE_DIR, exist_ok=True)
    # 7-Zip > patool > rarfile
    for sz in [r"C:\Program Files\7-Zip\7z.exe",
               r"C:\Program Files (x86)\7-Zip\7z.exe", "7z"]:
        try:
            subprocess.run([sz, "x", "-y", f"-o{CHILE_DIR}", rar], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[tamam] Chile 7-Zip ile acildi.")
            return
        except Exception:
            continue
    try:
        import patoolib
        patoolib.extract_archive(rar, outdir=CHILE_DIR)
        print("[tamam] Chile acildi.")
        return
    except Exception as e:
        print("[bilgi] patool ile acilamadi:", e)
    try:
        import rarfile
        with rarfile.RarFile(rar) as rf:
            rf.extractall(CHILE_DIR)
        print("[tamam] Chile acildi.")
        return
    except Exception as e:
        print("[bilgi] rarfile ile acilamadi:", e)
    print(f"[ELLE ACMA] '{rar}' dosyasini 7-Zip/WinRAR ile '{CHILE_DIR}' icine acin.")

def download_ohio():
    """Ohio OPSIYONEL'dir (CycleGAN havuzunu buyutur)."""
    # 1) Zaten var mi? (elle indirip koymus olabilirsin)
    if count_images(OHIO_DIR) > 0:
        print(f"[atlandi] Ohio zaten mevcut ({count_images(OHIO_DIR)} goruntu), indirilmeyecek.")
        return
    os.makedirs(DATA_MAIN, exist_ok=True)
    zp = os.path.join(DATA_MAIN, "ohio_files.zip")
    if not os.path.exists(zp) or os.path.getsize(zp) < 10000:
        ok = False
        for u in OHIO_URLS:
            try:
                _download(u, zp)
                ok = True
                break
            except Exception as e:
                print(f"[bilgi] Ohio indirilemedi ({u}): {e}")
        if not ok:
            print("""
[UYARI] Ohio seti otomatik indirilemedi (Zenodo 403). OPSIYONELDIR; Chile ile devam.
        Istersen elle indirip su klasore ac: data/eardrum_public_data/Ohio/
        Link: https://zenodo.org/records/4558155
""")
            return
    os.makedirs(OHIO_DIR, exist_ok=True)
    try:
        with zipfile.ZipFile(zp) as z:
            z.extractall(os.path.join(OHIO_DIR, "all_images"))
        print(f"[tamam] Ohio acildi ({count_images(OHIO_DIR)} goruntu).")
    except Exception as e:
        print("[uyari] Ohio zip acilamadi (atlaniyor):", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["chile", "ohio"], default=None)
    a = ap.parse_args()
    if a.only in (None, "chile"):
        download_chile()
    if a.only in (None, "ohio"):
        download_ohio()
    # === OZET: her setten kac goruntu ===
    c, o = count_images(CHILE_DIR), count_images(OHIO_DIR)
    print("\n==================== VERI OZETI ====================")
    print(f"  Chile : {c} goruntu")
    print(f"  Ohio  : {o} goruntu")
    print(f"  TOPLAM: {c + o} goruntu")
    print("====================================================")
    print("Veri kok dizini:", DATA_MAIN)
