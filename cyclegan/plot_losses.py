"""CycleGAN egitim eglerini cizer (results/cyclegan_loss_log.csv -> PNG)."""
import os, sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RESULTS

def main():
    path = os.path.join(RESULTS, "cyclegan_loss_log.csv")
    if not os.path.exists(path):
        print("[HATA] Loss kaydi yok:", path); return
    rows = list(csv.DictReader(open(path)))
    ep = [int(r["epoch"]) for r in rows]
    keys = ["G", "D", "cyc", "identity", "gan"]
    plt.figure(figsize=(10, 6))
    for k in keys:
        plt.plot(ep, [float(r[k]) for r in rows], label=k)
    plt.xlabel("Epoch"); plt.ylabel("Kayip (loss)")
    plt.title("CycleGAN Egitim Kayiplari")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(RESULTS, "cyclegan_loss_curves.png")
    plt.tight_layout(); plt.savefig(out, dpi=600)
    print("[kaydedildi]", out)

if __name__ == "__main__":
    main()
