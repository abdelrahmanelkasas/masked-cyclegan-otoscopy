"""
CycleGAN egitimi (parlama giderme, A=parlamali -> B=temiz).
RTX 3060 icin varsayilanlar (256px, batch=1, AMP).

Kullanim:
    python cyclegan/train.py --epochs 200
"""
import os, sys, argparse, itertools, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CYCLEGAN_DATA, CHECKPOINTS, RESULTS, IMG_SIZE, SEED
from cyclegan.models import Generator, Discriminator, init_weights
from cyclegan.datasets import UnpairedDataset
from cyclegan.utils import ReplayBuffer, LambdaLR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--decay_start", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--lambda_cyc", type=float, default=10.0)
    ap.add_argument("--lambda_id", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--no_resume", action="store_true",
                    help="onceki kayit olsa bile sifirdan basla")
    ap.add_argument("--data_dir", default=CYCLEGAN_DATA,
                    help="CycleGAN domain veri seti klasoru (varsayilan havuz)")
    ap.add_argument("--tag", default="",
                    help="cikti adlarina ek (orn: chile, ohio) -> ayri checkpoint/sonuc")
    a = ap.parse_args()
    SFX = f"_{a.tag}" if a.tag else ""

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[cihaz]", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

    G_AB = init_weights(Generator()).to(dev)   # parlamali -> temiz
    G_BA = init_weights(Generator()).to(dev)   # temiz -> parlamali
    D_A  = init_weights(Discriminator()).to(dev)
    D_B  = init_weights(Discriminator()).to(dev)

    mse, l1 = nn.MSELoss(), nn.L1Loss()
    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=a.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=a.lr, betas=(0.5, 0.999))
    sched = LambdaLR(a.epochs, a.decay_start)
    sG = torch.optim.lr_scheduler.LambdaLR(opt_G, sched.step)
    sD = torch.optim.lr_scheduler.LambdaLR(opt_D, sched.step)

    ds = UnpairedDataset(a.data_dir, "train", a.img_size)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    bufA, bufB = ReplayBuffer(), ReplayBuffer()
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    sample_dir = os.path.join(RESULTS, f"cyclegan_samples{SFX}"); os.makedirs(sample_dir, exist_ok=True)

    # ==== DEVAM (RESUME): onceki tam kayit varsa kaldigi yerden surdur ====
    ckpt_path = os.path.join(CHECKPOINTS, f"cyclegan_ckpt{SFX}.pth")
    start_epoch = 0
    if os.path.exists(ckpt_path) and not a.no_resume:
        ck = torch.load(ckpt_path, map_location=dev)
        G_AB.load_state_dict(ck["G_AB"]); G_BA.load_state_dict(ck["G_BA"])
        D_A.load_state_dict(ck["D_A"]);   D_B.load_state_dict(ck["D_B"])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        sG.load_state_dict(ck["sG"]); sD.load_state_dict(ck["sD"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"[DEVAM] Onceki kayittan devam ediliyor -> epoch {start_epoch}/{a.epochs}")
    else:
        print(f"[YENI] Sifirdan egitim baslatiliyor (0/{a.epochs}).")

    # epoch bazli loss kayit dosyasi (grafik icin)
    loss_csv = os.path.join(RESULTS, f"cyclegan_loss_log{SFX}.csv")
    if start_epoch == 0 or not os.path.exists(loss_csv):
        with open(loss_csv, "w") as f:
            f.write("epoch,G,D,cyc,identity,gan,lr\n")

    for ep in range(start_epoch, a.epochs):
        t0 = time.time()
        acc = {"G": 0.0, "D": 0.0, "cyc": 0.0, "id": 0.0, "gan": 0.0, "n": 0}
        for i, batch in enumerate(dl):
            real_A = batch["A"].to(dev); real_B = batch["B"].to(dev)
            valid = torch.ones((real_A.size(0), 1, 30, 30), device=dev)
            fake  = torch.zeros_like(valid)

            # ---- Generators ----
            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                loss_id = (l1(G_BA(real_A), real_A) + l1(G_AB(real_B), real_B)) * a.lambda_id
                fake_B = G_AB(real_A); fake_A = G_BA(real_B)
                loss_gan = mse(D_B(fake_B), valid) + mse(D_A(fake_A), valid)
                loss_cyc = (l1(G_BA(fake_B), real_A) + l1(G_AB(fake_A), real_B)) * a.lambda_cyc
                loss_G = loss_gan + loss_cyc + loss_id
            scaler.scale(loss_G).backward(); scaler.step(opt_G)

            # ---- Discriminators ----
            opt_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                fA = bufA.push_pop(fake_A.detach()); fB = bufB.push_pop(fake_B.detach())
                loss_D = (mse(D_A(real_A), valid) + mse(D_A(fA), fake) +
                          mse(D_B(real_B), valid) + mse(D_B(fB), fake)) * 0.5
            scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()

            acc["G"] += loss_G.item(); acc["D"] += loss_D.item()
            acc["cyc"] += loss_cyc.item(); acc["id"] += loss_id.item()
            acc["gan"] += loss_gan.item(); acc["n"] += 1
            if i % 50 == 0:
                print(f"ep{ep} it{i}/{len(dl)} G={loss_G.item():.3f} D={loss_D.item():.3f} cyc={loss_cyc.item():.3f}")
        sG.step(); sD.step()
        # epoch ortalamalarini CSV'ye yaz
        nb = max(acc["n"], 1)
        with open(loss_csv, "a") as f:
            f.write(f"{ep},{acc['G']/nb:.5f},{acc['D']/nb:.5f},{acc['cyc']/nb:.5f},"
                    f"{acc['id']/nb:.5f},{acc['gan']/nb:.5f},{opt_G.param_groups[0]['lr']:.6f}\n")
        if ep % a.sample_every == 0:
            save_image((torch.cat([real_A, fake_B], 0) * 0.5 + 0.5),
                       os.path.join(sample_dir, f"ep{ep:03d}.png"), nrow=real_A.size(0))
        torch.save(G_AB.state_dict(), os.path.join(CHECKPOINTS, f"G_AB{SFX}.pth"))
        torch.save(G_BA.state_dict(), os.path.join(CHECKPOINTS, f"G_BA{SFX}.pth"))
        # tam durum kaydi (devam edebilmek icin)
        torch.save({
            "epoch": ep,
            "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "sG": sG.state_dict(), "sD": sD.state_dict(),
            "scaler": scaler.state_dict(),
        }, ckpt_path)
        print(f"[ep{ep}] bitti {time.time()-t0:.0f}s  (devam noktasi kaydedildi: {ep+1}. epoch'tan surer)")
    print("[tamam] Egitim bitti. G_AB.pth = parlama giderici")

if __name__ == "__main__":
    main()
