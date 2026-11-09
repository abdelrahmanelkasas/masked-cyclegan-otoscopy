# Targeted masked CycleGAN for specular highlight removal in otoscopic images

Code for the article *"Targeted masked CycleGAN for specular highlight removal in otoscopic images
with leakage-free evaluation of eardrum-condition classification"* (Elkasas, Şeker & Özerdem;
submitted to *Scientific Reports*).

The method removes specular highlights (glare) from otoscopic images by blending the output of a
CycleGAN generator into the image **only inside detected glare regions** (feathered mask), and leaves
low-glare images unchanged. The evaluation uses grouped five-fold cross-validation that keeps
near-duplicate images in one fold and **retrains the CycleGAN inside each fold on training images only**,
so test images never influence any learnable step.

## 1. Environment

- Python 3.12 and one NVIDIA GPU (developed on an RTX 3060, CUDA 12.1). Fixed seed: 1234.

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Data (public; not redistributed here)

| Dataset | Images | Classes | Source |
|---|---|---|---|
| Dataset 1 (Chile) | 880 | chronic otitis media, earwax plug, myringosclerosis, normal | Viscaino et al., PLoS ONE 15, e0229226 (2020); [figshare](https://figshare.com/articles/dataset/Ear_imagery_database/11886630) |
| Dataset 2 (Ohio) | 454 | effusion, normal, tympanostomy tube | Camalan et al., PLoS ONE 15, e0232776 (2020); [Zenodo](https://zenodo.org/records/3595567) |

```
python data/download_data.py          # -> data/eardrum_public_data/{Chile,Ohio}
```

## 3. Reproducing the results

| Article item | Command | Output (`results/`) |
|---|---|---|
| Table 4, Fig. 3 (leakage-free five-fold CV, CycleGAN retrained per fold) | `python classification/run_perfold_cyclegan.py --dataset chile --models resnet50,efficientnet_b0,vit_b_16 --folds 5 --gan_epochs 60 --cls_epochs 30` (repeat with `--dataset ohio`) | `study_<dataset>_perfold_*.json` |
| Statistics in Table 4 and Supplementary Table S2 (no GPU needed) | `python analysis/compute_statistics.py` | `statistics_summary.json` (incl. cluster-robust CIs) |
| Table 3, Supplementary Table S3 and Fig. S3 (single grouped split, identical test images, training-only CycleGAN) | `python classification/single_split_leakfree.py` | `single_split_leakfree.json`, `*_perimage.csv` |
| Fig. 4 (Grad-CAM on held-out test images) | `python classification/gradcam_heldout.py --n 0` (all 177 test images) and `--n 12` (top-12 by glare) | `gradcam_heldout_*` |
| Dataset-level CycleGAN (Table 2, Fig. 2, Supplementary Figs. S1–S2) | `python preprocessing/build_cyclegan_dataset.py --only chile` then `python cyclegan/train.py --epochs 150 --decay_start 100 --data_dir data/cyclegan_chile --tag chile` (same for `ohio`) | `checkpoints/G_AB_<dataset>.pth` |
| Masked deglaring with the dataset-level model | `python cyclegan/apply_deglare.py --mask --src <raw dir> --dst <out dir> --ckpt checkpoints/G_AB_<dataset>.pth` | masked images |
| Table 2 (glare reduction, SSIM, PSNR) | `python cyclegan/eval_quality.py --orig <raw dir> --deg <masked dir> --tag <dataset>` | `gan_quality_<dataset>.json` |
| Supplementary Table S1 (glare-threshold sensitivity) | `python analysis/threshold_sensitivity.py` | `threshold_sensitivity_full.json` |

Runtime on an RTX 3060: about 1.5–2 h per fold for the five-fold pipeline (dominated by CycleGAN
training) and about 2–3 h for `single_split_leakfree.py`. A quick smoke test:
`python classification/single_split_leakfree.py --gan_epochs 2 --cls_epochs 1 --max_per_domain 20`.

**Published results.** The `results/` folder already contains the raw outputs reported in the article
(per-fold predictions and accuracies, statistics, image-quality metrics, Grad-CAM ratios, threshold
sensitivity and training logs), so `python analysis/compute_statistics.py` reproduces the statistics in
Table 4 and Supplementary Table S2 without a GPU.

## 4. Key implementation details

- **Near-duplicate-safe splitting** (`classification/grouped_split.py`): 16×16 average hash, Hamming
  distance ≤ 10 within a class, union-find clusters; whole clusters are assigned to one fold.
  With seed 1234 the test folds contain 139/193/148/202/198 (Chile) and 91/92/91/91/89 (Ohio) images.
- **Glare mask** (`preprocessing/glare_utils.py`): HSV, V ≥ 220 and S ≤ 60, then 3×3 opening and
  dilation. Feathered blending uses two 3×3 dilations and a 9×9 Gaussian kernel. Images with a glare
  ratio below 0.5% are left unchanged.
- **No test leakage**: domain construction, CycleGAN training and classifier training use training
  images only. Five-fold runs use no early stopping; single-split runs select checkpoints on the
  validation subset only.

## 5. Citation

If you use this code, please cite the article (citation details will be added on publication).

## License

MIT (see `LICENSE`). The datasets are subject to the licences of their original providers.
