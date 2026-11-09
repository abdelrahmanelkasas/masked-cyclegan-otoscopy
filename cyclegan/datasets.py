"""Unpaired (eslesmesiz) iki-domain veri yukleyici."""
import os, glob, random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

EXT = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

def _list(d):
    f = []
    for e in EXT:
        f += glob.glob(os.path.join(d, e))
    return sorted(f)

class UnpairedDataset(Dataset):
    def __init__(self, root, mode="train", img_size=256, augment=True):
        self.A = _list(os.path.join(root, mode + "A"))
        self.B = _list(os.path.join(root, mode + "B"))
        assert self.A and self.B, f"Veri bulunamadi: {root} ({mode})"
        tf = [T.Resize((img_size, img_size)), ]
        if augment and mode == "train":
            tf += [T.RandomHorizontalFlip()]
        tf += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        self.tf = T.Compose(tf)
    def __len__(self):
        return max(len(self.A), len(self.B))
    def _load(self, p):
        return self.tf(Image.open(p).convert("RGB"))
    def __getitem__(self, i):
        a = self._load(self.A[i % len(self.A)])
        b = self._load(self.B[random.randint(0, len(self.B) - 1)])
        return {"A": a, "B": b}
