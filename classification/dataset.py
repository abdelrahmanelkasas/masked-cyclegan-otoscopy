"""
Chile (veya deglared kopyasi) icin sinif veri seti.
Klasor yapisinda sinif adlari (CHILE_CLASSES) path icinde aranir.
Yolda 'Test' gecen goruntuler test, digerleri train/val olur.
"""
import os, sys, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CHILE_CLASSES, CLS_IMG_SIZE

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _label_of(path, classes):
    low = path.lower()
    # uzun sinif adini once dene (kismi eslesmeyi onler)
    for i in sorted(range(len(classes)), key=lambda k: -len(classes[k])):
        if classes[i].lower() in low:
            return i
    return None

def scan(root, classes=None):
    """Doner: [(path, label, is_test), ...]  classes verilmezse CHILE_CLASSES."""
    if classes is None:
        classes = CHILE_CLASSES
    items = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.lower().endswith(EXT):
                continue
            p = os.path.join(dp, f)
            lab = _label_of(p, classes)
            if lab is None:
                continue
            is_test = "test" in os.path.relpath(p, root).lower()
            items.append((p, lab, is_test))
    return items

def make_tf(train, size=CLS_IMG_SIZE):
    base = [T.Resize((size, size))]
    if train:
        base += [T.RandomHorizontalFlip(), T.RandomRotation(10),
                 T.ColorJitter(0.1, 0.1, 0.1)]
    base += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(base)

class EarDataset(Dataset):
    def __init__(self, items, train=True, size=CLS_IMG_SIZE):
        self.items = items
        self.tf = make_tf(train, size)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, lab, _ = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), lab
