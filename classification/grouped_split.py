"""
Grup-bazli (near-duplicate'a duyarli) train/val/test bolme.

Amac: birbirine COK benzeyen (near-duplicate) goruntuler ayni bolmede kalsin,
asla eğitim ile dogrulama/test arasinda bolunmesin. Boylece veri sizintisi
(data leakage) ortadan kalkar ve metrikler GERCEKCI olur.

Yontem: her goruntunun 16x16 average-hash'i; ayni sinif icinde Hamming mesafesi
<= thr olanlar birlestirilir (union-find) -> kume. Kumeler komple bir bolmeye atanir.
"""
import os, random
import cv2, numpy as np

def ahash(path, s=16):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (s, s))
    return (im > im.mean()).flatten()

def _cluster(items, thr=10):
    """items: [(path, label, is_test)] -> her oge icin kume kimligi (list)."""
    n = len(items)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    hashes = [ahash(p) for p, _, _ in items]
    by_label = {}
    for i, (_, lab, _) in enumerate(items):
        by_label.setdefault(lab, []).append(i)
    for lab, idxs in by_label.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if hashes[i] is None or hashes[j] is None:
                    continue
                if np.count_nonzero(hashes[i] != hashes[j]) <= thr:
                    union(i, j)
    return [find(i) for i in range(n)]

def grouped_split(items, seed=1234, val_frac=0.15, test_frac=0.20, thr=10):
    """Doner: (train_items, val_items, test_items) — near-duplicate kumeleri bolunmez."""
    cl = _cluster(items, thr)
    clusters = {}
    for i, c in enumerate(cl):
        clusters.setdefault(c, []).append(i)
    # kumeleri sinifa gore grupla (kume tek sinifta cunku ayni sinif icinde olusturuldu)
    by_label = {}
    for c, members in clusters.items():
        lab = items[members[0]][1]
        by_label.setdefault(lab, []).append(members)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for lab, cls in by_label.items():
        rng.shuffle(cls)
        total = sum(len(m) for m in cls)
        n_test, n_val = total * test_frac, total * val_frac
        at = av = 0
        for m in cls:
            if at < n_test:
                test += m; at += len(m)
            elif av < n_val:
                val += m; av += len(m)
            else:
                train += m
    pick = lambda idxs: [items[i] for i in idxs]
    return pick(train), pick(val), pick(test)

def grouped_kfold(items, k=5, seed=1234, thr=10):
    """Grup-bazli (near-duplicate kumeleri bolunmeyen) K-fold.
    Doner: liste halinde k adet (train_items, test_items) cifti.
    Kumeler siniflara gore round-robin fold'lara dagitilir (denge icin)."""
    cl = _cluster(items, thr)
    clusters = {}
    for i, c in enumerate(cl):
        clusters.setdefault(c, []).append(i)
    by_label = {}
    for c, members in clusters.items():
        by_label.setdefault(items[members[0]][1], []).append(members)
    rng = random.Random(seed)
    fold_of = {}  # item index -> fold id
    for lab, cls in by_label.items():
        rng.shuffle(cls)
        for fi, m in enumerate(cls):          # round-robin
            for idx in m:
                fold_of[idx] = fi % k
    folds = []
    for f in range(k):
        test = [items[i] for i in range(len(items)) if fold_of.get(i, 0) == f]
        train = [items[i] for i in range(len(items)) if fold_of.get(i, 0) != f]
        folds.append((train, test))
    return folds
