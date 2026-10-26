import os, json
import cv2
import numpy as np
from tqdm import tqdm
from descriptors import DESCRIPTORS

ROOT = r"data\objects\101_ObjectCategories"

# Liste des images (on exclut BACKGROUND_Google qui n'est pas une classe d'objet)
paths, classes = [], []
for c in sorted(os.listdir(ROOT)):
    if c == "BACKGROUND_Google":
        continue
    for f in sorted(os.listdir(os.path.join(ROOT, c))):
        if f.lower().endswith(".jpg"):
            paths.append(os.path.join(ROOT, c, f))
            classes.append(c)

# Calcul des descripteurs pour chaque image (phase off-line)
feats = {name: [] for name in DESCRIPTORS}
for p in tqdm(paths):
    img = cv2.imread(p)
    for name, fn in DESCRIPTORS.items():
        feats[name].append(fn(img))

# Sauvegarde
for name, L in feats.items():
    X = np.array(L, dtype="float32")
    np.save(f"features/obj_{name}.npy", X)
    print(f"{name:6s} : {X.shape}")

with open("features/obj_meta.json", "w", encoding="utf-8") as f:
    json.dump({"paths": paths, "classes": classes}, f)
print(len(paths), "images,", len(set(classes)), "classes")