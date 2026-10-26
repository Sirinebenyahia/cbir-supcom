import json
import numpy as np
import cv2
from tqdm import tqdm
from descriptors import cnn_batch

meta = json.load(open("features/obj_meta.json", encoding="utf-8"))
paths = meta["paths"]

BATCH = 32                                  # images traitées par paquets de 32
feats = []
for s in tqdm(range(0, len(paths), BATCH)):
    imgs = [cv2.imread(p) for p in paths[s:s + BATCH]]
    feats.append(cnn_batch(imgs))

X = np.vstack(feats)
np.save("features/obj_cnn.npy", X)
print("cnn :", X.shape)