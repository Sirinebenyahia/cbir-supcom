import os, json
import numpy as np
from tqdm import tqdm

ROOT = r"data\mirflickr"
N = 25000  # nombre d'images MIRFLICKR-25k

def read_vec(path):
    with open(path) as f:
        return [float(x) for x in f.read().split()]

eh, ht, paths, tags = [], [], [], []

for k in tqdm(range(1, N + 1)):
    idx = k - 1                                    # im1.jpg -> descripteur 0
    sub = os.path.join(str(idx // 10000), f"{idx}.txt")   # ex: im12345 -> 1\12344.txt

    eh.append(read_vec(os.path.join(ROOT, "features_edgehistogram", sub)))
    ht.append(read_vec(os.path.join(ROOT, "features_homogeneoustexture", sub)))
    paths.append(os.path.join(ROOT, "mirflickr", f"im{k}.jpg"))

    tag_file = os.path.join(ROOT, "mirflickr", "meta", "tags", f"tags{k}.txt")
    with open(tag_file, encoding="utf-8", errors="ignore") as f:
        tags.append([t.strip() for t in f if t.strip()])

# Vérification : toutes les images doivent avoir la même taille de descripteur
bad_eh = [i for i, v in enumerate(eh) if len(v) != 150]
bad_ht = [i for i, v in enumerate(ht) if len(v) != 43]
print("Descripteurs EH incorrects :", len(bad_eh), bad_eh[:10])
print("Descripteurs HT incorrects :", len(bad_ht), bad_ht[:10])

if not bad_eh and not bad_ht:
    os.makedirs("features", exist_ok=True)
    np.save("features/mir_eh.npy", np.array(eh, dtype="float32"))
    np.save("features/mir_ht.npy", np.array(ht, dtype="float32"))
    with open("features/mir_meta.json", "w", encoding="utf-8") as f:
        json.dump({"paths": paths, "tags": tags}, f)
    print("EH :", np.array(eh).shape, "| HT :", np.array(ht).shape)
    print("Image 1 :", paths[0], tags[0])
    print("Fichiers enregistrés dans features/")