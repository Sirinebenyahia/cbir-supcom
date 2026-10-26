import os, sys, time, pickle
import numpy as np
import faiss
from vptree_index import VPTree

sys.setrecursionlimit(20000)
os.makedirs("indexes", exist_ok=True)

# ---------- Base Objects : un VP-tree par descripteur ----------
print("=== Base Objects : VP-tree ===")
for name, metric in [("color", "l1"), ("lbp", "l1"), ("hog", "l2"), ("cnn", "l2")]:
    X = np.load(f"features/obj_{name}.npy")
    t = time.perf_counter()
    tree = VPTree(X, metric=metric)
    with open(f"indexes/vp_obj_{name}.pkl", "wb") as f:
        pickle.dump(tree, f)
    print(f"{name:6s} : VP-tree construit en {time.perf_counter() - t:.1f} s")

# ---------- Base MIRFLICKR : index FAISS IVF (100 partitions k-means) ----------
print("=== Base MIRFLICKR : FAISS IVF ===")
for name in ["eh", "ht"]:
    X = np.load(f"features/mir_{name}.npy").astype("float32")
    d = X.shape[1]
    t = time.perf_counter()
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)
    index.train(X)          # k-means : apprend les 100 centres de partitions
    index.add(X)            # range chaque image dans sa partition
    faiss.write_index(index, f"indexes/ivf_mir_{name}.faiss")
    print(f"{name:6s} : index IVF construit en {time.perf_counter() - t:.1f} s")