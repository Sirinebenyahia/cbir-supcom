import sys, time, pickle
import numpy as np
import faiss

sys.path.insert(0, "backend")
sys.setrecursionlimit(20000)
from vptree_index import VPTree   # nécessaire pour recharger les arbres

K = 10
rng = np.random.default_rng(0)

# ---------- Objects : recherche exhaustive vs VP-tree ----------
print("=== Base Objects (8677 images) : exhaustif vs VP-tree ===")
for name in ["color", "lbp", "hog"]:
    X = np.load(f"features/obj_{name}.npy")
    with open(f"indexes/vp_obj_{name}.pkl", "rb") as f:
        tree = pickle.load(f)
    qs = rng.choice(len(X), 50, replace=False)
    t_ex = t_vp = 0.0
    n_dist = 0
    identiques = 0
    for q in qs:
        t = time.perf_counter()
        if name == "hog":
            d = np.sqrt(((X - X[q]) ** 2).sum(axis=1))
        else:
            d = np.abs(X - X[q]).sum(axis=1)
        exact = set(np.argsort(d)[:K])
        t_ex += time.perf_counter() - t

        t = time.perf_counter()
        res = tree.search(X[q], K)
        t_vp += time.perf_counter() - t
        n_dist += tree.n_dist
        identiques += len(exact & {i for i, _ in res}) / K
    n = len(qs)
    print(f"{name:6s} | exhaustif {1000*t_ex/n:7.2f} ms | VP-tree {1000*t_vp/n:7.2f} ms"
          f" | distances calculées : {n_dist/n:5.0f} / {len(X)} ({100*n_dist/n/len(X):.0f} %)"
          f" | mêmes résultats : {100*identiques/n:.0f} %")

# ---------- MIRFLICKR : recherche exhaustive vs FAISS IVF ----------
print("\n=== Base MIRFLICKR (25000 images) : exhaustif vs FAISS IVF ===")
for name in ["eh", "ht"]:
    X = np.load(f"features/mir_{name}.npy").astype("float32")
    index = faiss.read_index(f"indexes/ivf_mir_{name}.faiss")
    qs = rng.choice(len(X), 100, replace=False)
    for nprobe in [1, 5, 10, 20]:
        index.nprobe = nprobe          # nombre de partitions visitées
        t_ex = t_ivf = 0.0
        rappel = 0.0
        for q in qs:
            t = time.perf_counter()
            d = ((X - X[q]) ** 2).sum(axis=1)
            exact = set(np.argsort(d)[:K])
            t_ex += time.perf_counter() - t

            t = time.perf_counter()
            _, I = index.search(X[q:q + 1], K)
            t_ivf += time.perf_counter() - t
            rappel += len(exact & set(I[0])) / K
        n = len(qs)
        print(f"{name} nprobe={nprobe:2d} | exhaustif {1000*t_ex/n:6.2f} ms"
              f" | IVF {1000*t_ivf/n:6.2f} ms | résultats exacts retrouvés : {100*rappel/n:.0f} %")