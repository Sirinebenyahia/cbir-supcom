import os, time
import numpy as np
import faiss

K = 10
rng = np.random.default_rng(0)
os.makedirs("eval", exist_ok=True)
lignes = ["Descripteur;nprobe;Exhaustif (ms);Index IVF (ms);Gain;Résultats exacts retrouvés (%)"]

for name in ["eh", "ht"]:
    X = np.load(f"features/mir1m_{name}.npy")
    ok = np.load(f"features/mir1m_{name}_ok.npy")
    ids = np.nonzero(ok)[0].astype("int64")
    Xv = X if ok.all() else X[ids]
    d = X.shape[1]

    # ---------- Construction : IVF à 1024 partitions (k-means) ----------
    t = time.perf_counter()
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 1024)
    index.train(Xv[rng.choice(len(ids), 100_000, replace=False)])   # k-means sur 100k vecteurs
    index.add_with_ids(Xv, ids)
    faiss.write_index(index, f"indexes/ivf_mir1m_{name}.faiss")
    print(f"\n{name} : index IVF (1024 partitions) sur {len(ids)} vecteurs "
          f"construit en {time.perf_counter() - t:.1f} s")

    # ---------- Mesure : recherche exhaustive vs index ----------
    qs = rng.choice(len(ids), 20, replace=False)
    for nprobe in [4, 16, 64]:
        index.nprobe = nprobe
        t_ex = t_ivf = rappel = 0.0
        for qi in qs:
            q = Xv[qi]
            t = time.perf_counter()
            dist = ((Xv - q) ** 2).sum(axis=1)
            exact = set(ids[np.argpartition(dist, K)[:K]].tolist())
            t_ex += time.perf_counter() - t

            t = time.perf_counter()
            _, I = index.search(q.reshape(1, -1), K)
            t_ivf += time.perf_counter() - t
            rappel += len(exact & set(I[0].tolist())) / K
        n = len(qs)
        ex_ms, ivf_ms, gain, rec = 1000 * t_ex / n, 1000 * t_ivf / n, t_ex / max(t_ivf, 1e-9), 100 * rappel / n
        print(f"{name} nprobe={nprobe:2d} | exhaustif {ex_ms:8.1f} ms | IVF {ivf_ms:6.2f} ms"
              f" | gain x{gain:6.0f} | résultats exacts retrouvés : {rec:.0f} %")
        lignes.append(f"{name.upper()};{nprobe};{ex_ms:.1f};{ivf_ms:.2f};x{gain:.0f};{rec:.0f}")
    del X, Xv, index

with open("eval/temps_index_1m.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(lignes) + "\n")
print("\nRésultats enregistrés dans eval/temps_index_1m.csv")