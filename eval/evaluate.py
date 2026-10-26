import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_QUERIES = 1000                      # nombre de requêtes tirées au hasard
rng = np.random.default_rng(42)

meta = json.load(open("features/obj_meta.json", encoding="utf-8"))
classes = np.array(meta["classes"])
FEATS = {n: np.load(f"features/obj_{n}.npy") for n in ["color", "lbp", "hog", "cnn"]}
METRIC = {"color": "l1", "lbp": "l1", "hog": "l2", "cnn": "l2"}
NOMS = {"color": "Couleur", "lbp": "LBP", "hog": "HOG", "cnn": "CNN"}
CONFIGS = [("color",), ("lbp",), ("hog",), ("cnn",),
           ("color", "lbp"), ("color", "hog"), ("lbp", "hog"),
           ("color", "lbp", "hog"), ("hog", "cnn"), ("color", "lbp", "hog", "cnn")]
LEVELS = np.linspace(0, 1, 11)        # niveaux de rappel 0, 0.1, ..., 1


def dist(name, q):
    diff = FEATS[name] - FEATS[name][q]
    if METRIC[name] == "l1":
        return np.abs(diff).sum(axis=1)
    return np.sqrt((diff ** 2).sum(axis=1))


queries = rng.choice(len(classes), N_QUERIES, replace=False)
stats = {c: {"P@10": [], "P@20": [], "R@20": [], "AP": [], "courbe": []} for c in CONFIGS}
ap_par_classe = {c: {} for c in CONFIGS}

t0 = time.time()
for n, q in enumerate(queries):
    D = {name: dist(name, q) for name in FEATS}
    pertinent = classes == classes[q]            # pertinent = même classe que la requête
    for cfg in CONFIGS:
        if len(cfg) == 1:
            score = D[cfg[0]].astype(float)
        else:                                    # fusion : distances normalisées puis moyennées
            score = sum(D[x] / (D[x].max() + 1e-8) for x in cfg) / len(cfg)
        score[q] = np.inf                        # on exclut la requête elle-même
        order = np.argsort(score)[:-1]
        rel = pertinent[order]
        n_rel = rel.sum()
        cum = np.cumsum(rel)
        precision = cum / np.arange(1, len(rel) + 1)
        rappel = cum / n_rel

        s = stats[cfg]
        s["P@10"].append(cum[9] / 10)
        s["P@20"].append(cum[19] / 20)
        s["R@20"].append(cum[19] / n_rel)
        ap = (precision * rel).sum() / n_rel     # Average Precision
        s["AP"].append(ap)
        s["courbe"].append([precision[rappel >= r].max() for r in LEVELS])
        ap_par_classe[cfg].setdefault(classes[q], []).append(ap)
    if (n + 1) % 100 == 0:
        print(f"{n + 1}/{N_QUERIES} requêtes évaluées ({time.time() - t0:.0f} s)")

# ---------- Tableau des résultats ----------
print(f"\n{'Descripteurs':30s} {'P@10':>7s} {'P@20':>7s} {'R@20':>7s} {'mAP':>7s}")
lignes = []
for cfg in CONFIGS:
    s = stats[cfg]
    nom = " + ".join(NOMS[x] for x in cfg)
    vals = [100 * np.mean(s[m]) for m in ["P@10", "P@20", "R@20", "AP"]]
    lignes.append((nom, vals))
    print(f"{nom:30s} " + " ".join(f"{v:6.1f}%" for v in vals))

os.makedirs("eval", exist_ok=True)
with open("eval/resultats_objects.csv", "w", encoding="utf-8") as f:
    f.write("descripteurs;P@10;P@20;R@20;mAP\n")
    for nom, vals in lignes:
        f.write(nom + ";" + ";".join(f"{v:.1f}" for v in vals) + "\n")

# ---------- Courbes rappel / précision ----------
plt.figure(figsize=(9, 6.5))
for cfg in CONFIGS:
    style = "-" if len(cfg) == 1 else "--"
    plt.plot(LEVELS, np.mean(stats[cfg]["courbe"], axis=0), style, marker="o", markersize=4,
             label=" + ".join(NOMS[x] for x in cfg))
plt.xlabel("Rappel")
plt.ylabel("Précision")
plt.title(f"Courbes rappel/précision — base Objects ({N_QUERIES} requêtes)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.savefig("rapport/courbe_rappel_precision.png", dpi=150, bbox_inches="tight")

# ---------- Classes les mieux / moins bien reconnues ----------
for cfg in [("lbp", "hog"), ("cnn",)]:
    apc = sorted(((c, np.mean(v)) for c, v in ap_par_classe[cfg].items()), key=lambda x: -x[1])
    print(f"\n[{' + '.join(NOMS[x] for x in cfg)}]")
    print("  5 classes les mieux reconnues :", ", ".join(f"{c} ({100*a:.0f}%)" for c, a in apc[:5]))
    print("  5 classes les moins bien reconnues :", ", ".join(f"{c} ({100*a:.0f}%)" for c, a in apc[-5:]))
print("\nFichiers : eval/resultats_objects.csv et rapport/courbe_rappel_precision.png")