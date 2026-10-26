import sys, json
import numpy as np
import cv2

# Chargement des descripteurs et métadonnées
eh = np.load("features/mir_eh.npy")
ht = np.load("features/mir_ht.npy")
meta = json.load(open("features/mir_meta.json", encoding="utf-8"))
paths, tags = meta["paths"], meta["tags"]

# Paramètres : python backend\test_search_mir.py <id_image> <eh|ht>
q = int(sys.argv[1]) if len(sys.argv) > 1 else 0        # id 0 = im1.jpg
desc = sys.argv[2] if len(sys.argv) > 2 else "eh"
X = eh if desc == "eh" else ht

# Recherche exhaustive : distance L1 entre la requête et toutes les images
d = np.abs(X - X[q]).sum(axis=1)
top = np.argsort(d)[:11]          # la requête elle-même + 10 plus proches

print(f"Requête : {paths[q]}  tags={tags[q][:5]}  descripteur={desc}")
for rang, i in enumerate(top):
    print(f"{rang:2d}  id={i:5d}  d={d[i]:.3f}  {tags[i][:5]}")

# Planche de résultats (2 lignes x 6 images)
vignettes = []
for i in top:
    img = cv2.resize(cv2.imread(paths[i]), (200, 150))
    cv2.putText(img, f"id {i}  d={d[i]:.2f}", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    vignettes.append(img)
vignettes.append(np.zeros_like(vignettes[0]))           # case vide pour faire 12
planche = np.vstack([np.hstack(vignettes[:6]), np.hstack(vignettes[6:])])

out = f"rapport/test_{desc}_{q}.jpg"
cv2.imwrite(out, planche)
print("Planche enregistrée :", out)