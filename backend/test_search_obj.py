import sys, json
import numpy as np
import cv2

meta = json.load(open("features/obj_meta.json", encoding="utf-8"))
paths, classes = meta["paths"], meta["classes"]

# Paramètres : python backend\test_search_obj.py <id_image> <color|lbp|hog>
q = int(sys.argv[1]) if len(sys.argv) > 1 else 0
desc = sys.argv[2] if len(sys.argv) > 2 else "color"
X = np.load(f"features/obj_{desc}.npy")

# Distance : L2 pour HOG, L1 pour les histogrammes (couleur, LBP)
if desc == "hog":
    d = np.sqrt(((X - X[q]) ** 2).sum(axis=1))
else:
    d = np.abs(X - X[q]).sum(axis=1)
top = np.argsort(d)[:11]                       # requête + 10 résultats

corrects = sum(classes[i] == classes[q] for i in top[1:])
print(f"Requête : id={q}  classe={classes[q]}  descripteur={desc}")
for rang, i in enumerate(top):
    ok = "OK " if classes[i] == classes[q] else "   "
    print(f"{rang:2d} {ok} id={i:5d}  d={d[i]:.3f}  {classes[i]}")
print(f"Précision@10 = {corrects}/10")

# Planche : cadre vert = bonne classe, rouge = mauvaise classe
vignettes = []
for i in top:
    img = cv2.resize(cv2.imread(paths[i]), (200, 150))
    couleur = (0, 200, 0) if classes[i] == classes[q] else (0, 0, 255)
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=couleur)
    cv2.putText(img, classes[i][:18], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    vignettes.append(img)
vignettes.append(np.zeros_like(vignettes[0]))
planche = np.vstack([np.hstack(vignettes[:6]), np.hstack(vignettes[6:])])

out = f"rapport/obj_{desc}_{q}.jpg"
cv2.imwrite(out, planche)
print("Planche enregistrée :", out)