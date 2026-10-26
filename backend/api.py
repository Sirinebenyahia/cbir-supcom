import os, sys, random
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search as S

app = FastAPI(title="Moteur CBIR - SUP'COM")
N_DISPO = len(S.META["mirflickr"]["paths"])        # images réellement téléchargées (25 000)


def fmt(base, res):
    return [{"id": i, "score": round(s, 4), "label": S.label(base, i),
             "url": f"/image/{base}/{i}"} for i, s in res]

def check(base, names):
    if base not in S.DESCS:
        raise HTTPException(400, f"base inconnue : {base}")
    if not names or any(n not in S.DESCS[base] for n in names):
        raise HTTPException(400, f"descripteurs valides pour {base} : {S.DESCS[base]}")


@app.get("/info")
def info():
    return {b: {"images": S.n_images(b), "descripteurs": S.DESCS[b]} for b in S.DESCS}


@app.get("/random")
def random_images(base: str = "objects", n: int = 15):
    """Images au hasard, pour choisir une image requête (base 1M : parmi les images disponibles)."""
    total = N_DISPO if base == "mirflickr1m" else S.n_images(base)
    ids = random.sample(range(total), n)
    return fmt(base, [(i, 0.0) for i in ids])


@app.get("/search/text")
def search_text(q: str, base: str = "mirflickr", k: int = 30, only_available: bool = False):
    """Recherche par mots-clés : AND / OR / NOT, préfixe, correction des fautes, tri BM25."""
    try:
        ids, used, corrections = S.text_search(base, q)
    except Exception:
        raise HTTPException(400, "requête mots-clés invalide")
    total = len(ids)
    if only_available and base == "mirflickr1m":
        ids = [i for i in ids if i < N_DISPO]
    return {"total": total, "query_used": used, "corrections": corrections,
            "results": fmt(base, [(i, 0.0) for i in ids[:k]])}


@app.get("/search/image")
def search_image(id: int, base: str = "objects", descs: str = "color", k: int = 20,
                 use_index: bool = True, q: str = "", only_available: bool = False):
    """Recherche par image de la base (+ mots-clés optionnels pour combiner ET)."""
    names = descs.split(",")
    check(base, names)
    restrict = None
    if q.strip():
        try:
            restrict = S.text_ids(base, q)
        except Exception:
            raise HTTPException(400, "requête mots-clés invalide")
    qvecs = {n: S.FEATS[(base, n)][id] for n in names}
    filtre = only_available and base == "mirflickr1m"
    # Sur le million, on cherche plus de voisins puis on garde ceux dont l'image est disponible
    res, ms = S.visual_search(base, qvecs, k * 50 if filtre else k, use_index, restrict)
    if filtre:
        res = [r for r in res if r[0] < N_DISPO][:k]
    return {"query": fmt(base, [(id, 0.0)])[0], "time_ms": round(ms, 2),
            "results": fmt(base, res)}


@app.post("/search/upload")
async def search_upload(file: UploadFile = File(...), descs: str = Form("color"),
                        k: int = Form(20), use_index: bool = Form(True)):
    """Recherche avec une image envoyée par l'utilisateur (base Objects)."""
    names = descs.split(",")
    check("objects", names)
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "image illisible")
    qvecs = {n: S.DESCRIPTORS[n](img).astype("float32") for n in names}
    res, ms = S.visual_search("objects", qvecs, k, use_index)
    return {"time_ms": round(ms, 2), "results": fmt("objects", res)}


def placeholder(img_id):
    """Vignette pour une image du million qui n'est pas téléchargée (seuls ses descripteurs le sont)."""
    img = np.full((180, 240, 3), 236, np.uint8)
    cv2.rectangle(img, (6, 6), (233, 173), (200, 200, 200), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Image non", (62, 75), font, 0.7, (110, 110, 110), 2)
    cv2.putText(img, "telechargee", (50, 105), font, 0.7, (110, 110, 110), 2)
    cv2.putText(img, f"#{img_id}", (80, 140), font, 0.6, (64, 64, 160), 2)
    _, buf = cv2.imencode(".png", img)
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/image/{base}/{id}")
def image(base: str, id: int):
    path = S.image_path(base, id)
    if path is None or not os.path.exists(path):
        return placeholder(id)
    return FileResponse(path)