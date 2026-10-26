import os, sys, re, json, time, pickle, sqlite3, difflib
from collections import defaultdict
import numpy as np
import faiss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.setrecursionlimit(20000)
from vptree_index import VPTree          # nécessaire pour recharger les VP-trees
from descriptors import DESCRIPTORS

ROOT = os.path.dirname(HERE)             # dossier cbir/
def P(*parts):
    return os.path.join(ROOT, *parts)

METRIC = {"color": "l1", "lbp": "l1", "hog": "l2", "cnn": "l2", "eh": "l2", "ht": "l2"}
DESCS = {"objects": ["color", "lbp", "hog", "cnn"],
         "mirflickr": ["eh", "ht"],
         "mirflickr1m": ["eh", "ht"]}
N_1M = 1_000_000

# ---------- Chargement au démarrage (phase on-line prête) ----------
FEATS, INDEX = {}, {}
for name in DESCS["objects"]:
    FEATS[("objects", name)] = np.load(P("features", f"obj_{name}.npy"))
    with open(P("indexes", f"vp_obj_{name}.pkl"), "rb") as f:
        INDEX[("objects", name)] = pickle.load(f)

for name in ["eh", "ht"]:
    # MIRFLICKR-25k
    FEATS[("mirflickr", name)] = np.load(P("features", f"mir_{name}.npy")).astype("float32")
    index = faiss.read_index(P("indexes", f"ivf_mir_{name}.faiss"))
    index.nprobe = 10
    INDEX[("mirflickr", name)] = index
    # MIRFLICKR-1M (1 000 000 de descripteurs MPEG-7)
    FEATS[("mirflickr1m", name)] = np.load(P("features", f"mir1m_{name}.npy"))
    index = faiss.read_index(P("indexes", f"ivf_mir1m_{name}.faiss"))
    index.nprobe = 16
    INDEX[("mirflickr1m", name)] = index

META = {
    "objects": json.load(open(P("features", "obj_meta.json"), encoding="utf-8")),
    "mirflickr": json.load(open(P("features", "mir_meta.json"), encoding="utf-8")),
}


def _db():
    return sqlite3.connect(P("cbir.db"))


# ---------- Fonctions utilitaires ----------
def n_images(base):
    return N_1M if base == "mirflickr1m" else len(META[base]["paths"])

def image_path(base, i):
    """Chemin de l'image, ou None si l'image n'est pas téléchargée (base 1M, id >= 25000)."""
    if base == "mirflickr1m":
        paths = META["mirflickr"]["paths"]
        return P(paths[i]) if i < len(paths) else None
    return P(META[base]["paths"][i])

def label(base, i):
    if base == "objects":
        return META["objects"]["classes"][i]
    if base == "mirflickr":
        return " ".join(META["mirflickr"]["tags"][i][:5])
    con = _db()
    row = con.execute("SELECT tags FROM tags1m_fts WHERE rowid = ?", (i,)).fetchone()
    con.close()
    return " ".join(row[0].split()[:5]) if row else ""


# ---------- Recherche textuelle (SQLite FTS5) ----------
_VOCAB = {}
MIN_PREFIXE = 5     # en dessous de 5 lettres : mot exact + pluriel (évite cat -> catalunya, cathedral...)

def _vocab(base):
    """Vocabulaire des mots-clés d'une base (chargé une seule fois)."""
    if base not in _VOCAB:
        con = _db()
        if base == "mirflickr1m":
            words = {r[0] for r in con.execute("SELECT term FROM tags1m_vocab")}
        else:
            words = set()
            for (t,) in con.execute("SELECT tags FROM tags_fts WHERE base = ?", (base,)):
                words.update(t.lower().split())
        con.close()
        by_first = defaultdict(list)
        for w in words:
            by_first[w[0]].append(w)
        _VOCAB[base] = (words, sorted(words), by_first)
    return _VOCAB[base]

def _terme(w):
    """Mot long : recherche par préfixe (sunset -> sunsets, sunsetbeach).
    Mot court : mot exact ou pluriel (cat -> cat, cats ; mais pas catalunya)."""
    if len(w) >= MIN_PREFIXE:
        return f"{w}*"
    return f"({w} OR {w}s)"

def expand_query(base, q):
    """Transforme la requête utilisateur, comme la recherche floue d'ElasticSearch :
    - correction des fautes de frappe (mot absent -> mots proches du vocabulaire) ;
    - variantes : préfixe pour les mots longs, pluriel pour les mots courts ;
    - opérateurs AND / OR / NOT et parenthèses conservés."""
    words, wlist, by_first = _vocab(base)
    out, corrections = [], {}
    for tok in re.findall(r"[()]|[^\s()]+", q):
        if tok in ("(", ")"):
            out.append(tok)
            continue
        if tok.upper() in ("AND", "OR", "NOT"):
            out.append(tok.upper())
            continue
        w = re.sub(r"\W", "", tok.lower())
        if not w:
            continue
        if w in words:
            out.append(_terme(w))
            continue
        pool = by_first.get(w[0], []) if len(wlist) > 200_000 else wlist   # gros vocabulaire : même 1re lettre
        candidats = [v for v in pool if abs(len(v) - len(w)) <= 2]
        proches = difflib.get_close_matches(w, candidats, n=3, cutoff=0.75)
        if proches:
            corrections[w] = proches[0]
            out.append("(" + " OR ".join(_terme(c) for c in proches) + ")")
        else:
            out.append(_terme(w))
    return " ".join(out), corrections

def text_search(base, q):
    """Retourne (ids triés par pertinence BM25, requête FTS5 utilisée, corrections)."""
    fts_q, corrections = expand_query(base, q)
    if not fts_q.strip():
        return [], fts_q, corrections
    con = _db()
    if base == "mirflickr1m":
        rows = con.execute("""SELECT rowid FROM tags1m_fts
                              WHERE tags1m_fts MATCH ? ORDER BY rank""", (fts_q,)).fetchall()
    else:
        rows = con.execute("""SELECT image_id FROM tags_fts
                              WHERE tags_fts MATCH ? AND base = ? ORDER BY rank""",
                           (fts_q, base)).fetchall()
    con.close()
    return [int(r[0]) for r in rows], fts_q, corrections

def text_ids(base, q):
    return text_search(base, q)[0]


# ---------- Recherche visuelle ----------
def distances(base, name, q, ids=None):
    """Distances entre q et les vecteurs de la base (ids=None : toute la base)."""
    X = FEATS[(base, name)]
    diff = (X if ids is None else X[ids]) - q
    if METRIC[name] == "l1":
        return np.abs(diff).sum(axis=1)
    return np.sqrt((diff ** 2).sum(axis=1))

def knn_index(base, name, q, k):
    """k plus proches voisins via l'index (VP-tree pour Objects, FAISS IVF pour MIRFLICKR)."""
    if base == "objects":
        return [i for i, _ in INDEX[(base, name)].search(q, k)]
    _, I = INDEX[(base, name)].search(q.reshape(1, -1).astype("float32"), k)
    return [int(i) for i in I[0] if i >= 0]

def visual_search(base, qvecs, k=20, use_index=True, restrict=None):
    """qvecs : {descripteur: vecteur requête}.
    restrict : liste d'ids autorisés (combinaison mots-clés ET image).
    Retourne ([(id, score)], temps en ms)."""
    t = time.perf_counter()
    names = list(qvecs)
    full = False

    # 1) Candidats
    if restrict is not None:
        cand = np.array(sorted(set(restrict)), dtype=int)       # filtre par mots-clés
    elif use_index:
        c = set()
        n_cand = k if len(names) == 1 else max(100, 5 * k)
        for n in names:
            c.update(knn_index(base, n, qvecs[n], n_cand))
        cand = np.array(sorted(c), dtype=int)
    else:
        cand = np.arange(n_images(base))                         # recherche exhaustive
        full = True
    if len(cand) == 0:
        return [], 1000 * (time.perf_counter() - t)

    # 2) Score : distance exacte, normalisée si plusieurs descripteurs (fusion)
    total = np.zeros(len(cand))
    for n in names:
        d = distances(base, n, qvecs[n], None if full else cand)
        if len(names) > 1:
            d = d / (d.max() + 1e-8)
        total += d
    total /= len(names)

    order = np.argsort(total)[:k]
    res = [(int(cand[o]), float(total[o])) for o in order]
    return res, 1000 * (time.perf_counter() - t)