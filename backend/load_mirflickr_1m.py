import os, re, sqlite3, zipfile
import numpy as np
from tqdm import tqdm

N = 1_000_000
ROOT = os.path.join("data", "mirflickr")
PAT = re.compile(r"(?:^|/)(\d+)/(\d+)\.txt$")      # ex : features_edgehistogram/12/123456.txt


def members(zf):
    """Liste (numéro d'image, nom du fichier dans le zip)."""
    out = []
    for name in zf.namelist():
        if "__MACOSX" in name:
            continue
        m = PAT.search(name)
        if m:
            out.append((int(m.group(2)), name))
    return out


def load_feature(zip_name, dim, out_name):
    """Lit les descripteurs MPEG-7 du million directement dans le zip (sans décompresser)."""
    X = np.zeros((N, dim), dtype="float32")
    ok = np.zeros(N, dtype=bool)
    bad = 0
    with zipfile.ZipFile(os.path.join(ROOT, zip_name)) as zf:
        mem = members(zf)
        print(f"{zip_name} : {len(mem)} fichiers (exemple : {mem[0][1]})")
        for idx, name in tqdm(mem, desc=out_name):
            vals = zf.read(name).split()
            if idx < N and len(vals) == dim:
                X[idx] = np.array(vals, dtype="float32")
                ok[idx] = True
            else:
                bad += 1
    np.save(f"features/mir1m_{out_name}.npy", X)
    np.save(f"features/mir1m_{out_name}_ok.npy", ok)
    print(f"{out_name} : {ok.sum()} descripteurs valides, {bad} ignorés")
    return X


eh = load_feature("features_edgehistogram.zip", 150, "eh")
ht = load_feature("features_homogeneoustexture.zip", 43, "ht")

# Vérification : les 25 000 premières lignes doivent être identiques à MIRFLICKR-25k
for name, X in [("eh", eh), ("ht", ht)]:
    ref = np.load(f"features/mir_{name}.npy")
    print(f"Vérification {name} sur les 25k : écart max = {np.abs(X[:25000] - ref).max():.6f}")
del eh, ht

# Tags du million -> table plein texte SQLite (rowid = numéro de l'image)
con = sqlite3.connect("cbir.db")
con.execute("DROP TABLE IF EXISTS tags1m_vocab")
con.execute("DROP TABLE IF EXISTS tags1m_fts")
con.execute("CREATE VIRTUAL TABLE tags1m_fts USING fts5(tags)")
with zipfile.ZipFile(os.path.join(ROOT, "tags.zip")) as zf:
    mem = members(zf)
    print(f"tags.zip : {len(mem)} fichiers (exemple : {mem[0][1]})")

    def rows():
        for idx, name in tqdm(mem, desc="tags"):
            txt = zf.read(name).decode("utf-8", errors="ignore")
            yield idx, " ".join(t.strip() for t in txt.splitlines() if t.strip())

    con.executemany("INSERT INTO tags1m_fts(rowid, tags) VALUES (?, ?)", rows())
# Vocabulaire des mots-clés (sert à la correction des fautes de frappe)
con.execute("CREATE VIRTUAL TABLE tags1m_vocab USING fts5vocab(tags1m_fts, 'row')")
con.commit()
n = con.execute("SELECT COUNT(*) FROM tags1m_fts WHERE tags != ''").fetchone()[0]
print(f"Tags : {n} images avec des tags indexées")
con.close()