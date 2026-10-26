import os, json, sqlite3

DB = "cbir.db"
if os.path.exists(DB):
    os.remove(DB)                      # on reconstruit la base à zéro
con = sqlite3.connect(DB)

con.execute("""CREATE TABLE images(base TEXT, id INTEGER, path TEXT, classe TEXT,
                                   PRIMARY KEY (base, id))""")
con.execute("CREATE VIRTUAL TABLE tags_fts USING fts5(base UNINDEXED, image_id UNINDEXED, tags)")

# --- MIRFLICKR : chemins + tags Flickr ---
mir = json.load(open("features/mir_meta.json", encoding="utf-8"))
con.executemany("INSERT INTO images VALUES (?,?,?,?)",
                [("mirflickr", i, p, None) for i, p in enumerate(mir["paths"])])
con.executemany("INSERT INTO tags_fts VALUES (?,?,?)",
                [("mirflickr", i, " ".join(t)) for i, t in enumerate(mir["tags"])])

# --- Objects (Caltech-101) : chemins + classe (utilisée comme mot-clé) ---
obj = json.load(open("features/obj_meta.json", encoding="utf-8"))
con.executemany("INSERT INTO images VALUES (?,?,?,?)",
                [("objects", i, p, c) for i, (p, c) in enumerate(zip(obj["paths"], obj["classes"]))])
con.executemany("INSERT INTO tags_fts VALUES (?,?,?)",
                [("objects", i, c.replace("_", " ")) for i, c in enumerate(obj["classes"])])

con.commit()
for base in ["mirflickr", "objects"]:
    n = con.execute("SELECT COUNT(*) FROM images WHERE base = ?", (base,)).fetchone()[0]
    print(f"{base:10s} : {n} images en base")