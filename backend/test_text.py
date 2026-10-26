import sys, sqlite3

con = sqlite3.connect("cbir.db")
q = sys.argv[1]

total = con.execute("SELECT COUNT(*) FROM tags_fts WHERE tags_fts MATCH ?", (q,)).fetchone()[0]
rows = con.execute("""SELECT t.image_id, i.path, t.tags
                      FROM tags_fts t JOIN images i ON i.id = t.image_id
                      WHERE tags_fts MATCH ? LIMIT 5""", (q,)).fetchall()

print(f"Requête « {q} » : {total} images trouvées")
for img_id, path, tg in rows:
    print(f"  id={img_id:5d}  {path}  [{tg[:60]}]")