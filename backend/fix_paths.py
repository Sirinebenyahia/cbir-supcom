import json

for f in ["features/mir_meta.json", "features/obj_meta.json"]:
    meta = json.load(open(f, encoding="utf-8"))
    meta["paths"] = [p.replace("\\", "/") for p in meta["paths"]]
    json.dump(meta, open(f, "w", encoding="utf-8"))
    print(f, ":", meta["paths"][0])