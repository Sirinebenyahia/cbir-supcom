import os
import html
import base64
import random
import requests
import pandas as pd
import streamlit as st

API = "http://127.0.0.1:8000"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="CBIR — Recherche d'images par le contenu",
                   page_icon="🔎", layout="wide")

# ======================= Style =======================
st.markdown("""<style>
.block-container {padding-top: 1.4rem; max-width: 1450px;}
.hero {background: linear-gradient(135deg,#0f2a5f 0%,#1e40af 60%,#2563eb 100%);
       color:#fff; padding:22px 28px; border-radius:16px; margin-bottom:14px;}
.hero h1 {color:#fff; font-size:1.85rem; margin:0 0 6px 0;}
.hero p {color:#dbeafe; margin:0; font-size:0.95rem;}
.card {border:2px solid #e2e8f0; border-radius:12px; padding:6px; background:#fff;
       box-shadow:0 1px 3px rgba(15,23,42,.08); margin-bottom:6px;}
.card img {width:100%; height:140px; object-fit:cover; border-radius:8px; display:block;}
.card .meta {display:flex; justify-content:space-between; align-items:center; gap:6px;
             font-size:.78rem; color:#475569; padding:6px 2px 0 2px;}
.card .rank {background:#1e40af; color:#fff; border-radius:999px; padding:1px 8px; font-weight:600;}
.card .lbl {white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1;}
.query {border:3px solid #f59e0b; border-radius:14px; padding:8px; background:#fffbeb;}
.query img {width:100%; border-radius:10px; display:block;}
.query .cap {font-size:.85rem; color:#92400e; padding-top:6px; text-align:center; font-weight:600;}
.legend {font-size:.82rem; color:#475569; margin-bottom:8px;}
.dot {display:inline-block; width:11px; height:11px; border-radius:3px; margin:0 5px 0 12px; vertical-align:middle;}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero">
<h1>🔎 Moteur d'indexation et de recherche d'images par le contenu</h1>
<p>Recherche textuelle et visuelle (CBIR) · Caltech-101, MIRFLICKR-25k et MIRFLICKR-1M (1 000 000 de descripteurs) ·
Descripteurs Couleur HSV, LBP, HOG, CNN et MPEG-7 · Index VP-Tree &amp; FAISS IVF · Client / serveur FastAPI (JSON)</p>
</div>""", unsafe_allow_html=True)


# ======================= Connexion au serveur =======================
@st.cache_data(ttl=60)
def api_info():
    return requests.get(f"{API}/info", timeout=5).json()

try:
    INFO = api_info()
except Exception:
    st.error("⚠️ Le serveur de recherche n'est pas joignable. Lancez d'abord : "
             "`uvicorn api:app --app-dir backend --port 8000`")
    st.stop()

NOMS = {"color": "Couleur HSV", "lbp": "Texture LBP", "hog": "Forme HOG", "cnn": "Deep CNN ResNet-50",
        "eh": "MPEG-7 Edge Histogram", "ht": "MPEG-7 Homogeneous Texture"}
DIMS = {"color": 128, "lbp": 59, "hog": 3780, "cnn": 2048, "eh": 150, "ht": 43}
BASES = {"objects": "📦 Objects — Caltech-101",
         "mirflickr": "🌄 MIRFLICKR-25k — MPEG-7",
         "mirflickr1m": "🌍 MIRFLICKR-1M — MPEG-7 (1 million)"}

ss = st.session_state
for key, val in {"results": [], "query": None, "msg": None, "picker": [], "base": None,
                 "last_ms": None, "total": 0, "compare": None}.items():
    ss.setdefault(key, val)

# ======================= Barre latérale =======================
with st.sidebar:
    st.markdown("### ⚙️ Paramètres de recherche")
    base = st.radio("Base d'images", [b for b in BASES if b in INFO], format_func=BASES.get)
    if ss.base != base:                                   # changement de base : réinitialisation
        ss.base, ss.results, ss.query, ss.msg, ss.picker, ss.compare = base, [], None, None, [], None
        ss.kw_filter = ""

    all_descs = INFO[base]["descripteurs"]
    descs = st.multiselect("Descripteurs", all_descs,
                           default=["cnn"] if base == "objects" else ["eh"],
                           format_func=lambda d: f"{NOMS[d]} ({DIMS[d]})", key=f"descs_{base}")
    k = st.slider("Nombre de résultats", 5, 50, 20)
    use_index = st.toggle("Recherche indexée (VP-Tree / FAISS)", value=True)
    st.caption("Désactiver l'index = recherche exhaustive sur toute la base.")
    only_av = False
    if base == "mirflickr1m":
        only_av = st.toggle("Afficher uniquement les images disponibles", value=True,
                            help="La recherche porte sur le million entier ; seuls les résultats "
                                 "dont l'image est téléchargée sont affichés.")

    st.divider()
    c1, c2 = st.columns(2)
    nb_img = INFO[base]["images"]
    c1.metric("Images", "1 M" if nb_img >= 1_000_000 else f"{nb_img:,}".replace(",", " "))
    c2.metric("Descripteurs", len(all_descs))
    if base == "mirflickr1m":
        st.caption("Recherche sur 1 000 000 de descripteurs MPEG-7. Seules les 25 000 premières "
                   "images sont téléchargées : les autres s'affichent comme « non téléchargée ».")
    with st.expander("ℹ️ Descripteurs et index"):
        st.markdown("""
| Descripteur | Attribut | Taille |
|---|---|---|
| Couleur HSV | couleur | 128 |
| LBP uniforme | texture | 59 |
| HOG | forme | 3780 |
| CNN ResNet-50 | sémantique (deep learning) | 2048 |
| MPEG-7 EH | contours | 150 |
| MPEG-7 HT | texture | 43 |

**Index** : VP-Tree (Objects, recherche exacte) ; FAISS IVF à 100 partitions (MIRFLICKR-25k, nprobe = 10)
et à 1024 partitions (MIRFLICKR-1M, nprobe = 16).

**Fusion** : distances normalisées puis moyennées.

**Mots-clés** : SQLite FTS5, tri BM25, préfixe et correction des fautes.
""")


# ======================= Appels au serveur =======================
def call_search(img_id, keywords="", index=None):
    return requests.get(f"{API}/search/image", timeout=120, params={
        "id": img_id, "base": base, "descs": ",".join(descs), "k": k,
        "use_index": use_index if index is None else index, "q": keywords,
        "only_available": only_av})

def search_by_id(img_id, keywords=""):
    if not descs:
        ss.msg = ("warning", "Sélectionnez au moins un descripteur dans la barre latérale.")
        return
    r = call_search(img_id, keywords)
    if not r.ok:
        ss.msg = ("error", str(r.json().get("detail")))
        return
    d = r.json()
    ss.query, ss.results, ss.last_ms, ss.compare = d["query"], d["results"], d["time_ms"], None
    filtre = f" · filtre mots-clés « {keywords} »" if keywords else ""
    mode = "indexée" if use_index else "exhaustive"
    dispo = " · images disponibles uniquement" if only_av else ""
    ss.msg = ("success", f"Recherche visuelle {mode} · {', '.join(NOMS[x] for x in descs)}{filtre}{dispo}")

def compare_index():
    times = {}
    for flag in (True, False):
        r = call_search(ss.query["id"], ss.get("kw_filter", ""), flag)
        times[flag] = r.json()["time_ms"]
    ss.compare = times

def card(r, rank=None, score=None, ok=None):
    border = {True: "#16a34a", False: "#dc2626"}.get(ok, "#e2e8f0")
    lbl = html.escape(r["label"]) or "(sans tag)"
    rank_html = f'<span class="rank">#{rank}</span>' if rank else ""
    score_html = f"<span>d = {score:.3f}</span>" if score is not None else ""
    return (f'<div class="card" style="border-color:{border}">'
            f'<img src="{API}{r["url"]}" loading="lazy">'
            f'<div class="meta">{rank_html}<span class="lbl" title="{lbl}">{lbl}</span>{score_html}</div></div>')

def query_html(src, caption):
    return f'<div class="query"><img src="{src}"><div class="cap">{html.escape(caption)}</div></div>'


# ======================= Onglets =======================
t_txt, t_img, t_up, t_eval, t_arch = st.tabs(
    ["🔤 Recherche textuelle", "🖼️ Recherche par image", "📤 Image externe",
     "📊 Évaluation", "🏗️ Architecture"])

# ---------- 1) Recherche textuelle ----------
with t_txt:
    exemple = ("airplanes · motorbikes · starfish" if base == "objects"
               else "sunset AND beach · dog OR cat · sunsset")
    c1, c2 = st.columns([5, 1])
    q = c1.text_input("Mots-clés", placeholder=f"Exemples : {exemple}", label_visibility="collapsed")
    if c2.button("Rechercher", type="primary", width="stretch") and q.strip():
        r = requests.get(f"{API}/search/text", params={"q": q, "base": base, "k": k,
                                                       "only_available": only_av}, timeout=60)
        if r.ok:
            d = r.json()
            ss.query, ss.results, ss.total, ss.compare = None, d["results"], d["total"], None
            corr = d.get("corrections") or {}
            txt_corr = ""
            if corr:
                txt_corr = " · ✏️ Correction automatique : " + ", ".join(
                    f"« {a} » → « {b} »" for a, b in corr.items())
            dispo = " · affichage : images disponibles uniquement" if only_av else ""
            ss.msg = ("info", f"Recherche textuelle « {q} » : {d['total']} images trouvées, "
                              f"{len(d['results'])} affichées (triées par pertinence){txt_corr}{dispo}")
        else:
            ss.msg = ("error", str(r.json().get("detail")))
    st.caption("Opérateurs logiques : **AND** (ET), **OR** (OU), **NOT**. Recherche tolérante aux fautes "
               "de frappe (*sunsset* → *sunset*) et aux variantes (*sunset* → *sunsets*). "
               "Résultats triés par pertinence (BM25). Cliquez sur « Similaires » pour passer en recherche visuelle.")

# ---------- 2) Recherche par image de la base (+ filtre mots-clés) ----------
with t_img:
    c1, c2 = st.columns([5, 1])
    c1.text_input("Filtre par mots-clés (optionnel)", key="kw_filter",
                  placeholder="Optionnel : combine mots-clés ET image (ex : sunset)")
    if c2.button("🎲 Nouvelles images", width="stretch") or not ss.picker:
        kw = ss.get("kw_filter", "").strip()
        if kw:      # avec un filtre : proposer des images qui correspondent aux mots-clés
            nb = 2000 if base == "mirflickr1m" else 200
            r = requests.get(f"{API}/search/text", params={"q": kw, "base": base, "k": nb,
                                                           "only_available": True}, timeout=60)
            pool = r.json()["results"] if r.ok else []
            ss.picker = random.sample(pool, min(10, len(pool)))
        else:       # sans filtre : images au hasard
            ss.picker = requests.get(f"{API}/random", params={"base": base, "n": 10}, timeout=10).json()
    st.caption("Choisissez une image de la base comme requête :")
    if not ss.picker:
        st.warning("Aucune image ne correspond à ce filtre dans cette base. "
                   "Modifiez les mots-clés (ex : airplanes, starfish pour Objects ; sunset, beach pour MIRFLICKR).")
    cols = st.columns(5)
    for n, r in enumerate(ss.picker):
        with cols[n % 5]:
            st.markdown(card(r), unsafe_allow_html=True)
            st.button("Choisir", key=f"pick_{r['id']}", on_click=search_by_id,
                      args=(r["id"], ss.get("kw_filter", "")), width="stretch")

# ---------- 3) Image externe ----------
with t_up:
    if base != "objects":
        st.info("La recherche par image externe est disponible sur la base **Objects** : les "
                "descripteurs MPEG-7 de MIRFLICKR sont fournis avec la base, pas calculés par le serveur.")
    else:
        c1, c2 = st.columns([2, 1])
        up = c1.file_uploader("Image requête (JPG ou PNG)", type=["jpg", "jpeg", "png"])
        if up:
            c2.image(up.getvalue(), caption="Aperçu", width=220)
            if c1.button("🔍 Rechercher les images similaires", type="primary"):
                if not descs:
                    ss.msg = ("warning", "Sélectionnez au moins un descripteur.")
                else:
                    r = requests.post(f"{API}/search/upload", timeout=120,
                                      files={"file": (up.name, up.getvalue())},
                                      data={"descs": ",".join(descs), "k": k, "use_index": use_index})
                    if r.ok:
                        d = r.json()
                        ss.query = {"upload": up.getvalue(), "label": up.name}
                        ss.results, ss.last_ms, ss.compare = d["results"], d["time_ms"], None
                        ss.msg = ("success", f"Recherche par image externe · {', '.join(NOMS[x] for x in descs)}")
                    else:
                        ss.msg = ("error", str(r.json().get("detail")))

# ---------- 4) Évaluation ----------
with t_eval:
    st.markdown("#### Évaluation sur la base Objects — 1 000 requêtes, 101 classes")
    csv_path = os.path.join(ROOT, "eval", "resultats_objects.csv")
    png_path = os.path.join(ROOT, "rapport", "courbe_rappel_precision.png")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")
        df.columns = ["Descripteurs", "P@10 (%)", "P@20 (%)", "R@20 (%)", "mAP (%)"]
        c1, c2 = st.columns([1, 1], gap="large")
        c1.dataframe(df, hide_index=True, width="stretch")
        c1.caption("Pertinent = même classe que l'image requête. Hasard ≈ 3 % de précision.")
        if os.path.exists(png_path):
            c2.image(png_path, caption="Courbes rappel / précision")
    else:
        st.info("Lancez `python eval\\evaluate.py` pour générer les résultats.")

    st.markdown("#### Gain apporté par l'index — MIRFLICKR-1M (1 000 000 de descripteurs)")
    csv_1m = os.path.join(ROOT, "eval", "temps_index_1m.csv")
    if os.path.exists(csv_1m):
        st.dataframe(pd.read_csv(csv_1m, sep=";"), hide_index=True, width="stretch")
        st.caption("FAISS IVF à 1024 partitions. nprobe = nombre de partitions visitées "
                   "(compromis vitesse / précision). Mesures issues de backend/build_index_1m.py.")
    st.markdown("#### Gain apporté par l'index — bases plus petites")
    st.markdown("""
| Base | Index | Recherche exhaustive | Recherche indexée | Résultats exacts retrouvés |
|---|---|---|---|---|
| MIRFLICKR-25k (EH) | FAISS IVF, nprobe = 10 | 8,9 ms | 0,15 ms | 99 % |
| MIRFLICKR-25k (HT) | FAISS IVF, nprobe = 10 | 2,9 ms | 0,05 ms | 100 % |
| Objects (LBP) | VP-Tree | 8 677 distances | 1 397 distances (16 %) | 100 % |
""")
    st.caption("Mesures issues du script backend/compare_times.py.")

# ---------- 5) Architecture ----------
with t_arch:
    st.markdown("#### Architecture client / serveur")
    st.graphviz_chart(r"""digraph {
        rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#eff6ff", color="#1e40af", fontname="Helvetica", fontsize=11];
        edge [color="#64748b", fontname="Helvetica", fontsize=9];
        Client [label="Client web\n(Streamlit)"];
        API [label="Serveur FastAPI\n(web service JSON)"];
        Moteur [label="Moteur de recherche\nfusion des descripteurs"];
        Index [label="Index\nVP-Tree (Objects)\nFAISS IVF (MIRFLICKR 25k / 1M)"];
        BD [label="Base SQLite FTS5\nchemins + mots-clés (BM25)"];
        Feat [label="Descripteurs .npy\n(phase off-line)"];
        Images [label="Images\n(répertoires)"];
        Client -> API [label="requête HTTP"];
        API -> Client [label="JSON : URLs triées"];
        API -> Moteur; Moteur -> Index; Moteur -> BD; Index -> Feat; API -> Images [label="/image"];
    }""")
    st.markdown("""
- **Phase off-line** : extraction des descripteurs (Couleur, LBP, HOG, CNN) et lecture des descripteurs MPEG-7
  du million, stockage en `.npy`, construction des index et de la base SQLite.
- **Phase on-line** : le client envoie la requête (mots-clés, image de la base ou image externe), le serveur
  calcule ou récupère les descripteurs, interroge l'index et renvoie la liste des URLs triées par similarité.
""")


# ======================= Résultats =======================
if ss.msg:
    kind, text = ss.msg
    getattr(st, kind)(text)

if ss.results:
    st.divider()
    visual = ss.query is not None
    left, right = st.columns([1, 4], gap="large")

    with left:
        if visual:
            st.markdown("#### Image requête")
            if "upload" in ss.query:
                src = "data:image/jpeg;base64," + base64.b64encode(ss.query["upload"]).decode()
                st.markdown(query_html(src, ss.query["label"]), unsafe_allow_html=True)
            else:
                st.markdown(query_html(API + ss.query["url"],
                                       f"id {ss.query['id']} · {ss.query['label'] or 'sans tag'}"),
                            unsafe_allow_html=True)
            st.metric("Temps de recherche", f"{ss.last_ms} ms")
            if base == "objects" and "upload" not in ss.query:
                autres = [r for r in ss.results if r["id"] != ss.query["id"]]
                bons = sum(r["label"] == ss.query["label"] for r in autres)
                st.metric("Précision (même classe)", f"{bons}/{len(autres)}",
                          f"{100 * bons / max(1, len(autres)):.0f} %", delta_color="off")
            if "upload" not in ss.query:
                st.button("⏱️ Comparer avec / sans index", on_click=compare_index, width="stretch")
                if ss.compare:
                    t_idx, t_ex = ss.compare[True], ss.compare[False]
                    st.markdown(f"**Indexée** : {t_idx} ms  \n**Exhaustive** : {t_ex} ms  \n"
                                f"**Rapport** : × {t_ex / max(t_idx, 1e-3):.1f}")
        else:
            st.markdown("#### Recherche textuelle")
            st.metric("Images trouvées", ss.total)
            st.caption("Cliquez sur « Similaires » pour passer en recherche visuelle.")

    with right:
        st.markdown("#### Résultats — du plus au moins similaire" if visual
                    else "#### Résultats — du plus au moins pertinent")
        check = visual and base == "objects" and "upload" not in ss.query
        if check:
            st.markdown('<div class="legend">Cadre :<span class="dot" style="background:#16a34a"></span>même classe'
                        '<span class="dot" style="background:#dc2626"></span>autre classe</div>',
                        unsafe_allow_html=True)
        cols = st.columns(5)
        for n, r in enumerate(ss.results):
            ok = (r["label"] == ss.query["label"]) if check else None
            with cols[n % 5]:
                st.markdown(card(r, rank=n + 1, score=r["score"] if visual else None, ok=ok),
                            unsafe_allow_html=True)
                st.button("🔍 Similaires", key=f"sim_{n}_{r['id']}", on_click=search_by_id,
                          args=(r["id"], ss.get("kw_filter", "")), width="stretch")