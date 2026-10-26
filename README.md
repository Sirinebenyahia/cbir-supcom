\# Moteur d'indexation et de recherche d'images par le contenu (CBIR)



Projet du cours \*\*Indexation et recherche dans des bases massives multimédia\*\* — SUP'COM.



Moteur de recherche d'images \*\*textuel et visuel\*\*, en architecture \*\*client / serveur\*\* :



\- recherche par \*\*mots-clés\*\* : opérateurs AND / OR / NOT, recherche par préfixe,

&#x20; \*\*correction automatique des fautes de frappe\*\* et tri par pertinence (\*\*BM25\*\*) ;

\- recherche par \*\*image de la base\*\*, par \*\*image externe\*\* (envoyée par l'utilisateur)

&#x20; ou \*\*combinée\*\* (mots-clés ET image) ;

\- choix d'un ou plusieurs \*\*descripteurs\*\* (Couleur, LBP, HOG, CNN, MPEG-7), fusionnés par

&#x20; moyenne des distances normalisées ;

\- résultats triés par similarité, affichés avec l'image requête, et \*\*relance\*\* d'une

&#x20; recherche depuis un résultat ;

\- \*\*index\*\* pour réduire le temps de recherche : VP-Tree et FAISS IVF, jusqu'à \*\*1 million\*\* d'images ;

\- \*\*évaluation\*\* quantitative : précision, rappel, mAP, courbes rappel / précision.



\---



\## Bases d'images



| Base | Taille | Descripteurs | Mots-clés |

|---|---|---|---|

| \*\*Caltech-101\*\* (base \*Objects\*) | 8 677 images, 101 classes | \*\*calculés\*\* : Couleur HSV (128), LBP (59), HOG (3780), CNN ResNet-50 (2048) | nom de la classe |

| \*\*MIRFLICKR-25k\*\* | 25 000 images | \*\*fournis\*\* (MPEG-7) : Edge Histogram (150), Homogeneous Texture (43) | tags Flickr |

| \*\*MIRFLICKR-1M\*\* | 1 000 000 de descripteurs (25 000 images téléchargées) | \*\*fournis\*\* (MPEG-7) : Edge Histogram (150), Homogeneous Texture (43) | tags Flickr (946 113 images) |



La catégorie `BACKGROUND\_Google` de Caltech-101 (images de fond sans objet) est exclue.



\---



\## Architecture



```

Client web (Streamlit) ⇄ HTTP / JSON ⇄ Serveur FastAPI ⇄ Moteur de recherche

&#x20;                                                         ├── Index : VP-Tree (Objects), FAISS IVF (MIRFLICKR 25k / 1M)

&#x20;                                                         ├── Base SQLite FTS5 (chemins + mots-clés, BM25)

&#x20;                                                         └── Descripteurs .npy (calculés off-line)

```



\- \*\*Phase off-line\*\* : extraction ou lecture des descripteurs, stockage en `.npy`, construction

&#x20; des index et de la base SQLite.

\- \*\*Phase on-line\*\* : le client envoie la requête, le serveur calcule ou récupère les descripteurs,

&#x20; interroge l'index et renvoie en JSON la liste des URLs triées par similarité.



\### Routes de l'API (documentation interactive : http://127.0.0.1:8000/docs)



| Route | Rôle |

|---|---|

| `GET /info` | bases disponibles, nombre d'images, descripteurs |

| `GET /random` | images au hasard, pour choisir une image requête |

| `GET /search/text` | recherche par mots-clés |

| `GET /search/image` | recherche par image de la base (+ filtre mots-clés optionnel) |

| `POST /search/upload` | recherche par image externe (base Objects) |

| `GET /image/{base}/{id}` | renvoie l'image |



\---



\## Structure du projet



```

cbir/

├── backend/

│   ├── descriptors.py          # descripteurs : Couleur HSV, LBP, HOG, CNN ResNet-50

│   ├── load\_mirflickr.py       # lecture des descripteurs MPEG-7 et des tags (25k)

│   ├── load\_mirflickr\_1m.py    # lecture du million de descripteurs et de tags (dans les zips)

│   ├── extract\_objects.py      # calcul de Couleur, LBP, HOG (base Objects)

│   ├── extract\_cnn.py          # calcul du descripteur CNN (base Objects)

│   ├── build\_db.py             # base SQLite + index plein texte FTS5

│   ├── vptree\_index.py         # implémentation du VP-Tree

│   ├── build\_index.py          # index VP-Tree (Objects) et FAISS IVF (MIRFLICKR-25k)

│   ├── build\_index\_1m.py       # index FAISS IVF sur 1 million + mesure des temps

│   ├── compare\_times.py        # gain de temps des index (Objects, MIRFLICKR-25k)

│   ├── search.py               # moteur de recherche (texte, visuel, fusion, filtrage)

│   ├── api.py                  # web service FastAPI (JSON)

│   ├── fix\_paths.py            # conversion des chemins au format "/" (Windows et Linux)

│   └── test\_\*.py               # scripts de test en ligne de commande

├── frontend/app.py             # client web Streamlit

├── eval/evaluate.py            # précision, rappel, mAP, courbes (base Objects)

├── .streamlit/config.toml      # thème de l'interface

├── features/                   # descripteurs .npy et métadonnées (générés)

├── indexes/                    # index VP-Tree et FAISS (générés)

├── data/                       # images et fichiers MIRFLICKR / Caltech-101

├── rapport/                    # figures et captures

├── cbir.db                     # base SQLite (générée)

├── requirements.txt

├── lancer.bat                  # lancement en un double-clic (Windows)

├── Dockerfile

├── docker-compose.yml

└── .dockerignore

```



\---



\## Installation



```bash

conda create -n cbir python=3.11 -y

conda activate cbir

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

```



\### Données à placer dans `data/`



\- `data/mirflickr/` (https://press.liacs.nl/mirflickr/mirdownload.html) :

&#x20; - `mirflickr25k.zip`, décompressé (images + tags des 25 000 images) ;

&#x20; - `tags.zip`, `features\_edgehistogram.zip`, `features\_homogeneoustexture.zip` : \*\*garder les zips\*\*

&#x20;   (le million est lu directement dedans) et les décompresser ;

\- `data/objects/101\_ObjectCategories/` : Caltech-101 (https://data.caltech.edu/records/mzrjq-6wc02).



\---



\## Phase off-line (une seule fois, dans cet ordre)



```bash

python backend\\load\_mirflickr.py       # descripteurs MPEG-7 + tags (25k)

python backend\\extract\_objects.py      # Couleur, LBP, HOG (Objects)

python backend\\extract\_cnn.py          # CNN ResNet-50 (Objects)

python backend\\build\_db.py             # base SQLite (recrée cbir.db)

python backend\\load\_mirflickr\_1m.py    # million de descripteurs + tags (après build\_db)

python backend\\build\_index.py          # index VP-Tree et FAISS (25k)

python backend\\build\_index\_1m.py       # index FAISS sur 1 million + mesures

python backend\\fix\_paths.py            # chemins compatibles Windows / Linux

```



\---



\## Lancement



\### Sous Windows

Double-cliquer sur \*\*`lancer.bat`\*\*, ou dans deux terminaux :

```bash

uvicorn api:app --app-dir backend --port 8000

streamlit run frontend\\app.py

```



\### Avec Docker

```bash

docker compose up --build

```

Le code est copié dans l'image ; les données volumineuses (`data/`, `features/`, `indexes/`, `cbir.db`)

sont montées en volumes depuis la machine hôte.



\### Accès

\- Interface : http://localhost:8501

\- Documentation de l'API : http://127.0.0.1:8000/docs



> Si le port 8000 est déjà utilisé, fermer les anciennes fenêtres du serveur avant de relancer.



\---



\## Résultats



\### Qualité de la recherche — base Objects (1 000 requêtes, 101 classes)



```bash

python eval\\evaluate.py

```



| Descripteurs | P@10 | P@20 | R@20 | mAP |

|---|---|---|---|---|

| Couleur | 25,1 % | 22,6 % | 2,1 % | 10,9 % |

| LBP | 29,0 % | 26,5 % | 2,2 % | 12,0 % |

| HOG | 41,1 % | 37,5 % | 4,1 % | 20,4 % |

| LBP + HOG | 43,7 % | 40,1 % | 4,0 % | 20,5 % |

| \*\*CNN\*\* | \*\*86,9 %\*\* | \*\*84,6 %\*\* | \*\*18,9 %\*\* | \*\*76,8 %\*\* |

| HOG + CNN | \*\*88,8 %\*\* | \*\*85,8 %\*\* | 18,5 % | 73,2 % |

| Couleur + LBP + HOG + CNN | 68,3 % | 62,3 % | 10,7 % | 41,5 % |



Une image est pertinente si elle appartient à la même classe que l'image requête

(un tirage au hasard donnerait environ 3 % de précision).



\### Temps de recherche — MIRFLICKR-1M (1 000 000 de descripteurs, FAISS IVF à 1024 partitions)



| Descripteur | nprobe | Exhaustif | Index IVF | Gain | Résultats exacts retrouvés |

|---|---|---|---|---|---|

| Edge Histogram | 4 | 407 ms | 0,47 ms | × 874 | 79 % |

| Edge Histogram | \*\*16\*\* | 375 ms | \*\*0,97 ms\*\* | \*\*× 387\*\* | \*\*94 %\*\* |

| Edge Histogram | 64 | 363 ms | 2,92 ms | × 124 | 100 % |

| Homogeneous Texture | 4 | 116 ms | 0,15 ms | × 758 | 87 % |

| Homogeneous Texture | \*\*16\*\* | 113 ms | \*\*0,32 ms\*\* | \*\*× 358\*\* | \*\*99 %\*\* |

| Homogeneous Texture | 64 | 115 ms | 1,03 ms | × 111 | 100 % |



Valeur retenue dans l'application : \*\*nprobe = 16\*\*.



\### Temps de recherche — bases plus petites



| Base | Index | Exhaustif | Indexé | Résultats exacts retrouvés |

|---|---|---|---|---|

| MIRFLICKR-25k (EH) | FAISS IVF, nprobe = 10 | 8,9 ms | 0,15 ms | 99 % |

| MIRFLICKR-25k (HT) | FAISS IVF, nprobe = 10 | 2,9 ms | 0,05 ms | 100 % |

| Objects (LBP) | VP-Tree | 8 677 distances | 1 397 distances (16 %) | 100 % |



\---



\## Technologies



Python 3.11 · OpenCV · scikit-image · PyTorch / torchvision (ResNet-50) · FAISS · NumPy ·

SQLite FTS5 · FastAPI · Uvicorn · Streamlit · pandas · matplotlib · Docker

