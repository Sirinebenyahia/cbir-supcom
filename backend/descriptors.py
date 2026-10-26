import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog


# ---------- Descripteurs "classiques" ----------

def color_hist(img, bins=(8, 4, 4)):
    """Histogramme couleur HSV 3D quantifié : 8 x 4 x 4 = 128 valeurs."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                     [0, 180, 0, 256, 0, 256]).flatten()
    return h / (h.sum() + 1e-8)                      # histogramme normalisé


def lbp_hist(img, P=8, R=1):
    """LBP uniforme (Ojala 2002) : 58 motifs uniformes + 1 non uniforme = 59 valeurs."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(g, P, R, method="nri_uniform")
    n_bins = P * (P - 1) + 3                         # = 59 pour P = 8
    h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    return h / (h.sum() + 1e-8)


def hog_desc(img):
    """HOG (Dalal & Triggs 2005) : image 64x128, cellules 8x8, blocs 2x2, 9 orientations.
    7 x 15 blocs x 36 valeurs = 3780 valeurs."""
    g = cv2.cvtColor(cv2.resize(img, (64, 128)), cv2.COLOR_BGR2GRAY)
    return hog(g, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")


# ---------- Descripteur CNN (deep learning, ResNet-50 pré-entraîné sur ImageNet) ----------
_CNN = {}

def _cnn_model():
    """Charge le réseau une seule fois (au premier appel)."""
    if "model" not in _CNN:
        import torch
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()      # on retire la couche de classification -> vecteur 2048
        model.eval()
        _CNN["model"], _CNN["prep"] = model, weights.transforms()
    return _CNN["model"], _CNN["prep"]


def cnn_batch(imgs):
    """Descripteurs CNN d'une liste d'images OpenCV (BGR) : matrice (n, 2048) normalisée L2."""
    import torch
    from PIL import Image
    model, prep = _cnn_model()
    x = torch.stack([prep(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in imgs])
    with torch.no_grad():
        f = model(x).numpy()
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    return f.astype("float32")


def cnn_desc(img):
    """Descripteur CNN d'une seule image : 2048 valeurs."""
    return cnn_batch([img])[0]


DESCRIPTORS = {"color": color_hist, "lbp": lbp_hist, "hog": hog_desc, "cnn": cnn_desc}


# ---------- Distances (anciennes fonctions, conservées pour compatibilité) ----------
N_DIST = [0]

def l1_pt(a, b):
    N_DIST[0] += 1
    return float(np.abs(a[1] - b[1]).sum())

def l2_pt(a, b):
    N_DIST[0] += 1
    return float(np.sqrt(((a[1] - b[1]) ** 2).sum()))