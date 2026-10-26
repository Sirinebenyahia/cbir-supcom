import heapq
import numpy as np


class VPTree:
    """VP-Tree (Vantage Point Tree) pour la recherche des k plus proches voisins.

    Construction : on choisit un point pivot (vantage point), on calcule la distance
    de tous les autres points à ce pivot, puis on coupe en deux selon la médiane mu :
      - "inside"  : points à distance <= mu du pivot
      - "outside" : points à distance >  mu du pivot
    et on recommence récursivement. Les petites feuilles (<= leaf_size) sont stockées telles quelles.

    Recherche : on descend d'abord dans la branche qui contient la requête, puis on
    n'explore l'autre branche que si la boule de rayon tau (distance du k-ième meilleur
    résultat) traverse la frontière mu -> élagage d'une grande partie de la base.
    """

    def __init__(self, X, metric="l1", leaf_size=20, seed=0):
        self.X = X
        self.metric = metric
        self.leaf_size = leaf_size
        self.rng = np.random.default_rng(seed)
        self.n_dist = 0                       # compteur de calculs de distance
        self.root = self._build(np.arange(len(X)))

    def _dist(self, idx, q):
        """Distances entre la requête q et les vecteurs X[idx] (calcul vectorisé numpy)."""
        self.n_dist += len(idx)
        diff = self.X[idx] - q
        if self.metric == "l1":
            return np.abs(diff).sum(axis=1)
        return np.sqrt((diff ** 2).sum(axis=1))

    def _build(self, idx):
        if len(idx) <= self.leaf_size:
            return {"leaf": idx}
        vp = idx[self.rng.integers(len(idx))]          # pivot choisi au hasard
        rest = idx[idx != vp]
        d = self._dist(rest, self.X[vp])
        mu = float(np.median(d))
        return {"vp": int(vp), "mu": mu,
                "inside": self._build(rest[d <= mu]),
                "outside": self._build(rest[d > mu])}

    def search(self, q, k=10):
        """Retourne les k plus proches voisins : liste de (id, distance) triée."""
        self.n_dist = 0
        heap = []                    # max-tas des k meilleurs : (-distance, id)
        tau = [np.inf]               # distance du k-ième meilleur résultat actuel

        def push(ids, ds):
            for i, d in zip(ids, ds):
                if len(heap) < k:
                    heapq.heappush(heap, (-float(d), int(i)))
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-float(d), int(i)))
            if len(heap) == k:
                tau[0] = -heap[0][0]

        def visit(node):
            if "leaf" in node:
                if len(node["leaf"]):
                    push(node["leaf"], self._dist(node["leaf"], q))
                return
            d = float(self._dist(np.array([node["vp"]]), q)[0])
            push([node["vp"]], [d])
            if d <= node["mu"]:
                first, second = node["inside"], node["outside"]
            else:
                first, second = node["outside"], node["inside"]
            visit(first)
            if abs(d - node["mu"]) <= tau[0]:     # élagage : sinon on ignore cette branche
                visit(second)

        visit(self.root)
        return [(i, -nd) for nd, i in sorted(heap, reverse=True)]