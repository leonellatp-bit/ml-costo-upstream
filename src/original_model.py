"""Modelo original: Hierarchical Geo-Shrinkage Gradient Boosting (HGSB).

Idea
----
En cada iteración de boosting, las variables categóricas jerárquicas (región → país,
región → elemento productivo, tipo → subtipo de reserva) se convierten en un valor
numérico: la **media del residuo actual de su categoría, contraída (shrinkage) hacia la
media de su categoría padre**. La contracción es la media posterior de un modelo
normal-normal (empirical Bayes / James-Stein):

    ẽ_c = (S_c + κ · π_c) / (n_c + κ)

con S_c la suma de residuos medios por activo de la categoría c, n_c su número de
activos, π_c la codificación de la categoría padre (o la media global en la raíz) y κ
la fuerza de contracción. Para las filas de entrenamiento la codificación se calcula
**fuera de pliegue por grupos de activos** (los activos se reparten al azar en K pliegues,
re-sorteados en cada iteración), de modo que el árbol no puede ver el residuo del activo
que intenta ajustar. (Dejar fuera solo el propio activo, LOO, filtra el objetivo de forma
inversa: dentro de una categoría la codificación decrece con el residuo propio.) Sobre esas codificaciones y las
variables numéricas se ajusta un árbol de regresión por histogramas con la ganancia de
segundo orden y los pesos de hoja regularizados w* = −G/(H + λ).

Ventajas que motivan el diseño (ver capítulo 5 del Jupyter Book):
1. Países o cuencas con uno o pocos activos (45 de 99 países) toman prestada
   información de su región en vez de ser ignorados (One-Hot con min_frequency) o
   sobreajustados.
2. Una categoría nunca vista en entrenamiento hereda la codificación de su padre.
3. El orden de las categorías por residuo medio da la partición binaria óptima para
   pérdida cuadrática (Fisher, 1958), así que una sola variable codificada sustituye a
   decenas de columnas One-Hot.

Todo el núcleo (codificación, árboles, boosting) está programado con numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# Jerarquía por defecto: columna -> columna padre (None = raíz)
DEFAULT_HIERARCHY = {
    "RegionName": None,
    "CountryName": "RegionName",
    "ProductionElementName": "RegionName",
    "ReserveTypeName": None,
    "SupplySubTypeName": "ReserveTypeName",
    "ReserveLocationName": None,
    "MemberOpec": None,
}
DEFAULT_NUMERIC = ["Year", "Production"]
LOG_NUMERIC = ("Production",)


# ----------------------------------------------------------------------
# Árbol de regresión por histogramas (estructura en arreglos planos)
# ----------------------------------------------------------------------
class _Tree:
    """Árbol binario guardado en arreglos: feature, threshold, left, right, value."""

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature, self.threshold, self.left, self.right, self.value = [], [], [], [], []

    def add_node(self):
        for a in (self.feature, self.threshold, self.left, self.right, self.value):
            a.append(-1 if a is not self.threshold and a is not self.value else 0.0)
        return len(self.feature) - 1

    def finalize(self):
        self.feature = np.asarray(self.feature, dtype=np.int64)
        self.threshold = np.asarray(self.threshold, dtype=float)
        self.left = np.asarray(self.left, dtype=np.int64)
        self.right = np.asarray(self.right, dtype=np.int64)
        self.value = np.asarray(self.value, dtype=float)
        return self

    def apply(self, Z: np.ndarray) -> np.ndarray:
        """Valor de hoja para cada fila de Z (recorrido vectorizado por niveles)."""
        node = np.zeros(len(Z), dtype=np.int64)
        while True:
            f = self.feature[node]
            internal = f >= 0
            if not internal.any():
                return self.value[node]
            idx = np.where(internal)[0]
            go_left = Z[idx, f[idx]] <= self.threshold[node[idx]]
            node[idx] = np.where(go_left, self.left[node[idx]], self.right[node[idx]])


def _bin_edges(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Cortes por cuantiles (únicos) de una columna."""
    q = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(q)


def _grow_tree(B: np.ndarray, edges: list, r: np.ndarray, rows: np.ndarray, *, max_depth: int,
               reg_lambda: float, gamma: float, min_child_weight: float, n_slots: int) -> _Tree:
    """Construye un árbol con la ganancia de 2.º orden (pérdida cuadrática: g = −r, h = 1).

    B : (n, F) índices de bin de cada fila y variable; edges[f] son los cortes de la variable f.
    rows : filas usadas (submuestreo).
    """
    tree = _Tree()
    F = B.shape[1]
    offs = (np.arange(F) * n_slots)[None, :]
    root = tree.add_node()
    stack = [(root, rows, 0)]
    while stack:
        node, idx, depth = stack.pop()
        G = r[idx].sum()  # suma de residuos (= −Σg)
        H = float(len(idx))
        tree.value[node] = G / (H + reg_lambda)  # w* = −G_g/(H+λ) con G_g = −Σr
        if depth >= max_depth or H < 2 * min_child_weight:
            continue
        flat = (B[idx] + offs).ravel()
        g_hist = np.bincount(flat, weights=np.repeat(r[idx], F), minlength=F * n_slots).reshape(F, n_slots)
        h_hist = np.bincount(flat, minlength=F * n_slots).reshape(F, n_slots).astype(float)
        GL = np.cumsum(g_hist, axis=1)[:, :-1]
        HL = np.cumsum(h_hist, axis=1)[:, :-1]
        GR, HR = G - GL, H - HL
        valid = (HL >= min_child_weight) & (HR >= min_child_weight)
        gain = 0.5 * (GL ** 2 / (HL + reg_lambda) + GR ** 2 / (HR + reg_lambda) - G ** 2 / (H + reg_lambda)) - gamma
        gain = np.where(valid, gain, -np.inf)
        # solo cortes que existen para la variable f
        for f in range(F):
            gain[f, len(edges[f]):] = -np.inf
        k = int(np.argmax(gain))
        f, b = divmod(k, n_slots - 1)
        if not np.isfinite(gain[f, b]) or gain[f, b] <= 0:
            continue
        left_mask = B[idx, f] <= b
        l, rr = tree.add_node(), tree.add_node()
        tree.feature[node], tree.threshold[node] = f, float(edges[f][b])
        tree.left[node], tree.right[node] = l, rr
        stack.append((l, idx[left_mask], depth + 1))
        stack.append((rr, idx[~left_mask], depth + 1))
    return tree.finalize()


# ----------------------------------------------------------------------
# Estimador
# ----------------------------------------------------------------------
class HierarchicalShrinkageBoosting(BaseEstimator, RegressorMixin):
    """Gradient boosting con codificación jerárquica contraída de las categóricas (HGSB).

    Parameters
    ----------
    n_estimators : int
        Número de árboles (iteraciones de boosting).
    learning_rate : float
        Tasa de aprendizaje η que multiplica cada árbol.
    max_depth : int
        Profundidad máxima de cada árbol.
    min_child_weight : float
        Mínimo de filas (hessiano) por hoja.
    reg_lambda : float
        Regularización L2 λ de los pesos de hoja.
    gamma : float
        Ganancia mínima para dividir un nodo (penaliza el número de hojas).
    subsample : float
        Fracción de filas (sin reemplazo) por árbol.
    colsample : float
        Fracción de variables por árbol.
    kappa : float
        Fuerza de contracción κ ≥ 0 (en "activos equivalentes") hacia la categoría padre.
    n_bins : int
        Número máximo de bins por variable.
    hierarchy : dict or None
        {columna: columna padre}. None usa ``DEFAULT_HIERARCHY``.
    use_hierarchy : bool
        Ablación: si False, toda categoría se contrae hacia la media global.
    oof_folds : int
        Pliegues de activos para la codificación fuera de pliegue de las filas de
        entrenamiento. ≤ 1 = codificación en muestra (ablación).
    encoding : {"hier", "onehot"}
        Ablación: "onehot" reemplaza la codificación por columnas indicadoras (boosting estándar).
    group_col : str or None
        Columna de X con el identificador del activo. Solo se usa para la codificación fuera de pliegue durante
        ``fit``; nunca es predictora. Si falta, cada fila es su propio grupo.
    random_state : int or None
        Semilla.
    """

    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5.0,
                 reg_lambda=1.0, gamma=0.0, subsample=0.8, colsample=1.0, kappa=3.0, n_bins=32,
                 hierarchy=None, numeric=None, use_hierarchy=True, oof_folds=5, encoding="hier",
                 group_col="CostProductionElementName", random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.kappa = kappa
        self.n_bins = n_bins
        self.hierarchy = hierarchy
        self.numeric = numeric
        self.use_hierarchy = use_hierarchy
        self.oof_folds = oof_folds
        self.encoding = encoding
        self.group_col = group_col
        self.random_state = random_state

    # ------------------------------------------------------------ validación
    def _check_params(self):
        if not (isinstance(self.n_estimators, (int, np.integer)) and self.n_estimators >= 1):
            raise ValueError("n_estimators debe ser un entero ≥ 1")
        if not (0 < self.learning_rate <= 1):
            raise ValueError("learning_rate debe estar en (0, 1]")
        if self.max_depth < 1:
            raise ValueError("max_depth debe ser ≥ 1")
        if self.kappa < 0 or self.reg_lambda < 0 or self.gamma < 0 or self.min_child_weight < 0:
            raise ValueError("kappa, reg_lambda, gamma y min_child_weight deben ser ≥ 0")
        if not (0 < self.subsample <= 1 and 0 < self.colsample <= 1):
            raise ValueError("subsample y colsample deben estar en (0, 1]")
        if self.encoding not in ("hier", "onehot"):
            raise ValueError("encoding debe ser 'hier' u 'onehot'")
        if not isinstance(self.oof_folds, (int, np.integer)) or self.oof_folds < 0:
            raise ValueError("oof_folds debe ser un entero ≥ 0")
        if self.n_bins < 2:
            raise ValueError("n_bins debe ser ≥ 2")

    def _check_X(self, X, fitting=False):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas.DataFrame con columnas nombradas")
        need = list(self.hier_) + list(self.num_)
        missing = [c for c in need if c not in X.columns]
        if missing:
            raise ValueError(f"Faltan columnas en X: {missing}")
        for c in self.num_:
            v = pd.to_numeric(X[c], errors="coerce")
            if np.isinf(v.to_numpy(dtype=float)).any():
                raise ValueError(f"La columna {c} contiene valores infinitos")
        return X

    # ------------------------------------------------------------ numéricas
    def _numeric_matrix(self, X):
        cols = []
        for c in self.num_:
            v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            miss = np.isnan(v)
            v = np.where(miss, self.num_median_[c], v)
            if c in LOG_NUMERIC:
                v = np.log1p(np.clip(v, 0, None))
            cols.append(v)
            if self.num_has_missing_[c]:
                cols.append(miss.astype(float))
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    # ------------------------------------------------------------ categóricas
    def _codes(self, X):
        """Códigos enteros por columna categórica (−1 = categoría no vista)."""
        out = {}
        for c in self.hier_:
            m = self.cat_index_[c]
            out[c] = X[c].astype(str).map(m).fillna(-1).to_numpy(dtype=np.int64)
        return out

    def _order(self):
        """Columnas ordenadas de modo que cada padre se procese antes que su hijo."""
        done, order = set(), []
        while len(order) < len(self.hier_):
            for c, p in self.hier_.items():
                if c not in done and (p is None or p in done):
                    order.append(c)
                    done.add(c)
        return order

    def _structure(self, codes, asset, A):
        """Categoría de cada activo y matriz padre W_c (fracción de activos del hijo en cada padre).

        Solo usa la composición de los activos (nunca el objetivo), así que no filtra información.
        """
        first_row = np.full(A, -1)
        first_row[asset[::-1]] = np.arange(len(asset))[::-1]
        self._cat_asset = {c: codes[c][first_row] for c in self.hier_}
        self._W = {}
        for c in self._order():
            p = self.hier_[c] if self.use_hierarchy else None
            if p is None:
                continue
            Kc, Kp = len(self.cat_index_[c]), len(self.cat_index_[p])
            W = np.zeros((Kc, Kp))
            np.add.at(W, (self._cat_asset[c], self._cat_asset[p]), 1.0)
            self._W[c] = W / np.maximum(W.sum(1, keepdims=True), 1)

    def _tables(self, r_asset, mask):
        """Tablas de codificación contraída calculadas con los activos ``mask``.

        ẽ_c = (S_c + κ·π_c) / (n_c + κ), π_c = W_c · ẽ_padre (o la media global en la raíz).
        """
        k = float(self.kappa)
        ra = r_asset[mask]
        root = float(ra.mean()) if mask.any() else 0.0
        tables = {}
        for c in self._order():
            p = self.hier_[c] if self.use_hierarchy else None
            K = len(self.cat_index_[c])
            ca = self._cat_asset[c][mask]
            S = np.bincount(ca, weights=ra, minlength=K)
            n = np.bincount(ca, minlength=K).astype(float)
            prior = np.full(K, root) if p is None else self._W[c] @ tables[p]
            if k > 0:
                tables[c] = (S + k * prior) / (n + k)
            else:
                tables[c] = np.where(n > 0, S / np.maximum(n, 1), prior)
        return tables, root

    def _encode_train(self, r_asset, asset, rng):
        """Codificaciones de entrenamiento fuera de pliegue (por grupos de activos).

        Los activos se reparten al azar en ``oof_folds`` pliegues (se re-sortean en cada
        iteración); la codificación de un activo se calcula sin los activos de su pliegue,
        de modo que no depende de su propio residuo. Con ``oof_folds`` ≤ 1 la codificación
        es en muestra (ablación).
        """
        A = len(r_asset)
        cols = self._order()
        full, root = self._tables(r_asset, np.ones(A, bool))
        enc_a = np.empty((A, len(cols)))
        if self.oof_folds <= 1:
            for j, c in enumerate(cols):
                enc_a[:, j] = full[c][self._cat_asset[c]]
        else:
            fold = rng.permutation(np.arange(A) % self.oof_folds)
            for f in range(self.oof_folds):
                inf = fold == f
                t, _ = self._tables(r_asset, ~inf)
                for j, c in enumerate(cols):
                    enc_a[inf, j] = t[c][self._cat_asset[c][inf]]
        return enc_a[asset], full, root

    def _onehot(self, codes):
        cols = []
        for c in self._order():
            K = len(self.cat_index_[c])
            oh = np.zeros((len(codes[c]), K))
            ok = codes[c] >= 0
            oh[np.where(ok)[0], codes[c][ok]] = 1.0
            cols.append(oh)
        return np.hstack(cols)

    # ------------------------------------------------------------ fit / predict
    def fit(self, X, y):
        """Ajusta el modelo.

        Parameters
        ----------
        X : DataFrame con las columnas categóricas de la jerarquía, las numéricas y,
            opcionalmente, ``group_col`` (identificador del activo, solo para la codificación fuera de pliegue).
        y : array-like (n,), objetivo continuo y finito.
        """
        self._check_params()
        self.hier_ = dict(DEFAULT_HIERARCHY if self.hierarchy is None else self.hierarchy)
        self.num_ = list(DEFAULT_NUMERIC if self.numeric is None else self.numeric)
        self._check_X(X, fitting=True)
        y = np.asarray(y, dtype=float).ravel()
        if len(y) != len(X):
            raise ValueError("X e y tienen distinto número de filas")
        if len(y) < 2 or not np.isfinite(y).all():
            raise ValueError("y debe tener al menos 2 valores finitos")
        rng = np.random.default_rng(self.random_state)

        # --- preprocesamiento propio (ajustado solo con entrenamiento)
        self.num_median_, self.num_has_missing_ = {}, {}
        for c in self.num_:
            v = pd.to_numeric(X[c], errors="coerce")
            self.num_median_[c] = float(v.median()) if v.notna().any() else 0.0
            self.num_has_missing_[c] = bool(v.isna().any())
        self.cat_index_ = {c: {v: i for i, v in enumerate(sorted(X[c].astype(str).unique()))} for c in self.hier_}
        Nm = self._numeric_matrix(X)
        codes = self._codes(X)
        if self.group_col is not None and self.group_col in X.columns:
            _, asset = np.unique(X[self.group_col].astype(str).to_numpy(), return_inverse=True)
        else:
            asset = np.arange(len(X))
        A = asset.max() + 1
        cnt_a = np.bincount(asset, minlength=A).astype(float)
        self._structure(codes, asset, A)

        n_slots = self.n_bins + 1
        num_edges = [_bin_edges(Nm[:, j], self.n_bins) for j in range(Nm.shape[1])]
        Bn = np.column_stack([np.searchsorted(e, Nm[:, j], side="left") for j, e in enumerate(num_edges)]) \
            if Nm.shape[1] else np.empty((len(X), 0), dtype=np.int64)
        if self.encoding == "onehot":
            OH = self._onehot(codes)
            oh_edges = [np.array([0.5])] * OH.shape[1]
            B_oh = (OH > 0.5).astype(np.int64)

        self.base_ = float(y.mean())
        F = np.full(len(y), self.base_)
        self.trees_, self.tables_, self.cols_used_, self.root_means_ = [], [], [], []
        n = len(y)
        for _ in range(self.n_estimators):
            r = y - F
            if self.encoding == "hier":
                r_asset = np.bincount(asset, weights=r, minlength=A) / cnt_a
                Ztr, tables, root = self._encode_train(r_asset, asset, rng)
                self.root_means_.append(root)
                cat_edges = [_bin_edges(Ztr[:, j], self.n_bins) for j in range(Ztr.shape[1])]
                Bc = np.column_stack([np.searchsorted(e, Ztr[:, j], side="left") for j, e in enumerate(cat_edges)])
                Zall, Ball, edges = np.hstack([Nm, Ztr]), np.hstack([Bn, Bc]), num_edges + cat_edges
            else:
                tables = None
                Zall, Ball, edges = np.hstack([Nm, OH]), np.hstack([Bn, B_oh]), num_edges + oh_edges
            p = Zall.shape[1]
            cols = np.sort(rng.choice(p, max(1, int(round(self.colsample * p))), replace=False))
            rows = np.sort(rng.choice(n, max(2, int(round(self.subsample * n))), replace=False)) \
                if self.subsample < 1 else np.arange(n)
            tree = _grow_tree(Ball[:, cols], [edges[j] for j in cols], r, rows, max_depth=self.max_depth,
                              reg_lambda=self.reg_lambda, gamma=self.gamma,
                              min_child_weight=self.min_child_weight, n_slots=n_slots)
            F += self.learning_rate * tree.apply(Zall[:, cols])
            self.trees_.append(tree)
            self.tables_.append(tables)
            self.cols_used_.append(cols)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _design(self, X, t):
        """Matriz de variables de la iteración t para filas nuevas (tablas completas)."""
        Nm = self._Nm_cache
        if self.encoding == "onehot":
            return np.hstack([Nm, self._OH_cache])
        codes, Zc = self._codes_cache, {}
        for c in self._order():
            p = self.hier_[c] if self.use_hierarchy else None
            full = self.tables_[t][c]
            ce = codes[c]
            if p is None:
                fb = np.full(len(ce), self.root_means_[t])
            else:
                fb = Zc[p]
            Zc[c] = np.where(ce >= 0, full[np.clip(ce, 0, None)], fb)
        return np.hstack([Nm, np.column_stack([Zc[c] for c in self._order()])])

    def staged_predict(self, X):
        """Generador con la predicción después de cada árbol (curva de convergencia)."""
        check_is_fitted(self, "trees_")
        self._check_X(X)
        self._Nm_cache = self._numeric_matrix(X)
        self._codes_cache = self._codes(X)
        if self.encoding == "onehot":
            self._OH_cache = self._onehot(self._codes_cache)
        out = np.full(len(X), self.base_)
        try:
            for t, tree in enumerate(self.trees_):
                Z = self._design(X, t)
                out = out + self.learning_rate * tree.apply(Z[:, self.cols_used_[t]])
                yield out
        finally:
            for a in ("_Nm_cache", "_codes_cache", "_OH_cache"):
                if hasattr(self, a):
                    delattr(self, a)

    def predict(self, X):
        """Predice el objetivo para filas nuevas. ``group_col`` se ignora."""
        out = np.full(len(X), getattr(self, "base_", 0.0))
        for out in self.staged_predict(X):
            pass
        return out
