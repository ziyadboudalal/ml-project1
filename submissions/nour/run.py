import os

import numpy as np

from helpers import load_csv_data, create_csv_submission

SEED = 42
DATA_PATH = os.environ.get("MICHD_DATA_PATH", "dataset/")
OUTPUT = "submission.csv"


CONFIG = {
    "logistic_pk": {
        "max_unique_categorical": 30,
        "zero_flags": True,
        "n_poly": 10,
        "flag_dropped_max_frac": 0.99,
        "max_nan_frac": 0.8,
    },
    "logistic_lambda": 0.0001,
    "logistic_pos_weight": 1.0,
    "logistic_interactions": 25,
    "gbdt_params": {
        "learning_rate": 0.1,
        "max_depth": 4,
        "reg_lambda": 1.0,
        "min_child_hess": 5.0,
        "subsample": 0.5,
        "colsample": 0.5,
    },
    "gbdt_trees": 209,
    "blend_weight": 0.7000000000000001,
    "threshold": 0.225,
}


METADATA_COLUMNS = [
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "SEQNO",
    "_PSU",
    "_STSTR",
]


FORCE_CATEGORICAL = ["_STATE"]


MISSING_CODES = {
    "_RACE": [9],
    "EMPLOY1": [9],
    "_AGEG5YR": [14],
    "_AGE65YR": [3],
    "PHYSHLTH": [77, 99],
    "MENTHLTH": [77, 99],
    "POORHLTH": [77, 99],
    "CHILDREN": [99],
    "AVEDRNK2": [77, 99],
    "DRNK3GE5": [77, 99],
    "MAXDRNKS": [77, 99],
    "WEIGHT2": [7777, 9999],
    "HEIGHT3": [7777, 9999],
    "ALCDAY5": [777, 999],
    "STRENGTH": [777, 999],
    "EXEROFT1": [777, 999],
    "EXEROFT2": [777, 999],
    "EXERHMM1": [777, 999],
    "EXERHMM2": [777, 999],
    "FRUITJU1": [777, 999],
    "FRUIT1": [777, 999],
    "FVBEANS": [777, 999],
    "FVGREEN": [777, 999],
    "FVORANG": [777, 999],
    "VEGETAB1": [777, 999],
    "_DRNKWEK": [99900],
}


FOOD_COLUMNS = ["FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"]
ZERO_CODES = {
    "PHYSHLTH": [88],
    "MENTHLTH": [88],
    "POORHLTH": [88],
    "CHILDREN": [88],
    "AVEDRNK2": [88],
    "DRNK3GE5": [88],
    "ALCDAY5": [888],
    "STRENGTH": [888],
    "EXEROFT1": [888],
    "EXEROFT2": [888],
    **{name: [300, 555] for name in FOOD_COLUMNS},
}


def per_month(v):
    out = v.copy()
    week = (v >= 101) & (v <= 199)
    month = (v >= 201) & (v <= 299)
    out[week] = (v[week] - 100) * 30 / 7
    out[month] = v[month] - 200
    return out


def per_day(v):
    out = v.copy()
    day = (v >= 101) & (v <= 199)
    week = (v >= 201) & (v <= 299)
    month = (v >= 301) & (v <= 399)
    out[day] = v[day] - 100
    out[week] = (v[week] - 200) / 7
    out[month] = (v[month] - 300) / 30
    return out


def weight_kg(v):
    out = v.copy()
    lbs = (v >= 50) & (v <= 999)
    kg = (v >= 9000) & (v <= 9998)
    out[lbs] = v[lbs] * 0.4536
    out[kg] = v[kg] - 9000
    return out


def height_cm(v):
    out = v.copy()
    ft = (v >= 200) & (v <= 711)
    cm = (v >= 9000) & (v <= 9998)
    out[ft] = ((v[ft] // 100) * 12 + v[ft] % 100) * 2.54
    out[cm] = v[cm] - 9000
    return out


def hours_minutes(v):
    return (v // 100) * 60 + v % 100


CONVERSIONS = {
    "ALCDAY5": per_month,
    "STRENGTH": per_month,
    "EXEROFT1": per_month,
    "EXEROFT2": per_month,
    "EXERHMM1": hours_minutes,
    "EXERHMM2": hours_minutes,
    "WEIGHT2": weight_kg,
    "HEIGHT3": height_cm,
    **{name: per_day for name in FOOD_COLUMNS},
}


def learn_heuristic_codes(x, feature_names, use_codebook=True):
    name_to_idx = {n: j for j, n in enumerate(feature_names)}
    codebook_idx = (
        {name_to_idx[n] for n in MISSING_CODES if n in name_to_idx}
        if use_codebook
        else set()
    )
    codes = {}
    for j in range(x.shape[1]):
        if j in codebook_idx:
            continue
        non_nan = x[:, j][~np.isnan(x[:, j])]
        if non_nan.size == 0:
            continue
        if feature_names[j].startswith("_") and np.unique(non_nan).size > 50:
            continue
        maxval = non_nan.max()
        if maxval <= 9:
            codes[j] = [7, 9]
        elif maxval <= 99:
            codes[j] = [77, 99]
        elif maxval <= 999:
            codes[j] = [777, 999]
    return codes


def apply_special_codes(x, feature_names, heuristic_codes, use_codebook=True):
    x = np.array(x, dtype=float)

    for j, codes in heuristic_codes.items():
        col = x[:, j]
        col[np.isin(col, codes)] = np.nan

    if not use_codebook:
        return x

    name_to_idx = {n: j for j, n in enumerate(feature_names)}
    for name, codes in MISSING_CODES.items():
        if name in name_to_idx:
            col = x[:, name_to_idx[name]]
            col[np.isin(col, codes)] = np.nan
    for name, codes in ZERO_CODES.items():
        if name in name_to_idx:
            col = x[:, name_to_idx[name]]
            col[np.isin(col, codes)] = 0.0
    for name, convert in CONVERSIONS.items():
        if name in name_to_idx:
            j = name_to_idx[name]
            x[:, j] = convert(x[:, j])
    return x


def continuous_matrix(x, cont_params):
    if not cont_params:
        return np.zeros((len(x), 0))
    cols = []
    for j, p in cont_params.items():
        col = np.where(np.isnan(x[:, j]), p["median"], x[:, j])
        col = np.clip(col, p["lower_bound"], p["upper_bound"])
        cols.append((col - p["mean"]) / p["std"])
    return np.column_stack(cols)


def fit_preprocessor(
    x,
    y,
    feature_names,
    max_nan_frac=0.5,
    max_unique_categorical=10,
    min_category_count=100,
    flag_threshold=0.05,
    use_codebook=True,
    drop_metadata=True,
    use_flags=True,
    flag_dropped_max_frac=0.0,
    clip=True,
    n_poly=0,
):
    feature_names = list(feature_names)
    n_features = x.shape[1]
    name_to_idx = {n: j for j, n in enumerate(feature_names)}

    special_codes = learn_heuristic_codes(x, feature_names, use_codebook)
    x = apply_special_codes(x, feature_names, special_codes, use_codebook)

    nan_fraction = np.isnan(x).mean(axis=0)
    n_unique = np.zeros(n_features, dtype=int)
    for j in range(n_features):
        column = x[:, j]
        n_unique[j] = np.unique(column[~np.isnan(column)]).size

    keep_columns = (nan_fraction <= max_nan_frac) & (n_unique > 1)
    metadata_idx = [name_to_idx[n] for n in METADATA_COLUMNS if n in name_to_idx]
    if drop_metadata:
        keep_columns[metadata_idx] = False

    is_categorical = n_unique <= max_unique_categorical
    for name in FORCE_CATEGORICAL:
        if name in name_to_idx:
            is_categorical[name_to_idx[name]] = True

    continuous_params = {}
    categorical_params = {}

    for j in np.flatnonzero(keep_columns):
        name = feature_names[j]
        column = x[:, j]
        non_nan = column[~np.isnan(column)]

        if is_categorical[j]:
            values, counts = np.unique(non_nan, return_counts=True)
            frequent = values[counts >= min_category_count]
            if frequent.size == 0:
                continue

            reference = frequent[np.argmax(counts[counts >= min_category_count])]
            has_other = non_nan.size < column.size or frequent.size < values.size
            columns = frequent[frequent != reference]
            if has_other:
                columns = np.append(columns, -1.0)
            if columns.size == 0:
                continue
            categorical_params[j] = {
                "name": name,
                "frequent": frequent,
                "reference": reference,
                "columns": columns,
            }

        else:
            median = np.median(non_nan)
            if clip:
                lower_bound, upper_bound = np.percentile(non_nan, [1, 99])
            else:
                lower_bound, upper_bound = -np.inf, np.inf
            clean_column = np.clip(
                np.where(np.isnan(column), median, column), lower_bound, upper_bound
            )
            std = clean_column.std()
            if std == 0:
                std = 1.0
            continuous_params[j] = {
                "name": name,
                "median": median,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "mean": clean_column.mean(),
                "std": std,
            }

    flag_candidates = []
    if use_flags:
        flag_candidates = [
            j for j in continuous_params if nan_fraction[j] > flag_threshold
        ]
        if flag_dropped_max_frac > 0:
            flag_candidates += [
                j
                for j in np.flatnonzero(~keep_columns)
                if max_nan_frac < nan_fraction[j] <= flag_dropped_max_frac
                and j not in metadata_idx
            ]

    missing_flag_columns, seen = [], set()
    for j in flag_candidates:
        key = np.packbits(np.isnan(x[:, j])).tobytes()
        if key not in seen:
            seen.add(key)
            missing_flag_columns.append(j)

    poly = None
    if n_poly > 0 and continuous_params:
        x_cont = continuous_matrix(x, continuous_params)
        y_centered = y - y.mean()
        corr = np.abs(x_cont.T @ y_centered) / len(y) / (y_centered.std() + 1e-12)
        top = np.argsort(corr)[::-1][:n_poly]
        squares = x_cont[:, top] ** 2
        poly = {
            "idx": top,
            "mean": squares.mean(axis=0),
            "std": squares.std(axis=0) + 1e-12,
        }

    return {
        "feature_names": feature_names,
        "use_codebook": use_codebook,
        "special_codes": special_codes,
        "missing_flag_columns": missing_flag_columns,
        "continuous": continuous_params,
        "categorical": categorical_params,
        "poly": poly,
    }


def transform(x, params):
    x = apply_special_codes(
        x, params["feature_names"], params["special_codes"], params["use_codebook"]
    )

    x_cont = continuous_matrix(x, params["continuous"])
    pieces = [np.ones((len(x), 1), dtype=np.float32), x_cont.astype(np.float32)]

    if params["poly"] is not None:
        p = params["poly"]
        pieces.append(
            ((x_cont[:, p["idx"]] ** 2 - p["mean"]) / p["std"]).astype(np.float32)
        )

    flag_idx = params["missing_flag_columns"]
    if len(flag_idx) > 0:
        pieces.append(np.isnan(x[:, flag_idx]).astype(np.float32))

    for j, p in params["categorical"].items():
        col = x[:, j]
        col = np.where(np.isin(col, p["frequent"]), col, -1.0)
        pieces.append((col[:, None] == p["columns"][None, :]).astype(np.float32))

    return np.concatenate(pieces, axis=1)


def fit_preprocessor_v3(x, y, feature_names, zero_flags=False, **kwargs):
    params = fit_preprocessor(x, y, feature_names, **kwargs)
    zero_flag_columns = []
    if zero_flags:
        idx = {n: j for j, n in enumerate(feature_names)}
        for name, codes in ZERO_CODES.items():
            if name in idx:
                frac = np.isin(x[:, idx[name]], codes).mean()
                if 0 < frac < 1:
                    zero_flag_columns.append((idx[name], codes))
    params["zero_flag_columns"] = zero_flag_columns
    return params


def transform_v3(x, params):
    tx = transform(x, params)
    if not params["zero_flag_columns"]:
        return tx
    flags = np.column_stack(
        [np.isin(x[:, j], codes) for j, codes in params["zero_flag_columns"]]
    )
    return np.concatenate([tx, flags.astype(np.float32)], axis=1)


def fit_interactions(tx, y, k):
    yc = y - y.mean()
    cols = tx[:, 1:]
    corr = np.abs(cols.T @ yc) / len(y) / (cols.std(axis=0) * yc.std() + 1e-12)
    top = 1 + np.argsort(corr)[::-1][:k]
    pairs = np.array([(a, b) for i, a in enumerate(top) for b in top[i + 1 :]])
    prods = tx[:, pairs[:, 0]] * tx[:, pairs[:, 1]]
    return {"pairs": pairs, "mean": prods.mean(axis=0), "std": prods.std(axis=0) + 1e-6}


def add_interactions(tx, inter):
    prods = (
        tx[:, inter["pairs"][:, 0]] * tx[:, inter["pairs"][:, 1]] - inter["mean"]
    ) / inter["std"]
    return np.concatenate([tx, prods.astype(np.float32)], axis=1)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def predict_logistic(tx, w):
    return sigmoid(tx @ w)


def newton_logistic_regression(
    y, tx, max_iters=15, lambda_=1e-4, pos_weight=1.0, tol=1e-7, chunk=20000
):
    n, d = tx.shape
    sw = np.where(y == 1, pos_weight, 1.0)
    sw = sw / sw.mean()

    def loss_fn(w):
        return np.mean(sw * np.logaddexp(0.0, (1 - 2 * y) * (tx @ w))) + lambda_ * w @ w

    w = np.zeros(d)
    loss = loss_fn(w)
    for _ in range(max_iters):
        p = sigmoid(tx @ w)
        grad = tx.T @ (sw * (p - y)) / n + 2 * lambda_ * w

        s = sw * p * (1 - p)
        hessian = 2 * lambda_ * np.eye(d)
        for a in range(0, n, chunk):
            block = tx[a : a + chunk].astype(np.float64)
            hessian += block.T @ (block * s[a : a + chunk, None]) / n
        step = np.linalg.solve(hessian, grad)

        t = 1.0
        new_loss = loss_fn(w - t * step)
        while new_loss > loss - 1e-4 * t * (grad @ step) and t > 1e-4:
            t /= 2
            new_loss = loss_fn(w - t * step)
        w = w - t * step

        converged = loss - new_loss < tol * max(1.0, loss)
        loss = new_loss
        if converged:
            break

    return w, np.mean(np.logaddexp(0.0, (1 - 2 * y) * (tx @ w)))


N_BINS = 64
NAN_BIN = N_BINS - 1


def fit_binner(x, feature_names, use_codebook=True, drop_metadata=True):
    feature_names = list(feature_names)
    codes = learn_heuristic_codes(x, feature_names, use_codebook)
    xc = apply_special_codes(x, feature_names, codes, use_codebook)
    metadata_idx = {
        feature_names.index(n) for n in METADATA_COLUMNS if n in feature_names
    }

    columns, edges = [], []
    for j in range(xc.shape[1]):
        if drop_metadata and j in metadata_idx:
            continue
        col = xc[:, j]
        non_nan = col[~np.isnan(col)]
        values = np.unique(non_nan)
        if values.size == 0 or (values.size == 1 and non_nan.size == col.size):
            continue
        if values.size <= NAN_BIN:
            e = (values[:-1] + values[1:]) / 2
        else:
            e = np.unique(
                np.percentile(non_nan, np.linspace(0, 100, NAN_BIN + 1)[1:-1])
            )
        columns.append(j)
        edges.append(e)
    return {
        "feature_names": feature_names,
        "codes": codes,
        "use_codebook": use_codebook,
        "columns": columns,
        "edges": edges,
    }


def bin_data(x, binner):
    xc = apply_special_codes(
        x, binner["feature_names"], binner["codes"], binner["use_codebook"]
    )
    out = np.empty((len(binner["columns"]), len(x)), dtype=np.uint8)
    for k, (j, e) in enumerate(zip(binner["columns"], binner["edges"])):
        col = xc[:, j]
        b = np.searchsorted(e, col, side="right")
        b[np.isnan(col)] = NAN_BIN
        out[k] = b
    return out


def build_tree(xb, g, h, features, max_depth, reg_lambda, min_child_hess):
    n_nodes = 2 ** (max_depth + 1) - 1
    split = np.zeros(n_nodes, dtype=bool)
    feat = np.zeros(n_nodes, dtype=np.int64)
    thr = np.zeros(n_nodes, dtype=np.int64)
    nan_left = np.zeros(n_nodes, dtype=bool)

    n = xb.shape[1]
    node = np.zeros(n, dtype=np.int64)
    active = np.arange(n)
    score = lambda G, H: G * G / (H + reg_lambda)

    hg_parent = hh_parent = None
    for depth in range(max_depth):
        if active.size == 0:
            break
        first, width = 2**depth - 1, 2**depth
        local = node[active] - first

        if depth == 0:
            rows, key, n_hist = active, local * N_BINS, 1
        else:
            is_left = local % 2 == 0
            rows, key, n_hist = (
                active[is_left],
                (local[is_left] // 2) * N_BINS,
                width // 2,
            )
        g_r, h_r = g[rows], h[rows]
        xr = xb[:, rows]
        hg_new = np.empty((n_hist, len(features), N_BINS))
        hh_new = np.empty((n_hist, len(features), N_BINS))
        for k, f in enumerate(features):
            kf = key + xr[f]
            hg_new[:, k, :] = np.bincount(
                kf, weights=g_r, minlength=n_hist * N_BINS
            ).reshape(n_hist, N_BINS)
            hh_new[:, k, :] = np.bincount(
                kf, weights=h_r, minlength=n_hist * N_BINS
            ).reshape(n_hist, N_BINS)
        if depth == 0:
            hg, hh = hg_new, hh_new
        else:

            parent_split = split[2 ** (depth - 1) - 1 : first][:, None, None]
            hg = np.empty((width, len(features), N_BINS))
            hh = np.empty((width, len(features), N_BINS))
            hg[0::2], hh[0::2] = hg_new, hh_new
            hg[1::2] = np.where(parent_split, hg_parent - hg_new, 0.0)
            hh[1::2] = np.where(parent_split, hh_parent - hh_new, 0.0)
        hg_parent, hh_parent = hg, hh

        G = hg[:, 0, :].sum(axis=1)[:, None, None]
        H = hh[:, 0, :].sum(axis=1)[:, None, None]
        cg, ch = np.cumsum(hg[:, :, :NAN_BIN], axis=2), np.cumsum(
            hh[:, :, :NAN_BIN], axis=2
        )
        ng, nh = hg[:, :, NAN_BIN:], hh[:, :, NAN_BIN:]

        gains = np.empty(cg.shape + (2,))

        for side, (GL, HL) in enumerate([(cg, ch), (cg + ng, ch + nh)]):
            HR = H - HL
            gain = score(GL, HL) + score(G - GL, HR) - score(G, H)
            gain[(HL < min_child_hess) | (HR < min_child_hess)] = -np.inf
            gains[..., side] = gain

        flat = gains.reshape(width, -1)
        best = flat.argmax(axis=1)
        best_gain = flat[np.arange(width), best]
        k_best, t_best, side_best = np.unravel_index(best, cg.shape[1:] + (2,))
        for i in np.flatnonzero(best_gain > 1e-12):
            nd = first + i
            split[nd] = True
            feat[nd] = features[k_best[i]]
            thr[nd] = t_best[i]
            nan_left[nd] = side_best[i] == 1

        go = split[node[active]]
        active = active[go]
        nd = node[active]
        b = xb[feat[nd], active]
        left = np.where(b == NAN_BIN, nan_left[nd], b <= thr[nd])
        node[active] = 2 * nd + 1 + (~left)

    G_leaf = np.bincount(node, weights=g, minlength=n_nodes)
    H_leaf = np.bincount(node, weights=h, minlength=n_nodes)
    value = -G_leaf / (H_leaf + reg_lambda)
    return {
        "split": split,
        "feat": feat,
        "thr": thr,
        "nan_left": nan_left,
        "value": value,
    }


def predict_tree(tree, xb):
    node = np.zeros(xb.shape[1], dtype=np.int64)
    idx = np.arange(xb.shape[1])
    while idx.size > 0:
        nd = node[idx]
        go = tree["split"][nd]
        idx, nd = idx[go], nd[go]
        if idx.size == 0:
            break
        b = xb[tree["feat"][nd], idx]
        left = np.where(b == NAN_BIN, tree["nan_left"][nd], b <= tree["thr"][nd])
        node[idx] = 2 * nd + 1 + (~left)
    return tree["value"][node]


def gbdt_fit(
    xb,
    y,
    xb_valid=None,
    y_valid=None,
    n_trees=500,
    learning_rate=0.1,
    max_depth=5,
    reg_lambda=1.0,
    min_child_hess=5.0,
    subsample=0.5,
    colsample=0.8,
    pos_weight=1.0,
    early_stopping=50,
    seed=SEED,
):
    rng = np.random.RandomState(seed)
    n_feat, n = xb.shape
    sw = np.where(y == 1, pos_weight, 1.0)
    p0 = np.clip(np.sum(sw * y) / np.sum(sw), 1e-6, 1 - 1e-6)
    base = np.log(p0 / (1 - p0))

    raw = np.full(n, base)
    has_valid = xb_valid is not None
    raw_valid = np.full(xb_valid.shape[1], base) if has_valid else None
    trees, best_loss, best_iter = [], np.inf, 0

    for t in range(n_trees):
        p = sigmoid(raw)
        g, h = sw * (p - y), sw * p * (1 - p)
        rows = np.sort(rng.choice(n, int(subsample * n), replace=False))
        features = np.sort(
            rng.choice(n_feat, max(1, int(colsample * n_feat)), replace=False)
        )
        tree = build_tree(
            xb[:, rows],
            g[rows],
            h[rows],
            features,
            max_depth,
            reg_lambda,
            min_child_hess,
        )
        tree["value"] *= learning_rate
        trees.append(tree)
        raw += predict_tree(tree, xb)

        if has_valid:
            raw_valid += predict_tree(tree, xb_valid)
            loss = np.mean(np.logaddexp(0.0, (1 - 2 * y_valid) * raw_valid))
            if loss < best_loss - 1e-7:
                best_loss, best_iter = loss, t + 1
            elif t + 1 - best_iter >= early_stopping:
                break

    if has_valid:
        trees = trees[:best_iter]
    return {"base": base, "trees": trees}


def gbdt_predict_proba(model, xb):
    raw = np.full(xb.shape[1], model["base"])
    for tree in model["trees"]:
        raw += predict_tree(tree, xb)
    return sigmoid(raw)


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    with open(os.path.join(DATA_PATH, "x_train.csv")) as f:
        feature_names = f.readline().strip().split(",")[1:]
    y01 = (y_train == 1).astype(float)
    cfg = CONFIG
    p_test = np.zeros(len(x_test))

    if cfg["blend_weight"] > 0:
        binner = fit_binner(x_train, feature_names)
        xb_full, xb_test = bin_data(x_train, binner), bin_data(x_test, binner)
        model = gbdt_fit(xb_full, y01, n_trees=cfg["gbdt_trees"], **cfg["gbdt_params"])
        p_test += cfg["blend_weight"] * gbdt_predict_proba(model, xb_test)
        del xb_full, xb_test
        print(f"GBDT done ({len(model['trees'])} trees)")

    if cfg["blend_weight"] < 1:
        params = fit_preprocessor_v3(x_train, y01, feature_names, **cfg["logistic_pk"])
        tx_full, tx_test = transform_v3(x_train, params), transform_v3(x_test, params)
        if cfg["logistic_interactions"] > 0:
            inter = fit_interactions(tx_full, y01, cfg["logistic_interactions"])
            tx_full, tx_test = add_interactions(tx_full, inter), add_interactions(
                tx_test, inter
            )
        w, _ = newton_logistic_regression(
            y01, tx_full, 15, cfg["logistic_lambda"], cfg["logistic_pos_weight"]
        )
        p_test += (1 - cfg["blend_weight"]) * predict_logistic(tx_test, w)
        print(f"logistic regression done ({tx_full.shape[1]} features)")

    y_pred = np.where(p_test >= cfg["threshold"], 1, -1)
    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(
        f"wrote {OUTPUT}: {len(y_pred)} rows, predicted positive rate {(y_pred == 1).mean():.4f}"
    )


if __name__ == "__main__":
    main()
