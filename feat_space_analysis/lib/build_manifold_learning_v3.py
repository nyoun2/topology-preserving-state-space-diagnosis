import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness as sk_trustworthiness


# ============================================================
# Utils: mkdir
# ============================================================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ============================================================
# (A) 평가 함수들 (Spearman 샘플링 + Jaccard + Continuity 근사 + Trustworthiness)
#    - user가 이전에 돌린 것과 동일한 계열
# ============================================================
def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_pairwise_distance_sample(
    X: np.ndarray,
    Y: np.ndarray,
    n_pairs: int = 200_000,
    seed: int = 0,
    chunk: int = 50_000,
) -> float:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    assert X.shape[0] == Y.shape[0]
    N = X.shape[0]
    if N < 3:
        return float("nan")

    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, N, size=n_pairs, endpoint=False)
    j_idx = rng.integers(0, N, size=n_pairs, endpoint=False)

    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    n_eff = i_idx.size
    if n_eff == 0:
        return float("nan")

    dx = np.empty(n_eff, dtype=np.float64)
    dy = np.empty(n_eff, dtype=np.float64)

    for s in range(0, n_eff, chunk):
        e = min(n_eff, s + chunk)
        a = i_idx[s:e]
        b = j_idx[s:e]
        dX = X[a] - X[b]
        dY = Y[a] - Y[b]
        dx[s:e] = np.sqrt(np.sum(dX * dX, axis=1))
        dy[s:e] = np.sqrt(np.sum(dY * dY, axis=1))

    rx = _rankdata_average_ties(dx)
    ry = _rankdata_average_ties(dy)

    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def knn_jaccard_overlap(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 20,
    metric_x: str = "euclidean",
    metric_y: str = "euclidean",
) -> float:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    N = X.shape[0]
    if N <= k + 1:
        return float("nan")

    nnx = NearestNeighbors(n_neighbors=min(N, k + 1), metric=metric_x).fit(X)
    nny = NearestNeighbors(n_neighbors=min(N, k + 1), metric=metric_y).fit(Y)

    idx_x = nnx.kneighbors(return_distance=False)[:, 1:]
    idx_y = nny.kneighbors(return_distance=False)[:, 1:]

    scores = np.empty(N, dtype=np.float64)
    for i in range(N):
        ax = set(idx_x[i].tolist())
        ay = set(idx_y[i].tolist())
        inter = len(ax.intersection(ay))
        union = len(ax.union(ay))
        scores[i] = inter / union if union > 0 else 0.0
    return float(scores.mean())


def continuity_approx(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 20,
    m: int = 200,
    metric_x: str = "euclidean",
    metric_y: str = "euclidean",
) -> float:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    N = X.shape[0]
    k = int(k)
    m = int(max(m, k))

    # continuity 공식 분모가 음수/0이 되는 구간 방지
    if N <= 3 * k + 1:
        return float("nan")

    nnx = NearestNeighbors(n_neighbors=min(N, k + 1), metric=metric_x).fit(X)
    nny = NearestNeighbors(n_neighbors=min(N, m + 1), metric=metric_y).fit(Y)

    knn_x = nnx.kneighbors(return_distance=False)[:, 1:]   # (N,k)
    mnn_y = nny.kneighbors(return_distance=False)[:, 1:]   # (N,m)

    rank_maps = []
    for i in range(N):
        rm = {int(j): r for r, j in enumerate(mnn_y[i], start=1)}
        rank_maps.append(rm)

    sum_term = 0.0
    for i in range(N):
        orig = knn_x[i]
        emb_topk = set(mnn_y[i][:k].tolist())
        rm = rank_maps[i]
        for j in orig:
            j = int(j)
            if j in emb_topk:
                continue
            r_ij = rm.get(j, m + 1)
            if r_ij > k:
                sum_term += (r_ij - k)

    denom = N * k * (2 * N - 3 * k - 1)
    C = 1.0 - (2.0 / denom) * sum_term
    return float(C)


def evaluate_umap_embedding(
    X_128d: np.ndarray,
    Z_2d: np.ndarray,
    ks=(10, 20, 50),
    spearman_pairs: int = 200_000,
    spearman_seed: int = 0,
    continuity_m: int = 200,
    metric_x: str = "euclidean",
    metric_z: str = "euclidean",
) -> dict:
    X = np.asarray(X_128d, dtype=np.float32)
    Z = np.asarray(Z_2d, dtype=np.float32)
    assert X.shape[0] == Z.shape[0]

    out = {}
    out["spearman_dist_sampled"] = spearman_pairwise_distance_sample(
        X, Z, n_pairs=spearman_pairs, seed=spearman_seed
    )

    out["per_k"] = {}
    for k in ks:
        k = int(k)
        tw = sk_trustworthiness(X, Z, n_neighbors=k, metric=metric_x)
        cj = knn_jaccard_overlap(X, Z, k=k, metric_x=metric_x, metric_y=metric_z)
        co = continuity_approx(X, Z, k=k, m=continuity_m, metric_x=metric_x, metric_y=metric_z)

        out["per_k"][k] = {
            "trustworthiness": float(tw),
            "continuity_approx": float(co),
            "knn_jaccard": float(cj),
        }
    return out


def evaluate_umap_embedding_subset(
    X: np.ndarray,
    Z: np.ndarray,
    idx: np.ndarray,
    ks=(10, 20, 50),
    spearman_pairs: int = 100_000,
    spearman_seed: int = 0,
    continuity_m: int = 200,
    metric_x: str = "euclidean",
    metric_z: str = "euclidean",
) -> dict:
    idx = np.asarray(idx, dtype=np.int32)
    Xs = X[idx]
    Zs = Z[idx]
    # subset이 작으면 pair 수 자동 축소
    n = Xs.shape[0]
    if n < 200:
        sp = min(spearman_pairs, 10_000)
    else:
        sp = min(spearman_pairs, max(10_000, n * 50))
    return evaluate_umap_embedding(
        Xs, Zs,
        ks=ks,
        spearman_pairs=sp,
        spearman_seed=spearman_seed,
        continuity_m=continuity_m,
        metric_x=metric_x,
        metric_z=metric_z,
    )


# ============================================================
# (B) Composite score: val 기준으로 best 결정
#    - trustworthiness/jaccard 중심, continuity 보조, spearman은 벌점(optional)
# ============================================================
def composite_score(metrics: dict, ks=(10, 20, 50), spearman_floor=0.30) -> float:
    per_k = metrics["per_k"]
    T = []
    J = []
    C = []
    for k in ks:
        T.append(per_k[k]["trustworthiness"])
        J.append(per_k[k]["knn_jaccard"])
        C.append(per_k[k]["continuity_approx"])
    Tm = float(np.nanmean(T))
    Jm = float(np.nanmean(J))
    Cm = float(np.nanmean(C))

    sp = float(metrics.get("spearman_dist_sampled", np.nan))
    penalty = 0.0
    if np.isfinite(sp):
        penalty = max(0.0, spearman_floor - sp)  # 너무 낮을 때만 벌점

    # 가중치는 상황에 맞춰 조절 가능
    score = 0.55 * Tm + 0.35 * Jm + 0.10 * Cm - 0.05 * penalty
    return float(score)


# ============================================================
# (C) Parametric manifold model (MLP)
# ============================================================
def build_parametric_embedder(input_dim=128, hidden=(256, 256, 128), out_dim=2, dropout=0.1):
    inp = keras.Input(shape=(input_dim,), name="x")
    x = inp
    for h in hidden:
        x = layers.Dense(h)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)
    z = layers.Dense(out_dim, name="z")(x)
    return keras.Model(inp, z, name="parametric_embedder")


# ============================================================
# (D) Build kNN edges + edge split
# ============================================================
def build_knn_edges(X, k=30, metric="euclidean"):
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(N, k + 1), metric=metric)
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)  # (N,k+1)

    dists = dists[:, 1:]  # (N,k)
    idx = idx[:, 1:]      # (N,k)

    sigma = dists[:, -1].astype(np.float32)
    sigma = np.maximum(sigma, 1e-6)

    src = np.repeat(np.arange(N, dtype=np.int32), k)
    dst = idx.reshape(-1).astype(np.int32)
    d_pos = dists.reshape(-1).astype(np.float32)

    edges = np.stack([src, dst], axis=1)  # (M,2)
    return edges, d_pos, sigma, idx  # idx는 negative sampling 회피용


def split_edges(edges, d_pos, val_edge_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    M = edges.shape[0]
    perm = rng.permutation(M)
    edges = edges[perm]
    d_pos = d_pos[perm]

    cut = int(M * (1.0 - val_edge_ratio))
    tr_edges, va_edges = edges[:cut], edges[cut:]
    tr_dpos, va_dpos = d_pos[:cut], d_pos[cut:]
    return (tr_edges, tr_dpos), (va_edges, va_dpos)


# ============================================================
# (E) Negative sampler (rejection sampling)
# ============================================================
def make_neg_sampler(N, idx_knn, seed=0):
    rng = np.random.default_rng(seed)
    knn_sets = [set(row.tolist()) for row in idx_knn]

    def sample(i_batch, num_neg):
        i_batch = np.asarray(i_batch, dtype=np.int32)
        B = i_batch.shape[0]
        neg = np.empty((B, num_neg), dtype=np.int32)
        for b, i in enumerate(i_batch):
            s = knn_sets[i]
            c = 0
            while c < num_neg:
                j = rng.integers(0, N)
                if j == i or j in s:
                    continue
                neg[b, c] = j
                c += 1
        return neg

    return sample

# ============================================================
# (E2) Temporal edges (sequential, step=1)
# ============================================================
def build_temporal_edges_sequential(N: int, step: int = 1) -> np.ndarray:
    step = int(max(1, step))
    if N <= step:
        return np.zeros((0, 2), dtype=np.int32)
    src = np.arange(0, N - step, dtype=np.int32)
    dst = src + step
    return np.stack([src, dst], axis=1)

# ============================================================
# (F) Train step: graph-contrastive(UMAP-ish) + stress + margin repulsion
# ============================================================
@tf.function
def train_step(
    model, opt,
    x_i, x_j, x_neg,
    d_ij, sigma_i, sigma_j,
    # temporal (optional)
    x_ti=None, x_tj=None, d_t=None, sigma_ti=None, sigma_tj=None,

    w_bce=1.0,
    w_stress=0.1,
    w_temp=0.02,          # << 작게 시작 권장
    neg_margin=1.0,
    eps=1e-6
):
    with tf.GradientTape() as tape:
        zi = model(x_i, training=True)  # (B,2)
        zj = model(x_j, training=True)  # (B,2)

        zneg = model(tf.reshape(x_neg, (-1, tf.shape(x_neg)[-1])), training=True)
        zneg = tf.reshape(zneg, (tf.shape(x_neg)[0], tf.shape(x_neg)[1], 2))  # (B,Kneg,2)

        dij_2 = tf.sqrt(tf.reduce_sum(tf.square(zi - zj), axis=1) + eps)  # (B,)
        dik_2 = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(zi, 1) - zneg), axis=2) + eps)  # (B,Kneg)

        # high-d edge weight (local scale)
        p = tf.exp(-tf.square(d_ij) / (sigma_i * sigma_j + eps))
        p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)

        # low-d similarity (UMAP kernel style)
        q_pos = 1.0 / (1.0 + tf.square(dij_2))
        q_pos = tf.clip_by_value(q_pos, 1e-6, 1.0 - 1e-6)

        q_neg = 1.0 / (1.0 + tf.square(dik_2))
        q_neg = tf.clip_by_value(q_neg, 1e-6, 1.0 - 1e-6)

        # BCE-like
        loss_pos = -tf.reduce_mean(p * tf.math.log(q_pos))
        loss_neg = -tf.reduce_mean(tf.math.log(1.0 - q_neg))
        loss_bce = loss_pos + loss_neg

        # stress (local distance shape control; log-space)
        t = d_ij / (sigma_i + eps)
        loss_stress = tf.reduce_mean(tf.square(tf.math.log(dij_2 + 1e-3) - tf.math.log(t + 1e-3)))

        # margin repulsion (stability)
        mean_neg_d = tf.reduce_mean(dik_2, axis=1)
        loss_margin = tf.reduce_mean(tf.nn.relu(neg_margin - mean_neg_d))

        # ---- gated temporal smoothness ----
        if (x_ti is not None) and (x_tj is not None):
            z_ti = model(x_ti, training=True)
            z_tj = model(x_tj, training=True)
            dt_2 = tf.sqrt(tf.reduce_sum(tf.square(z_ti - z_tj), axis=1) + eps)

            # gate weight (원공간에서 가까운 경우에만 temporal 항이 켜짐)
            w_gate = tf.exp(-tf.square(d_t) / (sigma_ti * sigma_tj + eps))
            w_gate = tf.stop_gradient(tf.clip_by_value(w_gate, 0.0, 1.0))

            loss_temp = tf.reduce_mean(w_gate * tf.math.log1p(tf.square(dt_2)))
        else:
            loss_temp = tf.constant(0.0, dtype=tf.float32)

        #loss = w_bce * loss_bce + w_stress * loss_stress + 0.05 * loss_margin
        loss = (w_bce * loss_bce) + (w_stress * loss_stress) + (0.05 * loss_margin) + (w_temp * loss_temp)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    #return loss, loss_bce, loss_stress, loss_pos, loss_neg, loss_margin
    return loss, loss_bce, loss_stress, loss_temp, loss_pos, loss_neg, loss_margin


# ============================================================
# (G) Fit with:
#  - edge split training
#  - train/val evaluation (point subset)
#  - composite score 기반 best 저장
#  - patience 조기 종료
# ============================================================
def fit_parametric_manifold_with_earlystop(
    X,
    out_dir="pm_umap_ckpt",
    input_dim=128,
    hidden=(256, 256, 128),
    out_dim=2,
    dropout=0.1,
    metric="euclidean",

    # graph / sampling
    k_graph=30,
    num_neg=15,
    batch_edges=2048,
    val_edge_ratio=0.2,

    # temporal smoothness
    use_temporal=True,
    temporal_step=1,      # t -> t+1 (6h 간격이라면 1이 6h)
    temporal_ratio=1.0,   # edge batch 크기 대비 temporal batch 크기 비율
    w_temp=0.02,          # << 작게 시작 권장 (0.005~0.03)

    # optimization
    lr=1e-3,
    epochs=200,
    w_stress=0.1,
    neg_margin=1.0,

    # evaluation
    eval_ks=(10, 20, 50),
    eval_every=1,
    spearman_pairs_train=120_000,
    spearman_pairs_val=120_000,
    continuity_m=200,

    # early stop
    patience=20,
    min_delta=1e-4,

    # evaluation point split
    val_point_ratio=0.2,

    seed=0,
    verbose=1,
):
    ensure_dir(out_dir)
    X = np.asarray(X, dtype=np.float32)
    N, D = X.shape
    assert D == input_dim, f"X dim={D}, expected={input_dim}"

    # -------------------------
    # (1) point split (evaluation 용)
    # -------------------------
    rng = np.random.default_rng(seed)
    perm_pts = rng.permutation(N)
    cutp = int(N * (1.0 - val_point_ratio))
    train_idx = np.sort(perm_pts[:cutp]).astype(np.int32)
    val_idx = np.sort(perm_pts[cutp:]).astype(np.int32)

    # -------------------------
    # (2) build kNN edges (전체에서) + edge split (training 용)
    # -------------------------
    edges, d_pos, sigma, idx_knn = build_knn_edges(X, k=k_graph, metric=metric)
    (tr_edges, tr_dpos), (va_edges, va_dpos) = split_edges(edges, d_pos, val_edge_ratio=val_edge_ratio, seed=seed)

    neg_sampler = make_neg_sampler(N, idx_knn, seed=seed)

    # -------------------------
    # (2-2) temporal edges (sequential)
    # -------------------------
    if use_temporal:
        temp_edges = build_temporal_edges_sequential(N, step=temporal_step)
        if temp_edges.shape[0] == 0:
            raise ValueError("Temporal edges가 0개입니다. N/temporal_step을 확인하세요.")
        Mtemp = temp_edges.shape[0]
    else:
        temp_edges = None
        Mtemp = 0

    # -------------------------
    # (3) model
    # -------------------------
    model = build_parametric_embedder(input_dim=input_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)
    opt = keras.optimizers.Adam(lr)

    best_score = -np.inf
    best_epoch = -1
    wait = 0

    best_path = os.path.join(out_dir, "best.weights.h5")
    last_path = os.path.join(out_dir, "last.weights.h5")
    log_path = os.path.join(out_dir, "train_log.jsonl")

    # epoch 로그 누적
    if os.path.exists(log_path):
        os.remove(log_path)

    def write_log(obj):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # -------------------------
    # (4) training loop
    # -------------------------
    Mtr = tr_edges.shape[0]
    for ep in range(1, epochs + 1):
        # shuffle edges each epoch
        perm_e = rng.permutation(Mtr)
        ep_edges = tr_edges[perm_e]
        ep_dpos = tr_dpos[perm_e]

        # shuffle temporal edges
        if use_temporal:
            perm_t = rng.permutation(Mtemp)
            ep_temp_edges = temp_edges[perm_t]
            tptr = 0

        loss_list = []
        bce_list = []
        stress_list = []
        temp_list = []

        # minibatch over edges
        for s in range(0, Mtr, batch_edges):
            e = min(Mtr, s + batch_edges)
            batch = ep_edges[s:e]
            d_ij = ep_dpos[s:e]
            B = batch.shape[0]

            i = batch[:, 0]
            j = batch[:, 1]

            neg_idx = neg_sampler(i, num_neg)  # (B,num_neg)

            x_i = X[i]
            x_j = X[j]
            x_neg = X[neg_idx]

            sig_i = sigma[i].astype(np.float32)
            sig_j = sigma[j].astype(np.float32)

            # temporal batch 구성
            if use_temporal:
                Bt = int(max(1, round(B * float(temporal_ratio))))
                if tptr + Bt > Mtemp:
                    tptr = 0
                t_batch = ep_temp_edges[tptr:tptr + Bt]
                tptr += Bt

                ti = t_batch[:, 0]
                tj = t_batch[:, 1]
                x_ti = X[ti]
                x_tj = X[tj]

                # 원공간 temporal distance + local scale
                d_t = np.sqrt(np.sum((x_ti - x_tj) ** 2, axis=1)).astype(np.float32)
                sig_ti = sigma[ti].astype(np.float32)
                sig_tj = sigma[tj].astype(np.float32)
            else:
                x_ti = x_tj = d_t = sig_ti = sig_tj = None

            loss, loss_bce, loss_stress, loss_temp, lp, ln, lm = train_step(
                model, opt,
                tf.convert_to_tensor(x_i),
                tf.convert_to_tensor(x_j),
                tf.convert_to_tensor(x_neg),
                tf.convert_to_tensor(d_ij.astype(np.float32)),
                tf.convert_to_tensor(sig_i),
                tf.convert_to_tensor(sig_j),

                x_ti=(tf.convert_to_tensor(x_ti) if use_temporal else None),
                x_tj=(tf.convert_to_tensor(x_tj) if use_temporal else None),
                d_t=(tf.convert_to_tensor(d_t) if use_temporal else None),
                sigma_ti=(tf.convert_to_tensor(sig_ti) if use_temporal else None),
                sigma_tj=(tf.convert_to_tensor(sig_tj) if use_temporal else None),

                w_bce=tf.convert_to_tensor(1.0, tf.float32),
                w_stress=tf.convert_to_tensor(w_stress, tf.float32),
                w_temp=tf.convert_to_tensor(w_temp, tf.float32),
                neg_margin=tf.convert_to_tensor(neg_margin, tf.float32),
            )

            loss_list.append(float(loss))
            bce_list.append(float(loss_bce))
            stress_list.append(float(loss_stress))
            temp_list.append(float(loss_temp))

        # epoch train loss summary
        loss_mean = float(np.mean(loss_list))
        bce_mean = float(np.mean(bce_list))
        stress_mean = float(np.mean(stress_list))
        temp_mean = float(np.mean(temp_list)) if len(temp_list) > 0 else 0.0
        

        if verbose:
            if use_temporal:
                print(f"[{ep:03d}/{epochs}] loss={loss_mean:.4f}  bce={bce_mean:.4f}  stress={stress_mean:.4f}  temp={temp_mean:.4f}")
            else:
                print(f"[{ep:03d}/{epochs}] loss={loss_mean:.4f}  bce={bce_mean:.4f}  stress={stress_mean:.4f}")

        # -------------------------
        # (5) evaluation + early stopping (val composite score)
        # -------------------------
        do_eval = (ep % eval_every == 0)
        metrics_tr = None
        metrics_va = None
        score_tr = None
        score_va = None

        if do_eval:
            # embed all once (빠르게)
            Z_all = model.predict(X, batch_size=4096, verbose=0).astype(np.float32)

            metrics_tr = evaluate_umap_embedding_subset(
                X, Z_all, train_idx,
                ks=eval_ks,
                spearman_pairs=spearman_pairs_train,
                spearman_seed=seed + ep,
                continuity_m=continuity_m,
                metric_x=metric,
                metric_z="euclidean",
            )
            metrics_va = evaluate_umap_embedding_subset(
                X, Z_all, val_idx,
                ks=eval_ks,
                spearman_pairs=spearman_pairs_val,
                spearman_seed=seed + 10_000 + ep,
                continuity_m=continuity_m,
                metric_x=metric,
                metric_z="euclidean",
            )

            score_tr = composite_score(metrics_tr, ks=eval_ks)
            score_va = composite_score(metrics_va, ks=eval_ks)

            if verbose:
                print(f"   score(train)={score_tr:.4f}  score(val)={score_va:.4f}  best(val)={best_score:.4f}")

            # save last
            model.save_weights(last_path)

            # update best
            if score_va > best_score + min_delta:
                best_score = score_va
                best_epoch = ep
                wait = 0

                model.save_weights(best_path)

                # best 메타 저장
                meta = {
                    "best_epoch": best_epoch,
                    "best_score_val": best_score,
                    "config": {
                        "input_dim": input_dim,
                        "hidden": list(hidden),
                        "out_dim": out_dim,
                        "dropout": dropout,
                        "metric": metric,
                        "k_graph": k_graph,
                        "num_neg": num_neg,
                        "w_stress": w_stress,
                        "w_temp": w_temp,
                        "use_temporal": bool(use_temporal),
                        "temporal_step": int(temporal_step),
                        "temporal_ratio": float(temporal_ratio),
                        "neg_margin": neg_margin,
                        "eval_ks": list(eval_ks),
                    },
                }
                with open(os.path.join(out_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            else:
                wait += 1

            # log jsonl
            write_log({
                "epoch": ep,
                "loss": loss_mean,
                "bce": bce_mean,
                "stress": stress_mean,
                "temp": temp_mean,
                "score_train": score_tr,
                "score_val": score_va,
                "metrics_train": metrics_tr,
                "metrics_val": metrics_va,
                "best_epoch": best_epoch,
                "best_score_val": best_score,
                "wait": wait,
            })

            if wait >= patience:
                if verbose:
                    print(f"[EARLY STOP] no val-score improvement for {patience} evals. best_epoch={best_epoch}, best_score={best_score:.4f}")
                break
        else:
            # eval 안하는 epoch도 log는 남길 수 있음
            write_log({
                "epoch": ep,
                "loss": loss_mean,
                "bce": bce_mean,
                "stress": stress_mean,
                "temp": temp_mean,
                "best_epoch": best_epoch,
                "best_score_val": best_score,
            })

    # training end: best weights 로드해서 반환
    if os.path.exists(best_path):
        model.load_weights(best_path)
    return {
        "model": model,
        "best_path": best_path,
        "last_path": last_path,
        "best_epoch": best_epoch,
        "best_score_val": best_score,
        "out_dir": out_dir,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


# ============================================================
# (H) Inference: saved weights로 128d -> 2d 투영
# ============================================================
# from tensorflow.keras.utils import model_to_dot
# from tensorflow.keras.utils import plot_model

def load_embedder_and_project(
    X_new: np.ndarray,
    weights_path: str,
    input_dim=128,
    hidden=(256, 256, 128),
    out_dim=2,
    dropout=0.1,
    batch_size=4096,
) -> np.ndarray:
    """
    X_new: (M,128)
    return Z_new: (M,2)
    """
    X_new = np.asarray(X_new, dtype=np.float32)
    assert X_new.ndim == 2 and X_new.shape[1] == input_dim

    model = build_parametric_embedder(input_dim=input_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)
    model.load_weights(weights_path)

    # print(model.summary())
    # dot = model_to_dot(model, show_shapes=True, show_layer_names=True, dpi=200)
    # dot.write_png("parametric_embedder_dot.png")
    # plot_model(
    #     model,
    #     to_file="parametric_embedder.png",
    #     show_shapes=True,
    #     show_layer_names=True,
    #     expand_nested=True,
    #     dpi=200
    # )

    Z_new = model.predict(X_new, batch_size=batch_size, verbose=0).astype(np.float32)
    return Z_new


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # X: (N,128) 준비되어 있다고 가정
    # result = fit_parametric_manifold_with_earlystop(X, out_dir="pm_run01", epochs=200, patience=20)
    # model = result["model"]
    # Z = model.predict(X, batch_size=4096)

    # 추론 예시:
    # Z_new = load_embedder_and_project(X_new, weights_path=result["best_path"])
    pass
