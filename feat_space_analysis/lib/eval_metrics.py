import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness as sk_trustworthiness
from collections import defaultdict
from itertools import combinations

from feat_space_analysis.lib.io_paths import glob_npy
from feat_space_analysis.lib.utils import append_jsonl, build_result_log
from feat_space_analysis.lib.feature_extract import _zscore_pair, _l2_normalize, _l2, _cosine_distance

def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    SciPy 없이 Spearman을 계산하기 위한 rankdata (average ties).
    반환 rank는 1..n (float).
    """
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i
        # 같은 값(동점) 구간 찾기
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # 평균 rank 할당 (1-indexed)
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
    """
    X: (N, D) 원 feature (예: 128d)
    Y: (N, d) 임베딩 (예: 2d)
    n_pairs 만큼 랜덤 pair를 뽑아, 거리의 Spearman rank correlation을 계산.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X.shape[0] == Y.shape[0]
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, N, size=n_pairs, endpoint=False)
    j_idx = rng.integers(0, N, size=n_pairs, endpoint=False)

    # i==j 제거 (거리 0 과잉 방지)
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    n_pairs_eff = i_idx.size
    if n_pairs_eff == 0:
        return np.nan

    dx = np.empty(n_pairs_eff, dtype=np.float64)
    dy = np.empty(n_pairs_eff, dtype=np.float64)

    # 메모리/속도 균형을 위해 chunk 계산
    for s in range(0, n_pairs_eff, chunk):
        e = min(n_pairs_eff, s + chunk)
        a = i_idx[s:e]
        b = j_idx[s:e]
        dX = X[a] - X[b]
        dY = Y[a] - Y[b]
        dx[s:e] = np.sqrt(np.sum(dX * dX, axis=1))
        dy[s:e] = np.sqrt(np.sum(dY * dY, axis=1))

    rx = _rankdata_average_ties(dx)
    ry = _rankdata_average_ties(dy)

    # Pearson corr of ranks = Spearman
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom == 0:
        return np.nan
    return float(np.dot(rx, ry) / denom)


def knn_jaccard_overlap(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 20,
    metric_x: str = "euclidean",
    metric_y: str = "euclidean",
) -> float:
    """
    각 점의 kNN 집합이 원공간과 임베딩공간에서 얼마나 겹치는지(Jaccard) 평균.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    N = X.shape[0]

    nnx = NearestNeighbors(n_neighbors=min(N, k + 1), metric=metric_x).fit(X)
    nny = NearestNeighbors(n_neighbors=min(N, k + 1), metric=metric_y).fit(Y)

    idx_x = nnx.kneighbors(return_distance=False)[:, 1:]  # self 제외
    idx_y = nny.kneighbors(return_distance=False)[:, 1:]  # self 제외

    # Jaccard: |A∩B| / |A∪B|
    # k가 작으니 set로 처리해도 충분히 빠름 (N~7300 기준)
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
    """
    Continuity의 근사 버전.
    - 원공간에서 kNN을 구함.
    - 임베딩공간에서는 mNN(>=k)을 구해 그 안에서의 순위(rank)를 사용.
    - 임베딩 mNN 밖으로 밀려난 점은 rank = m+1 로 취급(패널티 크게).
    보통 m을 5k~20k 정도로 잡으면 꽤 안정적.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    N = X.shape[0]
    k = int(k)
    m = int(max(m, k))

    if N <= 3 * k + 1:
        # continuity 공식에서 분모(2N-3k-1)가 0/음수 되는 케이스 회피
        # 작은 N에서는 의미가 약하니 NaN 반환
        return np.nan

    nnx = NearestNeighbors(n_neighbors=min(N, k + 1), metric=metric_x).fit(X)
    nny = NearestNeighbors(n_neighbors=min(N, m + 1), metric=metric_y).fit(Y)

    knn_x = nnx.kneighbors(return_distance=False)[:, 1:]  # (N,k)
    mnn_y = nny.kneighbors(return_distance=False)[:, 1:]  # (N,m)

    # mNN 내에서의 rank lookup: dict(이웃->순위)
    # rank는 1..m
    rank_maps = []
    for i in range(N):
        rm = {int(j): r for r, j in enumerate(mnn_y[i], start=1)}
        rank_maps.append(rm)

    # U_k(i): 원공간 kNN 중, 임베딩 kNN(여기서는 mNN 상위 k)에서 빠진 점들
    sum_term = 0.0
    for i in range(N):
        orig = knn_x[i]  # size k
        emb_topk = set(mnn_y[i][:k].tolist())
        rm = rank_maps[i]

        for j in orig:
            j = int(j)
            if j in emb_topk:
                continue
            r_ij = rm.get(j, m + 1)  # mNN 밖이면 m+1
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
    """
    (128d feature, 2d projection) 쌍에 대해
    - Spearman(거리 순위 보존, 샘플링)
    - Trustworthiness (sklearn)
    - Continuity(근사)
    - kNN Jaccard overlap
    을 한 번에 계산해서 dict로 반환.

    주의:
    - UMAP은 local-preserving이 목적이라 Spearman이 낮아도 Trustworthiness가 높으면 정상일 수 있음.
    - continuity_approx는 m(임베딩에서 고려하는 이웃 수)에 따라 값이 달라질 수 있음.
    """
    X = np.asarray(X_128d)
    Z = np.asarray(Z_2d)
    assert X.ndim == 2 and Z.ndim == 2
    assert X.shape[0] == Z.shape[0]

    out = {}

    # 1) Global-ish: Spearman (sampled pairs)
    out["spearman_dist_sampled"] = spearman_pairwise_distance_sample(
        X, Z, n_pairs=spearman_pairs, seed=spearman_seed
    )

    # 2) Local: Trustworthiness / Continuity / Jaccard for multiple k
    out["per_k"] = {}
    for k in ks:
        k = int(k)
        tw = sk_trustworthiness(X, Z, n_neighbors=k, metric=metric_x)
        cj = knn_jaccard_overlap(X, Z, k=k, metric_x=metric_x, metric_y=metric_z)
        co = continuity_approx(
            X, Z, k=k, m=continuity_m, metric_x=metric_x, metric_y=metric_z
        )
        out["per_k"][k] = {
            "trustworthiness": float(tw),
            "continuity_approx": float(co),
            "knn_jaccard": float(cj),
        }

    return out


def log_for_distance_metric(target_date, tag_name):

    data_dir = f"out_test_features/{tag_name}/{target_date}"
    
    # load Z_gt
    output_npy = os.path.join(data_dir, "final/z_ecmwf_manifold.npy")
    Z_gt = np.load(output_npy)
    
    # load Z_ai_list
    data_dir_ai = f"out_test_features/{tag_name}/{target_date}/proj"
    files = glob_npy(data_dir_ai, "pairs*.npy")
    cFile = len(files)

    Z_ai_list = []
    for a in range(cFile):
        output_npy = os.path.join(data_dir, f"final/ai_{a+1:02d}_manifold.npy")
        Z_ai = np.load(output_npy)
        Z_ai_list.append(Z_ai)

    lengths_list = [12, 24, 36, 48]
    rmse_best, dtw_best, rmse_list, dtw_list = calc_dtw_and_rmse_by_length(Z_gt, Z_ai_list, lengths_list)
    #dtw_list = np.array(dtw_list)
    #rmse_list = np.array(rmse_list)

    result_log = build_result_log(target_date, Z_gt[0], lengths_list, dtw_list, rmse_list, dtw_best, rmse_best)
    append_jsonl("D:/ext_data/logs/eval_log_detail.jsonl", result_log)
    print(result_log)

####################################
## DTW 계산
#####################################

def evaluate_statistics_prefeatures(target_date):
    # check distance,  9개 파일 한꺼번에 평가    
    data_dir = f"out_test_features/{target_date}/proj"
    files = glob_npy(data_dir, "pairs*.npy")
    # 1) 기본: L2 거리 -> cosine변경,
    res = rank_paired_files_and_daily_winner(files, metric="cosine", normalize="l2", topk_per_day=3)

    print("== Best overall (mean) ==")
    print(res["summary"]["best_overall_by_mean"])
    print("\n== Ranking by mean distance ==")
    for r in res["file_ranking_by_mean"]:
        print(r)
    print("\n== Ranking by daily wins ==")
    for r in res["file_ranking_by_wins"]:
        print(r)
    print(f"\n== Day record all ==")    
    for k in range(48):
        #print(f"\n== Day {k+1} record ==")    
        print(res["per_day"][k])


def rmse_2d_over_time(ecmwf_pts, aimd_pts):
    diff = ecmwf_pts - aimd_pts
    mse = np.mean(np.sum(diff**2, axis=1))
    return float(np.sqrt(mse))


def rank_paired_files_and_daily_winner(
    file_list,
    metric="l2",            # "l2" | "cosine"
    normalize=None,         # None | "l2" | "zscore"
    name0="ecmwf",
    name1="aimd",
    topk_per_day=3,
    return_all=False
):
    """
    file_list: list of npy paths, each (T,2,D).
    metric: distance function (smaller = better).
    normalize:
      - None: raw
      - "l2": L2-normalize vectors first
      - "zscore": per-dim zscore using both domains (within each file)
    """

    file_list = list(file_list)
    if not file_list:
        raise ValueError("file_list empty")

    # Load all -> dist matrix (F,T)
    dists = []
    names = []
    for p in file_list:
        arr = np.load(p)  # (T,2,D)
        if arr.ndim != 3 or arr.shape[1] != 2:
            raise ValueError(f"{p}: expected (T,2,D), got {arr.shape}")

        x0 = arr[:, 0, :].astype(np.float64, copy=False)
        x1 = arr[:, 1, :].astype(np.float64, copy=False)

        if normalize is None:
            a0, a1 = x0, x1
        elif normalize == "l2":
            a0, a1 = _l2_normalize(x0), _l2_normalize(x1)
        elif normalize == "zscore":
            a0, a1 = _zscore_pair(x0, x1)
        else:
            raise ValueError("normalize must be None, 'l2', or 'zscore'")

        if metric == "l2":
            dist = _l2(a0, a1)                 # (T,)
        elif metric == "cosine":
            dist = _cosine_distance(a0, a1)    # (T,)
        else:
            raise ValueError("metric must be 'l2' or 'cosine'")

        dists.append(dist.astype(np.float32))
        names.append(os.path.basename(p))

    dists = np.stack(dists, axis=0)  # (F,T)
    names = np.array(names)
    F, T = dists.shape

    # ----------------------------
    # (1) File-level ranking
    # ----------------------------
    mean_dist = dists.mean(axis=1)      # (F,)
    median_dist = np.median(dists, axis=1)
    p90_dist = np.percentile(dists, 90, axis=1)

    order = np.argsort(mean_dist)       # smaller better
    file_ranking = []
    for r, i in enumerate(order, start=1):
        file_ranking.append({
            "rank": r,
            "file": str(names[i]),
            "mean": float(mean_dist[i]),
            "median": float(median_dist[i]),
            "p90": float(p90_dist[i]),
        })

    # ----------------------------
    # (2) Day-level winners
    # ----------------------------
    winner_idx = np.argmin(dists, axis=0)           # (T,)
    winner_file = names[winner_idx]                 # (T,)
    winner_dist = dists[winner_idx, np.arange(T)]   # (T,)

    # top-k per day
    k = int(max(1, min(topk_per_day, F)))
    topk_idx = np.argsort(dists, axis=0)[:k, :]     # (k,T)
    topk_files = names[topk_idx]                    # (k,T)
    topk_dists = np.take_along_axis(dists, topk_idx, axis=0)  # (k,T)

    # win counts (how many days each file is best)
    win_counts = np.zeros(F, dtype=np.int32)
    for i in winner_idx:
        win_counts[i] += 1
    win_rate = win_counts / float(T)

    win_order = np.argsort(-win_counts)  # more wins better
    win_table = []
    for r, i in enumerate(win_order, start=1):
        win_table.append({
            "rank": r,
            "file": str(names[i]),
            "wins": int(win_counts[i]),
            "win_rate": float(win_rate[i]),
            "mean": float(mean_dist[i]),
        })

    # ----------------------------
    # (3) Summary
    # ----------------------------
    summary = {
        "num_files": int(F),
        "T": int(T),
        "metric": metric,
        "normalize": normalize,
        "best_overall_by_mean": file_ranking[0],
        "best_overall_by_median": {
            "file": str(names[np.argmin(median_dist)]),
            "median": float(np.min(median_dist)),
        },
        "best_overall_by_p90": {
            "file": str(names[np.argmin(p90_dist)]),
            "p90": float(np.min(p90_dist)),
        },
        "mean_dist_overall": float(dists.mean()),
    }

    # Build per-day records (readable)
    per_day = []
    for t in range(T):
        rec = {
            "day": int(t),
            "winner_file": str(winner_file[t]),
            "winner_dist": float(winner_dist[t]),
            "topk_files": [str(x) for x in topk_files[:, t]],
            "topk_dists": [float(x) for x in topk_dists[:, t]],
        }
        per_day.append(rec)

    out = {
        "summary": summary,
        "file_ranking_by_mean": file_ranking,   # 전체 평균거리 기준
        "file_ranking_by_wins": win_table,      # 날짜별 1등 횟수 기준
        "per_day": per_day,                     # 날짜별 winner/top-k
    }

    if return_all:
        out["arrays"] = {
            "names": names,
            "dists": dists,
            "mean_dist": mean_dist,
            "median_dist": median_dist,
            "p90_dist": p90_dist,
            "winner_idx": winner_idx,
            "topk_idx": topk_idx
        }

    return out

def check_good_performance_model(maps, target_date, model_list):
    status_str_list = []
    bAllGood = False
    good_models = []
    for k in model_list:
        # 여기에서 score: good/bad 를 결정할 수 있음
        status = query_condition_status_by_date(
            maps=maps,
            target_date=target_date,
            model_idx=k,
            thr=0.7
        )

        good_by_metric = defaultdict(list)
        for (metric, L), info in status.items():
            if info.get("good", False):
                good_by_metric[metric].append(L)

        parts = []
        for metric, lengths in good_by_metric.items():
            s = ", ".join(map(str, sorted(lengths)))
            parts.append(f"{metric}: {s}")

        result_str = " | ".join(parts)
        status_str_list.append(result_str)

        required_L = {12, 24, 36, 48}
        for metric, lengths in good_by_metric.items():
            if set(lengths) == required_L:
                good_models.append(k)
                bAllGood = True

    return status_str_list, bAllGood, good_models

def query_condition_status_by_date(
    maps: dict,
    target_date: str,
    model_idx: int,
    length_list=(12, 24, 36, 48),
    metrics=("dtw", "rmse"),
    thr: float = 0.7,
    date_key: str = "date",
):
    """
    maps[(model_idx, metric, L)] must contain:
      - "alpha": (M,)
      - "date":  (M,)  # target_date 문자열들

    return:
      result[(metric, L)] = {
        "alpha": float,
        "good": bool,
        "idx": int,   # record index within that cell
      }
    """
    out = {}
    for m in metrics:
        for L in length_list:
            cell = maps.get((model_idx, m, int(L)), None)
            if cell is None:
                out[(m, int(L))] = {"alpha": None, "good": None, "idx": None}
                continue

            if ("alpha" not in cell) or (date_key not in cell):
                raise KeyError(f"maps cell {(model_idx, m, int(L))} lacks 'alpha' or '{date_key}'")

            dates = cell[date_key]
            alpha = cell["alpha"]
            score = cell["score"]

            # 날짜 매칭
            # dates는 object array일 가능성이 있으니 str로 안전하게 비교
            mask = (dates.astype(str) == str(target_date))
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                out[(m, int(L))] = {"alpha": None, "good": None, "idx": None}
                continue
            if idxs.size > 1:
                # 같은 날짜가 중복으로 들어갔으면 첫 번째만 사용(원하면 평균 등으로 바꿀 수 있음)
                i = int(idxs[0])
            else:
                i = int(idxs[0])

            a = float(alpha[i])
            b = float(score[i])
            out[(m, int(L))] = {
                "alpha": a,
                #"good": bool(a >= thr),
                "good": bool(b <= 0.5),
                "idx": i,
            }
    return out


def _score_to_alpha(score: float, scores_all: np.ndarray, a_min=0.05, a_max=0.95, eps=1e-12) -> float:
    """낮은 score가 더 좋다고 가정. 조건별(한 날짜의 9개 모델) 상대강도로 alpha 결정."""
    smin = float(np.min(scores_all))
    smax = float(np.max(scores_all))
    if abs(smax - smin) < eps:
        return float(0.5 * (a_min + a_max))  # 모두 같으면 중간
    t = (score - smin) / (smax - smin)      # 0(best) ~ 1(worst)
    return float(a_min + (1.0 - t) * (a_max - a_min))


def _alpha_from_global_minmax(
    s: float,
    smin: float,
    smax: float,
    a_min: float = 0.05,
    a_max: float = 0.95,
    eps: float = 1e-12
) -> float:
    """
    낮을수록 좋은 score(=distance)라고 가정.
    smin(best) -> alpha=a_max, smax(worst) -> alpha=a_min
    """
    if not np.isfinite(s):
        return float(a_min)
    if not np.isfinite(smin) or not np.isfinite(smax) or abs(smax - smin) < eps:
        return float(0.5 * (a_min + a_max))
    t = (s - smin) / (smax - smin)  # 0(best) ~ 1(worst)
    a = a_min + (1.0 - t) * (a_max - a_min)
    return float(np.clip(a, a_min, a_max))





def _pairwise_cost_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: (T1, D), B: (T2, D)
    return: (T1, T2) where C[i,j] = ||A[i]-B[j]||_2
    """
    # broadcasting: (T1,1,D) - (1,T2,D) -> (T1,T2,D)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def dtw_distance(
    A: np.ndarray,
    B: np.ndarray,
    window: int | None = None,
    normalize: bool = True,
) -> float:
    """
    DTW distance between two trajectories in R^D.

    Args:
        A: (T1, D) numpy array
        B: (T2, D) numpy array
        window: Sakoe-Chiba band radius. If None, full DTW.
                Typical: window = int(0.1*max(T1,T2)) 정도부터 시작.
        normalize: if True, return (path-average) cost = total_cost / path_length
                   if False, return total_cost

    Returns:
        DTW distance (float)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A, B must be 2D arrays shaped (T, D).")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dim mismatch: A has D={A.shape[1]}, B has D={B.shape[1]}.")

    T1, D = A.shape
    T2, _ = B.shape

    C = _pairwise_cost_matrix(A, B)

    if window is None:
        window = max(T1, T2)  # effectively no constraint
    else:
        window = int(window)
        if window < 0:
            raise ValueError("window must be >= 0")

    # DP arrays
    INF = 1e18
    dp = np.full((T1 + 1, T2 + 1), INF, dtype=np.float64)
    steps = np.zeros((T1 + 1, T2 + 1), dtype=np.int32)  # path length to reach (i,j)
    dp[0, 0] = 0.0
    steps[0, 0] = 0

    for i in range(1, T1 + 1):
        j_start = max(1, i - window)
        j_end = min(T2, i + window)
        for j in range(j_start, j_end + 1):
            cost = C[i - 1, j - 1]

            # three predecessors: (i-1,j), (i,j-1), (i-1,j-1)
            prev_costs = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            k = int(np.argmin(prev_costs))
            if k == 0:
                dp[i, j] = prev_costs[0] + cost
                steps[i, j] = steps[i - 1, j] + 1
            elif k == 1:
                dp[i, j] = prev_costs[1] + cost
                steps[i, j] = steps[i, j - 1] + 1
            else:
                dp[i, j] = prev_costs[2] + cost
                steps[i, j] = steps[i - 1, j - 1] + 1

    total_cost = float(dp[T1, T2])
    path_len = int(steps[T1, T2])
    if path_len <= 0:
        # 이 케이스는 사실상 나오기 어렵지만 안전장치
        return total_cost

    return total_cost / path_len if normalize else total_cost



def compare_dtw_to_gt(
    gt_blue: np.ndarray,
    reds: list[np.ndarray] | np.ndarray,
    window: int | None = None,
    normalize: bool = True,
) -> dict:
    """
    gt_blue: (T,2) or (T,D)
    reds:
      - list of arrays [ (Ti,D), (Tj,D), ... ]  OR
      - stacked array (N,T,D)

    Returns:
      {
        "dtw": np.ndarray shape (N,),
        "best_idx": int,
        "best_dtw": float,
        "sorted_idx": np.ndarray (N,) ascending
      }
    """
    gt_blue = np.asarray(gt_blue, dtype=np.float64)

    if isinstance(reds, np.ndarray):
        if reds.ndim != 3:
            raise ValueError("If reds is ndarray, it must have shape (N,T,D).")
        red_list = [reds[i] for i in range(reds.shape[0])]
    else:
        red_list = [np.asarray(r, dtype=np.float64) for r in reds]

    scores = np.array(
        [dtw_distance(gt_blue, r, window=window, normalize=normalize) for r in red_list],
        dtype=np.float64
    )

    best_idx = int(np.argmin(scores))
    sorted_idx = np.argsort(scores)

    return {
        "dtw": scores,
        "best_idx": best_idx,
        "best_dtw": float(scores[best_idx]),
        "sorted_idx": sorted_idx+1,
    }



def calc_dtw_and_rmse_by_length(Z_gt, Z_ai_list, lengths_list):
    
    #out_rmse = {}
    #out_dtw = {}
    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    rmse_best = []
    dtw_best = []
    rmse_scores_list = []
    dtw_scores_list = []

    for L in lengths_list:
        ############
        # calc rmse
        ############
        rmse_list = []
        #distance_list = []
        for Z_ai in Z_ai_list:
            rmse_one = rmse_2d_over_time(Z_gt[1:L+1], Z_ai[1:L+1])
            #print(f"compare to ai_{a+1} vs. ecmwf => rmse => {np.round(rmse_one, 4)}")
            rmse_list.append(rmse_one)

        #     diff = Z_ai - Z_gt                      # (N, 2)
        #     dist = np.linalg.norm(diff, axis=1)     # (N,)  각 점의 거리
        #     distance_list.append(np.round(dist,3))

        # distance_arr = np.array(distance_list)
        # np.set_printoptions(suppress=True, precision=3)
        # print(distance_arr.T)
        
        # idx_min = np.argmin(distance_arr.T, axis=1)        

        # best_model_per_t = model_names[idx_min]
        # for m in range(len(best_model_per_t)):
        #     print(f"Date {m+1}: {best_model_per_t[m]}")
        # models, counts = np.unique(best_model_per_t, return_counts=True)
        # for m, c in zip(models, counts):
        #     print(f"{m}: {c}")
        # D = distance_arr.T      

        #print(f"==========length {L}: Models' RMSE Performance ==========")
        #print(np.round(rmse_list, 4))
        idx_rmse_min = int(np.argmin(rmse_list))
        sorted_rmse_idx = np.argsort(rmse_list)
        #print(f"best RMSE model: {model_names[idx_rmse_min]}")
        #print(f"Ranked RMSE: {sorted_rmse_idx+1}")

        ##########
        # calc dtw
        ##########
        out_dtw = compare_dtw_to_gt(
            gt_blue=Z_gt[1:L+1],
            reds=[z[1:L+1] for z in Z_ai_list],
            window=int(0.1 * len(Z_gt)),   # 없애려면 None
            normalize=True                   # True 추천 (길이 영향 줄임)
        )
        #print(f"==========length {L}: Models' DTW Performance ==========")
        #print("DTW scores:", out_dtw["dtw"])        
        best_idx = int(out_dtw["best_idx"])
        #print(f"best DTW model: {model_names[best_idx]}")
        #print("Ranked DTW:", out_dtw["sorted_idx"])

        
        # logging.info(f"==== Data Length {L} RMSE and DTW Performance ====")
        # logging.info(f"RMSE scores: {np.round(rmse_list, 4).tolist()}")
        # logging.info(f"best RMSE model: {model_names[idx_rmse_min]}")
        # logging.info(f"Ranked RMSE: {sorted_rmse_idx + 1}")        
        # logging.info(f"DTW scores: {np.round(out_dtw['dtw'], 4)}")
        # logging.info(f"best DTW model: {model_names[out_dtw['best_idx']]}")
        # logging.info(f"Ranked DTW: {out_dtw['sorted_idx']}")

        rmse_best.append(idx_rmse_min+1)
        dtw_best.append(best_idx+1)

        dtw_list = out_dtw["dtw"]
        dtw_list = np.array(dtw_list)
        
        rmse_scores_list.append(rmse_list)
        dtw_scores_list.append(dtw_list)

    print(f"RMSE: {rmse_best}")
    print(f"DTW: {dtw_best}")
    return rmse_best, dtw_best, rmse_scores_list, dtw_scores_list


## 계절별 응집성 평가를 위한 코드 ##

def standardize_2d(coords: np.ndarray) -> np.ndarray:
    """
    coords: (N, 2)
    각 축(x, y)을 독립적으로 z-score 표준화
    """
    coords = np.asarray(coords, dtype=float)
    mean = coords.mean(axis=0, keepdims=True)
    std = coords.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (coords - mean) / std


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    coords: (N, 2)
    return: (N, N) Euclidean distance matrix
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist


def compute_separability(coords: np.ndarray, labels: np.ndarray, normalize: bool = True):
    """
    coords : (N, 2) array
    labels : (N,) array, class label (총 4개라고 가정할 필요는 없고 일반적으로 동작)
    normalize : True이면 각 space별 x,y 축 z-score 표준화 후 계산

    return:
        {
            'within_mean': float,
            'between_mean': float,
            'separation_ratio': float,
            'within_by_class': dict,
            'between_by_classpair': dict,
            'n_points': int,
            'classes': list
        }
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if labels.ndim != 1 or len(labels) != len(coords):
        raise ValueError("labels must be of shape (N,) and match coords length")

    if normalize:
        coords_use = standardize_2d(coords)
    else:
        coords_use = coords.copy()

    dist = pairwise_distances(coords_use)
    classes = np.unique(labels)

    # 같은 class 내부 평균거리
    within_values = []
    within_by_class = {}

    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            within_by_class[c] = np.nan
            continue

        sub = dist[np.ix_(idx, idx)]
        tri = sub[np.triu_indices(len(idx), k=1)]  # 중복/대각 제외
        mean_val = float(np.mean(tri))
        within_by_class[c] = mean_val
        within_values.extend(tri.tolist())

    within_mean = float(np.mean(within_values)) if len(within_values) > 0 else np.nan

    # 다른 class 간 평균거리
    between_values = []
    between_by_classpair = {}

    for c1, c2 in combinations(classes, 2):
        idx1 = np.where(labels == c1)[0]
        idx2 = np.where(labels == c2)[0]

        sub = dist[np.ix_(idx1, idx2)]
        mean_val = float(np.mean(sub))
        between_by_classpair[(c1, c2)] = mean_val
        between_values.extend(sub.ravel().tolist())

    between_mean = float(np.mean(between_values)) if len(between_values) > 0 else np.nan

    if np.isnan(within_mean) or within_mean == 0:
        separation_ratio = np.nan
    else:
        separation_ratio = between_mean / within_mean

    return {
        "within_mean": within_mean,
        "between_mean": between_mean,
        "separation_ratio": separation_ratio,
        "within_by_class": within_by_class,
        "between_by_classpair": between_by_classpair,
        "n_points": len(coords),
        "classes": classes.tolist(),
    }


def compare_multiple_spaces(space_dict: dict, labels: np.ndarray, normalize: bool = True):
    """
    space_dict 예:
    {
        "space1": coords1,   # (N, 2)
        "space2": coords2,
        "space3": coords3
    }

    labels: 모든 space에 공통인 (N,) label
    """
    results = {}
    for name, coords in space_dict.items():
        results[name] = compute_separability(coords, labels, normalize=normalize)
    return results


def print_results(results: dict):
    """
    results: compare_multiple_spaces(...)의 결과
    """
    print("=" * 70)
    print(f"{'Space':<15} {'Within':>12} {'Between':>12} {'Ratio(B/W)':>15}")
    print("=" * 70)

    for name, res in results.items():
        print(
            f"{name:<15} "
            f"{res['within_mean']:>12.6f} "
            f"{res['between_mean']:>12.6f} "
            f"{res['separation_ratio']:>15.6f}"
        )

    print("=" * 70)

    # 응집성 기준: within_mean이 가장 작은 space
    best_compact = min(
        results.items(),
        key=lambda x: x[1]["within_mean"]
    )[0]

    # 분리비 기준: ratio가 가장 큰 space
    best_ratio = max(
        results.items(),
        key=lambda x: x[1]["separation_ratio"]
    )[0]

    print(f"가장 응집성이 좋은 space (within 최소): {best_compact}")
    print(f"가장 분리도가 좋은 space (between/within 최대): {best_ratio}")


##### temporal continuity를 정량화해서 평가하기 위한 코드 #####

def normalize_coords(coords, method="zscore"):
    """
    coords: (T, 2)
    method:
        - "zscore": 각 축별 평균 0, 표준편차 1
        - "minmax": 각 축별 [0, 1] 정규화
        - "none": 정규화 없음
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (T, 2)")

    if method == "zscore":
        mean = coords.mean(axis=0, keepdims=True)
        std = coords.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (coords - mean) / std

    elif method == "minmax":
        cmin = coords.min(axis=0, keepdims=True)
        cmax = coords.max(axis=0, keepdims=True)
        scale = cmax - cmin
        scale[scale == 0] = 1.0
        return (coords - cmin) / scale

    elif method == "none":
        return coords.copy()

    else:
        raise ValueError("method must be one of: 'zscore', 'minmax', 'none'")


def compute_trail_path_lengths(coords, trail_length):
    """
    coords: (T, 2), 시간 순서대로 정렬된 2D 좌표
    trail_length: trail에 포함되는 point 개수 N
                  예) N=5이면 5개 점을 잇는 4개 segment 길이의 합

    return:
        path_lengths: (T - trail_length + 1,) 각 sliding trail의 총 path length
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (T, 2)")
    if trail_length < 2:
        raise ValueError("trail_length must be >= 2")
    if len(coords) < trail_length:
        raise ValueError("trail_length must be <= number of points")

    # 인접 시점 간 segment length
    diffs = coords[1:] - coords[:-1]                      # (T-1, 2)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))      # (T-1,)

    # sliding sum으로 각 trail의 총 path length 계산
    # trail_length=N 이면 segment는 N-1개
    k = trail_length - 1
    csum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    path_lengths = csum[k:] - csum[:-k]

    return path_lengths


def analyze_temporal_continuity(
    space_dict,
    trail_length,
    normalize="zscore",
    bins=50,
    density=True
):
    """
    space_dict 예:
    {
        "space1": coords1,   # (T, 2)
        "space2": coords2,
        "space3": coords3
    }

    trail_length: trail에 포함되는 point 개수 N
    normalize: "zscore", "minmax", "none"
    bins: 히스토그램 bin 개수 또는 bin array
    density: True면 확률밀도, False면 count

    return:
        results = {
            "normalized_coords": {...},
            "path_lengths": {...},
            "histograms": {
                "bins": ...,
                "space1": ...,
                ...
            },
            "summary": {
                "space1": {
                    "mean": ...,
                    "median": ...,
                    "std": ...,
                    "min": ...,
                    "max": ...
                },
                ...
            }
        }
    """
    normalized_coords = {}
    path_lengths_dict = {}
    summary = {}

    # 1) 각 space 정규화 + trail path length 계산
    for name, coords in space_dict.items():
        coords_norm = normalize_coords(coords, method=normalize)
        normalized_coords[name] = coords_norm

        lengths = compute_trail_path_lengths(coords_norm, trail_length=trail_length)
        path_lengths_dict[name] = lengths

        summary[name] = {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
        }

    # 2) 공통 bin 설정
    all_lengths = np.concatenate(list(path_lengths_dict.values()))
    hist_values, bin_edges = np.histogram(all_lengths, bins=bins, density=density)

    histograms = {"bins": bin_edges}
    for name, lengths in path_lengths_dict.items():
        hist, _ = np.histogram(lengths, bins=bin_edges, density=density)
        histograms[name] = hist

    results = {
        "normalized_coords": normalized_coords,
        "path_lengths": path_lengths_dict,
        "histograms": histograms,
        "summary": summary,
    }

    return results


def print_temporal_continuity_summary(results, sort_by="mean"):
    """
    results: analyze_temporal_continuity(...)의 출력
    sort_by:
        - "mean"
        - "median"
    """
    summary = results["summary"]

    items = list(summary.items())
    items.sort(key=lambda x: x[1][sort_by])

    print("=" * 78)
    print(f"{'Space':<12} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("=" * 78)
    for name, s in items:
        print(
            f"{name:<12} "
            f"{s['mean']:>12.6f} "
            f"{s['median']:>12.6f} "
            f"{s['std']:>12.6f} "
            f"{s['min']:>12.6f} "
            f"{s['max']:>12.6f}"
        )
    print("=" * 78)
    print(f"가장 시간적 연속성이 좋은 space ({sort_by} 기준 최소): {items[0][0]}")