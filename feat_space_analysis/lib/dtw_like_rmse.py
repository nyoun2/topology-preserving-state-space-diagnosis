import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

################################
### 코어
#################################

def _pairwise_dist_matrix(X, Y, metric="euclidean"):
    """
    X: (N,D), Y:(M,D)
    return Dmat: (N,M)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if metric == "euclidean":
        # (N,1,D) - (1,M,D) -> (N,M,D)
        diff = X[:, None, :] - Y[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    elif metric == "sqeuclidean":
        diff = X[:, None, :] - Y[None, :, :]
        return np.sum(diff * diff, axis=-1)
    else:
        raise ValueError("metric must be 'euclidean' or 'sqeuclidean'")
    

def dtw_with_path(X, Y, metric="euclidean", window=None, return_mats=True):
    """
    Classic DTW (3-move) with optional Sakoe-Chiba window constraint.

    window:
      - None: unconstrained
      - int w: allow |i - j| <= w

    Returns:
      result dict:
        - dtw: final DTW cost
        - dtw_norm: dtw / path_length
        - path: list of (i,j) from (0,0) to (N-1,M-1)
        - path_len: len(path)
        - endpoint_reach:
            blue_end_step: first step index k where i==N-1
            red_end_step:  first step index k where j==M-1
            blue_end_frac: blue_end_step/(path_len-1)
            red_end_frac:  red_end_step/(path_len-1)
            end_hold_blue_steps: number of steps staying at i==N-1 (tail)
            end_hold_red_steps:  number of steps staying at j==M-1 (tail)
        - (optional) D: distance matrix
        - (optional) C: cumulative cost matrix
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    N, M = len(X), len(Y)
    if N == 0 or M == 0:
        raise ValueError("Empty sequence")

    D = _pairwise_dist_matrix(X, Y, metric=metric)

    # cumulative cost matrix with +inf padding for simpler transitions
    C = np.full((N, M), np.inf, dtype=np.float64)
    P = np.full((N, M, 2), -1, dtype=np.int32)  # backpointer: prev (pi,pj)

    # window constraint helper
    def _in_window(i, j):
        if window is None:
            return True
        return abs(i - j) <= int(window)

    # init
    if not _in_window(0, 0):
        raise ValueError("window too small: (0,0) not allowed")
    C[0, 0] = D[0, 0]
    P[0, 0] = (-1, -1)

    # fill DP
    for i in range(N):
        j_start = 0
        j_end = M - 1
        if window is not None:
            w = int(window)
            j_start = max(0, i - w)
            j_end = min(M - 1, i + w)

        for j in range(j_start, j_end + 1):
            if i == 0 and j == 0:
                continue

            # candidates: up (i-1,j), left (i,j-1), diag (i-1,j-1)
            best_cost = np.inf
            best_prev = (-1, -1)

            if i - 1 >= 0 and _in_window(i - 1, j):
                c = C[i - 1, j]
                if c < best_cost:
                    best_cost = c
                    best_prev = (i - 1, j)

            if j - 1 >= 0 and _in_window(i, j - 1):
                c = C[i, j - 1]
                if c < best_cost:
                    best_cost = c
                    best_prev = (i, j - 1)

            if i - 1 >= 0 and j - 1 >= 0 and _in_window(i - 1, j - 1):
                c = C[i - 1, j - 1]
                if c < best_cost:
                    best_cost = c
                    best_prev = (i - 1, j - 1)

            if np.isfinite(best_cost):
                C[i, j] = D[i, j] + best_cost
                P[i, j] = best_prev

    dtw = C[N - 1, M - 1]
    if not np.isfinite(dtw):
        raise ValueError("No valid DTW path found (window too tight?)")

    # backtrack path
    path = []
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi, pj = P[i, j]
        if pi == -1 and pj == -1:
            break
        i, j = int(pi), int(pj)
    path.reverse()

    path_len = len(path)
    dtw_norm = dtw / max(1, path_len)

    # endpoint reach metrics
    # "얼마나 빨리 endpoint에 도달했는지" = path에서 i==N-1, j==M-1 처음 등장 step
    blue_end_step = None
    red_end_step = None
    for k, (ii, jj) in enumerate(path):
        if blue_end_step is None and ii == N - 1:
            blue_end_step = k
        if red_end_step is None and jj == M - 1:
            red_end_step = k
        if blue_end_step is not None and red_end_step is not None:
            break

    # tail에서 endpoint에 머문 길이(정체)도 같이 뽑아두면 좋음
    end_hold_blue_steps = sum(1 for (ii, _) in path if ii == N - 1)
    end_hold_red_steps = sum(1 for (_, jj) in path if jj == M - 1)

    denom = max(1, path_len - 1)
    endpoint_reach = dict(
        blue_end_step=int(blue_end_step),
        red_end_step=int(red_end_step),
        blue_end_frac=float(blue_end_step / denom),
        red_end_frac=float(red_end_step / denom),
        end_hold_blue_steps=int(end_hold_blue_steps),
        end_hold_red_steps=int(end_hold_red_steps),
    )

    result = dict(
        dtw=float(dtw),
        dtw_norm=float(dtw_norm),
        path=path,
        path_len=int(path_len),
        endpoint_reach=endpoint_reach,
    )
    if return_mats:
        result["D"] = D
        result["C"] = C
    return result



################################
### 디버그/시각화
#################################

def print_dtw_diagnostics(result, name_blue="blue", name_red="red"):
    er = result["endpoint_reach"]
    print("========== DTW diagnostics ==========")
    print(f"DTW:      {result['dtw']:.6f}")
    print(f"DTW_norm: {result['dtw_norm']:.6f}  (DTW / path_len)")
    print(f"path_len: {result['path_len']}")
    print("--- endpoint reach (earlier = faster) ---")
    print(f"{name_blue} end reached at step: {er['blue_end_step']}  (frac={er['blue_end_frac']:.3f})")
    print(f"{name_red}  end reached at step: {er['red_end_step']}  (frac={er['red_end_frac']:.3f})")
    print("--- endpoint hold count (bigger = more tail-sticking) ---")
    print(f"steps with i=={name_blue}_last: {er['end_hold_blue_steps']}")
    print(f"steps with j=={name_red}_last:  {er['end_hold_red_steps']}")


def plot_dtw_path_on_matrix(result, which="C", title=None):
    """
    which: "C" (cumulative cost) or "D" (distance)
    """
    mat = result[which]
    path = result["path"]

    if which=='C':
        plt.figure(10, figsize=(7, 6))
    else:
        plt.figure(11, figsize=(7, 6))
    plt.clf()
    plt.imshow(mat, aspect="auto", origin="lower")  # (i,j) -> y,x
    xs = [j for (i, j) in path]
    ys = [i for (i, j) in path]
    plt.plot(xs, ys, linewidth=1.5)  # default style
    plt.xlabel("j (aimd index)")
    plt.ylabel("i (ecmwf index)")
    if title is None:
        title = f"DTW path on {'Cumulative cost' if which=='C' else 'Distance'} matrix"
    plt.title(title)
    plt.colorbar(label=which)
    plt.tight_layout()
    plt.pause(0.1)
    #plt.show()


def plot_dtw_alignment(X, Y, result, dims=(0, 1), title=None, max_links=400):
    """
    2D 시퀀스면 (x,y)로 궤적을 그리고, DTW path에 따라 매칭 선을 일부 그려줌.
    dims: 사용할 차원 인덱스 (0,1)
    max_links: 매칭 선이 너무 많으면 보기 힘드니 다운샘플링
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    path = result["path"]

    d0, d1 = dims
    plt.figure(12, figsize=(7, 6))
    plt.clf()
    plt.plot(X[:, d0], X[:, d1], marker="o", markersize=2, linewidth=1)
    plt.plot(Y[:, d0], Y[:, d1], marker="o", markersize=2, linewidth=1)

    # draw some match links
    if len(path) > max_links:
        idx = np.linspace(0, len(path) - 1, max_links).astype(int)
        path_draw = [path[k] for k in idx]
    else:
        path_draw = path

    for (i, j) in path_draw:
        plt.plot([X[i, d0], Y[j, d0]], [X[i, d1], Y[j, d1]], linewidth=0.5, alpha=0.5)

    if title is None:
        title = "DTW alignment (trajectory + matching links)"
    plt.title(title)
    plt.xlabel(f"dim {d0}")
    plt.ylabel(f"dim {d1}")
    plt.tight_layout()
    plt.pause(0.1) 
    #plt.waitforbuttonpress()
    #plt.close()
    #plt.show()

################################
### rmse-shift 비교
#################################

def plot_rmse_comparison_by_models(
    target_date: str,
    tag_name: str,
    model_list,
    lenTime: int = 49,
    var_name=("t", "z", "u", "v"),
    model_names=None,
    figsize=(8, 8),
):
    """
    여러 모델의 raw npy를 읽어 ecmwf(GT) 대비 RMSE를 계산하고,
    (2,2) subplot 형태로 변수별 RMSE(lead time) 비교 그림을 생성한다.
    """

    raw_data_path = os.path.normpath(
        os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev", f"{target_date}")
    )

    V = len(var_name)

    # -----------------------------
    # Load GT once: (T,H,W,V)
    # -----------------------------
    gt_img = []
    for k in range(lenTime):
        one_file = os.path.join(raw_data_path, f"ecmwf/ecmwf_t{k+1:04d}.npy")
        gt_img.append(np.load(one_file))
    gt_img = np.asarray(gt_img)

    if gt_img.ndim != 4 or gt_img.shape[-1] != V:
        raise ValueError(f"GT shape expected (T,H,W,{V}) but got {gt_img.shape}")

    # -----------------------------
    # Compute RMSE for each model
    # -----------------------------
    rmse_by_model = {}

    for m in model_list:
        model_img = []
        for k in range(lenTime):
            one_file = os.path.join(
                raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy"
            )
            model_img.append(np.load(one_file))
        model_img = np.asarray(model_img)

        if model_img.shape != gt_img.shape:
            raise ValueError(f"Model ai_{m:02d} shape {model_img.shape} != GT {gt_img.shape}")

        # RMSE: mean over H,W
        rmse = np.sqrt(np.mean((model_img - gt_img) ** 2, axis=(1, 2)))  # (T,V)
        rmse_by_model[m] = rmse

    # -----------------------------
    # Plot (2,2)
    # -----------------------------
    x = np.arange(1, lenTime + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize, clear=True)
    axes = axes.ravel()

    for c in range(V):
        ax = axes[c]
        for m in model_list:
            label = f"ai_{m:02d}"
            if model_names is not None:
                # model_idx 1~9 기준이면 m-1, 0~8이면 m
                if 1 <= m <= len(model_names):
                    label = model_names[m - 1]
                elif 0 <= m < len(model_names):
                    label = model_names[m]

            ax.plot(x, rmse_by_model[m][:, c], label=label)

        ax.set_title(f"RMSE | {var_name[c]}")
        ax.set_xlabel("lead time (1..49)")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(f"RMSE comparison | {target_date}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.show()
    plt.pause(0.1)


def plot_rmse_comparison_by_models_v2(
    target_date: str,
    tag_name: str,
    model_list,
    lenTime: int = 49,
    var_name=("t", "z", "u", "v"),
    model_names=None,
    figsize=(9, 9),
    shift_radius: int = 2,   # +-2
):
    """
    - 기본 RMSE(동일 time index 정렬)
    - shift-min RMSE: gt를 t±shift_radius 범위로 이동시켜 최소 RMSE 선택
    를 함께 계산/플롯한다.
    """

    raw_data_path = os.path.normpath(
        os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev", f"{target_date}")
    )
    V = len(var_name)

    # -----------------------------
    # Load GT once: (T,H,W,V)
    # -----------------------------
    gt_img = []
    for k in range(lenTime):
        one_file = os.path.join(raw_data_path, f"ecmwf/ecmwf_t{k+1:04d}.npy")
        gt_img.append(np.load(one_file))
    gt_img = np.asarray(gt_img)

    if gt_img.ndim != 4 or gt_img.shape[-1] != V:
        raise ValueError(f"GT shape expected (T,H,W,{V}) but got {gt_img.shape}")

    # helper: label
    def _label_for_model(m):
        label = f"ai_{m:02d}"
        if model_names is not None:
            if 1 <= m <= len(model_names):
                label = model_names[m - 1]
            elif 0 <= m < len(model_names):
                label = model_names[m]
        return label

    # -----------------------------
    # Compute RMSE for each model
    # -----------------------------
    rmse_base_by_model  = {}  # (T,V)
    rmse_shift_by_model = {}  # (T,V)

    shifts = np.arange(-shift_radius, shift_radius + 1, dtype=int)

    for m in model_list:
        # load model: (T,H,W,V)
        model_img = []
        for k in range(lenTime):
            one_file = os.path.join(raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy")
            model_img.append(np.load(one_file))
        model_img = np.asarray(model_img)

        if model_img.shape != gt_img.shape:
            raise ValueError(f"Model ai_{m:02d} shape {model_img.shape} != GT {gt_img.shape}")

        # (1) base RMSE (T,V)
        rmse_base = np.sqrt(np.mean((model_img - gt_img) ** 2, axis=(1, 2)))

        # (2) shift-min RMSE (T,V)
        rmse_shift = np.empty((lenTime, V), dtype=np.float64)

        # 각 t에 대해: gt[t+s] 중 가능한 s만 검사, RMSE 최소 선택
        for t in range(lenTime):
            # 후보 RMSE들을 (num_valid_shifts, V)로 쌓기
            candidates = []
            for s in shifts:
                tt = t + s
                if 0 <= tt < lenTime:
                    # RMSE over H,W -> (V,)
                    r = np.sqrt(np.mean((model_img[t] - gt_img[tt]) ** 2, axis=(0, 1)))
                    candidates.append(r)
            candidates = np.stack(candidates, axis=0)  # (K,V)
            rmse_shift[t] = candidates.min(axis=0)

        rmse_base_by_model[m]  = rmse_base
        rmse_shift_by_model[m] = rmse_shift

    # -----------------------------
    # Plot (2,2) : base vs shift-min
    # -----------------------------
    x = np.arange(1, lenTime + 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize, clear=True)
    axes = axes.ravel()

    for c in range(V):
        ax = axes[c]
        for m in model_list:
            label = _label_for_model(m)

            # base: 실선
            ax.plot(x, rmse_base_by_model[m][:, c], label=f"{label} (base)")

            # shift-min: 점선
            ax.plot(
                x,
                rmse_shift_by_model[m][:, c],
                linestyle="--",
                linewidth=1,
                label=f"{label} (shift±{shift_radius})",
            )

        ax.set_title(f"RMSE | {var_name[c]}")
        ax.set_xlabel(f"lead time (1..{lenTime})")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"RMSE comparison | {target_date} | base vs shift-min")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # plt.show() 대신 interactive 용
    plt.pause(0.1)





def calc_validtime_rmse_comparison_by_models(
    target_dates,
    tag_name: str,
    model_list,
    lenTime: int = 13,
    var_name=("t", "z", "u", "v"),
    valid_offsets=None,
    selected_offsets=(-12, -9, -6, -3, -1),
    model_names=None,
    base_dir="D:/ext_data",
):
    """
    여러 valid time target_dates에 대해,
    -12 day부터 -1 day까지의 valid-time RMSE를 계산한다.

    추가로 selected_offsets에 해당하는 offset만 뽑아서
    lead-time RMSE 표와 유사한 형태의 valid-time RMSE 표를 생성한다.

    데이터 구조 가정:
    model_img[0]  : valid time의 GT 또는 기준장
    model_img[1]  : -12 day forecast
    model_img[2]  : -11 day forecast
    ...
    model_img[12] : -1 day forecast
    """

    if model_names is None:
        model_names = np.array([
            "fnet_ifs", "fnet_kim", "fnet_um",
            "grph_ifs", "grph_kim", "grph_um",
            "pang_ifs", "pang_kim", "pang_um"
        ])
    else:
        model_names = np.asarray(model_names)

    if valid_offsets is None:
        valid_offsets = list(range(-12, 0))  # -12, -11, ..., -1

    V = len(var_name)

    if len(valid_offsets) != lenTime - 1:
        raise ValueError(
            f"len(valid_offsets)={len(valid_offsets)} must be lenTime-1={lenTime-1}. "
            f"Current lenTime={lenTime}"
        )

    # 선택 offset이 valid_offsets 안에 있는지 확인
    for off in selected_offsets:
        if off not in valid_offsets:
            raise ValueError(
                f"selected offset {off} is not in valid_offsets={valid_offsets}"
            )

    # offset 값 -> index 매핑
    # 예: -12 -> 0, -11 -> 1, ..., -1 -> 11
    offset_to_index = {
        off: i for i, off in enumerate(valid_offsets)
    }

    # rmse_values[m] = list of arrays, each shape (12,V)
    rmse_values = {
        m: []
        for m in model_list
    }

    for target_date in target_dates:
        raw_data_path = os.path.normpath(
            os.path.join(base_dir, f"{tag_name}", "Test_4var1lev", f"{target_date}")
        )

        print(f"current_date = {target_date}")

        for m in model_list:
            model_img = []

            for k in range(lenTime):
                one_file = os.path.join(
                    raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy"
                )
                model_img.append(np.load(one_file))

            model_img = np.asarray(model_img)

            if model_img.ndim != 4 or model_img.shape[-1] != V:
                raise ValueError(
                    f"[{target_date}] ai_{m:02d} shape expected (T,H,W,{V}) "
                    f"but got {model_img.shape}"
                )

            if model_img.shape[0] != lenTime:
                raise ValueError(
                    f"[{target_date}] ai_{m:02d} expected T={lenTime}, "
                    f"but got T={model_img.shape[0]}"
                )

            # valid time 기준장
            gt_img = model_img[0]        # (H,W,V)

            # forecast part: -12 day ~ -1 day
            forecast_img = model_img[1:] # (12,H,W,V)

            # offset별 변수별 RMSE: (12,V)
            rmse_offset = np.sqrt(
                np.mean((forecast_img - gt_img[None, :, :, :]) ** 2, axis=(1, 2))
            )

            rmse_values[m].append(rmse_offset)

    # ============================================================
    # 결과 정리
    # ============================================================
    rmse_mean = {}
    rmse_by_offset_mean = {}

    rows_mean = []
    rows_offset = []
    rows_selected = []

    for m in model_list:
        if 1 <= m <= len(model_names):
            display_name = model_names[m - 1]
        else:
            display_name = f"ai_{m:02d}"

        arr = np.asarray(rmse_values[m])  # (N_dates, 12, V)

        if arr.ndim != 3 or arr.shape[2] != V:
            raise ValueError(
                f"Unexpected RMSE array shape for model {m}: {arr.shape}"
            )

        # --------------------------------------------------------
        # 1) 전체 offset 평균: (V,)
        # --------------------------------------------------------
        mean_all = np.mean(arr, axis=(0, 1))
        rmse_mean[m] = mean_all

        row_mean = {
            "model_id": m,
            "model_name": display_name,
            "n_dates": arr.shape[0],
            "n_offsets": arr.shape[1],
            "offset_range": f"{valid_offsets[0]}d to {valid_offsets[-1]}d",
        }

        for vi, vname in enumerate(var_name):
            row_mean[f"rmse_{vname}"] = mean_all[vi]

        rows_mean.append(row_mean)

        # --------------------------------------------------------
        # 2) 전체 offset별 평균: (12,V)
        # --------------------------------------------------------
        mean_by_offset = np.mean(arr, axis=0)
        rmse_by_offset_mean[m] = {}

        for oi, offset in enumerate(valid_offsets):
            rmse_by_offset_mean[m][offset] = mean_by_offset[oi]

            row_offset = {
                "model_id": m,
                "model_name": display_name,
                "valid_offset_day": offset,
                "n_dates": arr.shape[0],
            }

            for vi, vname in enumerate(var_name):
                row_offset[f"rmse_{vname}"] = mean_by_offset[oi, vi]

            rows_offset.append(row_offset)

        # --------------------------------------------------------
        # 3) 선택 offset만 별도 표로 정리
        # --------------------------------------------------------
        for offset in selected_offsets:
            oi = offset_to_index[offset]

            row_selected = {
                "model_id": m,
                "model_name": display_name,
                "valid_offset_day": offset,
                "n_dates": arr.shape[0],
            }

            for vi, vname in enumerate(var_name):
                row_selected[f"rmse_{vname}"] = mean_by_offset[oi, vi]

            rows_selected.append(row_selected)

    df_mean = pd.DataFrame(rows_mean)
    df_offset = pd.DataFrame(rows_offset)
    df_selected = pd.DataFrame(rows_selected)

    return rmse_mean, rmse_by_offset_mean, df_mean, df_offset, df_selected

def plot_rmse_comparison_by_models_for_valid_trajectory(
    target_date: str,
    tag_name: str,
    model_list,
    lenTime: int = 13,
    var_name=("t", "z", "u", "v"),
    model_names=None,
    figsize=(7, 7),
    test_root=None,
):
    
    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    #test_root = D:/ext_data/{tag_name}/Test_4var1lev
    if test_root is None:        
        raw_data_path = os.path.normpath(
            os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev", f"{target_date}")
        )
    else:
        raw_data_path = os.path.normpath(
            os.path.join(test_root, f"{target_date}")
        )
    
    V = len(var_name)
    T = lenTime

    rmse_by_model = {}

    for m in model_list:
        model_img = []
        for k in range(T):
            one_file = os.path.join(
                raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy"
            )
            model_img.append(np.load(one_file))
        model_img = np.asarray(model_img)

        gt_img = model_img[0]       # (H,W,V)
        rmse_same = np.sqrt(np.mean((model_img - gt_img) ** 2, axis=(1, 2)))  # (12,V)
        rmse_by_model[m] = rmse_same

    # -----------------------------
    # Plot RMSE (2x2): same (solid) vs aligned (dashed) + AUC annotation
    # -----------------------------
    x = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 2, figsize=figsize, clear=True)
    axes = axes.ravel()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(model_list)}

    for c in range(V):
        ax = axes[c]
        var = var_name[c]

        y_text = 0.95
        dy = 0.07

        for mi, m in enumerate(model_list):
            color = model_colors[m]

            label = f"ai_{m:02d}"
            label = f"{model_names[m-1]}"
            if model_names is not None:
                if 1 <= m <= len(model_names):
                    label = model_names[m - 1]
                elif 0 <= m < len(model_names):
                    label = model_names[m]

            ax.plot(
                x, rmse_by_model[m][:, c],
                linestyle="-", linewidth=1.5, color=color, alpha=0.9,
                label=f"{label}"
            )
            
        ax.set_title(f"RMSE | {var}")
        ax.set_xlabel("lead time (1..T)")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        ax.invert_xaxis()   # x축 반전

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    mgr = plt.get_current_fig_manager()
    try:
        mgr.window.move(50, 10)
        mgr.window.resize(700, 700)
    except Exception:
        try:
            mgr.window.wm_geometry("700x700+50+10")
        except Exception:
            pass

    plt.pause(0.1)

def calc_leadtime_rmse_comparison_by_models(
    target_dates,
    tag_name: str,
    model_list,
    lenTime: int = 49,
    var_name=("t", "z", "u", "v"),
    lead_points=(12, 24, 36, 48),
    model_names=None,
    base_dir="D:/ext_data"
):
    """
    여러 예측시점 target_dates에 대해,
    지정된 lead point에서 GT와 AI 모델의 변수별 RMSE를 계산하고
    전체 평균 RMSE를 모델별/변수별/lead별로 반환한다.

    Parameters
    ----------
    target_dates : list[str]
        예측 초기시각 목록. 예: ["2020010100", "2020010200", ...]
    tag_name : str
        데이터 태그명
    model_list : list[int]
        비교할 ai 모델 번호 리스트. 예: [1, 2, 3]
    lenTime : int
        한 예측시점에서 읽을 forecast step 수
    var_name : tuple[str]
        변수명. 기본값 ("t", "z", "u", "v")
    lead_points : tuple[int]
        RMSE를 계산할 lead point 번호. 1-based 기준.
        예: 12, 24, 36, 48
    model_names : list[str] or None
        모델 표시 이름
    base_dir : str
        기본 데이터 경로

    Returns
    -------
    rmse_mean : dict
        rmse_mean[model_id][lead_point] = np.array shape (V,)
    df : pandas.DataFrame
        모델별, lead별, 변수별 평균 RMSE를 정리한 표
    """

    if model_names is None:
        model_names = np.array([
            "fnet_ifs", "fnet_kim", "fnet_um",
            "grph_ifs", "grph_kim", "grph_um",
            "pang_ifs", "pang_kim", "pang_um"
        ])
    else:
        model_names = np.asarray(model_names)

    V = len(var_name)

    # lead point는 1-based로 입력받고, 실제 array index는 0-based
    #lead_indices = [p - 1 for p in lead_points]
    lead_indices = lead_points #[p - 1 for p in lead_points]

    if max(lead_indices) >= lenTime:
        raise ValueError(
            f"lead_points={lead_points} requires lenTime >= {max(lead_points)}, "
            f"but lenTime={lenTime}"
        )

    # rmse_values[m][lead_point] = list of (V,) arrays over target_dates
    rmse_values = {
        m: {p: [] for p in lead_points}
        for m in model_list
    }

    for target_date in target_dates:
        raw_data_path = os.path.normpath(
            os.path.join(base_dir, f"{tag_name}", "Test_4var1lev", f"{target_date}")
        )
        print(f"current_date: {target_date}")

        # -----------------------------
        # Load GT once: (T,H,W,V)
        # -----------------------------
        gt_img = []
        for k in range(lenTime):
            one_file = os.path.join(raw_data_path, f"ecmwf/ecmwf_t{k+1:04d}.npy")
            gt_img.append(np.load(one_file))
        gt_img = np.asarray(gt_img)

        if gt_img.ndim != 4 or gt_img.shape[-1] != V:
            raise ValueError(
                f"[{target_date}] GT shape expected (T,H,W,{V}) but got {gt_img.shape}"
            )

        for m in model_list:
            model_img = []
            for k in range(lenTime):
                one_file = os.path.join(
                    raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy"
                )
                model_img.append(np.load(one_file))
            model_img = np.asarray(model_img)

            if model_img.shape != gt_img.shape:
                raise ValueError(
                    f"[{target_date}] Model ai_{m:02d} shape {model_img.shape} "
                    f"!= GT {gt_img.shape}"
                )

            # 전체 lead time에 대한 변수별 RMSE: (T,V)
            rmse_same = np.sqrt(
                np.mean((model_img - gt_img) ** 2, axis=(1, 2))
            )

            # 지정한 lead point만 저장
            for p, idx in zip(lead_points, lead_indices):
                rmse_values[m][p].append(rmse_same[idx])  # shape: (V,)

    # -----------------------------
    # Mean over all target_dates
    # -----------------------------
    rmse_mean = {
        m: {}
        for m in model_list
    }

    rows = []

    for m in model_list:
        if 1 <= m <= len(model_names):
            display_name = model_names[m - 1]
        else:
            display_name = f"ai_{m:02d}"

        for p in lead_points:
            arr = np.asarray(rmse_values[m][p])  # (N_dates, V)

            if arr.ndim != 2 or arr.shape[1] != V:
                raise ValueError(
                    f"Unexpected RMSE array shape for model {m}, lead {p}: {arr.shape}"
                )

            mean_rmse = np.mean(arr, axis=0)  # (V,)
            rmse_mean[m][p] = mean_rmse

            row = {
                "model_id": m,
                "model_name": display_name,
                "lead_point": p,
                "n_dates": arr.shape[0],
            }

            for vi, vname in enumerate(var_name):
                row[f"rmse_{vname}"] = mean_rmse[vi]

            rows.append(row)

    df = pd.DataFrame(rows)

    return rmse_mean, df
    
def plot_rmse_comparison_by_models(
    target_date: str,
    tag_name: str,
    model_list,
    lenTime: int = 49,
    var_name=("t", "z", "u", "v"),
    model_names=None,
    figsize=(8, 8),
    test_root=None,
):
    
    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    #test_root = D:/ext_data/{tag_name}/Test_4var1lev
    if test_root is None:        
        raw_data_path = os.path.normpath(
            os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev", f"{target_date}")
        )
    else:
        raw_data_path = os.path.normpath(
            os.path.join(test_root, f"{target_date}")
        )
    
    V = len(var_name)
    T = lenTime

    # -----------------------------
    # Load GT once: (T,H,W,V)
    # -----------------------------
    gt_img = []
    for k in range(T):
        one_file = os.path.join(raw_data_path, f"ecmwf/ecmwf_t{k+1:04d}.npy")
        gt_img.append(np.load(one_file))
    gt_img = np.asarray(gt_img)

    if gt_img.ndim != 4 or gt_img.shape[-1] != V:
        raise ValueError(f"GT shape expected (T,H,W,{V}) but got {gt_img.shape}")
    
    rmse_by_model = {}
    
    for m in model_list:
        # Load model: (T,H,W,V)
        model_img = []
        for k in range(T):
            one_file = os.path.join(
                raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy"
            )
            model_img.append(np.load(one_file))
        model_img = np.asarray(model_img)

        if model_img.shape != gt_img.shape:
            raise ValueError(f"Model ai_{m:02d} shape {model_img.shape} != GT {gt_img.shape}")

        # baseline same-time RMSE: gt[i] vs ai[i]
        rmse_same = np.sqrt(np.mean((model_img - gt_img) ** 2, axis=(1, 2)))  # (T,V)
        rmse_by_model[m] = rmse_same

    # -----------------------------
    # Plot RMSE (2x2): same (solid) vs aligned (dashed) + AUC annotation
    # -----------------------------
    x = np.arange(1, T + 1)    
    fig, axes = plt.subplots(2, 2, num=3, figsize=figsize, clear=True)    
    axes = axes.ravel()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(model_list)}

    for c in range(V):
        ax = axes[c]
        var = var_name[c]

        y_text = 0.95
        dy = 0.07

        for mi, m in enumerate(model_list):
            color = model_colors[m]

            label = f"ai_{m:02d}"
            label = f"{model_names[m-1]}"
            if model_names is not None:
                if 1 <= m <= len(model_names):
                    label = model_names[m - 1]
                elif 0 <= m < len(model_names):
                    label = model_names[m]

            ax.plot(
                x, rmse_by_model[m][:, c],
                linestyle="-", linewidth=1.5, color=color, alpha=0.9,
                label=f"{label}"
            )
            
        ax.set_title(f"RMSE | {var}")
        ax.set_xlabel("lead time (1..T)")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    mgr = plt.get_current_fig_manager()
    try:
        mgr.window.move(50, 10)
        mgr.window.resize(800, 800)
    except Exception:
        try:
            mgr.window.wm_geometry("800x800+50+10")
        except Exception:
            pass

    plt.pause(0.1)

def plot_rmse_comparison_by_models_monotone_shift_multivar(
    target_date: str,
    tag_name: str,
    model_list,
    lenTime: int = 49,
    var_name=("t", "z", "u", "v"),
    model_names=None,
    figsize=(8, 8),
    # ---- alignment params ----
    forward_window: int = 3,          # j_cur .. j_cur+forward_window
    require_all_vars_improve: bool = True,
    fallback_to_same_time: bool = True,
    score_reduce: str = "rel_softmin",  # "rel_mean" | "rel_min" | "rel_softmin"
    rel_eps: float = 1e-6,
    rel_alpha: float = 5.0,           # softmin 강도 (클수록 worst-case에 가까움)
    show: bool = False,
):
    """
    - GT i vs AI j 매칭 단조 증가(모노톤)
    - 후보: ai[j_cur .. j_cur+forward_window]
    - (옵션) 4변수 모두 동일시점(i,i)보다 RMSE 감소하는 후보 우선
    - (옵션) 현재 범위 최적이 동일시점보다 나쁘면 동일시점 쪽으로 회귀
    - reduce score는 baseline 대비 상대개선률 기반(스케일 영향 제거)
    """

    raw_data_path = os.path.normpath(
        os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev", f"{target_date}")
    )
    V = len(var_name)
    T = lenTime

    # -----------------------------
    # Load GT once: (T,H,W,V)
    # -----------------------------
    gt_img = []
    for k in range(T):
        one_file = os.path.join(raw_data_path, f"ecmwf/ecmwf_t{k+1:04d}.npy")
        gt_img.append(np.load(one_file))
    gt_img = np.asarray(gt_img)

    if gt_img.ndim != 4 or gt_img.shape[-1] != V:
        raise ValueError(f"GT shape expected (T,H,W,{V}) but got {gt_img.shape}")

    # -----------------------------
    # Helper: relative improvement score (lower is better)
    # -----------------------------
    def reduce_score(vec: np.ndarray, base: np.ndarray) -> float:
        """
        vec, base: (V,)
        rel_improve = (base - vec) / (base + eps)  # +면 개선
        score는 '작을수록 좋게' 반환
        """
        base = np.asarray(base, dtype=float)
        vec = np.asarray(vec, dtype=float)

        rel_improve = (base - vec) / (base + rel_eps)  # (V,)

        if score_reduce == "rel_mean":
            # 평균 상대개선률을 최대화 => score는 -mean
            return -float(np.mean(rel_improve))

        if score_reduce == "rel_min":
            # 최악 변수의 상대개선률을 최대화 => score는 -min
            return -float(np.min(rel_improve))

        if score_reduce == "rel_softmin":
            # worst-case에 가까운 부드러운 집계
            # rel_improve가 클수록 좋음. 따라서 -rel_improve에 softmax를 걸어 "큰 패널티"를 강조
            # score 작을수록 좋게 유지
            # (주의) rel_improve가 매우 크면 exp 언더/오버는 alpha가 너무 크지 않으면 괜찮음
            x = -rel_improve  # 작을수록 좋음(=rel_improve 클수록 좋음)
            # log-sum-exp
            m = np.min(rel_alpha * x)
            return float((m + np.log(np.sum(np.exp(rel_alpha * x - m)))) / rel_alpha)

        raise ValueError("score_reduce must be one of: 'rel_mean', 'rel_min', 'rel_softmin'")

    # -----------------------------
    # For each model: load + compute baseline RMSE + alignment
    # -----------------------------
    rmse_by_model = {}
    aligned_rmse_by_model = {}
    match_idx_by_model = {}

    for m in model_list:
        # Load model: (T,H,W,V)
        model_img = []
        for k in range(T):
            one_file = os.path.join(
                raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy"
            )
            model_img.append(np.load(one_file))
        model_img = np.asarray(model_img)

        if model_img.shape != gt_img.shape:
            raise ValueError(f"Model ai_{m:02d} shape {model_img.shape} != GT {gt_img.shape}")

        # baseline same-time RMSE: gt[i] vs ai[i]
        rmse_same = np.sqrt(np.mean((model_img - gt_img) ** 2, axis=(1, 2)))  # (T,V)
        rmse_by_model[m] = rmse_same

        def pair_rmse(i: int, j: int) -> np.ndarray:
            diff = model_img[j] - gt_img[i]
            return np.sqrt(np.mean(diff * diff, axis=(0, 1)))  # (V,)

        # -----------------------------
        # Monotone alignment (greedy, constrained forward window)
        # -----------------------------
        match_idx = np.zeros(T, dtype=int)
        aligned_rmse = np.zeros((T, V), dtype=np.float64)

        j_cur = 0
        for i in range(T):
            j0 = j_cur
            j1 = min(T - 1, j_cur + forward_window)
            cand_js = list(range(j0, j1 + 1))

            base_vec = rmse_same[i]  # (V,)
            base_score = reduce_score(base_vec, base_vec)  # = 0 근처(정의상)

            # 1) 개선(4변수 모두 감소) 후보 우선
            feasible = []
            for jj in cand_js:
                vec = pair_rmse(i, jj)
                if require_all_vars_improve:
                    if np.all(vec < base_vec):
                        feasible.append((jj, vec, reduce_score(vec, base_vec)))
                else:
                    feasible.append((jj, vec, reduce_score(vec, base_vec)))

            if len(feasible) > 0:
                feasible.sort(key=lambda x: (x[2], x[0]))
                best_j, best_vec, best_score = feasible[0]
            else:
                # 2) 없으면 현재 범위에서 score 최소
                tmp = []
                for jj in cand_js:
                    vec = pair_rmse(i, jj)
                    tmp.append((jj, vec, reduce_score(vec, base_vec)))
                tmp.sort(key=lambda x: (x[2], x[0]))
                best_j, best_vec, best_score = tmp[0]

                # 3) best가 baseline보다 나쁘면(= 상대개선률이 음수 위주면) 동일시점 쪽으로 회귀
                # base_score는 거의 0, best_score가 0보다 크면(=나쁨) 회귀로직 발동
                if fallback_to_same_time and (best_score > base_score):
                    best_j = min(cand_js, key=lambda jj: (abs(jj - i), jj))
                    best_vec = pair_rmse(i, best_j)
                    best_score = reduce_score(best_vec, base_vec)

            match_idx[i] = best_j
            aligned_rmse[i] = best_vec
            j_cur = best_j  # 단조 증가

        match_idx_by_model[m] = match_idx
        aligned_rmse_by_model[m] = aligned_rmse

    # -----------------------------
    # Summary & print
    # -----------------------------
    result = {
        "rmse_same": rmse_by_model,
        "rmse_aligned": aligned_rmse_by_model,
        "match_idx": match_idx_by_model,
    }
    summary = summarize_rmse_improvement(result, var_name=var_name)
    print(f"date: {target_date}", end=" ")
    print_summary(summary, model_names=model_names)

    # -----------------------------
    # Plot RMSE (2x2): same (solid) vs aligned (dashed) + AUC annotation
    # -----------------------------
    x = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 2, figsize=figsize, clear=True)
    axes = axes.ravel()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(model_list)}

    for c in range(V):
        ax = axes[c]
        var = var_name[c]

        y_text = 0.95
        dy = 0.07

        for mi, m in enumerate(model_list):
            color = model_colors[m]

            label = f"ai_{m:02d}"
            if model_names is not None:
                if 1 <= m <= len(model_names):
                    label = model_names[m - 1]
                elif 0 <= m < len(model_names):
                    label = model_names[m]

            ax.plot(
                x, rmse_by_model[m][:, c],
                linestyle="-", linewidth=1.5, color=color, alpha=0.9,
                label=f"{label} (same)"
            )
            ax.plot(
                x, aligned_rmse_by_model[m][:, c],
                linestyle="--", linewidth=1.0, color=color, alpha=0.9,
                label=f"{label} (aligned)"
            )

            auc_rel = summary[m]["per_var"][var]["auc_rel_drop_%"]
            ax.text(
                0.02, y_text - mi * dy,
                f"{label}: ΔAUC = {auc_rel:.1f}%",
                color=color, fontsize=9,
                transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

        ax.set_title(f"RMSE | {var}")
        ax.set_xlabel("lead time (1..T)")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"RMSE comparison (same vs aligned) | {target_date} | score={score_reduce}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # window positioning (Qt/Tk 둘 다 try)
    mgr = plt.get_current_fig_manager()
    try:
        mgr.window.move(50, 10)
        mgr.window.resize(900, 750)
    except Exception:
        try:
            mgr.window.wm_geometry("800x800+50+10")
        except Exception:
            pass

    # -----------------------------
    # Plot SHIFT
    # -----------------------------

    shift_directions = []
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 3), clear=True)
    gt_idx = np.arange(T)

    for m in model_list:
        mi_arr = match_idx_by_model[m]
        shift = mi_arr - gt_idx

        label = f"ai_{m:02d}"
        if model_names is not None:
            if 1 <= m <= len(model_names):
                label = model_names[m - 1]
            elif 0 <= m < len(model_names):
                label = model_names[m]
        print(f"{model_names[m]}, shift direction: {np.sum(shift)}, last position: {shift[-1]} ")
        ax2.plot(x, shift, marker="o", markersize=3, linewidth=1.5, label=label)
        shift_directions.append(np.sum(shift))

    ax2.axhline(0, linestyle="--", alpha=0.6)
    ax2.set_title(f"Monotone time-shift path (ai_idx - gt_idx) | {target_date}")
    ax2.set_xlabel("GT lead time (1..T)")
    ax2.set_ylabel("shift (steps)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()

    mgr2 = plt.get_current_fig_manager()
    try:
        mgr2.window.move(50, 730)
        mgr2.window.resize(900, 300)
    except Exception:
        try:
            mgr2.window.wm_geometry("800x300+50+730")
        except Exception:
            pass

    if show:
        plt.show()
    else:
        plt.pause(0.1)



    # -----------------------------
    # Full RMSE matrix + matched path visualization
    # -----------------------------
    plot_all_models_rmse_matrix_with_path(
        target_date=target_date,
        tag_name=tag_name,
        model_list=model_list,
        lenTime=lenTime,
        var_name=var_name,
        model_names=model_names,
        match_idx_by_model=match_idx_by_model,
        rmse_same_by_model=rmse_by_model,
        figsize=(9, 8),
        cmap="viridis",
        score_reduce=score_reduce,
        rel_eps=rel_eps,
        rel_alpha=rel_alpha,
        show=True,
    )

    return result, shift_directions


def summarize_rmse_improvement(result, var_name=("t","z","u","v")):
    rmse_same = result["rmse_same"]      # dict[m] -> (T,V)
    rmse_algn = result["rmse_aligned"]   # dict[m] -> (T,V)

    out = {}
    for m in rmse_same.keys():
        A = np.asarray(rmse_same[m], dtype=float)   # (T,V)
        B = np.asarray(rmse_algn[m], dtype=float)   # (T,V)
        d = A - B                                  # (T,V) : +면 개선

        # 안전장치
        eps = 1e-12
        rel = d / (A + eps)                         # (T,V) : 상대개선(비율)

        per_var = {}
        for vi, vn in enumerate(var_name):
            a = A[:, vi]
            b = B[:, vi]
            dd = d[:, vi]
            rr = rel[:, vi]

            # NaN 제외 (필요시)
            mask = np.isfinite(a) & np.isfinite(b)
            a, b, dd, rr = a[mask], b[mask], dd[mask], rr[mask]

            per_var[vn] = {
                "mean_abs_drop": float(np.mean(dd)),
                "median_abs_drop": float(np.median(dd)),
                "auc_drop": float(np.sum(dd)),  # 누적 감소량
                "mean_rel_drop_%": float(np.mean(rr) * 100.0),
                "auc_rel_drop_%": float((np.sum(a) - np.sum(b)) / (np.sum(a) + eps) * 100.0),
                "improved_ratio_%": float(np.mean(dd > 0) * 100.0),
            }

        # 전체(4변수 합산) 기준도 같이
        A_sum = np.sum(A, axis=1)  # (T,)
        B_sum = np.sum(B, axis=1)
        d_sum = A_sum - B_sum
        mask = np.isfinite(A_sum) & np.isfinite(B_sum)
        A_sum, B_sum, d_sum = A_sum[mask], B_sum[mask], d_sum[mask]

        out[m] = {
            "overall": {
                "mean_abs_drop": float(np.mean(d_sum)),
                "median_abs_drop": float(np.median(d_sum)),
                "auc_drop": float(np.sum(d_sum)),
                "auc_rel_drop_%": float((np.sum(A_sum) - np.sum(B_sum)) / (np.sum(A_sum) + eps) * 100.0),
                "improved_ratio_%": float(np.mean(d_sum > 0) * 100.0),
            },
            "per_var": per_var
        }

    return out

def print_summary(summary, model_names=None):
    # 보기 편하게 핵심만 출력(전체 기준)
    for m, d in summary.items():
        name = f"ai_{m:02d}"
        if model_names is not None:
            if 1 <= m <= len(model_names):
                name = model_names[m-1]
            elif 0 <= m < len(model_names):
                name = model_names[m]
        ov = d["overall"]
        print(f"\n[{name}]", end=" ")
        #print(f"  AUC drop (sum over time, 4vars): {ov['auc_drop']:.4f}")
        #print(f"  AUC relative drop (%):          {ov['auc_rel_drop_%']:.2f}%")
        #print(f"  mean abs drop (per step):       {ov['mean_abs_drop']:.4f}")
        #print(f"  improved ratio (% steps):       {ov['improved_ratio_%']:.1f}%")

        #변수별도 보고 싶으면 주석 해제
        for vn, vv in d["per_var"].items():
            print(f"    - {vn}: AUC rel {vv['auc_rel_drop_%']:.2f}%", end=" ")
            #print(f"    - {vn}: AUC {vv['auc_drop']:.4f}, rel {vv['auc_rel_drop_%']:.2f}%") #, improved {vv['improved_ratio_%']:.1f}%")
    print(" ") 

def build_full_rmse_matrix_multivar(gt_img: np.ndarray, model_img: np.ndarray) -> np.ndarray:
    """
    gt_img   : (T,H,W,V)
    model_img: (T,H,W,V)

    return
    ------
    rmse_mat : (T,T,V)
        rmse_mat[i,j,v] = RMSE( GT[i,:,:,v], AI[j,:,:,v] )
    """
    if gt_img.shape != model_img.shape:
        raise ValueError(f"shape mismatch: gt={gt_img.shape}, model={model_img.shape}")

    T, H, W, V = gt_img.shape
    rmse_mat = np.zeros((T, T, V), dtype=np.float32)

    for i in range(T):
        # model_img: (T,H,W,V), gt_img[i]: (H,W,V)
        diff = model_img - gt_img[i][None, ...]     # (T,H,W,V)
        rmse = np.sqrt(np.mean(diff * diff, axis=(1, 2)))   # (T,V)
        rmse_mat[i] = rmse

    return rmse_mat


def plot_rmse_matrix_with_matched_path_multivar(
    rmse_mat: np.ndarray,
    match_idx: np.ndarray,
    var_name=("t", "z", "u", "v"),
    model_label: str = "ai_model",
    target_date: str = "",
    rmse_same: np.ndarray = None,   # (T,V), optional
    score_reduce: str = "rel_softmin",
    rel_eps: float = 1e-6,
    rel_alpha: float = 5.0,
    figsize=(10, 8),
    cmap="viridis",
    show_values: bool = False,
):
    """
    rmse_mat  : (T,T,V)
    match_idx : (T,)
    rmse_same : (T,V), same-time baseline. 주어지면 relative-score map도 그림.
    """

    T, T2, V = rmse_mat.shape
    if T != T2:
        raise ValueError(f"rmse_mat must be (T,T,V), got {rmse_mat.shape}")
    if len(match_idx) != T:
        raise ValueError(f"match_idx length {len(match_idx)} != T {T}")

    def reduce_score(vec: np.ndarray, base: np.ndarray) -> float:
        base = np.asarray(base, dtype=float)
        vec = np.asarray(vec, dtype=float)

        rel_improve = (base - vec) / (base + rel_eps)

        if score_reduce == "rel_mean":
            return -float(np.mean(rel_improve))
        elif score_reduce == "rel_min":
            return -float(np.min(rel_improve))
        elif score_reduce == "rel_softmin":
            x = -rel_improve
            z = rel_alpha * x
            m = np.min(z)
            return float((m + np.log(np.sum(np.exp(z - m)))) / rel_alpha)
        else:
            raise ValueError("score_reduce must be 'rel_mean' | 'rel_min' | 'rel_softmin'")

    # -------------------------
    # panel 구성
    # 기본: 변수별 V개 + mean RMSE 1개 + score map 1개(가능할 때)
    # -------------------------
    n_panels = V + 1
    has_score_map = rmse_same is not None
    if has_score_map:
        n_panels += 1

    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    # 그냥 (2,2) 그래프로 대체
    ncols = 2
    nrows = 2
    bMeanRMSE = False

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, clear=True)
    axes = np.atleast_1d(axes).ravel()

    # heatmap extent를 1..T index처럼 보이게
    extent = [0.5, T + 0.5, 0.5, T + 0.5]
    path_x = match_idx + 1
    path_y = np.arange(T) + 1

    # -------------------------
    # 1) 변수별 RMSE heatmap
    # -------------------------
    for v in range(V):
        ax = axes[v]
        data = rmse_mat[:, :, v]  # rows=GT i, cols=AI j

        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        ax.plot(path_x, path_y, color="red", linewidth=1.0, marker="o", markersize=2)
        ax.set_title(f"{model_label} | RMSE matrix | {var_name[v]}")
        ax.set_xlabel("AI index j")
        ax.set_ylabel("GT index i")
        ax.grid(alpha=0.15)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("RMSE")

        if show_values and T <= 20:
            for i in range(T):
                for j in range(T):
                    ax.text(j + 1, i + 1, f"{data[i,j]:.2f}",
                            ha="center", va="center", fontsize=6, color="white")

    # -------------------------
    # 2) 평균 RMSE map
    # -------------------------
    if bMeanRMSE:
        idx_panel = V
        ax = axes[idx_panel]
        mean_rmse = np.mean(rmse_mat, axis=2)  # (T,T)
        im = ax.imshow(
            mean_rmse,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        ax.plot(path_x, path_y, color="red", linewidth=2.0, marker="o", markersize=3)
        ax.set_title(f"{model_label} | mean RMSE over vars")
        ax.set_xlabel("AI index j")
        ax.set_ylabel("GT index i")
        ax.grid(alpha=0.15)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("mean RMSE")

    # -------------------------
    # 3) relative score map (optional)
    # -------------------------
    if has_score_map:
        idx_panel += 1
        ax = axes[idx_panel]

        score_map = np.zeros((T, T), dtype=np.float32)
        for i in range(T):
            base_vec = rmse_same[i]  # (V,)
            for j in range(T):
                vec = rmse_mat[i, j]  # (V,)
                score_map[i, j] = reduce_score(vec, base_vec)

        im = ax.imshow(
            score_map,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="coolwarm",
        )
        ax.plot(path_x, path_y, color="black", linewidth=2.0, marker="o", markersize=3)
        ax.set_title(f"{model_label} | relative score map ({score_reduce})")
        ax.set_xlabel("AI index j")
        ax.set_ylabel("GT index i")
        ax.grid(alpha=0.15)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("score (lower is better)")

    # 남는 축 제거
    # for k in range(idx_panel + 1, len(axes)):
    #     fig.delaxes(axes[k])

    fig.suptitle(f"Full RMSE matrix + matched path | {model_label} | {target_date}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def plot_all_models_rmse_matrix_with_path(
    target_date: str,
    tag_name: str,
    model_list,
    lenTime: int = 49,
    var_name=("t", "z", "u", "v"),
    model_names=None,
    match_idx_by_model=None,
    rmse_same_by_model=None,
    figsize=(10, 8),
    cmap="viridis",
    score_reduce="rel_softmin",
    rel_eps=1e-6,
    rel_alpha=5.0,
    show=False,
):
    """
    기존 alignment 결과(match_idx_by_model, rmse_same_by_model)를 받아서
    각 모델의 full RMSE matrix + path를 그림.
    """

    raw_data_path = os.path.normpath(
        os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev", f"{target_date}")
    )
    V = len(var_name)
    T = lenTime

    # GT load
    gt_img = []
    for k in range(T):
        one_file = os.path.join(raw_data_path, f"ecmwf/ecmwf_t{k+1:04d}.npy")
        gt_img.append(np.load(one_file))
    gt_img = np.asarray(gt_img)

    if gt_img.ndim != 4 or gt_img.shape[-1] != V:
        raise ValueError(f"GT shape expected (T,H,W,{V}) but got {gt_img.shape}")

    figs = []

    for m in model_list:
        model_img = []
        for k in range(T):
            one_file = os.path.join(
                raw_data_path, f"ai_{m:02d}/ai_{m:02d}_t{k+1:04d}.npy"
            )
            model_img.append(np.load(one_file))
        model_img = np.asarray(model_img)

        rmse_mat = build_full_rmse_matrix_multivar(gt_img, model_img)

        label = f"ai_{m:02d}"
        if model_names is not None:
            if 1 <= m <= len(model_names):
                label = model_names[m - 1]
            elif 0 <= m < len(model_names):
                label = model_names[m]

        match_idx = None if match_idx_by_model is None else match_idx_by_model[m]
        rmse_same = None if rmse_same_by_model is None else rmse_same_by_model[m]

        if match_idx is None:
            raise ValueError("match_idx_by_model must be provided.")

        fig = plot_rmse_matrix_with_matched_path_multivar(
            rmse_mat=rmse_mat,
            match_idx=match_idx,
            var_name=var_name,
            model_label=label,
            target_date=target_date,
            rmse_same=None, #rmse_same,
            score_reduce=score_reduce,
            rel_eps=rel_eps,
            rel_alpha=rel_alpha,
            figsize=figsize,
            cmap=cmap,
        )
        figs.append(fig)

    def on_key(event):        
        if event.key in ["x"]:
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)

    if show:
        plt.show()
    else:
        plt.pause(0.1)

    return figs