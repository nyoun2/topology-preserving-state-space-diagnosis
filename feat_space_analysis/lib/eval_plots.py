import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import defaultdict
import logging
#import cartopy.crs as ccrs

import re
from collections import deque
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

from feat_space_analysis.lib.utils import convert_model_name, append_jsonl, build_result_log
from feat_space_analysis.lib.io_paths import glob_npy
from feat_space_analysis.lib.eval_metrics import _alpha_from_global_minmax, _score_to_alpha
from feat_space_analysis.lib.eval_metrics import query_condition_status_by_date, calc_dtw_and_rmse_by_length
from feat_space_analysis.lib.dtw_like_rmse import plot_rmse_comparison_by_models_monotone_shift_multivar
from feat_space_analysis.lib.build_manifold_learning_v3 import fit_parametric_manifold_with_earlystop, load_embedder_and_project


IDX_NAME = {
    1: "FNET-ifs",
    2: "FNET-kim",
    3: "FNET-um",
    4: "GRPH-ifs",
    5: "GRPH-kim",
    6: "GRPH-um",
    7: "PANG-ifs",
    8: "PANG-kim",
    9: "PANG-um",
}

def plot_points_on_sphere(
    S,
    color=None,
    title="UMAP projected onto sphere",
    point_size=10,
    alpha=0.75,
    show_sphere=True,
    sphere_resolution=60,
):
    """
    S: (N,3) on unit sphere
    color: None or (N,) array
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --- sphere surface (wireframe) ---
    if show_sphere:
        u = np.linspace(0, 2*np.pi, sphere_resolution)
        v = np.linspace(0, np.pi, sphere_resolution)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.25)

    # --- points ---
    if color is None:
        color = np.arange(len(S))

    sc = ax.scatter(
        S[:, 0], S[:, 1], S[:, 2],
        s=point_size,
        c=color,
        cmap="Spectral",
        alpha=alpha,
    )
    fig.colorbar(sc, ax=ax, shrink=0.75, label="Index" if color is not None else "")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 보기 좋게 축 비율 동일하게 (matplotlib 3D 기본이 왜곡됨)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()


def play_paired_features_with_zoom(
    A_2d,                 # (N,2) background feature space (ERA5)
    paired_list,          # list of paired 2D: each item -> (2,2) or dict/tuple
    *,
    pause_sec=0.8,
    margin_ratio=0.25,    # zoom 여백(점 범위의 비율)
    min_halfspan=0.05,    # 너무 가까울 때 최소 zoom 크기
    ecmwf_label="ECMWF",
    ai_label="AI",
    title_prefix="Paired features",
    show_connect_line=True,
):
    """
    A 전체 space 위에 (ECMWF, AI) 2점짜리 paired feature들을 B1..Bk 순서로 보여주고,
    오른쪽 zoom 창에 해당 2점 주변만 확대해서 순차 표시.

    paired_list 원소 허용 형태:
      - np.ndarray shape (2,2): [[x_ecmwf,y_ecmwf],[x_ai,y_ai]]
      - tuple/list 길이 2: (ecmwf_xy, ai_xy) where each is (2,)
      - dict: {"ecmwf": (2,), "ai": (2,)}  (키는 대소문자 무관하게 처리)
    """

    A = np.asarray(A_2d, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2:
        raise ValueError("A_2d must be shape (N,2)")

    def _parse_pair(p):
        if isinstance(p, np.ndarray):
            arr = np.asarray(p, dtype=float)
            if arr.shape == (2, 2):
                return arr[0], arr[1]
        if isinstance(p, (tuple, list)) and len(p) == 2:
            e = np.asarray(p[0], dtype=float).reshape(-1)
            a = np.asarray(p[1], dtype=float).reshape(-1)
            if e.size == 2 and a.size == 2:
                return e, a
        if isinstance(p, dict):
            keys = {str(k).lower(): k for k in p.keys()}
            if "ecmwf" in keys and "ai" in keys:
                e = np.asarray(p[keys["ecmwf"]], dtype=float).reshape(-1)
                a = np.asarray(p[keys["ai"]], dtype=float).reshape(-1)
                if e.size == 2 and a.size == 2:
                    return e, a
        raise ValueError("Each paired item must be (2,2) array, (ecmwf_xy, ai_xy), or {'ecmwf':..,'ai':..}")

    pairs = [ _parse_pair(p) for p in paired_list ]
    if len(pairs) == 0:
        raise ValueError("paired_list is empty")

    plt.ion()
    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(title_prefix)

    # --- background: A (ERA5 feature space) ---
    ax_all.set_title("All space (A) + current pair + ROI")
    ax_zoom.set_title("Zoom view (current pair)")
    ax_all.scatter(A[:, 0], A[:, 1], s=8, alpha=0.12)
    ax_zoom.scatter(A[:, 0], A[:, 1], s=10, alpha=0.08)  # zoom에도 faint 배경(원하면 제거)

    # --- artists (재사용) ---
    # all
    scat_e_all = ax_all.scatter([], [], s=80, marker="x")
    scat_a_all = ax_all.scatter([], [], s=80, marker="o")
    line_all = ax_all.plot([], [], linewidth=1.5, alpha=0.8)[0] if show_connect_line else None
    roi_rect = Rectangle((0, 0), 1, 1, fill=False, linewidth=1.5, alpha=0.9)
    ax_all.add_patch(roi_rect)

    # zoom
    scat_e_z = ax_zoom.scatter([], [], s=140, marker="x")
    scat_a_z = ax_zoom.scatter([], [], s=140, marker="o")
    line_z = ax_zoom.plot([], [], linewidth=2.0, alpha=0.9)[0] if show_connect_line else None

    # --- 전체 범위 고정(원하면 주석 처리) ---
    xpad = (A[:, 0].max() - A[:, 0].min()) * 0.03
    ypad = (A[:, 1].max() - A[:, 1].min()) * 0.03
    ax_all.set_xlim(A[:, 0].min() - xpad, A[:, 0].max() + xpad)
    ax_all.set_ylim(A[:, 1].min() - ypad, A[:, 1].max() + ypad)

    for k, (e_xy, a_xy) in enumerate(pairs, start=1):
        e_xy = np.asarray(e_xy, float)
        a_xy = np.asarray(a_xy, float)

        # --- ALL 창 업데이트 ---
        scat_e_all.set_offsets(e_xy[None, :])
        scat_a_all.set_offsets(a_xy[None, :])
        if show_connect_line:
            line_all.set_data([e_xy[0], a_xy[0]], [e_xy[1], a_xy[1]])

        # --- ZOOM 범위 계산 ---
        xs = np.array([e_xy[0], a_xy[0]])
        ys = np.array([e_xy[1], a_xy[1]])
        cx, cy = xs.mean(), ys.mean()
        span_x = max(xs.max() - xs.min(), 2 * min_halfspan)
        span_y = max(ys.max() - ys.min(), 2 * min_halfspan)
        half_x = (span_x * (1.0 + margin_ratio)) / 2.0
        half_y = (span_y * (1.0 + margin_ratio)) / 2.0

        x0, x1 = cx - half_x, cx + half_x
        y0, y1 = cy - half_y, cy + half_y

        # ROI 박스(ALL)에 표시
        roi_rect.set_xy((x0, y0))
        roi_rect.set_width(x1 - x0)
        roi_rect.set_height(y1 - y0)

        # --- ZOOM 창 업데이트 ---
        ax_zoom.set_xlim(x0, x1)
        ax_zoom.set_ylim(y0, y1)
        scat_e_z.set_offsets(e_xy[None, :])
        scat_a_z.set_offsets(a_xy[None, :])
        if show_connect_line:
            line_z.set_data([e_xy[0], a_xy[0]], [e_xy[1], a_xy[1]])

        # 타이틀/범례 느낌 텍스트
        ax_zoom.set_title(f"Zoom (B{k})   {ecmwf_label}: x  |  {ai_label}: o")
        ax_all.set_xlabel("dim-1")
        ax_all.set_ylabel("dim-2")
        ax_zoom.set_xlabel("dim-1")
        ax_zoom.set_ylabel("dim-2")

        fig.canvas.draw_idle()
        plt.pause(pause_sec)

    plt.ioff()
    plt.show()


def show_raw_data_comparison(target_date, tag_name, model_idx, model_name):
    raw_data_path = os.path.normpath(
        os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev", f"{target_date}")
    )

    lenTime = 49
    V = 4

    # -----------------------------
    # Load all images: (T, H, W, V)
    # -----------------------------
    model_img = []
    for k in range(lenTime):
        one_file = os.path.join(
            raw_data_path, f"ai_{model_idx:02d}/ai_{model_idx:02d}_t{k+1:04d}.npy"
        )
        model_img.append(np.load(one_file))

    gt_img = []
    for k in range(lenTime):
        one_file = os.path.join(raw_data_path, f"ecmwf/ecmwf_t{k+1:04d}.npy")
        gt_img.append(np.load(one_file))

    model_img = np.asarray(model_img)
    gt_img = np.asarray(gt_img)

    # -----------------------------
    # (2) Shared imshow scales: per-variable vmin/vmax over ALL time, GT+MODEL
    # -----------------------------
    vmin = np.empty(V, dtype=np.float64)
    vmax = np.empty(V, dtype=np.float64)
    for c in range(V):
        vmin[c] = min(gt_img[..., c].min(), model_img[..., c].min())
        vmax[c] = max(gt_img[..., c].max(), model_img[..., c].max())

    # (선택) outlier 때문에 스케일이 망가지면 퍼센타일로 바꿔라
    # for c in range(V):
    #     allc = np.concatenate([gt_img[..., c].ravel(), model_img[..., c].ravel()])
    #     vmin[c] = np.percentile(allc, 1)
    #     vmax[c] = np.percentile(allc, 99)

    # -----------------------------
    # (1) RMSE per variable over time: (T, V)
    # -----------------------------
    rmse = np.sqrt(np.mean((model_img - gt_img) ** 2, axis=(1, 2)))  # (49,4)

    var_name = ["t", "z", "u", "v"]

    # -----------------------------
    # 3x4 figure
    # -----------------------------
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), num=3, clear=True)
    state = {"idx": 0}
    

    def draw(idx):
        # --- GT row ---
        one_gt = gt_img[idx]
        for c in range(V):
            ax = axes[0, c]
            ax.cla()
            ax.imshow(one_gt[:, :, c], vmin=vmin[c], vmax=vmax[c])
            ax.set_title(f"GT | {var_name[c]}")
            ax.set_xticks([]); ax.set_yticks([])

        # --- Model row ---
        one_model = model_img[idx]
        for c in range(V):
            ax = axes[1, c]
            ax.cla()
            ax.imshow(one_model[:, :, c], vmin=vmin[c], vmax=vmax[c])
            ax.set_title(f"Model | {var_name[c]}")
            ax.set_xticks([]); ax.set_yticks([])
        

        fig.suptitle(
            f"{target_date} | model={model_name} | lead_time={idx+1} | "
            f"RMSE(t,z,u,v)=({rmse[idx,0]:.3g}, {rmse[idx,1]:.3g}, {rmse[idx,2]:.3g}, {rmse[idx,3]:.3g})"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ["right", "d"]:
            state["idx"] = (state["idx"] + 1) % lenTime
            draw(state["idx"])
        elif event.key in ["left", "a"]:
            state["idx"] = (state["idx"] - 1) % lenTime
            draw(state["idx"])
        elif event.key in [" "]:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(state["idx"])
    plt.tight_layout()

    mgr = plt.get_current_fig_manager()
    mgr.window.move(2030, 30)
    mgr.window.resize(1400, 800)

    plt.show()

def show_next_figures():
    plt.waitforbuttonpress()

def show_trajectory_on_featuremap(target_date, tag_name, model_list, maps, infer_out_root=None):

    #data_dir = f"out_test_features/{tag_name}/{target_date}"
    if infer_out_root is None:
        data_dir_base = f"out_test_features/{tag_name}/{target_date}"
    else:        
        data_dir_base = os.path.normpath(os.path.join(infer_out_root, target_date))

    # load Z_all
    output_npy = os.path.join(data_dir_base, "final/era5_manifold_z.npy")
    Z_all = np.load(output_npy)

    # load Z_gt
    output_npy = os.path.join(data_dir_base, "final/z_ecmwf_manifold.npy")
    Z_gt = np.load(output_npy)
    
    # load Z_ai_list
    #data_dir_ai = f"out_test_features/{tag_name}/{target_date}/proj"
    data_dir_ai = os.path.join(data_dir_base, "proj")
    files = glob_npy(data_dir_ai, "pairs*.npy")
    cFile = len(files)
    Z_ai_all = []
    for a in range(cFile):
        output_npy = os.path.join(data_dir_base, f"final/ai_{a+1:02d}_manifold.npy")
        Z_ai = np.load(output_npy)
        Z_ai_all.append(Z_ai)

    ###########################################
    ### AIMD path figure,
    ###########################################

    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    
    status_str_list = []
    for k in model_list:
        # 여기에서 score: good/bad 를 결정할 수 있음
        target_date_alone = target_date.split('_')[0]
        status = query_condition_status_by_date(
            maps=maps,
            target_date=target_date_alone,
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

    # --------------------------------------------------
    # model_list로 필터링 (1-based → 0-based)
    # --------------------------------------------------
    model_idx0 = [i - 1 for i in model_list]
    Z_ai_list = [Z_ai_all[i] for i in model_idx0]
    

    ### draw all models in a single figure
    state = {"bRMSEview": False}
    # fig = plt.figure(num=5+model_list[0], figsize=(9, 9), clear=True)
    # ax = fig.add_subplot(111)
    fig_num = 5 + model_list[0]
    fig = plt.figure(num=fig_num)
    fig.clf()
    fig.set_size_inches(9, 9, forward=True)
    ax = fig.add_subplot(111)

    def toggle_view():
        state["bRMSEview"] = not state["bRMSEview"]
        print("rmse view open" if state["bRMSEview"] else "rmse view closed")

    def draw(bRMSEview):
        ax.cla()

        # if bRMSEview:
        #     print("show raw data comparison")
        #     # 전체 모델에 대해 raw comparison을 띄우고 싶지 않으면
        #     # 대표 1개만 띄우거나, 아래를 주석 처리하는 편이 낫다.
        #     # 예: 첫 번째 모델만 표시
        #     show_raw_data_comparison(
        #         target_date=target_date,
        #         tag_name=tag_name,
        #         model_idx=model_list[0],
        #         model_name=model_names[model_list[0] - 1]
        #     )

        # 전체 배경 점
        ax.scatter(
            Z_all[:, 0], Z_all[:, 1],
            c="gray", s=2, alpha=0.25, label="all"
        )

        # GT 시작/끝/경로
        ax.scatter(
            Z_gt[0, 0], Z_gt[0, 1],
            c="green", s=40, alpha=0.9, label="GT start"
        )
        ax.scatter(
            Z_gt[-1, 0], Z_gt[-1, 1],
            c="orange", s=40, alpha=0.9, label="GT end"
        )
        ax.scatter(
            Z_gt[:, 0], Z_gt[:, 1],
            c="black", s=8, alpha=0.8, label="GT points"
        )
        ax.plot(
            Z_gt[:, 0], Z_gt[:, 1],
            color="black",
            alpha=0.8,
            linewidth=1.2,
            linestyle="--",
            label="GT trajectory"
        )

        # 모델별 색상
        # model 수가 많아져도 자동으로 색이 순환되게 tab10 사용
        cmap = plt.get_cmap("tab10")
        n_models = len(Z_ai_list)

        title_lines = [f"{target_date}"]

        for i, Z_ai in enumerate(Z_ai_list):
            color = cmap(i % 10)
            model_idx = model_list[i]
            model_name = model_names[model_idx - 1]
            status_str = status_str_list[i]

            color = cmap((model_list[0]-1)%10) # 3개 모델에 따라 색상 조정하기 위함

            # GT와 AI 동일 시점 연결선
            ax.plot(
                np.stack([Z_gt[:, 0], Z_ai[:, 0]], axis=1).T,
                np.stack([Z_gt[:, 1], Z_ai[:, 1]], axis=1).T,
                color=color,
                alpha=0.20,
                linewidth=0.8,
                linestyle=":"
            )

            # AI trajectory line
            ax.plot(
                Z_ai[:, 0], Z_ai[:, 1],
                color=color,
                alpha=0.9,
                linewidth=1.5,
                linestyle="-",
                label=f"{model_name}"
            )

            # AI points
            ax.scatter(
                Z_ai[:, 0], Z_ai[:, 1],
                color=color,
                s=10,
                alpha=0.85
            )

            # AI 시작점 강조
            ax.scatter(
                Z_ai[0, 0], Z_ai[0, 1],
                color=color,
                s=35,
                alpha=1.0,
                edgecolors="k",
                linewidths=0.4
            )

            title_lines.append(f"{model_name}: {status_str}")

        ax.set_title("\n".join(title_lines), fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        fig.canvas.draw_idle()


    def on_key(event):
        if event.key in ["d"]:
            toggle_view()
            draw(state["bRMSEview"])
        elif event.key in [" "]:
            plt.close("all")


    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(state["bRMSEview"])

    mgr = plt.get_current_fig_manager()
    try:# model_list에 따라 figure 위치 조정하여 출력 (3개 모델에 최적화됨)
        mgr.window.move(200+(model_list[0]-1)*170, 30)
        mgr.window.resize(510, 510)
    except Exception:
        pass

    #plt.show()
    plt.pause(0.1)
    #plt.waitforbuttonpress()

    return Z_gt, Z_ai_list
    

    # ### draw single models..
    # model_names = model_names[model_idx0]
    # state =  {"idx": 0, "bRMSEview": False}  # 현재 보고 있는 AI index
    # #fig, ax = plt.subplots(figsize=(8, 8))
    # fig = plt.figure(num=5, figsize=(9, 9), clear=True)
    # ax = fig.add_subplot(111)

    
    # def toggle_view():
    #     state["bRMSEview"] = not state["bRMSEview"]
    #     print("rmse view open" if state["bRMSEview"] else "rmse view closed")
        
    
    # def draw(idx, bRMSEview):
    #     ax.cla()

    #     if bRMSEview:
    #         print("show raw data comparison")
    #         show_raw_data_comparison(target_date, tag_name, idx+1, model_names[idx])

    #     Z_ai = Z_ai_list[idx]

    #     ax.scatter(Z_all[:, 0], Z_all[:, 1],
    #             c="gray", s=2, alpha=0.3, label="all")        
    #     ax.plot(
    #         np.stack([Z_gt[:, 0], Z_ai[:, 0]], axis=1).T,
    #         np.stack([Z_gt[:, 1], Z_ai[:, 1]], axis=1).T,
    #         color="gray",
    #         alpha=0.7,
    #         linewidth=1,
    #         linestyle="--"
    #     )
    #     plt.plot(
    #         np.stack([Z_ai[:-1, 0], Z_ai[1:, 0]], axis=1).T,
    #         np.stack([Z_ai[:-1, 1], Z_ai[1:, 1]], axis=1).T,
    #         color="red",
    #         alpha=0.7,
    #         linewidth=1,
    #         linestyle="-"
    #     )

    #     ax.scatter(Z_gt[0, 0], Z_gt[0, 1],
    #             c="green", s=30, alpha=0.7, label="start_date")
    #     ax.scatter(Z_gt[-1, 0], Z_gt[-1, 1],
    #             c="orange", s=30, alpha=0.7, label="end_date")
    #     ax.scatter(Z_gt[:, 0], Z_gt[:, 1],
    #             c="blue", s=5, alpha=0.7, label="ecmwf")
    #     ax.scatter(Z_ai[:, 0], Z_ai[:, 1],
    #             c="red", s=5, alpha=0.7, label=f"ai_{idx+1}")
        
    #     plt.plot(
    #         np.stack([Z_gt[:-1, 0], Z_gt[1:, 0]], axis=1).T,
    #         np.stack([Z_gt[:-1, 1], Z_gt[1:, 1]], axis=1).T,
    #         color="black",
    #         alpha=0.7,
    #         linewidth=1,
    #         linestyle="--"
    #     )

    #     ax.set_title(f"{target_date}, {model_names[idx]}, good conditions: {status_str_list[idx]}") # | RMSE = {rmse:.4f}, ({mask.sum()})")
    #     ax.legend()
    #     fig.canvas.draw_idle()

    # def on_key(event):
    #     if event.key in ["right"]:
    #         state["idx"] = (state["idx"] + 1) % len(Z_ai_list)
    #         draw(state["idx"], state["bRMSEview"])
    #     elif event.key in ["left"]:
    #         state["idx"] = (state["idx"] - 1) % len(Z_ai_list)
    #         draw(state["idx"], state["bRMSEview"])
    #     elif event.key in ["d"]:
    #         # detail rmse view
    #         toggle_view()
    #         draw(state["idx"], state["bRMSEview"])
    #     elif event.key in [" "]:
    #         plt.close("all")

    # fig.canvas.mpl_connect("key_press_event", on_key)
    # draw(state["idx"], state["bRMSEview"])
    
    # mgr = plt.get_current_fig_manager()
    # mgr.window.move(900, 30)
    # mgr.window.resize(900, 900)

    # plt.show()
    # # plt.pause(0.1)
    # # plt.close("all")

    
def show_valid_trajectory_on_featuremap(target_date, tag_name, model_list,
                                        infer_out_root=None):

    if infer_out_root is None:
        data_dir_base = f"out_test_features/{tag_name}/{target_date}"
    else:        
        data_dir_base = os.path.normpath(os.path.join(infer_out_root, target_date))

    # load Z_all
    output_npy = os.path.join(data_dir_base, "final/era5_manifold_z.npy")
    Z_all = np.load(output_npy)
    
    # load Z_ai_list
    #data_dir_ai = f"out_test_features/{tag_name}/{target_date}/proj"
    data_dir_ai = os.path.join(data_dir_base, "proj")
    #files = glob_npy(data_dir_ai, "pairs*.npy")
    files = glob_npy(data_dir_ai, "Z_*.npy")
    cFile = len(files)

    Z_ai_all = []
    for a in range(cFile):
        output_npy = os.path.join(data_dir_base, f"final/ai_{a+1:02d}_manifold.npy")
        Z_ai = np.load(output_npy)
        Z_ai_all.append(Z_ai)

    ###########################################
    ### AIMD path figure,
    ###########################################

    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    # --------------------------------------------------
    # model_list로 필터링 (1-based → 0-based)
    # --------------------------------------------------
    model_idx0 = [i - 1 for i in model_list]
    Z_ai_list = [Z_ai_all[i] for i in model_idx0]

    ### draw all models in a single figure
    state = {"bRMSEview": False}
    #fig = plt.figure(num=5+model_list[0], figsize=(9, 9), clear=True)
    #ax = fig.add_subplot(111)
    fig_num = 5 + model_list[0]
    fig = plt.figure(num=fig_num)
    fig.clf()
    fig.set_size_inches(9, 9, forward=True)
    ax = fig.add_subplot(111)

    def toggle_view():
        state["bRMSEview"] = not state["bRMSEview"]
        print("rmse view open" if state["bRMSEview"] else "rmse view closed")

    def draw(bRMSEview):
        ax.cla()

        # 전체 배경 점
        ax.scatter(
            Z_all[:, 0], Z_all[:, 1],
            c="gray", s=2, alpha=0.25, label="all"
        )        

        # 모델별 색상
        # model 수가 많아져도 자동으로 색이 순환되게 tab10 사용
        cmap = plt.get_cmap("tab10")
        n_models = len(Z_ai_list)

        title_lines = [f"{target_date}"]

        for i, Z_ai in enumerate(Z_ai_list):
            color = cmap(i % 10)
            model_idx = model_list[i]
            model_name = model_names[model_idx - 1]
            color = cmap((model_list[0]-1) % 10)

            # AI trajectory line
            ax.plot(
                Z_ai[:, 0], Z_ai[:, 1],
                color=color,
                alpha=0.9,
                linewidth=1.5,
                linestyle="-",
                label=f"{model_name}"
            )

            # AI points
            ax.scatter(
                Z_ai[:, 0], Z_ai[:, 1],
                color=color,
                s=10,
                alpha=0.85
            )

            # AI 시작점 강조
            ax.scatter(
                Z_ai[0, 0], Z_ai[0, 1],
                color="black",
                s=35,
                alpha=1.0,
                edgecolors="k",
                linewidths=0.4
            )

            title_lines.append(f"{model_name}")

        ax.set_title(" / ".join(title_lines), fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ["d"]:
            toggle_view()
            draw(state["bRMSEview"])
        elif event.key in [" "]:
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(state["bRMSEview"])

    # mgr = plt.get_current_fig_manager()
    # try:
    #     mgr.window.move(750, 30)
    #     mgr.window.resize(900, 900)
    # except Exception:
    #     pass

    # #plt.show()
    # plt.pause(0.1)

    mgr = plt.get_current_fig_manager()
    try:# model_list에 따라 figure 위치 조정하여 출력 (3개 모델에 최적화됨)
        mgr.window.move(200+(model_list[0]-1)*170, 30)
        mgr.window.resize(510, 510)
    except Exception:
        pass

    #plt.show()
    plt.pause(0.1)
    #plt.waitforbuttonpress()


def evaluate_performance_and_show_featuremap(target_date, tag_name, bShow=True, bOnlyTrack=False):

    # Final Stage: parametric manifold learning, 추론 과정
    # z_post features to 2d loc. data via parametric umap 

    data_dir = f"out_test_features/{tag_name}/{target_date}"
    EXTRACT_DIR = f"model_all/{tag_name}"

    #input_npy = "out_features/ERA5_proj_z.npy" # 학습데이터 확인
    input_npy = f"model_all/{tag_name}/era5_proj_z.npy" # 학습데이터 확인
    os.makedirs(os.path.join(data_dir, "final"), exist_ok=True)
    output_npy = os.path.join(data_dir, "final/era5_manifold_z.npy")
    
    #z_pre_all = np.load(input_npy)
    #proc_parametric_umap(input_npy, output_npy, toTrain=False, tag_name=tag_name)
    # umap_projection(
    #     feature_path=input_npy,
    #     mode="fit", #"fit", "transform"
    #     umap_mode=1,
    #     model_path=f"{EXTRACT_DIR}/umap_model.pkl",
    #     save_path=output_npy,
    #     show_plot=False
    # )
    # Z_all = np.load(output_npy)

    z_pre_all = np.load(input_npy)
    best_path = "pm_run02/best.weights.h5"
    Z_all = load_embedder_and_project(z_pre_all, weights_path=best_path)
    np.save(output_npy, Z_all) 


    # ## 여기에 feature space에 대한 평가 결과를 표시 ##
    # X = np.load(input_npy)
    # Z = np.load(output_npy)
    # metrics = evaluate_umap_embedding(X, Z, ks=(10,20,50), spearman_pairs=200_000, continuity_m=300)
    # print(metrics)


    # plt.figure(figsize=(8, 8))
    # plt.scatter(Z_all[:, 0], Z_all[:, 1], c="gray", s=2, alpha=0.3, label="all")    
    # plt.xlabel("z1")
    # plt.ylabel("z2")
    # plt.title("EfficientNet based Feature Representation")
    # plt.legend()
    # plt.tight_layout()
    # #plt.waitforbuttonpress()
    # plt.pause(0.1)
    
    if not bOnlyTrack: # ecmwf와 pair 정보가 없음
        input_npy = os.path.join(data_dir, "proj/z_ecmwf.npy")
        output_npy = os.path.join(data_dir, "final/z_ecmwf_manifold.npy")
        #z_pre_ecmwf = np.load(input_npy)
        #proc_parametric_umap(input_npy, output_npy, toTrain=False, tag_name=tag_name)
        # umap_projection(
        #     feature_path=input_npy,
        #     mode="transform", #"fit", "transform"
        #     umap_mode=1,
        #     model_path=f"{EXTRACT_DIR}/umap_model.pkl",
        #     save_path=output_npy,
        #     show_plot=False
        # )
        # Z_gt = np.load(output_npy)

        z_pre_ecmwf = np.load(input_npy)
        best_path = "pm_run02/best.weights.h5"
        Z_gt = load_embedder_and_project(z_pre_ecmwf, weights_path=best_path)
        np.save(output_npy, Z_gt)
        
        
        plt.figure(1, figsize=(8, 8))
        plt.clf()
        plt.scatter(Z_all[:, 0], Z_all[:, 1], c="gray", s=2, alpha=0.3, label="all") 
        plt.scatter(Z_gt[0, 0], Z_gt[0, 1],
                c="green", s=30, alpha=0.7, label="start_date")
        plt.scatter(Z_gt[-1, 0], Z_gt[-1, 1],
                c="orange", s=30, alpha=0.7, label="end_date")
        plt.scatter(Z_gt[:, 0], Z_gt[:, 1], c="blue", s=5, alpha=0.7, label="ecmwf")
        #plt.waitforbuttonpress()
        plt.plot(
                np.stack([Z_gt[:-1, 0], Z_gt[1:, 0]], axis=1).T,
                np.stack([Z_gt[:-1, 1], Z_gt[1:, 1]], axis=1).T,
                color="black",
                alpha=0.7,
                linewidth=1,
                linestyle="--"
            )
        plt.title(f"target date: {target_date}")
        plt.pause(0.1)

    # # temporary, era5 vs. ecmwf
    # era5_s_idx = 7257
    # plt.figure(figsize=(8, 8))
    # plt.scatter(Z_all[:, 0], Z_all[:, 1], c="gray", s=2, alpha=0.3, label="all") 
    # plt.scatter(Z_gt[0, 0], Z_gt[0, 1],
    #         c="green", s=30, alpha=0.7, label="start_date")
    # plt.scatter(Z_gt[-1, 0], Z_gt[-1, 1],
    #         c="orange", s=30, alpha=0.7, label="end_date")
    # plt.scatter(Z_gt[:, 0], Z_gt[:, 1], c="blue", s=5, alpha=0.7, label="ecmwf")
    # plt.scatter(Z_all[era5_s_idx:era5_s_idx+48, 0], Z_all[era5_s_idx:era5_s_idx+48, 1], 
    #             c="red", s=5, alpha=0.7, label="era5_matched")
    # #plt.waitforbuttonpress()
    # plt.plot(
    #         np.stack([Z_gt[:, 0], Z_all[era5_s_idx:era5_s_idx+48, 0]], axis=1).T,
    #         np.stack([Z_gt[:, 1], Z_all[era5_s_idx:era5_s_idx+48, 1]], axis=1).T,
    #         color="black",
    #         alpha=0.7,
    #         linewidth=1
    #     )
    # plt.pause(0.1)
    
    data_dir = f"out_test_features/{tag_name}/{target_date}/proj"
    files = glob_npy(data_dir, "Z_ai*.npy")
    cFile = len(files)

    distance_list = []
    rmse_list = []
    Z_ai_list = []
    
    for a in range(cFile):
        input_npy = "tmp.npy"
        x = np.load(files[a])
        # x_second = x[:, 1, :]        
        # np.save(input_npy, x_second)
        np.save(input_npy, x)
        
        data_dir = f"out_test_features/{tag_name}/{target_date}"
        #output_npy = os.path.join(data_dir, f"final/pairs_ai_{a+1:02d}_manifold.npy")
        output_npy = os.path.join(data_dir, f"final/ai_{a+1:02d}_manifold.npy")

        #proc_parametric_umap(input_npy, output_npy, toTrain=False, tag_name=tag_name)
        # umap_projection(
        #     feature_path=input_npy,
        #     mode="transform", #"fit", "transform"
        #     umap_mode=1,
        #     model_path=f"{EXTRACT_DIR}/umap_model.pkl",
        #     save_path=output_npy,
        #     show_plot=False
        # )
        # Z_ai = np.load(output_npy)

        z_pre_ai = np.load(input_npy)
        best_path = "pm_run02/best.weights.h5"
        Z_ai = load_embedder_and_project(z_pre_ai, weights_path=best_path)
        np.save(output_npy, Z_ai)

        Z_ai_list.append(Z_ai)

        # rmse_one = rmse_2d_over_time(Z_gt, Z_ai)
        # #print(f"compare to ai_{a+1} vs. ecmwf => rmse => {np.round(rmse_one, 4)}")
        # rmse_list.append(rmse_one)

        # diff = Z_ai - Z_gt                      # (N, 2)
        # dist = np.linalg.norm(diff, axis=1)     # (N,)  각 점의 거리
        # distance_list.append(np.round(dist,3))

        # plt.figure(figsize=(8, 8))    
        # plt.clf()
        # plt.scatter(Z_all[:, 0], Z_all[:, 1], c="gray", s=2, alpha=0.3, label="all")
        # plt.plot(
        #     np.stack([Z_gt[:, 0], Z_ai[:, 0]], axis=1).T,
        #     np.stack([Z_gt[:, 1], Z_ai[:, 1]], axis=1).T,
        #     color="black",
        #     alpha=0.7,
        #     linewidth=1
        # )
        # plt.scatter(Z_gt[:, 0], Z_gt[:, 1], c="blue", s=5, alpha=0.7, label="ecmwf")
        # plt.scatter(Z_ai[:, 0], Z_ai[:, 1], c="red", s=5, alpha=0.7, label=f"ai_{a+1}")
        # # 같은 index끼리 선 연결 (핵심)
        
        # plt.legend()
        # plt.title(f"compare to ai_{a+1} vs. ecmwf => rmse => {np.round(rmse_one, 4)}")
        # #plt.waitforbuttonpress()
        # plt.pause(0.1)

    logging.info(f"/n{tag_name}: {target_date}")
    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])
    lengths_list = [12, 24, 36, 48]

    if not bOnlyTrack:
        rmse_best, dtw_best, rmse_list, dtw_list = calc_dtw_and_rmse_by_length(Z_gt, Z_ai_list, lengths_list)

        result_log = build_result_log(target_date, Z_gt[0], lengths_list, dtw_list, rmse_list, dtw_best, rmse_best)
        append_jsonl("D:/ext_data/logs/eval_log.jsonl", result_log)
        print(result_log)
    

    # distance_arr = np.array(distance_list)
    # np.set_printoptions(suppress=True, precision=3)
    # print(distance_arr.T)
    
    # idx_min = np.argmin(distance_arr.T, axis=1)
    # model_names = np.array([
    #     "fnet_ifs","fnet_kim","fnet_um",
    #     "grph_ifs","grph_kim","grph_um",
    #     "pang_ifs","pang_kim","pang_um"
    # ])

    # best_model_per_t = model_names[idx_min]
    # for m in range(len(best_model_per_t)):
    #     print(f"Date {m+1}: {best_model_per_t[m]}")
    # models, counts = np.unique(best_model_per_t, return_counts=True)
    # for m, c in zip(models, counts):
    #     print(f"{m}: {c}")
    # D = distance_arr.T      

    # print("========== Models' RMSE Performance ==========")
    # print(np.round(rmse_list, 4))
    # idx_rmse_min = int(np.argmin(rmse_list))
    # sorted_rmse_idx = np.argsort(rmse_list)
    # print(f"best RMSE model: {model_names[idx_rmse_min]}")
    # print(f"Ranked RMSE: {sorted_rmse_idx+1}")
    
    # result_dtw = compare_dtw_to_gt(
    #     gt_blue=Z_gt,
    #     reds=Z_ai_list,
    #     window=int(0.1 * len(Z_gt)),   # 없애려면 None
    #     normalize=True                   # True 추천 (길이 영향 줄임)
    # )
    # print("========== Models' DTW Performance ==========")
    # print("DTW scores:", result_dtw["dtw"])
    # #print("Best idx:", result_dtw["best_idx"], "Best dtw:", np.round(result_dtw["best_dtw"],4))
    # best_idx = int(result_dtw["best_idx"])
    # print(f"best DTW model: {model_names[best_idx]}")
    # print("Ranked DTW:", result_dtw["sorted_idx"])

    # logging.info(f"/n{tag_name}: {target_date}")
    # logging.info("==== Models' RMSE Performance ====")
    # logging.info(np.round(rmse_list, 4).tolist())
    # logging.info(f"best RMSE model: {model_names[idx_rmse_min]}")
    # logging.info(f"Ranked RMSE: {sorted_rmse_idx + 1}")
    # logging.info("==== Models' DTW Performance =====")
    # logging.info(f"DTW scores: {result_dtw['dtw']}")
    # logging.info(f"best DTW model: {model_names[result_dtw['best_idx']]}")
    # logging.info(f"Ranked DTW: {result_dtw['sorted_idx']}")

    # blue = Z_gt
    # for k in range(len(Z_ai_list)):
    #     red = Z_ai_list[k]
    #     result_new = dtw_with_path(blue, red, metric="euclidean", window=None)
    #     print(f"ai_{k+1} - DTW path_len: {result_new['path_len']}")
    #     #print_dtw_diagnostics(result_new, name_blue="ecmwf", name_red="aimd")
    #     plot_dtw_path_on_matrix(result_new, which="D")
    #     #plot_dtw_path_on_matrix(result_new, which="C")
    #     plot_dtw_alignment(blue, red, result_new, dims=(0,1))
    #     plt.waitforbuttonpress()

    ###########################################
    ### AIMD path figure,
    ###########################################

    print("len(Z_ai_list):", len(Z_ai_list))
    print("len(model_names):", len(model_names))

    state = {"idx": 0}  # 현재 보고 있는 AI index
    #fig, ax = plt.subplots(figsize=(8, 8))
    fig = plt.figure(num=2, figsize=(8, 8), clear=True)
    ax = fig.add_subplot(111)
    
    def draw(idx):
        ax.cla()

        #mask = D[:, idx] <= 2.0

        Z_ai = Z_ai_list[idx]
        #rmse = rmse_list[idx]

        ax.scatter(Z_all[:, 0], Z_all[:, 1],
                c="gray", s=2, alpha=0.3, label="all")        

        if not bOnlyTrack:
            ax.plot(
                np.stack([Z_gt[:, 0], Z_ai[:, 0]], axis=1).T,
                np.stack([Z_gt[:, 1], Z_ai[:, 1]], axis=1).T,
                color="gray",
                alpha=0.7,
                linewidth=1,
                linestyle="--"
            )
        

        # nidx = np.where(mask)[0]
        # for i in nidx:
        #     ax.plot(
        #         [Z_gt[i, 0], Z_ai[i, 0]],
        #         [Z_gt[i, 1], Z_ai[i, 1]],
        #         color="black",
        #         alpha=0.7,
        #         linewidth=1                
        #     )
        plt.plot(
            np.stack([Z_ai[:-1, 0], Z_ai[1:, 0]], axis=1).T,
            np.stack([Z_ai[:-1, 1], Z_ai[1:, 1]], axis=1).T,
            color="red",
            alpha=0.7,
            linewidth=1,
            linestyle="-"
        )

        # ax.plot(
        #     np.stack([Z_gt[:, 0], Z_ai[:, 0]], axis=1).T,
        #     np.stack([Z_gt[:, 1], Z_ai[:, 1]], axis=1).T,
        #     color="black",
        #     alpha=0.7,
        #     linewidth=1,
        #     linestyle="--"
        # )

        if not bOnlyTrack:
            ax.scatter(Z_gt[0, 0], Z_gt[0, 1],
                    c="green", s=30, alpha=0.7, label="start_date")
            ax.scatter(Z_gt[-1, 0], Z_gt[-1, 1],
                    c="orange", s=30, alpha=0.7, label="end_date")
            ax.scatter(Z_gt[:, 0], Z_gt[:, 1],
                    c="blue", s=5, alpha=0.7, label="ecmwf")
        else:
            ax.scatter(Z_ai[0, 0], Z_ai[0, 1],
                    c="orange", s=30, alpha=0.7, label="end_date")
            
        ax.scatter(Z_ai[:, 0], Z_ai[:, 1],
                c="red", s=5, alpha=0.7, label=f"ai_{idx+1}")

        # if idx == len(Z_ai_list):
        #     idx = idx % len(Z_ai_list)
        ax.set_title(f"{model_names[idx]}: ai_{idx+1} vs ecmwf") # | RMSE = {rmse:.4f}, ({mask.sum()})")
        ax.legend()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ["right", "d"]:
            state["idx"] = (state["idx"] + 1) % len(Z_ai_list)
            draw(state["idx"])
        elif event.key in ["left", "a"]:
            state["idx"] = (state["idx"] - 1) % len(Z_ai_list)
            draw(state["idx"])
        elif event.key in [" "]:
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(state["idx"])

    if bShow:
        plt.show()
    else:
        plt.pause(0.1)

    #best_idx = int(result_dtw["best_idx"]) + 1
    #return best_idx


def build_points_by_map(records: list[dict], length_list=(12, 24, 36, 48), metrics=("dtw", "rmse")):
    """
    records에서 (metric, length)별로
      - XY: (M,2)
      - best_idx: (M,)
      - target_date: (M,)  # 필요시 annotation용

    return:
      maps[(metric, L)] = dict(XY=..., idx=..., date=...)
    """
    maps = {}
    for m in metrics:
        for L in length_list:
            maps[(m, int(L))] = {"XY": [], "idx": [], "date": []}

    for rec in records:
        td = rec.get("target_date", "")
        pos = rec.get("pos2d", None)
        lengths = rec.get("lengths", {})

        if pos is None or len(pos) != 2:
            continue  # 혹은 raise로 바꿔도 됨
        x, y = float(pos[0]), float(pos[1])

        for L in length_list:
            Ls = str(int(L))
            if Ls not in lengths:
                continue
            for m in metrics:
                if m not in lengths[Ls]:
                    continue
                best_idx = int(lengths[Ls][m])  # 1-based라고 가정
                maps[(m, int(L))]["XY"].append([x, y])
                maps[(m, int(L))]["idx"].append(best_idx)
                maps[(m, int(L))]["date"].append(td)

    # numpy 변환
    for k, v in maps.items():
        v["XY"] = np.array(v["XY"], dtype=np.float64) if len(v["XY"]) else np.zeros((0, 2), dtype=np.float64)
        v["idx"] = np.array(v["idx"], dtype=np.int32) if len(v["idx"]) else np.zeros((0,), dtype=np.int32)
        v["date"] = np.array(v["date"], dtype=object) if len(v["date"]) else np.zeros((0,), dtype=object)

    return maps


def plot_8_maps(
    Z_space: np.ndarray,
    maps: dict,
    length_list=(12, 24, 36, 48),
    metrics=("dtw", "rmse"),
    title: str | None = None,
    idx_min: int = 1,
    idx_max: int = 9,
    annotate: bool = False,
    annotate_max: int = 30,   # 너무 많으면 지저분해서 제한
):
    """
    2x4 figure:
      row0: metric=dtw, col=length_list
      row1: metric=rmse, col=length_list
    """

    IDX_NAME = {
        1: "FNET-ifs",
        2: "FNET-kim",
        3: "FNET-um",
        4: "GRPH-ifs",
        5: "GRPH-kim",
        6: "GRPH-um",
        7: "PANG-ifs",
        8: "PANG-kim",
        9: "PANG-um"
    }

    # index별 색(1~9)을 안정적으로 만들기
    # tab10을 쓰면 1~9는 다르게 잘 나옴
    base = plt.get_cmap("tab10")
    colors = [base(i) for i in range(idx_max + 1)]  # 0 포함 (안 쓰는 용)
    cmap = ListedColormap(colors)
    bounds = np.arange(idx_min - 0.5, idx_max + 1.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    # 공통 배경(feature space)
    # xlim = (np.min(Z_space[:, 0]), np.max(Z_space[:, 0]))
    # ylim = (np.min(Z_space[:, 1]), np.max(Z_space[:, 1]))

    for r, m in enumerate(metrics):
        for c, L in enumerate(length_list):
            ax = axes[r, c]
            ax.scatter(Z_space[:, 0], Z_space[:, 1], s=4, alpha=0.15)  # 배경

            data = maps[(m, int(L))]
            XY = data["XY"]
            idx = data["idx"]

            if XY.shape[0] > 0:
                sc = ax.scatter(XY[:, 0], XY[:, 1], c=idx, cmap=cmap, norm=norm, s=20, alpha=0.95)

                if annotate:
                    # 너무 많으면 앞에서 일부만
                    n_anno = min(XY.shape[0], annotate_max)
                    for i in range(n_anno):
                        ax.text(XY[i, 0], XY[i, 1], str(idx[i]), fontsize=8)

            ax.set_title(f"{m.upper()} | L={int(L)}")
            # ax.set_xlim(*xlim)
            # ax.set_ylim(*ylim)
            # ax.set_xlabel("x-axis")
            # ax.set_ylabel("y-axis")
            ax.grid(alpha=0.2)

    # 공통 colorbar (best index)
    # 마지막 scatter 객체를 찾아서 colorbar 연결
    mappable = None
    for ax in axes.ravel()[::-1]:
        colls = ax.collections
        if len(colls) >= 2:  # 배경+오버레이
            mappable = colls[-1]
            break
    if mappable is not None:
        #cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
        #cbar.set_label("Best index (1-based)")
        make_index_legend(
            ax=axes[1, 3],          # (row=1, col=3) → rmse, L=48
            idx_name_map=IDX_NAME,
            cmap=cmap,
            norm=norm
        )

    if title:
        fig.suptitle(title, fontsize=16)

    plt.show()



def build_points_by_model_maps_global(
    records: list[dict],
    length_list=(12, 24, 36, 48),
    metrics=("dtw", "rmse"),
    n_models: int = 9,
    a_min: float = 0.05,
    a_max: float = 0.95,
    top_ratio: float | None = None,
):
    """
    조건(metric, L)별로 records 전체(날짜 전체, 모델 전체)의 score 분포에서
    global min/max를 계산한 뒤 alpha를 결정한다.

    return maps[(model_idx, metric, L)] = dict(
        XY=(M,2), alpha=(M,), score=(M,), date=(M,)
    )
    model_idx: 1..n_models (1-based)
    """
    if top_ratio is not None:
        if not (0.0 < float(top_ratio) < 1.0):
            raise ValueError("top_ratio must be in (0,1), e.g. 0.1 for top 10%")
        
    # ---------------------------
    # 1-pass: 조건별 score 수집
    # ---------------------------
    pool = {(m, int(L)): [] for m in metrics for L in length_list}

    for rec in records:
        pos = rec.get("pos2d", None)
        lengths = rec.get("lengths", {})
        if pos is None or len(pos) != 2:
            continue

        for L in length_list:
            Ls = str(int(L))
            if Ls not in lengths:
                continue
            block = lengths[Ls]
            scores_block = block.get("scores", {})
            if not isinstance(scores_block, dict):
                continue

            for m in metrics:
                scores = scores_block.get(m, None)
                if scores is None:
                    continue
                scores = np.asarray(scores, dtype=np.float64)
                if scores.shape[0] != n_models:
                    continue
                pool[(m, int(L))].append(scores)  # (n_models,)

    # global min/max per condition
    cond_minmax = {}
    for key, arrs in pool.items():
        if len(arrs) == 0:
            cond_minmax[key] = (np.nan, np.nan)
            continue
        all_scores = np.concatenate(arrs, axis=0)  # (num_records*n_models,)
        smin = float(np.nanmin(all_scores))
        smax = float(np.nanmax(all_scores))
        cond_minmax[key] = (smin, smax)

    # ---------------------------
    # 2-pass: maps 생성 (alpha는 global min/max 사용)
    # ---------------------------
    maps = {}
    for k in range(1, n_models + 1):
        for m in metrics:
            for L in length_list:
                maps[(k, m, int(L))] = {"XY": [], "alpha": [], "score": [], "date": []}

    for rec in records:
        td = rec.get("target_date", "")
        pos = rec.get("pos2d", None)
        lengths = rec.get("lengths", {})

        if pos is None or len(pos) != 2:
            continue
        x, y = float(pos[0]), float(pos[1])

        for L in length_list:
            Ls = str(int(L))
            if Ls not in lengths:
                continue

            block = lengths[Ls]
            scores_block = block.get("scores", {})
            if not isinstance(scores_block, dict):
                continue

            for m in metrics:
                scores = scores_block.get(m, None)
                if scores is None:
                    continue
                scores = np.asarray(scores, dtype=np.float64)
                if scores.shape[0] != n_models:
                    continue

                smin, smax = cond_minmax[(m, int(L))]

                for k in range(1, n_models + 1):
                    s = float(scores[k - 1])
                    a = _alpha_from_global_minmax(s, smin, smax, a_min=a_min, a_max=a_max)

                    key = (k, m, int(L))
                    maps[key]["XY"].append([x, y])
                    maps[key]["alpha"].append(a)
                    maps[key]["score"].append(s)
                    maps[key]["date"].append(td)

    # numpy 변환
    for key, v in maps.items():
        v["XY"] = np.asarray(v["XY"], dtype=np.float64) if len(v["XY"]) else np.zeros((0, 2), dtype=np.float64)
        v["alpha"] = np.asarray(v["alpha"], dtype=np.float64) if len(v["alpha"]) else np.zeros((0,), dtype=np.float64)
        v["score"] = np.asarray(v["score"], dtype=np.float64) if len(v["score"]) else np.zeros((0,), dtype=np.float64)
        v["date"] = np.asarray(v["date"], dtype=object) if len(v["date"]) else np.zeros((0,), dtype=object)

    # ---------------------------
    # 3-pass(선택): 조건별 alpha threshold 계산
    # ---------------------------
    cond_thr = None
    if top_ratio is not None:
        q = 100.0 * (1.0 - float(top_ratio))  # 예: top 10% -> 90 percentile
        cond_thr = {}
        for m in metrics:
            for L in length_list:
                # 모델 전체 alpha를 합쳐서 조건별 threshold 1개만 만듦(권장)
                alphas = []
                for k in range(1, n_models + 1):
                    a = maps[(k, m, int(L))]["alpha"]
                    if a.size > 0:
                        alphas.append(a)
                if len(alphas) == 0:
                    cond_thr[(m, int(L))] = np.nan
                else:
                    all_a = np.concatenate(alphas, axis=0)
                    cond_thr[(m, int(L))] = np.round(float(np.percentile(all_a, q)),4)
                    if cond_thr[(m, int(L))] > (1-top_ratio):
                        cond_thr[(m, int(L))] = 1-top_ratio


    # 필요하면 min/max도 함께 반환 (디버깅/리포트용)
    return maps, cond_minmax, cond_thr



def build_points_by_model_maps(
    records: list[dict],
    length_list=(12, 24, 36, 48),
    metrics=("dtw", "rmse"),
    n_models: int = 9,
    a_min: float = 0.05,
    a_max: float = 0.95,
):
    """
    return maps[(model_idx, metric, L)] = dict(
        XY=(M,2), alpha=(M,), score=(M,), date=(M,)
    )
    model_idx: 1..n_models (1-based)
    """
    maps = {}
    for k in range(1, n_models + 1):
        for m in metrics:
            for L in length_list:
                maps[(k, m, int(L))] = {"XY": [], "alpha": [], "score": [], "date": []}

    for rec in records:
        td = rec.get("target_date", "")
        pos = rec.get("pos2d", None)
        lengths = rec.get("lengths", {})

        if pos is None or len(pos) != 2:
            continue
        x, y = float(pos[0]), float(pos[1])

        for L in length_list:
            Ls = str(int(L))
            if Ls not in lengths:
                continue

            block = lengths[Ls]
            scores_block = block.get("scores", {})
            if not isinstance(scores_block, dict):
                continue

            for m in metrics:
                scores = scores_block.get(m, None)
                if scores is None:
                    continue
                scores = np.asarray(scores, dtype=np.float64)
                if scores.shape[0] != n_models:
                    # 모델 수가 9가 아닌 경우(데이터 이상) 스킵
                    continue

                # 각 모델 k(1..9)의 점을 각각의 모델-지도에 누적
                for k in range(1, n_models + 1):
                    s = float(scores[k - 1])
                    a = _score_to_alpha(s, scores, a_min=a_min, a_max=a_max)

                    key = (k, m, int(L))
                    maps[key]["XY"].append([x, y])
                    maps[key]["alpha"].append(a)
                    maps[key]["score"].append(s)
                    maps[key]["date"].append(td)

    # numpy 변환
    for key, v in maps.items():
        v["XY"] = np.asarray(v["XY"], dtype=np.float64) if len(v["XY"]) else np.zeros((0, 2), dtype=np.float64)
        v["alpha"] = np.asarray(v["alpha"], dtype=np.float64) if len(v["alpha"]) else np.zeros((0,), dtype=np.float64)
        v["score"] = np.asarray(v["score"], dtype=np.float64) if len(v["score"]) else np.zeros((0,), dtype=np.float64)
        v["date"] = np.asarray(v["date"], dtype=object) if len(v["date"]) else np.zeros((0,), dtype=object)

    return maps



def plot_model_8_maps(
    Z_space: np.ndarray,
    maps: dict,
    model_idx: int,
    alpha_thre: list[float],
    length_list=(12, 24, 36, 48),
    metrics=("dtw", "rmse"),
    title: str | None = None,
    point_size: int = 18,
):
    """
    2x4 figure:
      row0: metric=dtw, col=length_list
      row1: metric=rmse, col=length_list
    model_idx: 1..9
    """

    good_thre = 0.4642

    if model_idx not in IDX_NAME:
        raise ValueError("model_idx must be 1..9")

    # 모델 고정 색상(진한색). alpha만 데이터로 변하게.
    base = plt.get_cmap("tab10")
    #base_rgb = np.array(base((model_idx - 1) % 10)[:3], dtype=np.float64)
    RED_RGB = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    good_points = []
    alpha_by_cell = [[None for _ in range(4)] for _ in range(2)]

    for r, m in enumerate(metrics):
        for c, L in enumerate(length_list):
            ax = axes[r, c]
            ax.scatter(Z_space[:, 0], Z_space[:, 1], c='gray', s=4, alpha=0.12)  # 배경

            data = maps.get((model_idx, m, int(L)), None)
            if data is None:
                ax.set_title(f"{m.upper()} | L={int(L)}")
                ax.grid(alpha=0.2)
                continue

            XY = data["XY"]
            alpha = data["alpha"]
            score = data["score"]

            if XY.shape[0] > 0:
                # # 점마다 alpha 적용된 RGBA 만들기
                # rgba = np.zeros((XY.shape[0], 4), dtype=np.float64)
                # rgba[:, :3] = RED_RGB[None, :]
                # rgba[:, 3] = np.clip(alpha, 0.0, 1.0)
                alpha_clip = np.clip(alpha, 0.0, 1.0)

                alpha_by_cell[r][c] = alpha_clip  # ✅ 히스토그램용 저장

                #thr = alpha_thre #np.percentile(alpha_clip, 70.0)
                thr = alpha_thre[(m, int(L))]
                #thr = np.percentile(alpha_clip, alpha_thre)

                rgba = np.zeros((XY.shape[0], 4), dtype=np.float64)

                # 기본: red
                rgba[:, 0] = 1.0  # R
                rgba[:, 1] = 0.0  # G
                rgba[:, 2] = 0.0  # B

                # 상위 10%: green
                #mask_top = alpha_clip >= thr
                mask_top = score <= good_thre
                rgba[mask_top, 0] = 0.0
                rgba[mask_top, 1] = 1.0
                rgba[mask_top, 2] = 0.0

                # alpha는 그대로 score 강도
                rgba[:, 3] = alpha_clip

                ax.scatter(XY[:, 0], XY[:, 1], s=point_size, c=rgba, edgecolors="none")
                ax.scatter(XY[mask_top, 0], XY[mask_top, 1], s=point_size, 
                           c=rgba[mask_top], edgecolors="black", linewidths=0.3)

            ax.set_title(f"{m.upper()} | L={int(L)}, ({np.sum(mask_top)})")
            ax.grid(alpha=0.2)

            good_points.append(np.sum(mask_top))

            
    if title is None:
        title = f"Model {model_idx}: {IDX_NAME[model_idx]} (alpha = relative score strength)"
    fig.suptitle(title, fontsize=16)
    #plt.show()
    plt.waitforbuttonpress()

    # -----------------------
    # Figure 2: 2x4 HISTOGRAMS
    # -----------------------
    fig_hist, axes_hist = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    hist_bin_step=0.05
    bins = np.arange(0.0, 1.0 + hist_bin_step, hist_bin_step)  # 0.0..1.0 step 0.1

    # (1) 전체 subplot에서 최대 count 미리 계산
    global_max_count = 0
    for r in range(2):
        for c in range(4):
            a = alpha_by_cell[r][c]
            if a is not None and a.size > 0:
                counts, _ = np.histogram(a, bins=bins)
                global_max_count = max(global_max_count, counts.max())

    # 여유를 조금 주는 게 보기 좋음
    ymax = int(global_max_count * 1.1) if global_max_count > 0 else 1

    for r, m in enumerate(metrics):
        for c, L in enumerate(length_list):
            axh = axes_hist[r, c]
            a = alpha_by_cell[r][c]
            if a is None:
                a = np.array([], dtype=np.float64)

            thr = alpha_thre[(m, int(L))]

            if a.size > 0:
                axh.hist(a, bins=bins, edgecolor="black", rwidth=0.85, histtype="bar")
                axh.axvline(thr, linestyle="--", linewidth=1.5, label=f"thr={thr}")
                axh.set_xlim(0.0, 1.0)
                axh.legend(fontsize=9, loc="upper left")
            else:
                axh.text(0.5, 0.5, "no data", ha="center", va="center", transform=axh.transAxes)
                axh.set_xlim(0.0, 1.0)

            axh.set_title(f"{m.upper()} | L={int(L)}")
            axh.set_ylim(0, ymax)
            axh.set_xlabel("alpha")
            axh.set_ylabel("count")
            axh.grid(alpha=0.2)

    fig_hist.suptitle(f"Model {model_idx} alpha histograms (bin={hist_bin_step})", fontsize=16)

    #plt.show()
    plt.waitforbuttonpress()

    return good_points


def plot_two_models_8_maps_4classes_fixed_thr(
    Z_space: np.ndarray,
    maps: dict,
    modelA: int,
    modelB: int,
    cond_thr: list[float],
    length_list=(12, 24, 36, 48),
    metrics=("dtw", "rmse"),
    title: str | None = None,
    point_size: int = 18,    
    alpha_mode: str = "max",   # "max" | "mean" | "A" | "B"
    season_label: np.ndarray | None = None,   # 추가
):
    """
    2x4 figure:
      row0: metric=dtw, col=length_list
      row1: metric=rmse, col=length_list

    한 좌표(같은 날짜의 pos2d)에 대해 A/B의 alpha로 good 판정을 하고,
    4분류 색상으로 한 번에 표시:
      - both good: green
      - A only good: red
      - B only good: blue
      - both bad: gray

    good 판정은 기존 방식 반영: alpha_clip >= thr (thr 고정)
    """

    good_thre = 0.4642 # 90 percentile of shift data validation set

    # 4-class 색상 (RGB)
    RGB_AONLY = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # red
    RGB_BONLY = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # blue
    RGB_BOTH  = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # green
    RGB_BAD   = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # black

    def _combine_alpha(aA, aB):
        if alpha_mode == "max":
            return np.maximum(aA, aB)
        if alpha_mode == "mean":
            return 0.5 * (aA + aB)
        if alpha_mode == "A":
            return aA
        if alpha_mode == "B":
            return aB
        raise ValueError("alpha_mode must be one of: 'max','mean','A','B'")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    # 기존 코드 스타일: good_points 카운트 모으기
    good_points_A = []
    good_points_B = []
    good_points_both = []
    alpha_by_cell = [[{"A": None, "B": None} for _ in range(4)] for _ in range(2)]

    # 추가: 계절별 count 저장
    #season_counts_by_cell = [[None for _ in range(4)] for _ in range(2)]
    season_counts = {}

    for r, m in enumerate(metrics):
        for c, L in enumerate(length_list):
            ax = axes[r, c]
            ax.scatter(Z_space[:, 0], Z_space[:, 1], c="gray", s=4, alpha=0.12)

            dataA = maps.get((modelA, m, int(L)), None)
            dataB = maps.get((modelB, m, int(L)), None)

            if (dataA is None) or (dataB is None):
                ax.set_title(f"{m.upper()} | L={int(L)} (no data)")
                ax.grid(alpha=0.2)
                continue

            XY_A = dataA.get("XY", None)
            aA = dataA.get("alpha", None)
            sA = dataA.get("score", None)
            XY_B = dataB.get("XY", None)
            aB = dataB.get("alpha", None)
            sB = dataB.get("score", None)

            if XY_A is None or aA is None or XY_B is None or aB is None:
                ax.set_title(f"{m.upper()} | L={int(L)} (bad format)")
                ax.grid(alpha=0.2)
                continue

            # 좌표가 동일하다는 가정: XY는 A의 것을 사용
            XY = XY_A
            if XY.shape[0] == 0:
                ax.set_title(f"{m.upper()} | L={int(L)} (empty)")
                ax.grid(alpha=0.2)
                continue

            # alpha clip
            alphaA = np.clip(np.asarray(aA, dtype=np.float64), 0.0, 1.0)
            alphaB = np.clip(np.asarray(aB, dtype=np.float64), 0.0, 1.0)

            # (혹시 길이 다르면 안전하게 최소 길이로 자름)
            n = min(XY.shape[0], alphaA.shape[0], alphaB.shape[0])
            XY = XY[:n]
            alphaA = alphaA[:n]
            alphaB = alphaB[:n]

            # 히스토그램용 저장(원하면 나중에 바로 2x4 hist 뽑을 수 있음)
            alpha_by_cell[r][c]["A"] = alphaA
            alpha_by_cell[r][c]["B"] = alphaB

            # 기존 방식: thr 고정, mask_top = alpha>=thr
            thr = cond_thr[(m, int(L))]
            goodA = alphaA >= thr
            goodB = alphaB >= thr

            goodA = sA <= good_thre
            goodB = sB <= good_thre

            mask_both = goodA & goodB
            mask_Aonly = goodA & (~goodB)
            mask_Bonly = (~goodA) & goodB
            mask_bad = (~goodA) & (~goodB)

            # 추가: 계절별 카운트
            if season_label is not None:
                season_arr = np.asarray(season_label)[:n]

                cell_counts = {}
                for s in [1, 2, 3, 4]:
                    m_season = (season_arr == s)
                    cell_counts[s] = {
                        "both_good": int(np.sum(mask_both & m_season)),
                        "A_only": int(np.sum(mask_Aonly & m_season)),
                        "B_only": int(np.sum(mask_Bonly & m_season)),
                        "all_bad": int(np.sum(mask_bad & m_season)),
                    }
                season_counts[(m, int(L))] = cell_counts

            # 색 + alpha 결합
            alphaC = np.clip(_combine_alpha(alphaA, alphaB), 0.0, 1.0)

            rgba = np.zeros((n, 4), dtype=np.float64)
            rgba[mask_bad,  :3] = RGB_BAD[None, :]
            rgba[mask_Aonly,:3] = RGB_AONLY[None, :]
            rgba[mask_Bonly,:3] = RGB_BONLY[None, :]
            rgba[mask_both, :3] = RGB_BOTH[None, :]
            rgba[:, 3] = alphaC

            ax.scatter(XY[:, 0], XY[:, 1], s=point_size, c=rgba, edgecolors="none")
            # both good만 테두리 강조(가장 중요한 영역)
            ax.scatter(
                XY[mask_both, 0], XY[mask_both, 1],
                s=point_size * 1.25,
                c=rgba[mask_both],
                edgecolors="black",
                linewidths=0.35
            )

            # 기존 스타일: subplot 제목에 카운트 표시
            nA = int(np.sum(goodA))
            nB = int(np.sum(goodB))
            nBoth = int(np.sum(mask_both))

            if m == 'dtw':
                ax.set_title(f"Alignment-based | L={int(L)} (A={nA}, B={nB}, both={nBoth})")
            else:
                ax.set_title(f"Index-aligned | L={int(L)} (A={nA}, B={nB}, both={nBoth})")
            ax.grid(alpha=0.2)

            good_points_A.append(nA)
            good_points_B.append(nB)
            good_points_both.append(nBoth)

    # Legend (한 번만)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=RGB_BOTH,  markersize=10, label='both good'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=RGB_AONLY, markersize=10, label='A good only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=RGB_BONLY, markersize=10, label='B good only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=RGB_BAD,   markersize=10, label='both bad'),
    ]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=10)

    
        
    if title is None:
        title = f"A={convert_model_name(modelA)} vs B={convert_model_name(modelB)} | 4-class map (thr={0.5}, alpha={alpha_mode})"
    fig.suptitle(title, fontsize=16)

    plt.waitforbuttonpress()

    # 필요하면 카운트 반환 (기존 good_points처럼)
    return {
        "good_points_A": good_points_A,
        "good_points_B": good_points_B,
        "good_points_both": good_points_both,
        "alpha_by_cell": alpha_by_cell,
        "season_counts": season_counts,   # 추가
    }
    


def plot_two_models_8_hists(
    maps: dict,
    modelA: int,
    modelB: int,
    length_list=(12, 24, 36, 48),
    metrics=("dtw", "rmse"),
    thr_mode="top30",
    thr_fixed=0.9,
    bin_step=0.1,
    title=None,
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    bins = np.arange(0.0, 1.0 + bin_step, bin_step)

    def get_alpha(model, m, L):
        d = maps.get((model, m, int(L)))
        if d is None or d["alpha"].size == 0:
            return np.array([], dtype=np.float64)
        return np.clip(d["alpha"], 0.0, 1.0)

    def get_thr(alpha):
        if alpha.size == 0:
            return 1.0
        if thr_mode == "fixed":
            return float(thr_fixed)
        return float(np.percentile(alpha, 70.0))

    # ---- y축 전역 최대치 계산(8개 subplot 공통) ----
    global_max = 0
    for m in metrics:
        for L in length_list:
            aA = get_alpha(modelA, m, L)
            aB = get_alpha(modelB, m, L)
            if aA.size:
                global_max = max(global_max, np.histogram(aA, bins=bins)[0].max())
            if aB.size:
                global_max = max(global_max, np.histogram(aB, bins=bins)[0].max())
    ymax = int(global_max * 1.1) if global_max > 0 else 1

    # ---- plotting ----
    for r, m in enumerate(metrics):
        for c, L in enumerate(length_list):
            ax = axes[r, c]
            aA = get_alpha(modelA, m, L)
            aB = get_alpha(modelB, m, L)

            if aA.size:
                ax.hist(
                    aA, bins=bins,
                    rwidth=0.85,
                    histtype="bar",
                    edgecolor="black",
                    alpha=0.45,     # ✅ 겹쳐 보이게
                    color="red",
                    label="A"
                )
                ax.axvline(get_thr(aA), color="red", linestyle="--", linewidth=1.5)

            if aB.size:
                ax.hist(
                    aB, bins=bins,
                    rwidth=0.85,
                    histtype="bar",
                    edgecolor="black",
                    alpha=0.45,     # ✅ 겹쳐 보이게
                    color="blue",
                    label="B"
                )
                ax.axvline(get_thr(aB), color="blue", linestyle="--", linewidth=1.5)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0, ymax)
            ax.set_title(f"{m.upper()} | L={int(L)}")
            ax.set_xlabel("alpha")
            ax.set_ylabel("count")
            ax.grid(alpha=0.2)

    # legend는 한 번만
    axes[0, 0].legend(loc="upper left", fontsize=10)

    if title is None:
        title = f"Alpha histograms: Model A={convert_model_name(modelA)} vs Model B={convert_model_name(modelB)} (bin={bin_step}, shared y)"
    fig.suptitle(title, fontsize=16)
    plt.show()


def make_index_legend(ax, idx_name_map, cmap, norm):
    """
    ax에 index → name legend를 추가
    """
    handles = []
    for idx, name in idx_name_map.items():
        color = cmap(norm(idx))
        h = mlines.Line2D(
            [], [], 
            color=color, marker='o', linestyle='None',
            markersize=8,
            label=f"{idx}: {name}"
        )
        handles.append(h)

    ax.legend(
        handles=handles,
        title="Best index",
        loc="upper right",
        fontsize=9,
        title_fontsize=10,
        frameon=True
    )

def is_on_or_after(month, day, boundary):
    """현재 날짜가 boundary(월,일) 이상인지 비교"""
    b_m, b_d = boundary
    return (month > b_m) or (month == b_m and day >= b_d)

def date_to_season(t):
    """
    최적 탐색된 season boundary를 기준으로 계절 id 반환
    1: DJF (겨울)
    2: MAM (봄)
    3: JJA (여름)
    4: SON (가을)
    """
    # 계절 경계 (기후학적 연도 기준)
    SPRING_START = (3, 1) #(3, 7)   # 03-07 (3, 1) #
    SUMMER_START = (6, 1) #(6, 13)  # 06-13 (6, 1) #
    AUTUMN_START = (9, 1) # (9, 19)  # 09-16 (9, 1) #
    WINTER_START = (12, 1) #(11, 23) # 11-22 (12, 1) #

    m, d = t.month, t.day
    # 봄 시작 ~ 여름 시작 전
    if is_on_or_after(m, d, SPRING_START) and not is_on_or_after(m, d, SUMMER_START):
        return 2  # MAM
    # 여름 시작 ~ 가을 시작 전
    if is_on_or_after(m, d, SUMMER_START) and not is_on_or_after(m, d, AUTUMN_START):
        return 3  # JJA
    # 가을 시작 ~ 겨울 시작 전
    if is_on_or_after(m, d, AUTUMN_START) and not is_on_or_after(m, d, WINTER_START):
        return 4  # SON
    # 나머지 = 겨울 (11/22 이후 또는 03/07 이전)
    return 1  # DJF

def set_season_npy():
    # 실제 라벨 생성
    N = 7308
    #from datetime import datetime, timedelta
    start = datetime(2020, 1, 1, 0, 0)

    labels = []
    for i in range(N):
        t = start + timedelta(hours=6 * i)
        season = date_to_season(t)
        labels.append(season)

    labels = np.array(labels)  # shape: (7308,)
    np.save('season_label_regular.npy', labels)

def set_season_npy_365():
    # 실제 라벨 생성
    N = 365
    #from datetime import datetime, timedelta
    start = datetime(2025, 1, 1, 0, 0)

    labels = []
    for i in range(N):
        t = start + timedelta(hours=24 * i)
        season = date_to_season(t)
        labels.append(season)

    labels = np.array(labels)  # shape: (365,)
    np.save('season_label_365.npy', labels)

def validate_season_correspondence():

    set_season_npy()
    label_path = "season_label.npy"
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"
    feature_paths = [f"model_all/{tag_name}/feature_space_comparison/without_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_PML_era5.npy"]
    
    for umap_path in feature_paths:
        plot_2d_points_from_npy(umap_path, label_path,
            title="2D coords + class", save_path="coords_with_class.png")
        
    plt.show()

def plot_2d_points_from_npy(points_path, labels_path, **kwargs):
    """
    npy 파일에서 바로 불러서 그리는 래퍼 함수
    points_path: 2D 좌표가 들어있는 npy (shape: (N, 2))
    labels_path: 0~35 class 레이블 npy (shape: (N,))
    """
    points = np.load(points_path)
    labels = np.load(labels_path)
    plot_2d_points_with_classes(points, labels, **kwargs)

def plot_2d_points_with_classes(points, classes,
                                title=None,
                                figsize=(7, 7),
                                save_path=None,
                                show=True):
    """
    points: (N, 2) 배열 (2차원 좌표)
    classes: (N,) 배열, 0 ~ 35 사이의 정수 레이블
    """
    points = np.asarray(points)
    classes = np.asarray(classes)

    assert points.shape[0] == classes.shape[0], "좌표 개수와 레이블 개수가 다릅니다."
    assert points.shape[1] == 2, "points는 (N, 2) 형태여야 합니다."

    fig, ax = plt.subplots(figsize=figsize)

    # N개 class를 표현하기 위한 colormap
    uniq = np.unique(classes)
    nClass = len(uniq)
    #cmap = plt.get_cmap('tab20', nClass)  # tab20을 36개로 쪼개서 사용
    cmap = plt.get_cmap("nipy_spectral", nClass)     # 100개 색상 팔레트


    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=classes,
        cmap=cmap,
        s=1,
        alpha=0.8
    )

    # 색 -> class 번호를 보여주는 colorbar
    cbar = fig.colorbar(sc, ticks=np.arange(0, nClass, 10))
    cbar.ax.set_ylabel("class")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        #plt.show()
        plt.pause(0.1)
    #plt.close(fig)


def plot_umap_manual_multi(
        single_umap_paths,
        combined_umap_paths,  # str 또는 list[str] (여러 개 비교 가능)
        index_order=None,  # None이면 0..T-1 또는 (group_ids 있을 때) 0..(T_per_group-1)
        trail_len=10,
        pause_sec=0.1,
        bg_alpha=0.15,
        point_size=36,
        start_idx=0,  # 시작 frame을 지정, index_order 기준
        feature_labels=None,  # 예: ["z","r","t","cc"] 등. None이면 파일명에서 추출
        combined_labels=None,  # 예: ["post","flow","alt"] 등. None이면 파일명에서 추출
        group_ids=None,
):
    """
    왼쪽: 단일 UMAP들을 N개 행으로 배치 (0개일 수도 있음)
    오른쪽: 통합 UMAP들을 열로 배치(각 열은 모든 행을 세로 병합하여 큰 패널)

    Args:
        single_umap_paths (list[str]): 길이 >=0, 각 (T_total, 2)
        combined_umap_paths (str|list[str]): 1개 이상, 각 (T_total, 2)
        group_ids (array-like|None): 길이 T_total, 각 행이 속한 그룹 ID (1,2,3,...)
            - None: 기존 방식 유지 (한 번에 하나의 인덱스만 움직임)
            - not None: 각 그룹이 같은 길이의 시계열이라고 가정하고,
            같은 시간 인덱스(랭크)에 해당하는 모든 그룹의 점을 동시에 표시
    """

    # --- 입력 정규화 ---
    if isinstance(single_umap_paths, (str, os.PathLike)):
        single_umap_paths = [str(single_umap_paths)]
    if not isinstance(single_umap_paths, (list, tuple)):
        raise ValueError("single_umap_paths는 리스트 또는 튜플이어야 합니다.")
    if isinstance(combined_umap_paths, (str, os.PathLike)):
        combined_umap_paths = [str(combined_umap_paths)]
    if not isinstance(combined_umap_paths, (list, tuple)) or len(combined_umap_paths) < 1:
        raise ValueError("combined_umap_paths는 1개 이상의 경로를 포함해야 합니다.")

    # --- 데이터 로드 ---
    singles = [np.load(p) for p in single_umap_paths]
    combos = [np.load(p) for p in combined_umap_paths]

    # --- 검증: shape (T,2) & 동일 T ---
    T_list = [arr.shape[0] for arr in combos]  # 최소 1개 있음
    if len(singles) > 0:
        T_list.extend([arr.shape[0] for arr in singles])
    if len(set(T_list)) != 1:
        raise ValueError(f"모든 UMAP의 길이(T)가 같아야 합니다. 현재 길이들: {T_list}")
    if any(arr.shape[1] != 2 for arr in singles + combos):
        raise ValueError("각 UMAP 배열의 shape는 (T, 2) 여야 합니다.")
    T_total = T_list[0]

    # ------------------------------------------------------------------
    # group_ids 전처리
    # ------------------------------------------------------------------
    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if group_ids.shape[0] != T_total:
            raise ValueError(
                f"group_ids 길이({group_ids.shape[0]})가 UMAP 길이(T_total={T_total})와 다릅니다."
            )
        unique_groups = np.unique(group_ids)
        G = len(unique_groups)

        # 각 그룹별 인덱스 목록
        group_index_lists = []
        for g in unique_groups:
            idx_g = np.where(group_ids == g)[0]
            idx_g.sort()
            group_index_lists.append(idx_g)

        # 모든 그룹 길이가 같다고 가정 (필요하면 relax 가능)
        lens = [len(idx_g) for idx_g in group_index_lists]
        if len(set(lens)) != 1:
            raise ValueError(
                f"모든 그룹이 같은 길이의 시계열이라 가정합니다. 현재 길이들: {lens}"
            )
        T_per_group = lens[0]  # 한 그룹당 시계열 길이

        # 그룹별 색상 정의
        cmap = plt.get_cmap("tab10")
        group_colors = [cmap(i % cmap.N) for i in range(G)]

        # 각 row가 어느 그룹에 속하는지 기록 (0..G-1)
        group_idx_of_row = np.zeros(T_total, dtype=int)
        for j, g in enumerate(unique_groups):
            group_idx_of_row[group_ids == g] = j

        # 시간 랭크(0..T_per_group-1) -> 프레임 순서
        if index_order is None:
            index_order = np.arange(T_per_group, dtype=int)
        else:
            index_order = np.asarray(index_order, dtype=int)
            if index_order.ndim != 1:
                raise ValueError("index_order는 1차원 정수 배열이어야 합니다.")
            if np.any((index_order < 0) | (index_order >= T_per_group)):
                raise ValueError(
                    f"group_ids가 있을 때 index_order 값은 0~{T_per_group-1} 범위여야 합니다."
                )

        num_steps = len(index_order)

        # start_idx 체크는 프레임 개수 기준
        if not (0 <= start_idx < num_steps):
            raise ValueError(f"start_idx는 0~{num_steps - 1} 범위여야 합니다.")

    else:
        # ------------------------------------------------------------------
        # 기존 방식: group_ids가 없을 때 index_order는 행 인덱스
        # ------------------------------------------------------------------
        if index_order is None:
            index_order = np.arange(T_total, dtype=int)
        else:
            index_order = np.asarray(index_order, dtype=int)
            if index_order.ndim != 1:
                raise ValueError("index_order는 1차원 정수 배열이어야 합니다.")
            if np.any((index_order < 0) | (index_order >= T_total)):
                raise ValueError("index_order에 유효 범위를 벗어난 인덱스가 있습니다.")

        if not (0 <= start_idx < len(index_order)):
            raise ValueError(f"start_idx는 0~{len(index_order) - 1} 범위여야 합니다.")

    # --- 전역 TRAIL_LEN ---
    global TRAIL_LEN
    TRAIL_LEN = int(trail_len)

    # --- 라벨 추출 ---
    def infer_label(p):
        name = os.path.splitext(os.path.basename(p))[0]
        # 1️⃣ "umap_2d_z_global" → z
        m = re.search(r'umap[_\-]?\d*d[_\-]?([A-Za-z0-9]+?)(?:[_\-](?:global|local|combined|all))?$', name)
        if m:
            return m.group(1)
        # 2️⃣ "umap_z_global" → z
        m2 = re.search(r'umap[_\-]?([A-Za-z0-9]+?)(?:[_\-](?:global|local|combined|all))?$', name)
        if m2:
            return m2.group(1)
        # 3️⃣ fallback: 마지막 토큰
        m3 = re.search(r'[_\-]([A-Za-z0-9]+)$', name)
        if m3:
            return m3.group(1)
        return name

    N_single = len(singles)
    M = len(combos)

    if feature_labels is None and N_single > 0:
        feature_labels = [infer_label(p) for p in single_umap_paths]
    if N_single > 0 and len(feature_labels) != N_single:
        raise ValueError(f"feature_labels 길이({len(feature_labels)})가 single_umap_paths 길이({N_single})와 같아야 합니다.")
    if combined_labels is None:
        combined_labels = [infer_label(p) for p in combined_umap_paths]
    if len(combined_labels) != M:
        raise ValueError("combined_labels 길이는 combined_umap_paths 길이와 같아야 합니다.")

    # --- Figure 레이아웃 ---
    if N_single > 0:
        fig_w = 6 + 6 * M
        fig_h = max(6, 2.2 * N_single)
        width_ratios = [1.0] + [2.0] * M
        height_ratios = [1.0] * N_single
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(
            N_single, 1 + M,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=0.08, hspace=0.18
        )
        axs_left = [fig.add_subplot(gs[i, 0]) for i in range(N_single)]
        axs_right = [fig.add_subplot(gs[:, j]) for j in range(1, 1 + M)]
    else:
        # single이 0개면 오른쪽 패널만
        fig_w = 6 * M
        fig_h = 6 #8
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(1, M, wspace=0.08)
        axs_left = []
        axs_right = [fig.add_subplot(gs[0, j]) for j in range(M)]


    # group_ids가 있을 때 그룹 범례 추가
    if group_ids is not None:
        legend_axes = []
        if len(axs_left) > 0:
            legend_axes.append(axs_left[0])
        if len(axs_right) > 0:
            legend_axes.append(axs_right[0])

        handles = [
            Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markerfacecolor=group_colors[i],
                markeredgecolor='k',
                markersize=6,
                label=f"group {g}"
            )
            for i, g in enumerate(unique_groups)
        ]

        for ax in legend_axes:
            ax.legend(handles=handles, loc='upper right', fontsize=8, frameon=False)    

    # --- 패널 초기화 ---
    def init_panel(ax, data, title):
        ax.scatter(data[:, 0], data[:, 1], s=6, alpha=bg_alpha, linewidths=0)
        ax.set_title(title, fontsize=12)
        xmin, ymin = data.min(axis=0)
        xmax, ymax = data.max(axis=0)
        mx = 0.05 * max(1e-8, xmax - xmin)
        my = 0.05 * max(1e-8, ymax - ymin)
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)
        ax.set_xticks([])
        ax.set_yticks([])
        scat_cur = ax.scatter([], [], s=point_size, c=[[0.1, 0.7, 1.0, 1.0]],
                            edgecolors='k', linewidths=0.6, zorder=3)
        scat_prev = ax.scatter([], [], s=point_size, c=[[0.3, 0.3, 0.3, 0.8]],
                            edgecolors='none', zorder=2)
        scat_trail = ax.scatter([], [], s=int(point_size * 0.6),
                                edgecolors='none', zorder=1)
        line_traj, = ax.plot([], [], lw=1.6, alpha=0.95, color='orange', zorder=0)

        if group_ids is None:
            # 기존 단일 그룹 모드: 한 색의 Line2D
            line_traj, = ax.plot([], [], lw=1.6, alpha=0.95, color='orange', zorder=0)
        else:
            # 여러 그룹 모드: 그룹별 색상을 가지는 LineCollection
            line_traj = LineCollection([], linewidths=1.4, alpha=0.9, zorder=0)
            ax.add_collection(line_traj)

        return scat_cur, scat_prev, scat_trail, line_traj
    
    def index_to_datetime_str(idx: int, fmt: str = "%Y-%m-%d %H:%M"):
        """
        index=0 → 2020-01-01 00:00
        index=1 → 2020-01-01 06:00
        ...
        index n → 2020-01-01 00:00 + 6*n hours
        """
        base = datetime(2020, 1, 1, 0, 0)
        dt = base + timedelta(hours=6 * idx)
        return dt.strftime(fmt)

    left_artists = []
    if N_single > 0:
        left_titles = [f"UMAP #{i + 1} ({lbl})" for i, lbl in enumerate(feature_labels)]
        left_artists = [init_panel(ax, data, ttl) for ax, data, ttl in zip(axs_left, singles, left_titles)]

    right_artists = []
    for ax, combo, lbl in zip(axs_right, combos, combined_labels):
        right_artists.append(init_panel(ax, combo, f"Combined UMAP ({lbl})"))

    
    # --- deque 준비 ---
    if group_ids is not None:
        # 패널(좌/우) × 그룹 수 만큼 deque를 둔다.
        left_deques = [[deque(maxlen=TRAIL_LEN) for _ in unique_groups] for _ in range(N_single)]
        right_deques = [[deque(maxlen=TRAIL_LEN) for _ in unique_groups] for _ in range(M)]
    else:
        # 기존 단일 그룹 모드
        left_deques = [deque(maxlen=TRAIL_LEN) for _ in range(N_single)]
        right_deques = [deque(maxlen=TRAIL_LEN) for _ in range(M)]


    # ------------------------------------------------------------------
    # 여러 인덱스를 동시에 업데이트하는 헬퍼 (group_ids 있을 때 사용)
    # idx_list: unique_groups 순서와 align된 전역 인덱스 리스트
    # group_deques: [deque(그룹1), deque(그룹2), ...]
    # ------------------------------------------------------------------
    def update_panel_multi(sc_cur, sc_prev, sc_tr, ln, arr, idx_list, group_deques):
        idx_list = np.asarray(idx_list, dtype=int)
        if idx_list.size == 0:
            return

        # 1) 그룹별 deque에 현재 인덱스를 추가하고,
        #    모든 group의 과거 인덱스를 한 리스트로 합침
        all_trail_indices = []
        for j, idx in enumerate(idx_list):
            dq = group_deques[j]      # j번째 그룹의 deque
            dq.append(int(idx))
            all_trail_indices.extend(dq)

        # 2) 현재 위치: 모든 그룹의 현재 위치를 한 번에 표시
        cur_xy = arr[idx_list, :]
        sc_cur.set_offsets(cur_xy)
        # 현재 점 색을 그룹별 색으로
        colors_cur = np.array([group_colors[group_idx_of_row[idx]]
                                for idx in idx_list])
        sc_cur.set_facecolors(colors_cur)

        # 3) tail scatter (과거 점들)
        if len(all_trail_indices) > 0:
            all_trail_indices = np.asarray(all_trail_indices, dtype=int)
            trail_xy = arr[all_trail_indices, :]

            sc_tr.set_offsets(trail_xy)
            sc_prev.set_offsets(trail_xy)

            trail_colors = []
            prev_colors = []
            for idx in all_trail_indices:
                g_idx = group_idx_of_row[idx]
                base = np.array(group_colors[g_idx])
                # trail: 옅은 색
                c_tr = base.copy()
                c_tr[3] = 0.25
                # prev: 좀 더 진한 색
                c_prev = base.copy()
                c_prev[3] = 0.6
                trail_colors.append(c_tr)
                prev_colors.append(c_prev)

            sc_tr.set_facecolors(trail_colors)
            sc_prev.set_facecolors(prev_colors)

            # 4) 궤적 선: 그룹별로 색을 다르게, LineCollection에 segment 넣기
            segs = []
            seg_colors = []
            for j, dq in enumerate(group_deques):
                if len(dq) < 2:
                    continue
                pts = arr[list(dq), :]     # 이 그룹의 궤적 좌표 (시간순)
                # (n-1, 2, 2) 형태의 segments 생성
                segs_j = np.stack([pts[:-1], pts[1:]], axis=1)
                segs.append(segs_j)
                # 이 그룹의 모든 segment에 동일 그룹 색
                seg_colors.extend([group_colors[j]] * (len(dq) - 1))

            if len(segs) > 0:
                segs = np.concatenate(segs, axis=0)
                ln.set_segments(segs)
                ln.set_color(seg_colors)
            else:
                ln.set_segments([])
        else:
            sc_tr.set_offsets([])
            sc_prev.set_offsets([])
            ln.set_segments([])


    # --- 초기 프레임 (start_idx 반영) ---
    if group_ids is not None:
        if len(index_order) > 0:
            rank = int(index_order[start_idx])
            # 같은 시간 랭크에 해당하는 모든 그룹의 실제 행 인덱스
            idx_multi = [idx_g[rank] for idx_g in group_index_lists]  # len = #groups

            for (sc_cur, sc_prev, sc_tr, ln), arr, group_dqs in zip(left_artists, singles, left_deques):
                update_panel_multi(sc_cur, sc_prev, sc_tr, ln, arr, idx_multi, group_dqs)
            for artists, arr, group_dqs in zip(right_artists, combos, right_deques):
                update_panel_multi(*artists, arr, idx_multi, group_dqs)

            fig.suptitle(
                f"Step {start_idx + 1}/{len(index_order)} | Rank: {rank}, {index_to_datetime_str(rank)} | Groups: {len(group_index_lists)}",
                fontsize=13
            )
            plt.pause(0.01)
    else:
        # 기존 단일 인덱스 방식 그대로
        if len(index_order) > 0:
            init_idx = int(index_order[start_idx])
            for (sc_cur, sc_prev, sc_tr, ln), arr, dq in zip(left_artists, singles, left_deques):
                update_umap_marker(sc_cur, sc_prev, sc_tr, ln, arr, init_idx, dq)
            for artists, arr, dq in zip(right_artists, combos, right_deques):
                update_umap_marker(*artists, arr, init_idx, dq)
            fig.suptitle(f"Step {start_idx + 1}/{len(index_order)} | Index: {init_idx}, {index_to_datetime_str(init_idx)}", fontsize=13)
            plt.pause(0.01)


    # --- 수동 루프 ---
    step = start_idx
    try:
        while plt.fignum_exists(fig.number):
            if step >= len(index_order):
                break

            if group_ids is not None:
                # 시간 랭크 기반 multi 업데이트
                rank = int(index_order[step])
                idx_multi = [idx_g[rank] for idx_g in group_index_lists]

                for (sc_cur, sc_prev, sc_tr, ln), arr, group_dqs in zip(left_artists, singles, left_deques):
                    update_panel_multi(sc_cur, sc_prev, sc_tr, ln, arr, idx_multi, group_dqs)
                for artists, arr, group_dqs in zip(right_artists, combos, right_deques):
                    update_panel_multi(*artists, arr, idx_multi, group_dqs)

                fig.suptitle(
                    f"Step {step + 1}/{len(index_order)} | Rank: {rank}, {index_to_datetime_str(rank)} | Groups: {len(group_index_lists)}",
                    fontsize=13
                )
            else:
                # 기존 단일 인덱스 기반 업데이트
                idx = int(index_order[step])

                for (sc_cur, sc_prev, sc_tr, ln), arr, dq in zip(left_artists, singles, left_deques):
                    update_umap_marker(sc_cur, sc_prev, sc_tr, ln, arr, idx, dq)
                for artists, arr, dq in zip(right_artists, combos, right_deques):
                    update_umap_marker(*artists, arr, idx, dq)

                fig.suptitle(f"Step {step + 1}/{len(index_order)} | Index: {idx}, {index_to_datetime_str(idx)}", fontsize=13)

            if pause_sec == 0:
                plt.waitforbuttonpress()
            else:
                plt.pause(pause_sec)
            if not plt.fignum_exists(fig.number):
                break

            step += 1
    except KeyboardInterrupt:
        pass


    return fig

## ============================= ##
## 다중 umap - combined plot 표시
#################################
def update_umap_marker(scat_cur, scat_prev, scat_trail, line_traj,
                        umap_coords, idx, trail_deque):
    """현재/이전/꼬리/경로 동기화"""
    xy_cur = umap_coords[idx:idx + 1, :]
    scat_cur.set_offsets(xy_cur)

    prev_idx = idx - 1 if idx > 0 else 0
    xy_prev = umap_coords[prev_idx:prev_idx + 1, :]
    scat_prev.set_offsets(xy_prev)

    trail_deque.append(idx)
    while len(trail_deque) > TRAIL_LEN:
        trail_deque.popleft()

    trail_pts = umap_coords[np.array(trail_deque, dtype=int), :]
    scat_trail.set_offsets(trail_pts)

    # 오래된→최근 점 순서로 알파 증가
    k = len(trail_deque)
    alphas = np.linspace(0.25, 0.85, k)
    facecolors = np.tile(np.array([[1.0, 0.55, 0.0, 1.0]]), (k, 1))
    facecolors[:, 3] = alphas
    scat_trail.set_facecolors(facecolors)

    line_traj.set_data(trail_pts[:, 0], trail_pts[:, 1])

def plot_distance_for_valid_time_comparison(tag_name, folder_target_date, model_list=None, bShow=True):

    ## z_ai path comparison    
    data_dir = f"out_test_features/{tag_name}/{folder_target_date}"
    Z_ai_list = []    
    for a in range(9):
        z_ai_npy = os.path.join(data_dir, f"final/ai_{a+1:02d}_manifold.npy")
        Z_ai = np.load(z_ai_npy)
        Z_ai_list.append(Z_ai)
    # -------------------------------------------------
    # 2. 하나의 그래프에 9개 distance plot
    # -------------------------------------------------
    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    # model_list가 없으면 전체 사용
    if model_list is None:
        model_list = list(range(1, len(Z_ai_list) + 1))

    if not bShow:
        T = Z_ai_list[0].shape[0]
        # lead time (예: 48, 42, ... , 0)
        lead_days = np.arange(-12, 1)
        dist_list = []
        #for i, Z_ai in enumerate(Z_ai_list):
        for i in model_list:
            Z_ai = Z_ai_list[i-1]
            ref = Z_ai[0]
            
            #dist = np.linalg.norm(Z_ai - ref, axis=1)
            dist = np.linalg.norm(Z_ai - ref, axis=1)[::-1]
            dist_list.append(dist)

    else:
        plt.figure(10, figsize=(7,3))
        plt.clf()
        
        T = Z_ai_list[0].shape[0]

        # lead time (예: 48, 42, ... , 0)
        lead_days = np.arange(-12, 1)
        dist_list = []
        #for i, Z_ai in enumerate(Z_ai_list):
        for i in model_list:
            Z_ai = Z_ai_list[i-1]
            ref = Z_ai[0]
            
            #dist = np.linalg.norm(Z_ai - ref, axis=1)
            dist = np.linalg.norm(Z_ai - ref, axis=1)[::-1]
            dist_list.append(dist)

            plt.plot(
                lead_days,
                dist,
                marker='o',
                markersize=3,
                linewidth=1,
                label=f"{model_names[i-1]}"
            )

        plt.xlabel("Days Before Target Date")
        plt.ylabel("Distance from Z_ai[0]")
        plt.title("Distance from Initial Point in Feature Space")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.pause(0.1)

        mgr = plt.get_current_fig_manager()
        try:
            mgr.window.move(50, 650)
            mgr.window.resize(700, 300)
        except Exception:
            try:
                mgr.window.wm_geometry("700x300+50+650")
            except Exception:
                pass

        plt.show()
    

    return dist_list


def plot_final_window_boxplots(
    total_dist_list,
    model_names=None,
    valid_times=[-1, -6, -12],
    figsize=(18, 5),
    showfliers=False,
    title_prefix="Valid-time distance distribution",
    drop_last_zero=True,
):
    """
    total_dist_list shape:
        (n_cases, n_models, n_leads)
        예: (352, 9, 13)

    가정:
        마지막 lead 값이 모두 0이면 drop_last_zero=True로 제외
    """

    arr = np.asarray(total_dist_list, dtype=float)

    if arr.ndim != 3:
        raise ValueError(f"3차원 배열이어야 합니다. 현재 shape={arr.shape}")

    n_cases, n_models, n_leads = arr.shape

    if model_names is None:
        model_names = [f"model_{i+1}" for i in range(n_models)]

    if len(model_names) != n_models:
        raise ValueError(
            f"model_names 길이({len(model_names)})가 모델 개수({n_models})와 다릅니다."
        )

    # 마지막 lead가 모두 0이면 제외
    if drop_last_zero:
        arr = arr[:, :, :-1]   # (cases, models, leads-1)

    n_cases, n_models, n_leads = arr.shape

    if n_leads < 3:
        raise ValueError("유효한 lead time이 최소 3개 이상 필요합니다.")

    # (cases, models, leads) -> (models, leads, cases)
    arr = np.transpose(arr, (1, 2, 0))   # shape: (n_models, n_leads, n_cases)

    # 마지막 1/2/3 step 요약
    final_1 = arr[:, valid_times[0], :]              # (n_models, n_cases)
    final_2 = arr[:, valid_times[1]:, :].mean(axis=1)
    final_3 = arr[:, valid_times[2]:, :].mean(axis=1)

    panel_data = [final_1, final_2, final_3]
    panel_titles = [
        f"Final {-valid_times[0]}-step",
        f"Final {-valid_times[1]}-step mean",
        f"Final {-valid_times[2]}-step mean",
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    group_colors = []
    for i in range(n_models):
        if i < 3:
            group_colors.append(color_cycle[0])  # fnet
        elif i < 6:
            group_colors.append(color_cycle[1])  # grph
        else:
            group_colors.append(color_cycle[2])  # pang

    for ax, data, panel_title in zip(axes, panel_data, panel_titles):
        bp = ax.boxplot(
            [data[i] for i in range(n_models)],
            tick_labels=model_names,
            patch_artist=True,
            showfliers=showfliers,
            medianprops=dict(linewidth=1.5, color="black"),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=1.0),
        )

        for patch, c in zip(bp["boxes"], group_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)

        ax.set_title(panel_title)
        ax.set_ylabel("Distance")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(title_prefix, fontsize=13)
    plt.show()

    return fig, axes

#####
## 일기도 그리기: 850hPa
#####
def to_2d_lonlat(lon, lat):
    """
    lon, lat가 1D이면 meshgrid로 2D 생성
    이미 2D이면 그대로 반환
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lon.ndim == 2 and lat.ndim == 2:
        lon2d, lat2d = lon, lat
    else:
        raise ValueError("lon, lat는 둘 다 1D이거나 둘 다 2D여야 합니다.")
    return lon2d, lat2d

# def to_2d_lonlat(lon, lat):
#     """
#     lon, lat가 1D이면 meshgrid로 2D 생성
#     이미 2D이면 그대로 반환
#     """
#     lon = np.asarray(lon)
#     lat = np.asarray(lat)

#     if lon.ndim == 1 and lat.ndim == 1:
#         lon2d, lat2d = np.meshgrid(lon, lat)
#     elif lon.ndim == 2 and lat.ndim == 2:
#         lon2d, lat2d = lon, lat
#     else:
#         raise ValueError("lon, lat는 둘 다 1D이거나 둘 다 2D여야 합니다.")

#     return lon2d, lat2d

def convert_geopotential_to_height(z):
    """
    z가 geopotential(m^2/s^2)이면 geopotential height(m)로 변환.
    이미 값 범위가 height처럼 보이면 그대로 사용.
    """
    z = np.asarray(z)
    # 경험적 판별:
    # 850hPa height는 대체로 1000~1800 m 수준
    # geopotential은 대체로 10000~20000 m^2/s^2 수준
    if np.nanmean(z) > 3000:
        return z / 9.80665
    return z

# def convert_geopotential_to_height(z):
#     """
#     geopotential [m^2/s^2] -> geopotential height [m]
#     """
#     return np.asarray(z) / 9.80665

def potential_temperature(t_k, p_hpa=850.0):
    """
    잠재온위(theta, K)
    """
    Rd_cp = 0.286  # R_d / c_p
    return t_k * (1000.0 / p_hpa) ** Rd_cp

# def potential_temperature(t, p_hpa=850.0):
#     """
#     t: Kelvin
#     """
#     kappa = 0.286
#     return np.asarray(t) * (1000.0 / p_hpa) ** kappa


def _draw_850hpa_on_ax(
    ax,
    lon2d,
    lat2d,
    t,
    z,
    u,
    v,
    use_theta=False,
    barb_skip=8,
    title=None,
    shaded_levels=None,
    hgt_levels=None,
    cmap="turbo"
):
    """
    하나의 ax에 850hPa chart를 그림.
    colorbar는 여기서 만들지 않음.
    return: contourf handle (shared colorbar용)
    """
    t = np.asarray(t)
    z = np.asarray(z)
    u = np.asarray(u)
    v = np.asarray(v)

    hgt = convert_geopotential_to_height(z)

    if use_theta:
        shaded = potential_temperature(t, p_hpa=850.0)
    else:
        shaded = t - 273.15

    cf = ax.contourf(
        lon2d,
        lat2d,
        shaded,
        levels=shaded_levels,
        cmap=cmap,
        extend="both",
        #transform=ccrs.PlateCarree()
    )

    cs = ax.contour(
        lon2d,
        lat2d,
        hgt,
        levels=hgt_levels,
        colors="k",
        linewidths=1.2, #0.9,
        #transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, fmt="%d", fontsize=7)

    ax.barbs(
        lon2d[::barb_skip, ::barb_skip],
        lat2d[::barb_skip, ::barb_skip],
        u[::barb_skip, ::barb_skip],
        v[::barb_skip, ::barb_skip],
        length=6,
        linewidth=0.8,
        #transform=ccrs.PlateCarree()
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title if title is not None else "")

    return cf

def plot_850hpa_panel(
    lon,
    lat,
    field_list,
    title_list,
    use_theta=False,
    barb_skip=8,
    figsize=(14, 10),
    fig_num=2,
    ncols=2
):
    """
    gt + 여러 model을 하나의 figure에 subplot으로 표시

    Parameters
    ----------
    lon, lat : 1D or 2D
    field_list : list of dict
        각 원소는 {"t":..., "z":..., "u":..., "v":...} 형태
    title_list : list of str
        각 패널 제목
    use_theta : bool
    barb_skip : int
    figsize : tuple
    fig_num : int
    ncols : int
    """
    n_panels = len(field_list)
    nrows = math.ceil(n_panels / ncols)

    lon2d, lat2d = to_2d_lonlat(lon, lat)

    # -----------------------------
    # 공통 shading range 계산
    # -----------------------------
    shaded_all = []
    hgt_all = []

    for fields in field_list:
        t = np.asarray(fields["t"])
        z = np.asarray(fields["z"])

        if use_theta:
            shaded = potential_temperature(t, p_hpa=850.0)
        else:
            shaded = t - 273.15

        hgt = convert_geopotential_to_height(z)

        shaded_all.append(shaded)
        hgt_all.append(hgt)

    shaded_min = min(np.nanmin(x) for x in shaded_all)
    shaded_max = max(np.nanmax(x) for x in shaded_all)

    # 너무 촘촘하지 않게 2단위 간격
    vmin = np.floor(shaded_min / 2) * 2
    vmax = np.ceil(shaded_max / 2) * 2
    shaded_levels = np.arange(vmin, vmax + 2, 2)

    hmin = min(np.nanmin(x) for x in hgt_all)
    hmax = max(np.nanmax(x) for x in hgt_all)
    hmin = np.floor(hmin / 30) * 30
    hmax = np.ceil(hmax / 30) * 30
    hgt_levels = np.arange(hmin, hmax + 30, 30)

    # -----------------------------
    # figure 재사용
    # -----------------------------
    if plt.fignum_exists(fig_num):
        fig = plt.figure(fig_num, figsize=figsize, clear=True)
    else:
        fig = plt.figure(num=fig_num, figsize=figsize)

    axes = fig.subplots(nrows, ncols, squeeze=False)
    axes_flat = axes.ravel()

    cf_last = None

    for i, (fields, title) in enumerate(zip(field_list, title_list)):
        ax = axes_flat[i]

        cf_last = _draw_850hpa_on_ax(
            ax=ax,
            lon2d=lon2d,
            lat2d=lat2d,
            t=fields["t"],
            z=fields["z"],
            u=fields["u"],
            v=fields["v"],
            use_theta=use_theta,
            barb_skip=barb_skip,
            title=title,
            shaded_levels=shaded_levels,
            hgt_levels=hgt_levels,
            cmap="turbo"
        )

    # 남는 subplot 숨기기
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # 공통 colorbar 하나만
    if cf_last is not None:
        # subplot 영역을 조금 줄여서 오른쪽 공간 확보
        fig.subplots_adjust(right=0.88)
        # colorbar 전용 axes 생성 (x, y, width, height)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

        cbar = fig.colorbar(
            cf_last,
            cax=cax
        )
        if use_theta:
            cbar.set_label("Potential Temperature (K)")
        else:
            cbar.set_label("Temperature (°C)")

    #fig.tight_layout()
    plt.pause(0.1)

def plot_850hpa_chart(
    lon,
    lat,
    t,
    z,
    u,
    v,
    use_theta=False,
    barb_skip=8,
    figsize=(10, 8),
    title=None,
    fig_num=2
):
    """
    850hPa 일기도:
      - shaded: 온도(°C) 또는 잠재온위(K)
      - contour: geopotential height(m)
      - barbs: wind (u,v)

    Parameters
    ----------
    lon, lat : 1D or 2D array
    t, z, u, v : 2D array, shape (ny, nx)
    use_theta : bool
        True면 잠재온위(theta) 채색
        False면 온도(°C) 채색
    barb_skip : int
        바람깃 간격 축소용
    """

    USE_CARTOPY = False

    lon2d, lat2d = to_2d_lonlat(lon, lat)

    t = np.asarray(t)
    z = np.asarray(z)
    u = np.asarray(u)
    v = np.asarray(v)

    hgt = convert_geopotential_to_height(z)  # m

    if use_theta:
        shaded = potential_temperature(t, p_hpa=850.0)
        shaded_label = "Potential Temperature (K)"
        # 보기 좋은 레벨 자동 설정
        vmin = np.floor(np.nanmin(shaded) / 2) * 2
        vmax = np.ceil(np.nanmax(shaded) / 2) * 2
        shaded_levels = np.arange(vmin, vmax + 2, 2)
    else:
        shaded = t - 273.15
        shaded_label = "Temperature (°C)"
        vmin = np.floor(np.nanmin(shaded) / 2) * 2
        vmax = np.ceil(np.nanmax(shaded) / 2) * 2
        shaded_levels = np.arange(vmin, vmax + 2, 2)

    # 850hPa 고도선 간격
    hmin = np.floor(np.nanmin(hgt) / 30) * 30
    hmax = np.ceil(np.nanmax(hgt) / 30) * 30
    hgt_levels = np.arange(hmin, hmax + 30, 30)

    if USE_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)

        ax.set_extent(
            [np.nanmin(lon2d), np.nanmax(lon2d), np.nanmin(lat2d), np.nanmax(lat2d)],
            crs=proj
        )

        ax.coastlines(resolution="50m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        cf = ax.contourf(
            lon2d, lat2d, shaded,
            levels=shaded_levels,
            cmap="turbo",
            extend="both",
            transform=proj
        )

        cs = ax.contour(
            lon2d, lat2d, hgt,
            levels=hgt_levels,
            colors="k",
            linewidths=1.0,
            transform=proj
        )
        ax.clabel(cs, fmt="%d", fontsize=8)

        ax.barbs(
            lon2d[::barb_skip, ::barb_skip],
            lat2d[::barb_skip, ::barb_skip],
            u[::barb_skip, ::barb_skip],
            v[::barb_skip, ::barb_skip],
            length=5.5,
            linewidth=0.6,
            transform=proj
        )

    else:
        #fig, ax = plt.subplots(figsize=figsize)
        #fig = plt.figure(num=2, figsize=figsize, clear=True)        
        fig_num = fig_num
        if plt.fignum_exists(fig_num):
            fig = plt.figure(fig_num, clear=True)   # 기존 figure 가져오기
        else:
            fig = plt.figure(num=fig_num, figsize=figsize)
        ax = fig.add_subplot(111)

        cf = ax.contourf(
            lon2d, lat2d, shaded,
            levels=shaded_levels,
            cmap="turbo",
            extend="both"
        )

        cs = ax.contour(
            lon2d, lat2d, hgt,
            levels=hgt_levels,
            colors="k",
            linewidths=0.6 #1.0
        )
        ax.clabel(cs, fmt="%d", fontsize=8)

        ax.barbs(
            lon2d[::barb_skip, ::barb_skip],
            lat2d[::barb_skip, ::barb_skip],
            u[::barb_skip, ::barb_skip],
            v[::barb_skip, ::barb_skip],
            length=4, #5.5,
            linewidth=0.6
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = plt.colorbar(cf, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label(shaded_label)

    if title is None:
        if use_theta:
            title = "850 hPa Geopotential Height + Wind + Potential Temperature"
        else:
            title = "850 hPa Geopotential Height + Wind + Temperature"

    ax.set_title(title)
    plt.tight_layout()
    #plt.show()
    #plt.waitforbuttonpress()
    plt.pause(1)

def plot_weathermap_by_models(target_date, tag_name, model_list,
                              lenTime: int = 49, var_name=("t", "z", "u", "v"),
                            model_names=None, figsize=(8, 8),):

    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

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
    
    model_img_list = []
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
        
        model_img_list.append(model_img)

    
    lon = np.arange(105.25, 147, 0.25)   # 길이 180
    lat = np.arange(58.75, 16, -0.25)   # 길이 180

    # for k in range(T):
    #     one_gt = gt_img[k]
    #     t = one_gt[:,:,0]
    #     z = one_gt[:,:,1]
    #     u = one_gt[:,:,2]
    #     v = one_gt[:,:,3]

    #     plot_850hpa_chart(lon=lon, lat=lat,
    #         t=t, z=z, u=u, v=v,
    #         use_theta=False,   # True로 바꾸면 잠재온위
    #         barb_skip=8,
    #         figsize=(8, 7),
    #         title=f"{target_date} 850hPa Chart: gt (time {k})",
    #         fig_num=2
    #     )

    #     for p in range(len(model_list)):
    #         one_model_img = model_img_list[p]
    #         title_str = f"{target_date} 850hPa Chart: model {model_names[model_list[p]-1]} (time {k})"

    #         one_md = one_model_img[k]
    #         t = one_md[:,:,0]
    #         z = one_md[:,:,1]
    #         u = one_md[:,:,2]
    #         v = one_md[:,:,3]

    #         plot_850hpa_chart(lon=lon, lat=lat,
    #             t=t, z=z, u=u, v=v,
    #             use_theta=False,   # True로 바꾸면 잠재온위
    #             barb_skip=8,
    #             figsize=(8, 7),
    #             title=title_str,
    #             fig_num=3+p
    #         )
    #     plt.waitforbuttonpress()

    for k in range(T):
        field_list = []
        title_list = []
        # GT
        one_gt = gt_img[k]
        field_list.append({
            "t": one_gt[:, :, 0],
            "z": one_gt[:, :, 1],
            "u": one_gt[:, :, 2],
            "v": one_gt[:, :, 3],
        })
        title_list.append(f"{target_date}\nGT (time {k})")

        # Models
        for p in range(len(model_list)):
            one_model_img = model_img_list[p]
            one_md = one_model_img[k]

            field_list.append({
                "t": one_md[:, :, 0],
                "z": one_md[:, :, 1],
                "u": one_md[:, :, 2],
                "v": one_md[:, :, 3],
            })

            model_name = model_names[model_list[p] - 1]
            title_list.append(f"{target_date}\n{model_name} (time {k})")

        # 한 figure에 모두 표시
        plot_850hpa_panel(
            lon=lon,
            lat=lat,
            field_list=field_list,
            title_list=title_list,
            use_theta=False,
            barb_skip=8,
            figsize=(14, 10),
            fig_num=2,
            ncols=2   # 4개면 2x2, 3개면 2x2에서 하나 비움
        )

        plt.waitforbuttonpress()


def plot_temporal_continuity_histogram(
    results,
    figsize=(8, 5),
    alpha=0.4,
    linewidth=2.0,
    show_kde_style=True,
    title=None,
):
    """
    analyze_temporal_continuity(...)의 결과를 받아
    각 space의 trail path length histogram을 공통 bin으로 그린다.

    parameters
    ----------
    results : dict
        analyze_temporal_continuity(...)의 반환값
    figsize : tuple
        figure size
    alpha : float
        histogram 투명도
    linewidth : float
        선 두께
    show_kde_style : bool
        True이면 histogram 위에 bin center 기준 선 그래프도 같이 그림
    title : str or None
        제목. None이면 자동 생성
    """
    histograms = results["histograms"]
    summary = results["summary"]

    bin_edges = histograms["bins"]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=figsize)

    space_names = [k for k in histograms.keys() if k != "bins"]

    for name in space_names:
        hist = histograms[name]

        # 막대형 histogram
        ax.hist(
            bin_centers,
            bins=bin_edges,
            weights=hist,
            alpha=alpha,
            label=f"{name} (mean={summary[name]['mean']:.4f})"
        )

        # 선 형태 overlay
        if show_kde_style:
            ax.plot(
                bin_centers,
                hist,
                linewidth=linewidth
            )

    ax.set_xlabel("Trail path length")
    ax.set_ylabel("Density" if np.all([np.sum(histograms[n]) > 0 for n in space_names]) else "Count")

    if title is None:
        title = "Distribution of trail path lengths across feature spaces"
    ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_temporal_continuity_step_hist(
    results,
    figsize=(8, 5),
    linewidth=2.0,
    title="Distribution of trail path lengths across feature spaces",
    xlabel="Trail path length",
    ylabel="Density",
    xlim=None,
    ylim=None,
    show_mean_vline=False,
):
    histograms = results["histograms"]
    summary = results["summary"]
    bin_edges = histograms["bins"]

    fig, ax = plt.subplots(figsize=figsize)

    space_names = [k for k in histograms.keys() if k != "bins"]

    for name in space_names:
        lengths = results["path_lengths"][name]

        ax.hist(
            lengths,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=linewidth,
            label=f"{name} (mean={summary[name]['mean']:.4f}, std={summary[name]['std']:.4f})"
        )

        if show_mean_vline:
            ax.axvline(
                summary[name]["mean"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.8
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.show()



def plot_pair_distance_histograms(
    dist_dict,
    bins=40,
    density=True,
    figsize=(15, 4),
    linewidth=2.0,
    xlim=None,
    ylim=None,
    sharex=False,
    sharey=True,
):
    """
    dist_dict 예시:
    {
        "ref": {
            "space1": dist_ref_1,   # shape (N,)
            "space2": dist_ref_2,
            "space3": dist_ref_3,
        },
        "noise": {
            "space1": dist_noise_1,
            "space2": dist_noise_2,
            "space3": dist_noise_3,
        },
        "shift": {
            "space1": dist_shift_1,
            "space2": dist_shift_2,
            "space3": dist_shift_3,
        }
    }
    """

    pair_types = list(dist_dict.keys())

    fig, axes = plt.subplots(
        1, len(pair_types),
        figsize=figsize,
        sharex=sharex,
        sharey=sharey
    )

    if len(pair_types) == 1:
        axes = [axes]

    for ax, pair_type in zip(axes, pair_types):
        space_data = dist_dict[pair_type]

        # 같은 subplot 내에서는 공통 bin 사용
        all_vals = np.concatenate([np.asarray(v).ravel() for v in space_data.values()])
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

        for space_name, dist in space_data.items():
            dist = np.asarray(dist).ravel()

            ax.hist(
                dist,
                bins=bin_edges,
                density=density,
                histtype="step",
                linewidth=linewidth,
                label=f"{space_name} (mean={np.mean(dist):.4f})"
            )

            # 평균 위치 표시
            ax.axvline(
                np.mean(dist),
                linestyle="--",
                linewidth=1.0,
                alpha=0.8
            )

        ax.set_title(pair_type)
        ax.set_xlabel("Pair distance")
        ax.grid(True, alpha=0.25)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel("Density" if density else "Count")
    axes[-1].legend(frameon=True)

    plt.tight_layout()
    plt.show()




def plot_pair_distance_ecdf(
    dist_dict,
    figsize=(15, 4),
    linewidth=2.5,
    xlim=None,
    ylim=(0, 1),
    sharex=False,
    sharey=True,
    show_median_vline=False,
):
    """
    dist_dict 예시:
    {
        "ref": {
            "space1": dist_ref_1,   # shape (N,)
            "space2": dist_ref_2,
            "space3": dist_ref_3,
        },
        "noise": {
            "space1": dist_noise_1,
            "space2": dist_noise_2,
            "space3": dist_noise_3,
        },
        "shift": {
            "space1": dist_shift_1,
            "space2": dist_shift_2,
            "space3": dist_shift_3,
        }
    }
    """

    pair_types = list(dist_dict.keys())

    fig, axes = plt.subplots(
        1, len(pair_types),
        figsize=figsize,
        sharex=sharex,
        sharey=sharey
    )

    if len(pair_types) == 1:
        axes = [axes]

    for ax, pair_type in zip(axes, pair_types):
        space_data = dist_dict[pair_type]

        for space_name, dist in space_data.items():
            dist = np.asarray(dist).ravel()
            x = np.sort(dist)
            y = np.arange(1, len(x) + 1) / len(x)

            ax.plot(
                x, y,
                linewidth=linewidth,
                label=f"{space_name}"
                #label=f"{space_name} (mean={np.mean(dist):.4f})"
            )

            if show_median_vline:
                ax.axvline(
                    np.median(dist),
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8
                )

        ax.set_title(pair_type)
        ax.set_xlabel("Pair distance")
        ax.grid(True, alpha=0.25)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel("Cumulative probability")
    axes[-1].legend(frameon=True)

    plt.tight_layout()
    plt.show()


############################################
## distance metric comparison ##
############################################

import numpy as np
import matplotlib.pyplot as plt


def compute_distance_matrix(gt_traj, fcst_traj):
    gt = np.asarray(gt_traj, dtype=float)
    fc = np.asarray(fcst_traj, dtype=float)
    diff = gt[:, None, :] - fc[None, :, :]
    return np.linalg.norm(diff, axis=2)   # shape: (Tg, Tf)


def dtw_path_from_distance_matrix(dist_mat):
    Tg, Tf = dist_mat.shape
    acc = np.full((Tg, Tf), np.inf, dtype=float)
    acc[0, 0] = dist_mat[0, 0]

    for i in range(1, Tg):
        acc[i, 0] = dist_mat[i, 0] + acc[i - 1, 0]
    for j in range(1, Tf):
        acc[0, j] = dist_mat[0, j] + acc[0, j - 1]

    for i in range(1, Tg):
        for j in range(1, Tf):
            acc[i, j] = dist_mat[i, j] + min(
                acc[i - 1, j],      # up
                acc[i, j - 1],      # left
                acc[i - 1, j - 1],  # diag
            )

    i, j = Tg - 1, Tf - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        candidates = []
        if i > 0 and j > 0:
            candidates.append((acc[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((acc[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((acc[i, j - 1], i, j - 1))

        _, i, j = min(candidates, key=lambda x: x[0])
        path.append((i, j))

    path.reverse()
    return acc, path


def representative_alignment_by_gt(path, dist_mat):
    Tg, Tf = dist_mat.shape
    matched_j = np.full(Tg, -1, dtype=int)
    align_dist = np.full(Tg, np.nan, dtype=float)

    bucket = {i: [] for i in range(Tg)}
    for i, j in path:
        bucket[i].append(j)

    for i in range(Tg):
        js = bucket[i]
        best_j = min(js, key=lambda j: dist_mat[i, j])
        matched_j[i] = best_j
        align_dist[i] = dist_mat[i, best_j]

    return matched_j, align_dist


def diagonal_distance(dist_mat):
    Tg, Tf = dist_mat.shape
    T = max(Tg, Tf)
    out = np.full(T, np.nan, dtype=float)
    n = min(Tg, Tf)
    out[:n] = np.diag(dist_mat[:n, :n])
    return out


def plot_alignment_vs_index_distance(
    gt_traj,
    fcst_traj,
    title="Lead-time trajectory distance comparison",
    cmap="viridis",
    figsize=(13, 5),
    show_local_distance=False,
):
    dist_mat = compute_distance_matrix(gt_traj, fcst_traj)  # (gt, fcst)
    acc, path = dtw_path_from_distance_matrix(dist_mat)
    matched_j, align_dist = representative_alignment_by_gt(path, dist_mat)
    idx_dist = diagonal_distance(dist_mat)

    cum_idx = np.nancumsum(idx_dist)
    cum_align = np.nancumsum(align_dist)

    fig, axes = plt.subplots(1, 2, num=7, figsize=figsize, clear=True)
    ax0, ax1 = axes

    # -------------------------
    # Left: distance matrix
    # x-axis = GT, y-axis = Forecast
    # -------------------------
    # dist_mat shape = (gt, fcst) 이므로 transpose해서 표시
    im = ax0.imshow(
        dist_mat.T,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )

    path_i = np.array([p[0] for p in path])  # gt
    path_j = np.array([p[1] for p in path])  # fcst

    # x=gt, y=forecast
    ax0.plot(path_i, path_j, color="red", linewidth=2.0, label="DTW best path")

    n = min(dist_mat.shape[0], dist_mat.shape[1])
    ax0.plot(
        np.arange(n),
        np.arange(n),
        linestyle="--",
        linewidth=1.5,
        color="white",
        label="Index-aligned diagonal",
    )

    ax0.scatter(
        np.arange(len(matched_j)),   # x = gt index
        matched_j,                   # y = forecast index
        s=18,
        color="cyan",
        edgecolors="black",
        linewidths=0.4,
        label="Representative match",
        zorder=3,
    )

    ax0.set_title("Distance matrix with alignment path")
    ax0.set_xlabel("GT index")
    ax0.set_ylabel("Forecast index")
    ax0.legend(loc="best", fontsize=8)

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Euclidean distance")

    # -------------------------
    # Right: distance curves by GT index
    # -------------------------
    x = np.arange(len(align_dist))

    ax1.plot(x, cum_idx[:len(x)], linewidth=2.2, label="Index-based cumulative")
    ax1.plot(x, cum_align, linewidth=2.2, label="Alignment-based cumulative")

    if show_local_distance:
        ax1.plot(
            x, idx_dist[:len(x)],
            linestyle="--", alpha=0.45,
            label="Index-based local"
        )
        ax1.plot(
            x, align_dist,
            linestyle="--", alpha=0.45,
            label="Alignment-based local"
        )

    ax1.set_title("Distance change along GT index")
    ax1.set_xlabel("GT index")
    ax1.set_ylabel("Distance")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    # suptitle 위치 수정
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    result = {
        "distance_matrix": dist_mat,
        "accumulated_cost": acc,
        "dtw_path": path,
        "matched_forecast_index_by_gt": matched_j,
        "index_distance": idx_dist,
        "alignment_distance": align_dist,
        "cum_index_distance": cum_idx,
        "cum_alignment_distance": cum_align,
    }
    
    plt.pause(0.1)
    plt.waitforbuttonpress()

    return fig, axes, result