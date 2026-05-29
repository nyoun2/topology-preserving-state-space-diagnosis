import os
import time
import numpy as np
import matplotlib.pyplot as plt

from feat_space_analysis.lib.utils import ensure_dir
from feat_space_analysis.lib.io_paths import list_date_folders
from feat_space_analysis.lib.main_process import run_feature_inference
from feat_space_analysis.lib.preprocess import save_vars_from_ecmwf_and_aimd
from feat_space_analysis.lib.eval_metrics import log_for_distance_metric, compare_multiple_spaces, print_results
from feat_space_analysis.lib.eval_metrics import analyze_temporal_continuity, print_temporal_continuity_summary
from feat_space_analysis.lib.eval_plots import plot_temporal_continuity_histogram, plot_temporal_continuity_step_hist
from feat_space_analysis.lib.eval_plots import evaluate_performance_and_show_featuremap, plot_distance_for_valid_time_comparison
from feat_space_analysis.lib.eval_plots import validate_season_correspondence, show_valid_trajectory_on_featuremap
from feat_space_analysis.lib.eval_plots import plot_final_window_boxplots, show_next_figures
from feat_space_analysis.lib.navigator import navigate_on_feats, navigate_on_feats_temporal, navigate_on_feats_and_replay
from feat_space_analysis.lib.dtw_like_rmse import plot_rmse_comparison_by_models_for_valid_trajectory, calc_validtime_rmse_comparison_by_models
from feat_space_analysis.lib.build_manifold_learning_v3 import load_embedder_and_project
from feat_space_analysis.lib.io_paths import glob_npy

# for tracking target
from feat_space_analysis.lib.io_paths import make_file_list_for_same_target_date
from feat_space_analysis.lib.preprocess import save_controlled_vars, crop_dataset, rearrange_folder

# for lead time evaluation
from feat_space_analysis.lib.utils import load_logs_json, load_logs_jsonl
from feat_space_analysis.lib.data_loading import load_feature_space
from feat_space_analysis.lib.eval_plots import show_trajectory_on_featuremap, build_points_by_model_maps_global
from feat_space_analysis.lib.dtw_like_rmse import plot_rmse_comparison_by_models
from feat_space_analysis.lib.eval_metrics import check_good_performance_model

## GT까지의 valid-time comparison을 위한 작업
def calc_valid_time_comparison(day_interval=1, start_date="2025011300", end_date="2025123000"):
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"

    data_folder_dir = "D:/ext_data/aimd4var"
    date_list = list_date_folders(data_folder_dir)
    start = start_date #"2025011500" #"2025010100"
    end   = end_date # "2025123000" #"2025011000"
    date_list = [d for d in date_list if start <= d <= end]

    total_dist_list = []
    for target_date in date_list[::day_interval]:
        folder_target_date = f"{target_date}_target"
        dist_list = plot_distance_for_valid_time_comparison(tag_name, folder_target_date, model_list=None, bShow=False)
        total_dist_list.append(dist_list)

    nCases = len(total_dist_list)
    mean_dist = np.mean(total_dist_list, axis=0)
    print(mean_dist)

    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    plt.figure(figsize=(8,6))

    # lead time (예: -12 ... -1)
    lead_days = np.arange(-12, 1)

    for i in range(len(model_names)):

        dist = mean_dist[i]   # lead 방향 맞추기

        plt.plot(
            lead_days,
            dist,
            marker='o',
            markersize=3,
            linewidth=1,
            label=model_names[i]
        )

    plt.xlabel("Days Before Target Date")
    plt.ylabel("Distance from Z_gt")
    plt.title(f"Mean Distance from Initial Point in Feature Space ({nCases} cases)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    #plt.show()
    plt.pause(0.1)

    # boxplot 그리기
    fig, axes = plot_final_window_boxplots(
        total_dist_list,
        model_names=model_names,
        valid_times=[-1, -6, -12],
        figsize=(19, 5),
        showfliers=False,
        title_prefix="Distribution of feature-space distance near target date",
        drop_last_zero=True,
    )

def build_input_data_dir(input_root_dir, tag_name, input_subdir):
    """
    Build the directory containing inference input folders.

    Example
    -------
    D:/ext_data/{tag_name}/Test_4var1lev
    """
    return os.path.normpath(
        os.path.join(input_root_dir, tag_name, input_subdir)
    )


def build_processed_tag_dir(processed_root_dir, tag_name):
    """
    Build the directory containing processed inference outputs.

    Example
    -------
    out_test_features/{tag_name}
    """
    return os.path.normpath(
        os.path.join(processed_root_dir, tag_name)
    )


def list_available_target_dates(
    search_dir,
    target_folder_suffix="",
    include_dates=None,
):
    """
    Search target-date folders from a directory.

    Parameters
    ----------
    search_dir : str
        Directory containing target-date folders.

    target_folder_suffix : str
        Folder suffix.
        Lead-time: ""
        Valid-time: "_target" or later "_valid"

    include_dates : list[str] or None
        If provided, only these dates are used.

    Returns
    -------
    list[str]
        Dates without suffix.

    Examples
    --------
    Lead-time folder:
        2025010100
        -> date = 2025010100

    Valid-time folder:
        2025011300_target
        -> date = 2025011300
    """
    search_dir = os.path.normpath(search_dir)

    if not os.path.isdir(search_dir):
        raise FileNotFoundError(f"search_dir not found: {search_dir}")

    all_folders = [
        f for f in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, f))
    ]

    date_list = []

    for folder in all_folders:
        if target_folder_suffix:
            if not folder.endswith(target_folder_suffix):
                continue
            date_part = folder[: -len(target_folder_suffix)]
        else:
            date_part = folder

        if len(date_part) == 10 and date_part.isdigit():
            date_list.append(date_part)

    date_list = sorted(date_list)

    if include_dates is not None:
        include_dates = [str(d) for d in include_dates]
        missing = [d for d in include_dates if d not in date_list]

        if missing:
            print(f"[WARN] Some requested dates were not found in {search_dir}: {missing}")

        date_list = [d for d in include_dates if d in date_list]

    if len(date_list) == 0:
        raise RuntimeError(
            f"No target folders found in {search_dir} "
            f"with suffix '{target_folder_suffix}'"
        )

    return date_list

def infer_and_evaluate_lead_time_trajectory(
    show_only=True,
    tag_name="CL_era5_REF-ecmwf_COND-noise-shift_v1",
    input_root_dir="data/input",
    input_subdir="Test_4var1lev",
    processed_root_dir="data/out_test_features",
    feature_npy=None,
    log_path=None,
    model_list=None,
    top_ratio=0.15,
    include_dates=None,
    target_folder_suffix="_lead",
):
    """
    Infer and evaluate lead-time trajectories.

    Directory convention
    --------------------
    Inference input:
        {input_root_dir}/{tag_name}/{input_subdir}/{target_date}

    Processed output:
        {processed_root_dir}/{tag_name}/{target_date}

    Parameters
    ----------
    show_only : bool
        True:
            Use existing processed results in processed_root_dir.
        False:
            Re-run inference using input_root_dir/{tag_name}/{input_subdir}.

    target_folder_suffix : str
        Lead-time normally uses "".
    """
    #ensure_dir(processed_root_dir)

    if model_list is None:
        model_list = [1, 4, 7]

    input_data_dir = build_input_data_dir(
        input_root_dir=input_root_dir,
        tag_name=tag_name,
        input_subdir=input_subdir,
    )

    processed_tag_dir = build_processed_tag_dir(
        processed_root_dir=processed_root_dir,
        tag_name=tag_name,
    )

    if feature_npy is None:
        feature_npy = os.path.join(
            processed_root_dir,
            tag_name,            
            "era5_manifold_z.npy",
        )

    if log_path is None:
        log_path = os.path.join(
            "data",
            "logs",
            "eval_log_detail.jsonl",
        )

    feature_npy = os.path.normpath(feature_npy)
    log_path = os.path.normpath(log_path)

    print("[Lead-time trajectory]")
    print(f"  tag_name             : {tag_name}")
    print(f"  input_data_dir       : {input_data_dir}")
    print(f"  processed_tag_dir    : {processed_tag_dir}")
    print(f"  feature_npy          : {feature_npy}")
    print(f"  log_path             : {log_path}")
    print(f"  model_list           : {model_list}")
    print(f"  show_only            : {show_only}")
    print(f"  include_dates        : {include_dates}")
    print(f"  target_folder_suffix : {target_folder_suffix}")
    print(f"  top_ratio            : {top_ratio}")

    if not os.path.exists(feature_npy):
        raise FileNotFoundError(f"Feature-space file not found: {feature_npy}")

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Evaluation log file not found: {log_path}")

    # Load archived feature-space coordinates.
    _ = load_feature_space(feature_npy)

    # Load trajectory metric logs and build background/model maps.
    if log_path.endswith(".jsonl"):
        records = load_logs_jsonl(log_path)
    else:
        records = load_logs_json(log_path)

    maps, cond_minmax, cond_thr = build_points_by_model_maps_global(
        records,
        top_ratio=top_ratio,
    )

    print("[Metric ranges]")
    print(cond_minmax)

    # Select target dates.
    if show_only:
        search_dir = processed_tag_dir
    else:
        search_dir = input_data_dir

    date_list = list_available_target_dates(
        search_dir=search_dir,
        target_folder_suffix=target_folder_suffix,
        include_dates=include_dates,
    )

    print(f"  search_dir           : {search_dir}")
    print(f"  n_dates              : {len(date_list)}")
    print(f"  dates                : {date_list}")

    for target_date in date_list:
        folder_target_date = f"{target_date}{target_folder_suffix}"

        print(f"==== lead target_date: {target_date} ====")
        print(f"  folder_target_date: {folder_target_date}")

        if not show_only:
            start_time = time.perf_counter()

            # Input must already exist:
            # {input_root_dir}/{tag_name}/{input_subdir}/{target_date}
            input_target_dir = os.path.join(input_data_dir, folder_target_date)

            if not os.path.isdir(input_target_dir):
                raise FileNotFoundError(f"Input target folder not found: {input_target_dir}")

            mode = "infer_set"
            run_feature_inference(mode, folder_target_date, tag_name,
                    test_root=input_data_dir,
                    infer_out_root=processed_tag_dir,)
            
            manifold_embedding(tag_name, folder_target_date, bValid=False,
                               infer_out_root=processed_tag_dir)

            elapsed = time.perf_counter() - start_time
            print(f"Elapsed time: {elapsed:.3f} sec")

        # No bAllGood filtering in public workflow.
        plot_rmse_comparison_by_models(
            folder_target_date,
            tag_name,
            model_list,
            test_root=input_data_dir,
        )

        for one_model in model_list:
            show_trajectory_on_featuremap(
                folder_target_date,
                tag_name,
                [one_model],
                maps,
                infer_out_root=processed_tag_dir,
            )

        show_next_figures()


def infer_and_evaluate_valid_time_trajectory(
    show_only=True,
    tag_name="CL_era5_REF-ecmwf_COND-noise-shift_v1",
    input_root_dir="data/input",
    input_subdir="Test_4var1lev",
    processed_root_dir="data/out_test_features",
    model_list=None,
    include_dates=None,
    target_folder_suffix="_valid",
):
    """
    Infer and evaluate valid-time trajectories.

    Directory convention
    --------------------
    Inference input:
        {input_root_dir}/{tag_name}/{input_subdir}/{target_date}{suffix}

    Processed output:
        {processed_root_dir}/{tag_name}/{target_date}{suffix}

    Parameters
    ----------
    show_only : bool
        True:
            Use existing processed results in processed_root_dir.
        False:
            Re-run inference using input_root_dir/{tag_name}/{input_subdir}.

    target_folder_suffix : str
        Current valid-time convention: "_target".
        Later this can be changed to "_valid" in YAML.
    """
    #ensure_dir(processed_root_dir)

    if model_list is None:
        model_list = [1, 4, 7]

    input_data_dir = build_input_data_dir(
        input_root_dir=input_root_dir,
        tag_name=tag_name,
        input_subdir=input_subdir,
    )

    processed_tag_dir = build_processed_tag_dir(
        processed_root_dir=processed_root_dir,
        tag_name=tag_name,
    )

    print("[Valid-time trajectory]")
    print(f"  tag_name             : {tag_name}")
    print(f"  input_data_dir       : {input_data_dir}")
    print(f"  processed_tag_dir    : {processed_tag_dir}")
    print(f"  model_list           : {model_list}")
    print(f"  show_only            : {show_only}")
    print(f"  include_dates        : {include_dates}")
    print(f"  target_folder_suffix : {target_folder_suffix}")

    # Select target dates.
    if show_only:
        search_dir = processed_tag_dir
    else:
        search_dir = input_data_dir

    date_list = list_available_target_dates(
        search_dir=search_dir,
        target_folder_suffix=target_folder_suffix,
        include_dates=include_dates,
    )

    print(f"  search_dir           : {search_dir}")
    print(f"  n_dates              : {len(date_list)}")
    print(f"  dates                : {date_list}")

    for target_date in date_list:
        folder_target_date = f"{target_date}{target_folder_suffix}"

        print(f"==== valid target_date: {target_date} ====")
        print(f"  folder_target_date: {folder_target_date}")

        if not show_only:
            start_time = time.perf_counter()

            # Input must already exist:
            # {input_root_dir}/{tag_name}/{input_subdir}/{target_date}{suffix}
            input_target_dir = os.path.join(input_data_dir, folder_target_date)

            if not os.path.isdir(input_target_dir):
                raise FileNotFoundError(f"Input target folder not found: {input_target_dir}")

            mode = "infer_set"
            run_feature_inference(mode, folder_target_date, tag_name,
                    test_root=input_data_dir,
                    infer_out_root=processed_tag_dir,)
            
            manifold_embedding(tag_name, folder_target_date, bValid=True,
                               infer_out_root=processed_tag_dir)

            elapsed = time.perf_counter() - start_time
            print(f"Elapsed time: {elapsed:.3f} sec")

        plot_rmse_comparison_by_models_for_valid_trajectory(
            folder_target_date,
            tag_name,
            model_list,
            test_root=input_data_dir,
        )

        for one_model in model_list:
            show_valid_trajectory_on_featuremap(
                folder_target_date,
                tag_name,
                [one_model],
                infer_out_root=processed_tag_dir,
            )

        show_next_figures()

def manifold_embedding(tag_name, target_date, bValid=False, infer_out_root=None):

    # Final Stage: parametric manifold learning, 추론 과정
    # z_post features to 2d loc. data via parametric umap 
    
    if infer_out_root is None:
        infer_out_root = f"out_test_features/{tag_name}/{target_date}"
        data_dir_base = os.path.normpath(infer_out_root)
    else:
        data_dir_base = os.path.normpath(
            os.path.join(infer_out_root, target_date)
        )    
    #data_dir = f"out_test_features/{tag_name}/{target_date}"
    EXTRACT_DIR = f"data/model_all/{tag_name}"

    #input_npy = "out_features/ERA5_proj_z.npy" # 학습데이터 확인
    input_npy = f"data/model_all/{tag_name}/era5_proj_z.npy" # 학습데이터 확인
    os.makedirs(os.path.join(data_dir_base, "final"), exist_ok=True)
    output_npy = os.path.join(data_dir_base, "final/era5_manifold_z.npy")
    
    z_pre_all = np.load(input_npy)
    best_path = f"data/model_all/{tag_name}/pm_run02/best.weights.h5"
    Z_all = load_embedder_and_project(z_pre_all, weights_path=best_path)
    np.save(output_npy, Z_all) 

    if not bValid:
        input_npy = os.path.join(data_dir_base, "proj/z_ecmwf.npy")
        output_npy = os.path.join(data_dir_base, "final/z_ecmwf_manifold.npy")
        
        z_pre_ecmwf = np.load(input_npy)
        best_path = f"data/model_all/{tag_name}/pm_run02/best.weights.h5"
        Z_gt = load_embedder_and_project(z_pre_ecmwf, weights_path=best_path)
        np.save(output_npy, Z_gt)
        
    
    data_dir_proj = os.path.join(data_dir_base, "proj")
    #data_dir = f"out_test_features/{tag_name}/{target_date}/proj"
    files = glob_npy(data_dir_proj, "Z_ai*.npy")
    cFile = len(files)

    distance_list = []
    rmse_list = []
    Z_ai_list = []
    
    for a in range(cFile):
        input_npy = "tmp.npy"
        x = np.load(files[a])
        np.save(input_npy, x)
                
        output_npy = os.path.join(data_dir_base, f"final/ai_{a+1:02d}_manifold.npy")

        z_pre_ai = np.load(input_npy)
        best_path = f"data/model_all/{tag_name}/pm_run02/best.weights.h5"
        Z_ai = load_embedder_and_project(z_pre_ai, weights_path=best_path)
        np.save(output_npy, Z_ai)
        Z_ai_list.append(Z_ai)

        

def infer_and_evaluate_target_dates():
    ## start date를 중심으로 연속적인 테스트를 진행
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"

    data_dir = "D:/ext_data/aimd4var"
    out_dir = os.path.normpath(os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev"))
    ensure_dir(out_dir)

    date_list = list_date_folders(data_dir)

    start = "2025010100" #"2025010100"
    end   = "2025123100" #"2025011000"
    date_list = [d for d in date_list if start <= d <= end]    

    for target_date in date_list: #[::10]:

        start = time.perf_counter()
        save_vars_from_ecmwf_and_aimd(
            target_date,
            data_dir=data_dir,
            out_dir=out_dir,
            bSameLead=False,
            control_aimd=True
        )

        mode = "infer_set"
        run_feature_inference(mode, target_date, tag_name)

        bShow = True # False: pause(0.1), True: show()
        evaluate_performance_and_show_featuremap(target_date, tag_name, bShow=bShow)
        
        elapsed = time.perf_counter() - start
        if not bShow:
            print(f"Elapsed time: {elapsed:.3f} sec")

        ##only for log
        #log_for_distance_metric(target_date, tag_name)


def navigate_feature_space():
    # feature space의 한지점을 마우스로 표시하면 가장 가까운 지점을 찾고, 이후에 그 지점에 있는 점들에서 
    # 일기도를 만들어낸다. 마우스가 움직이면서 위치가 달라지면 가장 가까운 지점은 달라지게 되고, 
    # 일기도를 그리는 raw data가 달라지면서 최종적으로 새로운 일기도로 바뀐다.
    # 마우스는 클릭한 상태에서 움직이는 경로가 표시되고, 지점을 거쳤던 정보들도 색깔이 바뀐다.

    # -----------------------------------------------------
    # 예시 1) feature space 좌표
    # 실제로는 사용자가 이미 가진 2D embedding 결과를 넣으면 됨
    # shape = (N, 2)
    # -----------------------------------------------------
    # load Z_all
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"
    data_dir = f"model_all/{tag_name}/feature_space_comparison"
    output_npy = os.path.join(data_dir, "with_CL_PML_era5.npy")
    Z_all = np.load(output_npy)
    N = Z_all.shape[0]
    
    # -----------------------------------------------------
    # 예시 2) 각 점에 대응되는 raw data 파일 목록
    # 실제 파일 경로로 교체해야 함
    # 각 npz 파일 안에 lon, lat, t, z, u, v가 있어야 함
    # -----------------------------------------------------
    #file_list = [f"./raw_data/sample_{i:03d}.npz" for i in range(N)]
    #file_list = [f"d:/ext_data/ecmwf_4var_crop_idx/ecmwf_t{i:04d}.npy" for i in range(N)] # ecmwf
    file_list = [f"ext_data/out_npy_era5_850_cropped/cropped_vzuv850_t{i:04d}.npy" for i in range(N)] # era5
    
    #navigate_on_feats(Z_all, file_list)
    #navigate_on_feats_temporal(Z_all, file_list)
    navigate_on_feats_and_replay(Z_all, file_list)

def validate_season_labels():
    
    validate_season_correspondence()

    return 0

    label_path = "season_label.npy"
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"
    feature_paths = [f"model_all/{tag_name}/feature_space_comparison/without_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_PML_era5.npy"]

    labels = np.load(label_path)
    coords1 = np.load(feature_paths[0])
    coords2 = np.load(feature_paths[1])
    coords3 = np.load(feature_paths[2])

    space_dict = {
        "space1": coords1,
        "space2": coords2,
        "space3": coords3,
    }

    results = compare_multiple_spaces(space_dict, labels, normalize=True)
    print_results(results)

def validate_temporal_continuity():

    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"
    feature_paths = [f"model_all/{tag_name}/feature_space_comparison/without_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_PML_era5.npy"]

    coords1 = np.load(feature_paths[0])
    coords2 = np.load(feature_paths[1])
    coords3 = np.load(feature_paths[2])

    space_dict = {
        "WO_CL_UMAP": coords1,   # shape: (T, 2)
        "W_CL_UMAP": coords2,
        "W_CL_PME": coords3,
    }

    trail_length_list = [12, 8, 16, 20, 4]

    for len_trail in trail_length_list:
        print(f"trail length = {len_trail}")
        # 예: 8개 시점 trail = 6시간 간격 기준 총 42시간 구간
        results = analyze_temporal_continuity(
            space_dict=space_dict,
            trail_length=len_trail, #8,
            normalize="zscore",
            bins=40,
            density=True
        )

        print_temporal_continuity_summary(results, sort_by="mean")

        # # 각 space별 trail path length 배열
        # lengths1 = results["path_lengths"]["space1"]
        # lengths2 = results["path_lengths"]["space2"]
        # lengths3 = results["path_lengths"]["space3"]

        # # 공통 histogram bin
        # bins = results["histograms"]["bins"]
        # hist1 = results["histograms"]["space1"]
        # hist2 = results["histograms"]["space2"]
        # hist3 = results["histograms"]["space3"]

        #plot_temporal_continuity_histogram(results)
        plot_temporal_continuity_step_hist(results, xlim=(0, 10))
        

#if __name__ == "__main__":

    # target_date = None
    # tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"

    # # parametric manifold learning
    # input_npy = f"model_all/{tag_name}/era5_proj_z.npy" # 학습데이터 확인
    # X = np.load(input_npy)

    # # model = fit_parametric_manifold(X, k=30, num_neg=10, epochs=60, w_stress=0.1, lr=1e-3)
    # # Z = model.predict(X, batch_size=4096)
    # result = fit_parametric_manifold_with_earlystop(X, out_dir="pm_run02", epochs=200, patience=20)
    # model = result["model"]
    # Z = model.predict(X, batch_size=4096)
    # #추론 예시:
    # #Z_new = load_embedder_and_project(X_new, weights_path=result["best_path"])

    # output_npy = f"model_all/{tag_name}/era5_manifold_2d_v2.npy" # 학습데이터 확인
    # np.save(output_npy, Z)
    # ## 여기에 feature space에 대한 평가 결과를 표시 ##
    # X = np.load(input_npy)    
    # Z = np.load(output_npy)
    # metrics = evaluate_umap_embedding(X, Z, ks=(10,20,50), spearman_pairs=200_000, continuity_m=300)
    # print(metrics)

    # exit()


    # # ## fused.npy 비교
    # base_ecmwf = np.load(f"model_all/{tag_name}/ecmwf_proj_z.npy")
    # test_ecmwf = np.load(f"out_test_features/{tag_name}/2024121900/proj/z_ecmwf.npy")
    # sample = base_ecmwf[7257:7257+48]

    # base_ecmwf = np.load(f"model_all/{tag_name}/ecmwf_fused.npy")
    # test_ecmwf = np.load(f"out_test_features/{tag_name}/2024121900/fused/ecmwf_fused.npy")
    # sample = base_ecmwf[7257:7257+48]

    # for ti in range(7255, 7260):
    #     print(f"ti: {ti}")
    #     #### raw npy 비교
    #     base_ecmwf = np.load(f"D:/ext_data/ecmwf_4var_crop_idx/ecmwf_t{ti}.npy")
    #     test_ecmwf = np.load(f"D:/ext_data/{tag_name}/Test_4var1lev/2024121900/ecmwf/ecmwf_t0001.npy")
    #     # sample = base_ecmwf[7257:7257+48]        
    #     fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    #     imgs = [base_ecmwf[:,:,0], base_ecmwf[:,:,1], base_ecmwf[:,:,2], base_ecmwf[:,:,3], 
    #             test_ecmwf[:,:,0], test_ecmwf[:,:,1], test_ecmwf[:,:,2], test_ecmwf[:,:,3]]
    #     for i, ax in enumerate(axes.flat):
    #         ax.imshow(imgs[i])
    #         ax.set_title(f"Image {i+1}")
    #         ax.axis("off")
    #     plt.tight_layout()
    #     plt.show()

    # exit()