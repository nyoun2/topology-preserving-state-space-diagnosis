import os
import numpy as np
import time

from feat_space_analysis.lib.utils import ensure_dir, load_logs_json, load_logs_jsonl
from feat_space_analysis.lib.io_paths import list_date_folders
from feat_space_analysis.lib.data_loading import load_feature_space
from feat_space_analysis.lib.main_process import run_feature_inference
from feat_space_analysis.lib.preprocess import save_vars_from_ecmwf_and_aimd
from feat_space_analysis.lib.eval_plots import show_trajectory_on_featuremap, build_points_by_model_maps_global, show_next_figures
from feat_space_analysis.lib.eval_plots import plot_model_8_maps, plot_two_models_8_hists, plot_two_models_8_maps_4classes_fixed_thr
from feat_space_analysis.lib.eval_plots import plot_rmse_comparison_by_models_monotone_shift_multivar
from feat_space_analysis.lib.eval_plots import plot_weathermap_by_models, plot_alignment_vs_index_distance, set_season_npy_365
from feat_space_analysis.lib.eval_metrics import check_good_performance_model
from feat_space_analysis.lib.dtw_like_rmse import plot_rmse_comparison_by_models, calc_leadtime_rmse_comparison_by_models




# 9개 모델에 대하여, rmse 기준 0.5 이하의 good point들을 표시하기.
# best 2 models에 대하여, 둘 사이의 공간 분포를 상호 비교하기
def plot_map_dominance():

    target_date = None
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"
    
    data_dir = f"out_test_features/{tag_name}/2025010100/final"
    feature_npy = os.path.join(data_dir, "era5_manifold_z.npy")
    
    log_path = "D:/ext_data/logs/eval_log_detail.jsonl"  # 또는 .json

    Z_space = load_feature_space(feature_npy)

    # jsonl / json 중 택1
    if log_path.endswith(".jsonl"):
        records = load_logs_jsonl(log_path)
    else:
        records = load_logs_json(log_path)

    #maps = build_points_by_model_maps(records)
    alpha_thre = 0.15
    maps, cond_minmax, cond_thr = build_points_by_model_maps_global(records, top_ratio=alpha_thre)
    print(cond_minmax)
    #print(cond_thr)

    
    # ###### 9개 model 전체 테스트, feature map에서의 분포를 확인 #####
    # total_good_points = []
    # for k in range(1, 10):
    #     good_points = plot_model_8_maps(Z_space, maps, model_idx=k, alpha_thre=cond_thr)
    #     total_good_points.append(good_points)
    # total_good_points = np.array(total_good_points)
    # print(total_good_points)

    set_season_npy_365()
    season_label = np.load("season_label_365.npy")    
    out = plot_two_models_8_maps_4classes_fixed_thr(Z_space, maps, modelA=4, modelB=7,
                                    cond_thr=cond_thr, alpha_mode="max",
                                    season_label=season_label)
    print(out["season_counts"][("dtw", 12)])
    print(out["season_counts"][("dtw", 24)])
    print(out["season_counts"][("dtw", 36)])
    print(out["season_counts"][("dtw", 48)])
    plot_two_models_8_hists(maps, modelA=4, modelB=7, thr_mode="top30", bin_step=0.05, thr_fixed=alpha_thre)


# bDTW_RMSE
def compare_trajectories(bOnlyShow=True):

    target_date = None
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"


    ### 9개 모델을 8개 조건에 따라서, 개별 모델별 trajectory metric을 표시
    data_dir = f"out_test_features/{tag_name}/2025010100/final"
    feature_npy = os.path.join(data_dir, "era5_manifold_z.npy")
    
    log_path = "D:/ext_data/logs/eval_log_detail.jsonl"  # 또는 .json

    Z_space = load_feature_space(feature_npy)

    # jsonl / json 중 택1
    if log_path.endswith(".jsonl"):
        records = load_logs_jsonl(log_path)
    else:
        records = load_logs_json(log_path)

    #maps = build_points_by_model_maps(records)
    alpha_thre = 0.15
    maps, cond_minmax, cond_thr = build_points_by_model_maps_global(records, top_ratio=alpha_thre)
    print(cond_minmax)
    #print(cond_thr)

    ## start date를 중심으로 연속적인 테스트를 진행
    data_dir = "D:/ext_data/aimd4var"
    out_dir = os.path.normpath(os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev"))
    ensure_dir(out_dir)

    date_list = list_date_folders(data_dir)

    start = "2025010100" #"2025022300" #"2025010100"
    end   = "2025123100" #"2025011000"
    date_list = [d for d in date_list if start <= d <= end]

    # ## 여기에서, 전체 기간에 대한 RMSE를 계산하도록 한다. # 이 부분을 임시로 제거
    # model_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # rmse_mean, df_rmse = calc_leadtime_rmse_comparison_by_models(
    #     target_dates=date_list,
    #     tag_name=tag_name,
    #     model_list=model_list,
    #     lenTime=49,
    #     lead_points=(12, 24, 36, 48),
    #     var_name=("t", "z", "u", "v")
    # )
    # print(df_rmse)


    nTest = 0
    for target_date in date_list[::4]:

        if not bOnlyShow:
            # extract_features -> convert_state_space
            start = time.perf_counter()
            save_vars_from_ecmwf_and_aimd(
                target_date,
                data_dir=data_dir,
                out_dir=out_dir,
                bSameLead=False,
                control_aimd=True
            )
            mode = "infer_set"
            main(mode, target_date, tag_name)

        # evaluate model_list
        model_list = [1, 4, 7]
        

        # DTW/RMSE 어느 하나가 모두 good condition일때
        result_str, bAllGood, good_models = check_good_performance_model(maps, target_date, model_list)
        bAllGood = True # 전체실행을 위해
        if not bAllGood:
            continue
        else:
            nTest += 1
            print(f"all_good_trajectory:{nTest}: {target_date}, model_id {good_models}")
            # if good_model != 7:
            #     continue
        
            #model_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            plot_rmse_comparison_by_models(target_date, tag_name, model_list)

            #Z_gt, Z_ai_list = show_trajectory_on_featuremap(target_date, tag_name, model_list, maps)
        

            ###### DTW-metric show ######
            for one_model in model_list:
                one_list = [one_model]
                # trajectory는 단일 모델별로 표시하도록 함. one_list에 둘 이상이 들어가면 한꺼번에 표현됨
                Z_gt, Z_ai_list = show_trajectory_on_featuremap(target_date, tag_name, one_list, maps)
            show_next_figures()
            

        #     # gt_traj, fcst_traj: shape (48, 2)
        #     # trajectory 비교하여, distance metric에 따른 차이 확인할 수 있도록, 1:1로만 매치됨
            # fig, axes, out = plot_alignment_vs_index_distance(
            #     Z_gt, Z_ai_list[0], title="Lead-time trajectory: alignment-based vs index-aligned distance")
            


        
        # # 여기에, GT/AI 모델의 raw data를 활용한 일기도를 그린다.
        # plot_weathermap_by_models(target_date, tag_name, model_list)
    

def calc_rmse_temporal_shift():

    target_date = None
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"

    ## start date를 중심으로 연속적인 테스트를 진행
    data_dir = "D:/ext_data/aimd4var"
    out_dir = os.path.normpath(os.path.join("D:/ext_data", f"{tag_name}", "Test_4var1lev"))
    ensure_dir(out_dir)

    date_list = list_date_folders(data_dir)

    date_list = ["2025013100", "2025030200", "2025062500", "2025102800"]

    start = "2025010100" #"2025010100"
    end   = "2025123100" #"2025011000"
    date_list = [d for d in date_list if start <= d <= end]

    model_names = np.array([
        "fnet_ifs","fnet_kim","fnet_um",
        "grph_ifs","grph_kim","grph_um",
        "pang_ifs","pang_kim","pang_um"
    ])

    all_shift_directions = []
    
    for target_date in date_list[::1]:
                
        model_list = [1, 4, 7]

        shift_directions = None
    
        #plot_rmse_comparison_by_models_v2(target_date, tag_name, model_list, model_names=model_names)
        result, shift_directions = plot_rmse_comparison_by_models_monotone_shift_multivar(target_date, tag_name, model_list, model_names=model_names, show=False)
        #summary = summarize_rmse_improvement(result, var_name=("t","z","u","v"))
        #print_summary(summary, model_names=model_names)
        all_shift_directions.append(shift_directions)

    
    all_shift_directions = np.array(all_shift_directions)
    print(all_shift_directions)

    np.save("shift_direction.npy", all_shift_directions)
    np.savetxt("shift_direction.csv", all_shift_directions, delimiter=",")

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

    #exit()

    