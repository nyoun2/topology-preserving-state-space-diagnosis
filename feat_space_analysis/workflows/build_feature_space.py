import os
import numpy as np
import matplotlib.pyplot as plt


from feat_space_analysis.lib.build_manifold_learning_v3 import load_embedder_and_project, fit_parametric_manifold_with_earlystop
from feat_space_analysis.lib.manifold_train import umap_projection
from feat_space_analysis.lib.eval_metrics import evaluate_umap_embedding
from feat_space_analysis.lib.eval_plots import plot_umap_manual_multi, plot_pair_distance_histograms, plot_pair_distance_ecdf
from feat_space_analysis.lib.main_process import run_feature_inference


def compare_two_training_data(tag_name, src_npy, comp_npy, comp_tag, bParametric=True, fit_umap=False):

    data_dir = f"out_test_features/{tag_name}"
    EXTRACT_DIR = f"model_all/{tag_name}"
    
    #input_npy = "out_features/ERA5_proj_z.npy" # 학습데이터 확인
    input_npy = src_npy #f"model_all/{tag_name}/ERA5_proj_z.npy" # 학습데이터 확인    
    output_npy = os.path.join(data_dir, "train_source_manifold_z.npy")
    #z_pre_all = np.load(input_npy)
    if bParametric:
        #proc_parametric_umap(input_npy, output_npy, toTrain=False, tag_name=tag_name)        
        output_npy = os.path.join(data_dir, f"parametric_map_era5_{tag_name}.npy")
        # proc_parametric_manifold_v2(
        #     input_npy, output_npy,              
        #     tag_name=tag_name,
        #     toTrain=False,
        #     output_dim=2, 
        #     metric="mahalanobis"
        # )
        source_proj = np.load(input_npy)
        best_path = "pm_run02/best.weights.h5"
        #Z_all = load_embedder_and_project(source_proj, weights_path=result["best_path"])
        Z_all = load_embedder_and_project(source_proj, weights_path=best_path)
        #np.save(output_npy, Z_all[:5840])
        np.save(output_npy, Z_all)
        
    else:
        if fit_umap:
            umap_projection(
                feature_path=src_npy,
                mode="fit", #"fit", "transform"
                umap_mode=1,
                model_path=f"{EXTRACT_DIR}/umap_model.pkl",
                save_path=output_npy,
                show_plot=False
            )
        else:
            umap_projection(
                feature_path=src_npy,
                mode="transform", #"fit", "transform"
                umap_mode=1,
                model_path=f"{EXTRACT_DIR}/umap_model.pkl",
                save_path=output_npy,
                show_plot=False
            )

        # base_feat = np.load(src_npy)
        # N = base_feat.shape[0]
        # t_index = np.arange(N)  # 시간순이면 이걸로 충분

        # Z_all = fit_umap_base_spatiotemporal(
        #     base_feat=base_feat,
        #     model_dir="umap_base_st",
        #     t_index=t_index,
        #     gamma_time=0.2,
        #     n_neighbors=60,
        #     min_dist=0.05,
        #     n_components=2,
        # )
        # np.save(output_npy, Z_all)

    Z_all = np.load(output_npy)

    plt.figure(figsize=(8, 8))
    plt.scatter(Z_all[:, 0], Z_all[:, 1], c="gray", s=2, alpha=0.3, label="all")    
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("EfficientNet based Feature Representation")
    plt.legend()
    plt.tight_layout()
    #plt.waitforbuttonpress()
    plt.pause(0.1)

    dist_seq = np.linalg.norm(Z_all[1:] - Z_all[:-1], axis=1)    

    plt.figure(figsize=(6, 4))
    plt.hist(dist_seq, bins=20, edgecolor="black", alpha=0.7)

    counts, bin_edges = np.histogram(dist_seq, bins=20)
    for i in range(len(counts)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        print(f"[{left:.4f}, {right:.4f}): {counts[i]}개")
    
    mask = dist_seq < 0.75
    print(f"temporal continuity")
    print(f"[INFO] selected points: {mask.sum()} / {len(mask)}, {np.round(mask.sum() / len(mask), 3)}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(Z_all[:, 0], Z_all[:, 1],
                c="gray", s=2, alpha=0.3, label="all")
    # 연결되는 점들만 강조 표시 (선택)
    plt.scatter(Z_all[:-1][mask, 0], Z_all[:-1][mask, 1],
                c="blue", s=6, alpha=0.6, label="start")
    plt.scatter(Z_all[1:][mask, 0], Z_all[1:][mask, 1],
                c="red", s=6, alpha=0.6, label="end")
    # 선 연결
    for a, b in zip(Z_all[:-1][mask], Z_all[1:][mask]):
        plt.plot([a[0], b[0]],
                [a[1], b[1]],
                color="black", alpha=0.7, linewidth=1)
    plt.legend()
    #plt.show()
    plt.pause(0.1)

    #####---------------------------------------------------######

    input_npy = comp_npy # f"model_all/{tag_name}/ecmwf_proj_z.npy" # 학습데이터 확인    
    output_npy = os.path.join(data_dir, "train_compare_manifold_z.npy")
    #z_pre_all = np.load(input_npy)
    if bParametric:
        #proc_parametric_umap(input_npy, output_npy, toTrain=False, tag_name=tag_name)
        output_npy = os.path.join(data_dir, f"parametric_map_{comp_tag}_{tag_name}.npy")
        # proc_parametric_manifold_v2(
        #     input_npy, output_npy,              
        #     tag_name=tag_name,
        #     toTrain=False,
        #     output_dim=2, 
        #     metric="mahalanobis"
        # )
        compare_proj = np.load(input_npy)        
        best_path = "pm_run02/best.weights.h5"
        #Z_all = load_embedder_and_project(source_proj, weights_path=result["best_path"])
        Z_all_compare = load_embedder_and_project(compare_proj, weights_path=best_path)        
        # Z_all_compare = infer_embedding(
        #     model_path="pm_model_base/embedder.keras",
        #     X=compare_proj,                 # (M,D)
        #     apply_0_10_map=True,             # 저장된 0~10 좌표계로 변환
        # )
        #np.save(output_npy, Z_all_compare[:5840])
        np.save(output_npy, Z_all_compare)
        # embedder = keras.models.load_model("pm_model_base/embedder.keras")
        # compare_proj = np.load(input_npy)
        # Z_all_compare = embedder(compare_proj.astype("float32"), training=False).numpy()
        # np.save(output_npy, Z_all_compare)
        
    else:
        umap_projection(
            feature_path=comp_npy,
            mode="transform",
            umap_mode=1,
            model_path=f"{EXTRACT_DIR}/umap_model.pkl",
            save_path=output_npy,
            show_plot=False
        )
        # ref_feat = np.load(comp_npy)
        # N = ref_feat.shape[0]
        # t_index = np.arange(N)  # 시간순이면 이걸로 충분
        # Z_all_compare = transform_umap_base_spatiotemporal(ref_feat, "umap_base_st", t_index)
        # np.save(output_npy, Z_all_compare)

    Z_all_compare = np.load(output_npy)

    diff = Z_all - Z_all_compare                      # (N, 2)
    dist = np.linalg.norm(diff, axis=1)   # (N,)
    mean_d = dist.mean()
    median_d = np.median(dist)
    p90 = np.quantile(dist, 0.9)

    plt.figure(figsize=(6, 4))
    plt.hist(dist, bins=20, edgecolor="black", alpha=0.7)

    counts, bin_edges = np.histogram(dist, bins=20)
    for i in range(len(counts)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        print(f"[{left:.4f}, {right:.4f}): {counts[i]}개")

    plt.axvline(mean_d, color="red", linestyle="--", label=f"mean={mean_d:.3f}")
    plt.axvline(median_d, color="green", linestyle="--", label=f"median={median_d:.3f}")
    plt.axvline(p90, color="purple", linestyle=":", label=f"p90={p90:.3f}")

    plt.xlabel("distance")
    plt.ylabel("count")
    plt.title("Distance distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #plt.show()
    #plt.pause(0.1)

    plt.figure(figsize=(8, 8))
    if comp_tag == "translated":
        mask = dist < 0.75
    else:
        mask = dist < 0.5
    print(f"{comp_tag}")
    print(f"[INFO] selected points: {mask.sum()} / {len(mask)}, {np.round(mask.sum() / len(mask), 3)}")
        
    plt.scatter(Z_all[:, 0], Z_all[:, 1], c="gray", s=2, alpha=0.3, label="all")
    plt.scatter(Z_all[mask, 0], Z_all[mask, 1], c="blue", s=5, alpha=0.3, label="all")
    plt.scatter(Z_all_compare[mask, 0], Z_all_compare[mask, 1], c="red", s=5, alpha=0.3, label="compare") 
    for a, b in zip(Z_all[mask], Z_all_compare[mask]):
        plt.plot([a[0], b[0]], [a[1], b[1]], color="black", alpha=0.7, linewidth=1)
    plt.pause(0.1)
    #plt.show()

    # plt.clf()
    # plt.scatter(Z_all[:, 0], Z_all[:, 1], c="gray", s=2, alpha=0.3, label="all")
    # for p in range(0, len(Z_all), 100):
    #     if dist[p] < 1.0:
    #         plt.scatter(Z_all[p, 0], Z_all[p, 1], c="blue", s=5, alpha=0.3, label="all")
    #         plt.scatter(Z_all_compare[p, 0], Z_all_compare[p, 1], c="red", s=5, alpha=0.3, label="compare") 
    #         plt.plot(
    #             [Z_all[p, 0], Z_all_compare[p, 0]],
    #             [Z_all[p, 1], Z_all_compare[p, 1]],
    #             color="black",
    #             alpha=0.7,
    #             linewidth=1
    #         )
    #         plt.pause(0.1)
    # plt.show()


    # plt.clf()
    # for p in range(len(Z_all_compare)):
    #     plt.scatter(Z_all[p, 0], Z_all[p, 1], c="blue", s=5, alpha=0.3, label="all")
    #     plt.scatter(Z_all_compare[p, 0], Z_all_compare[p, 1], c="red", s=5, alpha=0.3, label="compare")
        
    #     plt.plot(
    #         [Z_all[p, 0], Z_all_compare[p, 0]],
    #         [Z_all[p, 1], Z_all_compare[p, 1]],
    #         color="black",
    #         alpha=0.7,
    #         linewidth=1
    #     )
    #     plt.pause(0.01)        
    #     #plt.waitforbuttonpress()

    # plt.show()

    return dist, Z_all

def run_parametric_manifold_learning(tag_name):

    input_npy = f"model_all/{tag_name}/era5_proj_z.npy" # 학습데이터 확인
    X = np.load(input_npy)

    # model = fit_parametric_manifold(X, k=30, num_neg=10, epochs=60, w_stress=0.1, lr=1e-3)
    # Z = model.predict(X, batch_size=4096)
    result = fit_parametric_manifold_with_earlystop(X, out_dir="pm_run02", epochs=200, patience=20)
    model = result["model"]
    Z = model.predict(X, batch_size=4096)
    #추론 예시:
    #Z_new = load_embedder_and_project(X_new, weights_path=result["best_path"])

    output_npy = f"model_all/{tag_name}/era5_manifold_2d_v2.npy" # 학습데이터 확인
    np.save(output_npy, Z)
    ## 여기에 feature space에 대한 평가 결과를 표시 ##
    X = np.load(input_npy)    
    Z = np.load(output_npy)
    metrics = evaluate_umap_embedding(X, Z, ks=(10,20,50), spearman_pairs=200_000, continuity_m=300)
    print(metrics)


def run_training_feature_spaces():

    target_date = None
    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"
    #tag_name = "CL_era5_REF-ecmwf_COND-noise-shift-blur_v1"


    mode = "extract"  # "extract" | "train_all" | "infer_set" | "infer_aug"   
    main(mode, target_date, tag_name)

    mode = "train_all"  # "extract" | "train_all" | "infer_set" | "infer_aug"   
    main(mode, target_date, tag_name)

    run_parametric_manifold_learning(tag_name)
    ## 여기까지 하면 최종 학습 완료 ##

    ## 여기부터는 feature space의 성능 평가 ##
    
    mode = "infer_aug"  # "extract" | "train_all" | "infer_set" | "infer_aug"
    main(mode, target_date, tag_name)


def compare_two_dataset_on_featurespace():
    # _proj_z.npy는 contrastive learning을 거친 128d features.
    # parametric=True/False에 따라 umap 혹은 manifold learning을 거쳐, 2d feature space에서 비교.

    tag_name = "CL_era5_REF-ecmwf_COND-noise-shift_v1"

    bTest_mode = [False, False, True]

    #### Task One # WO_CL_UMAP
    if bTest_mode[0]:
        bParametric = False
        # ### fused_z ### (without contrastive learning)
        ## train data mapping test
        src_npy = f"model_all/{tag_name}/era5_fused.npy" # 학습데이터 확인
        comp_npy = f"model_all/{tag_name}/ecmwf_fused.npy" # ecmwf        
        dist_ref_1, Z_one = compare_two_training_data(tag_name, src_npy, comp_npy, "ecmwf", bParametric=bParametric, fit_umap=True)

        comp_npy = f"model_all/{tag_name}/noise_fused.npy" # noise    
        dist_noise_1, Z_one = compare_two_training_data(tag_name, src_npy, comp_npy, "noise", bParametric=bParametric)

        comp_npy = f"model_all/{tag_name}/translated_fused.npy" # translated    
        dist_shift_1, Z_one = compare_two_training_data(tag_name, src_npy, comp_npy, "translated", bParametric=bParametric)

    #### Task Two # W_CL_UMAP
    if bTest_mode[1]:
        bParametric = False
        ### proj_z ### (with contrastive learning)
        ## train data mapping test
        src_npy = f"model_all/{tag_name}/era5_proj_z.npy" # 학습데이터 확인
        comp_npy = f"model_all/{tag_name}/ecmwf_proj_z.npy" # ecmwf        
        dist_ref_2, Z_two = compare_two_training_data(tag_name, src_npy, comp_npy, "ecmwf", bParametric=bParametric, fit_umap=True)

        comp_npy = f"model_all/{tag_name}/noise_proj_z.npy" # noise    
        dist_noise_2, Z_two = compare_two_training_data(tag_name, src_npy, comp_npy, "noise", bParametric=bParametric)

        comp_npy = f"model_all/{tag_name}/translated_proj_z.npy" # translated    
        dist_shift_2, Z_two = compare_two_training_data(tag_name, src_npy, comp_npy, "translated", bParametric=bParametric)

    #### Task Three # W_CL_PME
    if bTest_mode[2]:
        bParametric = True
        ### proj_z ### (with contrastive learning)
        ## train data mapping test
        src_npy = f"model_all/{tag_name}/era5_proj_z.npy" # 학습데이터 확인
        comp_npy = f"model_all/{tag_name}/ecmwf_proj_z.npy" # ecmwf    
        dist_ref_3, Z_three = compare_two_training_data(tag_name, src_npy, comp_npy, "ecmwf", bParametric=bParametric, fit_umap=True)

        comp_npy = f"model_all/{tag_name}/noise_proj_z.npy" # noise
        dist_noise_3, Z_three = compare_two_training_data(tag_name, src_npy, comp_npy, "noise", bParametric=bParametric)

        comp_npy = f"model_all/{tag_name}/translated_proj_z.npy" # translated
        dist_shift_3, Z_three = compare_two_training_data(tag_name, src_npy, comp_npy, "translated", bParametric=bParametric)

    if not bTest_mode[0] and not bTest_mode[1] and bTest_mode[2]:
        test_dist_shift = dist_shift_3[5840:]

        dist_val = np.asarray(test_dist_shift, dtype=float)
        dist_val = dist_val[np.isfinite(dist_val)]

        plist = [50, 75, 80, 90, 95, 99]
        vals = np.percentile(dist_val, plist)

        for p, v in zip(plist, vals):
            print(f"{p:>2} percentile : {v:.6f}")

        
        # show graph
        p90 = np.percentile(dist_val, 90)
        p95 = np.percentile(dist_val, 95)

        plt.figure(figsize=(7, 4))
        plt.hist(dist_val, bins=50, alpha=0.7)
        plt.axvline(p90, linestyle="--", label=f"90th = {p90:.4f}")
        plt.axvline(p95, linestyle="--", label=f"95th = {p95:.4f}")
        plt.xlabel("Shift-pair distance")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if bTest_mode[0] and bTest_mode[1] and bTest_mode[2]:
        ## now, Z_one, Z_two, Z_three가 있고
        ## dist_ref, dist_noise, dist_shift가 각각 있을 때, 이걸 histogram으로 각각 표시하고자 한다.
        dist_dict = {
            "ref": {
                "WO_CL_UMAP": dist_ref_1,
                "W_CL_UMAP": dist_ref_2,
                "W_CL_PME": dist_ref_3,
            },
            "noise": {
                "WO_CL_UMAP": dist_noise_1,
                "W_CL_UMAP": dist_noise_2,
                "W_CL_PME": dist_noise_3,
            },
            "shift": {
                "WO_CL_UMAP": dist_shift_1,
                "W_CL_UMAP": dist_shift_2,
                "W_CL_PME": dist_shift_3,
            }
        }
        #plot_pair_distance_histograms(dist_dict, bins=40, density=True)
        plot_pair_distance_ecdf(dist_dict, xlim=(0, 1.5), sharex=True, sharey=True)
        #plot_pair_distance_ecdf(dist_dict, xlim=(0, 10), sharex=True, sharey=True)
    

def umap_viewer_single_and_combined(variable_list=None, groups_ids=None):

    ################################
    ## UMAP viewer : new version
    
    if variable_list is None:
        single_umap_paths = []
    else:
        single_umap_paths = [f"out_fourier/umap_2d_{var}_global.npy" for var in variable_list]

    #tag_name = 'fivelevels'
    #tag_name = 'efficientnet' #'ERA5cropped'
    tag_name = 'CL_era5_REF-ecmwf_COND-noise-shift_v1'
    combined_umap_paths = [        
        f"model_all/{tag_name}/feature_space_comparison/without_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_umap_era5.npy",
        f"model_all/{tag_name}/feature_space_comparison/with_CL_PML_era5.npy",
        #"umap_base_st/Z_base.npy"
    ]
    _labels=[        
        "without_CL and UMAP",
        "with_CL and UMAP",
        "with_CL and manifold_learning",
    ]
    groups_idx = None
    if groups_ids != None:
        groups_idx = np.load(groups_ids)
    plot_umap_manual_multi(single_umap_paths, combined_umap_paths,
                                    trail_len=15, pause_sec=0.1, combined_labels=_labels,
                                    group_ids = groups_idx)