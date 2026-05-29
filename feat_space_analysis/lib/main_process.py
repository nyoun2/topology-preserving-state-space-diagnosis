
import os
import numpy as np
import time
import tensorflow as tf
import json

from feat_space_analysis.lib.utils import ensure_dir
from feat_space_analysis.lib.data_loading import NoisyWrapperProvider, NoisyWrapperShiftProvider
from feat_space_analysis.lib.data_loading import ReferenceNpyFolderProvider, TranslatedWrapperProvider
from feat_space_analysis.lib.data_loading import GaussianBlurWrapperProvider
from feat_space_analysis.lib.preprocess import find_missing_mask_in_t_folder, compute_channel_mean_std
from feat_space_analysis.lib.preprocess import NpyFolderProvider, preprocess_standardize_with_stats
from feat_space_analysis.lib.feature_extract import MultiVarFeatureExtractor, load_projection_head_from_ckpt
from feat_space_analysis.lib.feature_extract import extract_fused_features, PairSampler, train_contrastive_on_relationship
from feat_space_analysis.lib.manifold_train import build_view_config, export_best_z_for_manifest_views
from feat_space_analysis.lib.manifold_train import umap_projection, export_best_z_all

from feat_space_analysis.lib.train_total_contrastive_learning import run_train_mode_flexible
from feat_space_analysis.lib.build_parameterized_umap import proc_parametric_umap


################################
### Main
#################################
def run_feature_inference(mode, target_date=None, tag_name=None,
    test_root=None,
    infer_out_root=None,):
    # ========================================================
    # CONFIG
    # ========================================================
    MODE = mode #"extract"  # "extract" | "train" | "infer_set" | "train_all"

    #H, W = 181, 177
    H, W = 181-10, 177-10 # cropped image
    V = 4

    EXTRACT_DIR = f"data/model_all/{tag_name}" #"out_features"
    ensure_dir(EXTRACT_DIR)

    EXTRACT_BATCH = 16
    WEIGHTS = "imagenet"
    BACKBONE_TRAINABLE = False
    ADAPTER_TRAINABLE = False
    EMBED_DIM = 256
    FUSED_DIM = 256
    FUSION_MODE = "concat_linear"

    # 핵심: EfficientNet 내부 preprocessing은 끄고,
    #         ERA5 기반 채널 mean/std로 입력 표준화
    EFFNET_INCLUDE_PREPROCESSING = False

    # --- ERA5 data ---
    #ERA5_DIR = r"out_npy_era5_850"
    ERA5_DIR = r"ext_data/out_npy_era5_850_cropped"
    FILE_PATTERN = "*.npy"
    era5_provider = NpyFolderProvider(
        folder=ERA5_DIR, H=H, W=W, V=V,
        pattern=FILE_PATTERN, sort=True, mmap_mode="r"
    )
    N = len(era5_provider)
    DATASET_INDICES = np.arange(N, dtype=np.int32)
    # cropped_shifted provider
    era5_raw_provider = NpyFolderProvider(
        "ext_data/out_npy_era5_850", H=181, W=177, V=4, 
        pattern=FILE_PATTERN, sort=True, mmap_mode="r"
    )
    # --- noisy ERA5 ---
    # noise_provider = NoisyWrapperProvider(
    #     base_provider=era5_provider, noise_std=0.02, seed=123, clip=None
    # )    
    
    # --- reference (ecmwf) --- 
    nTotal = 7308
    src_dir = 'D:/ext_data/ecmwf_4var_crop_idx' #dst_dir
    mask, ECMWF_MISSING = find_missing_mask_in_t_folder(src_dir, nTotal)
    ecmwf_provider = ReferenceNpyFolderProvider(
        folder=src_dir,
        H=H, W=W, V=V,
        n_total=nTotal,
        missing_indices=ECMWF_MISSING,
        mmap_mode="r",
        prefix="ecmwf",   # 파일명이 ecmwf_t0000.npy 형태면 prefix="ecmwf"
    )

    DATASETS = {
        "era5": era5_provider,
        "ecmwf": ecmwf_provider,
    }

    DATASETS["noise"] = NoisyWrapperProvider(era5_provider, noise_std=0.01, seed=123)
    #DATASETS["blur"] = BlurWrapperProvider(base_provider, sigma=0.8, seed=123)
    DATASETS["blur"] = GaussianBlurWrapperProvider(era5_provider, sigma_range=(0.5, 1.0), seed=123)
    DATASETS["translated"] = TranslatedWrapperProvider(era5_provider, era5_raw_provider, seed=123, clip=None, n=10)
    #DATASETS["zoom"] = ZoomWrapperProvider(base_provider, max_zoom=0.05, seed=123)


    BASE_VIEW = "era5"
    REFERENCE_VIEWS = ["ecmwf"]     # 없으면 []
    DISABLE_VIEWS = [] #["zoom"]        # optional
    FAMILY_RULES = {
        "appearance": ["noise", "blur"], #["noise", "blur", "jitter"],
        "phase": ["translated"], #["trans", "shift", "translated"],
        "scale": [], #["zoom", "scale"],
    }

    TRAIN = dict(
        batch_size=64,
        epochs=1000,
        patience=100,
        lr=1e-3,
        temperature=0.1,
        noise_std=0.01,
        dropout_rate=0.10,
        proj_hidden=256,
        proj_dim=128,
        val_ratio=0.2,
        seed=0,
    )

    REL = dict(rule="cross_aligned", A="era5", B="ecmwf", lag=1)
    CKPT_DIR = os.path.join(EXTRACT_DIR, "checkpoints_rel") #"checkpoints_rel"

    # --- infer set config ---    
    if test_root is None:
        test_root = f"D:/ext_data/{tag_name}/Test_4var1lev"
    if infer_out_root is None:
        infer_out_root = f"out_test_features/{tag_name}"
    TEST_ROOT = os.path.normpath(test_root)
    ensure_dir(TEST_ROOT)
    TEST_SET_NAME = target_date
    TEST_MEMBERS = ["ecmwf"] + [f"ai_{i:02d}" for i in range(1, 10)]
    TEST_FILE_PATTERN = "*.npy"
    INFER_OUT_ROOT = os.path.normpath(infer_out_root)
    ensure_dir(INFER_OUT_ROOT)

    # TEST_ROOT = f"D:/ext_data/{tag_name}/Test_4var1lev" #r"test_root"          # 예: test_root/20250107_00/ecmwf, ai_01..ai_09
    # ensure_dir(f"D:/ext_data/{tag_name}")
    # ensure_dir(TEST_ROOT)
    # TEST_SET_NAME = target_date # "2025030100"
    # TEST_MEMBERS = ["ecmwf"] + [f"ai_{i:02d}" for i in range(1, 10)]
    # TEST_FILE_PATTERN = "*.npy"       # 각 폴더 안에 48개
    # INFER_OUT_ROOT = f"out_test_features/{tag_name}"  # set별 결과 저장
    # ensure_dir(INFER_OUT_ROOT)

    tf.keras.backend.clear_session()
    tf.keras.backend.set_image_data_format("channels_last")

    if MODE == "extract":
        s_time = time.time()

        extractor = MultiVarFeatureExtractor(
            num_vars=V,
            input_hw=(H, W),
            embed_dim=EMBED_DIM,
            fused_dim=FUSED_DIM,
            backbone_trainable=BACKBONE_TRAINABLE,
            adapter_trainable=ADAPTER_TRAINABLE,
            weights=WEIGHTS,
            fusion_mode=FUSION_MODE,
        )
        _ = extractor(np.zeros((1, H, W, V), dtype=np.float32), training=False)

        # --- normalization stats ---
        mean_v, std_v = compute_channel_mean_std(
            provider=era5_provider,
            indices=DATASET_INDICES,
            H=H, W=W, V=V,
            batch_size=EXTRACT_BATCH,
            clip_percentile=None
        )
        norm_stats = {"mean": mean_v.tolist(), "std": std_v.tolist(), "eps": 1e-6, "clip": None}

        def PREPROCESS_FN(x, idx=None):
            return preprocess_standardize_with_stats(x, norm_stats)

        feature_paths = {}
        for name, provider in DATASETS.items():
            out_path = os.path.join(EXTRACT_DIR, f"{name}_fused.npy")
            extract_fused_features(
                extractor=extractor,
                x_provider=provider,
                indices=DATASET_INDICES,
                H=H, W=W, V=V,
                out_path=out_path,
                batch_size=EXTRACT_BATCH,
                preprocess_fn=PREPROCESS_FN,
            )
            feature_paths[name] = out_path

        # ==============================
        # ★ extractor weights 저장
        # ==============================
        extractor_wpath = os.path.join(EXTRACT_DIR, f"extractor_{tag_name}.weights.h5")
        extractor.save_weights(extractor_wpath)
        print(f"[extract] saved extractor weights: {extractor_wpath}")

        view_config = build_view_config(
            feature_paths=feature_paths,
            base_view=BASE_VIEW,
            reference_views=REFERENCE_VIEWS,
            disable_views=DISABLE_VIEWS,
            family_rules=FAMILY_RULES,
        )

        manifest = {
            "shape": {"H": H, "W": W, "V": V, "N": int(len(DATASET_INDICES))},
            "extractor": {
                "weights": WEIGHTS,
                "embed_dim": EMBED_DIM,
                "fused_dim": FUSED_DIM,
                "fusion_mode": FUSION_MODE,
                "weights_path": extractor_wpath,   # ★ 추가
            },
            "normalization": norm_stats,
            "feature_paths": feature_paths,
            "view_config": view_config,
        }
        # reference view별 valid index 기록 (contrastive 학습용)
        valid_indices = {}
        if isinstance(ecmwf_provider, ReferenceNpyFolderProvider):
            valid_indices["ecmwf"] = ecmwf_provider.valid_indices().tolist()
        if valid_indices:
            manifest["valid_indices"] = valid_indices

        with open(os.path.join(EXTRACT_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[extract] done. time={time.time()-s_time:.1f}s")

    elif MODE == "train_all":
        best_weights = run_train_mode_flexible(EXTRACT_DIR, CKPT_DIR, TRAIN)

        #best_weights = f"{EXTRACT_DIR}/checkpoints_rel/_best_finetune.weights.h5"

        export_best_z_for_manifest_views(
            manifest_path=os.path.join(EXTRACT_DIR, "manifest.json"),
            best_proj_weights=best_weights,
            out_dir=EXTRACT_DIR,               # 또는 os.path.join(EXTRACT_DIR, "proj")
            proj_hidden=TRAIN["proj_hidden"],
            proj_dim=TRAIN["proj_dim"],
            dropout_rate=0.0,
            batch_size=1024,
        )

        input_npy = f"{EXTRACT_DIR}/era5_proj_z.npy"
        output_npy = f"{EXTRACT_DIR}/parametric_map_era5_{tag_name}.npy"
        
        umap_projection(
            feature_path=input_npy,
            mode="fit", #"fit", "transform"
            umap_mode=1,
            model_path=f"{EXTRACT_DIR}/umap_model.pkl",
            save_path=output_npy,
            show_plot=True
        )

        #proc_parametric_umap(input_npy, output_npy, toTrain=True, tag_name=tag_name)
        # proc_parametric_manifold_v2(
        #     input_npy, output_npy, 
        #     tag_name=tag_name, 
        #     toTrain=True,             
        #     output_dim=2, 
        #     metric="mahalanobis"
        # )
        
        # train_parametric_manifold(
        #     X_base_npy=input_npy,
        #     X_ref=None,
        #     epochs=1000,
        #     lr=1e-3,
        #     lambda_temp=0.0, #0.05,
        #     lambda_ref=0.0,
        #     out_dir="pm_model_base",
        #     num_pairs=8192,
        #     k_nn=5,
        #     pair_seed=0,
        #     plot_range=(0.0, 10.0),
        # )


    elif MODE == "train":
        manifest_path = os.path.join(EXTRACT_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest not found. run extract first: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        feature_paths = manifest["feature_paths"]

        sampler = PairSampler(
            rule=REL["rule"],
            A=REL["A"],
            B=REL.get("B", None),
            lag=REL.get("lag", 1)
        )

        run_dir = os.path.join(CKPT_DIR, f"{REL['rule']}_{REL['A']}_{REL.get('B', REL['A'])}_lag{REL.get('lag', 1)}")
        train_contrastive_on_relationship(
            feature_paths=feature_paths,
            sampler=sampler,
            batch_size=TRAIN["batch_size"],
            epochs=TRAIN["epochs"],
            patience=TRAIN["patience"],
            lr=TRAIN["lr"],
            temperature=TRAIN["temperature"],
            noise_std=TRAIN["noise_std"],
            dropout_rate=TRAIN["dropout_rate"],
            proj_hidden=TRAIN["proj_hidden"],
            proj_dim=TRAIN["proj_dim"],
            ckpt_dir=run_dir,
            val_ratio=TRAIN["val_ratio"],
            seed=TRAIN["seed"]
        )

        export_best_z_all(
            fused_path=f"{EXTRACT_DIR}/era5_fused.npy",
            best_proj_weights=f"{run_dir}/final_best_proj.weights.h5",
            out_path=f"{EXTRACT_DIR}/ERA5_proj_z.npy",
            proj_hidden=TRAIN["proj_hidden"],
            proj_dim=TRAIN["proj_dim"]
        )

        export_best_z_all(
            fused_path=f"{EXTRACT_DIR}/ecmwf_fused.npy",
            best_proj_weights=f"{run_dir}/final_best_proj.weights.h5",
            out_path=f"{EXTRACT_DIR}/ecmwf_proj_z.npy",
            proj_hidden=TRAIN["proj_hidden"],
            proj_dim=TRAIN["proj_dim"]
        )

        input_npy = f"{EXTRACT_DIR}/ERA5_proj_z.npy"
        output_npy = f"out_fourier/parametric_umap_z_post_{tag_name}.npy"
        proc_parametric_umap(input_npy, output_npy, toTrain=True, tag_name=tag_name)

    elif MODE == "infer_set":
        """
        TEST_ROOT/TEST_SET_NAME/
          ecmwf/*.npy (48)
          ai_01/*.npy (48)
          ...
          ai_09/*.npy (48)

        Output:
          out_test_features/TEST_SET_NAME/
            fused/ecmwf_fused.npy, ai_01_fused.npy, ...
            proj/z_ecmwf.npy
            proj/pairs_ai_01.npy ... (48,2,proj_dim)
        """        

        manifest_path = os.path.join(EXTRACT_DIR, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        norm_stats = manifest["normalization"]
        ext_cfg = manifest["extractor"]

        def PREPROCESS_FN(x, idx=None):
            return preprocess_standardize_with_stats(x, norm_stats)

        extractor = MultiVarFeatureExtractor(
            num_vars=V,
            input_hw=(H, W),
            embed_dim=int(ext_cfg["embed_dim"]),
            fused_dim=int(ext_cfg["fused_dim"]),
            backbone_trainable=False,
            adapter_trainable=False,
            weights=ext_cfg["weights"],
            fusion_mode=ext_cfg["fusion_mode"],
        )

        # build
        _ = extractor(np.zeros((1, H, W, V), dtype=np.float32), training=False)

        # ==============================
        # ★ extractor weights 로드
        # ==============================
        wpath = ext_cfg.get("weights_path", None)
        if wpath is None or not os.path.exists(wpath):
            raise FileNotFoundError(f"Extractor weights not found: {wpath}")

        extractor.load_weights(wpath)
        print(f"[infer_set] loaded extractor weights: {wpath}")

        # projection head load (run_dir를 네가 지정)
        # 여기서는 REL 기반 폴더를 사용한다고 가정
        #run_dir = os.path.join(CKPT_DIR, f"{REL['rule']}_{REL['A']}_{REL.get('B', REL['A'])}_lag{REL.get('lag', 1)}")
        run_dir = f"{EXTRACT_DIR}/checkpoints_rel"

        # test set providers 만들기
        set_dir = os.path.join(TEST_ROOT, TEST_SET_NAME)
        if not os.path.isdir(set_dir):
            raise FileNotFoundError(f"test set dir not found: {set_dir}")

        base_mem = None
        providers = {}
        for mem in TEST_MEMBERS:
            mem_dir = os.path.join(set_dir, mem)
            if not os.path.isdir(mem_dir):
                print(f"member dir not found: {mem_dir}")
                continue
                #raise FileNotFoundError(f"member dir not found: {mem_dir}")
            prov = NpyFolderProvider(folder=mem_dir, H=H, W=W, V=V, pattern=TEST_FILE_PATTERN, sort=True, mmap_mode="r")
            if len(prov) < 1:
                print(f"{mem} has no files: {len(prov)}")
                #continue
                raise ValueError(f"{mem} has no files: {len(prov)}")
            providers[mem] = prov
            if base_mem == None:
                base_mem = mem

        # T = 48
        # idx48 = np.arange(T, dtype=np.int32)

        #T = min(len(p) for p in providers.values())  # 모든 멤버 공통 길이
        #T = len(providers["ecmwf"])
        T = len(providers[base_mem])
        # 또는 ecmwf 기준으로만 자르려면: T = len(providers["ecmwf"])
        idxN = np.arange(T, dtype=np.int32)
        print(f"[infer_set] T={T} (common length across members)")

        out_set_root = os.path.join(INFER_OUT_ROOT, TEST_SET_NAME)
        out_fused_dir = os.path.join(out_set_root, "fused")
        out_proj_dir = os.path.join(out_set_root, "proj")
        ensure_dir(out_set_root)
        ensure_dir(out_fused_dir)
        ensure_dir(out_proj_dir)

        # 1) fused 추출 저장 (N개씩)
        fused_paths = {}
        for mem, prov in providers.items():
            out_path = os.path.join(out_fused_dir, f"{mem}_fused.npy")
            extract_fused_features(
                extractor=extractor,
                x_provider=prov,
                indices=idxN,
                H=H, W=W, V=V,
                out_path=out_path,
                batch_size=EXTRACT_BATCH,
                preprocess_fn=PREPROCESS_FN,
                dtype_out="float32",
                verbose=True
            )
            fused_paths[mem] = out_path

        # # 2) projection으로 z & pairs 저장
        # #    input_dim은 fused feature dim
        # D = int(np.load(fused_paths["ecmwf"], mmap_mode="r").shape[1])
        # proj = load_projection_head_from_ckpt(run_dir, input_dim=D, proj_hidden=TRAIN["proj_hidden"], proj_dim=TRAIN["proj_dim"])

        # # ecmwf z 따로 저장 (재사용)
        # ecmwf_fused = np.load(fused_paths["ecmwf"], mmap_mode="r")
        # z_ecmwf = proj(np.asarray(ecmwf_fused[:T], np.float32), training=False).numpy().astype(np.float32)
        # z_ecmwf_path = os.path.join(out_proj_dir, "z_ecmwf.npy")
        # np.save(z_ecmwf_path, z_ecmwf)
        # print(f"[infer] saved {z_ecmwf_path} shape={z_ecmwf.shape}")

        # # ai별 pairs 저장
        # #for mem in TEST_MEMBERS:
        # for mem in providers.keys():
        #     if mem == "ecmwf":
        #         continue
        #     out_pairs = os.path.join(out_proj_dir, f"pairs_{mem}.npy")
        #     export_paired_z_for_set(
        #         fused_ecmwf_path=fused_paths["ecmwf"],
        #         fused_other_path=fused_paths[mem],
        #         proj=proj,
        #         out_path_pairs=out_pairs,
        #         batch_size=1024
        #     )
        # ==============================
        # 2) projection으로 z & pairs 저장 (best weights 기반)
        # ==============================

        # best projection weights 경로를 "명시적으로" 잡아라
        # (당신 프로젝트에서 파일명이 final_best_proj.weights.h5 라고 했으니 그 기준)
        best_proj_weights = os.path.join(run_dir, "_best_finetune.weights.h5")
        if not os.path.exists(best_proj_weights):
            raise FileNotFoundError(f"best proj weights not found: {best_proj_weights}")

        # ecmwf z 저장
        z_ecmwf_path = os.path.join(out_proj_dir, "z_ecmwf.npy")
        if base_mem == "ecmwf":            
            export_best_z_all(
                fused_path=fused_paths["ecmwf"],
                best_proj_weights=best_proj_weights,
                out_path=z_ecmwf_path,
                proj_hidden=int(TRAIN["proj_hidden"]),
                proj_dim=int(TRAIN["proj_dim"]),
                dropout_rate=0.0,           # deterministic
                batch_size=1024
            )

        # ai 멤버도 각자 z 저장 + pairs 저장
        proj_dim = int(TRAIN["proj_dim"])

        if base_mem == "ecmwf":
            # z_ecmwf는 이후 pairs 생성에 재사용
            Z_ecmwf = np.load(z_ecmwf_path, mmap_mode="r")   # (T, proj_dim) expected
            if Z_ecmwf.shape[0] != T:
                # 혹시 T를 ecmwf 길이로 잡았는데 fused가 더 길거나 짧으면 여기서 강제 정렬
                T = min(T, Z_ecmwf.shape[0])
                print(f"[infer_set] adjusted T based on z_ecmwf: T={T}")

        for mem in providers.keys():
            if mem == "ecmwf":
                continue

            # 1) z_mem 저장
            z_mem_path = os.path.join(out_proj_dir, f"z_{mem}.npy")
            export_best_z_all(
                fused_path=fused_paths[mem],
                best_proj_weights=best_proj_weights,
                out_path=z_mem_path,
                proj_hidden=int(TRAIN["proj_hidden"]),
                proj_dim=proj_dim,
                dropout_rate=0.0,
                batch_size=1024
            )

            if base_mem == "ecmwf":
                # 2) pairs_mem 저장: (T,2,proj_dim)
                Z_mem = np.load(z_mem_path, mmap_mode="r")
                Tm = min(T, Z_mem.shape[0], Z_ecmwf.shape[0])

                out_pairs = os.path.join(out_proj_dir, f"pairs_{mem}.npy")

                tmp = out_pairs + ".mmap"
                mm = np.memmap(tmp, mode="w+", dtype="float32", shape=(Tm, 2, proj_dim))
                mm[:, 0, :] = np.asarray(Z_ecmwf[:Tm], dtype=np.float32)
                mm[:, 1, :] = np.asarray(Z_mem[:Tm], dtype=np.float32)
                mm.flush()

                pairs = np.array(mm)
                np.save(out_pairs, pairs)

                del mm
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            
                print(f"[infer_set] saved {out_pairs} shape={pairs.shape}")

        # infer manifest 저장
        infer_manifest = {
            "test_set": TEST_SET_NAME,
            #"members": TEST_MEMBERS,
            "members": sorted(list(providers.keys())),  # 실제 처리된 멤버
            "T": T,
            "normalization_used": norm_stats,
            "extractor_cfg": ext_cfg,
            "run_dir": run_dir,
            "best_proj_weights": best_proj_weights,   # ✅ 추가
            "outputs": {
                "fused_paths": fused_paths,
                "z_ecmwf": z_ecmwf_path,
                "pairs_dir": out_proj_dir,
                "z_paths": {mem: os.path.join(out_proj_dir, f"z_{mem}.npy") for mem in providers.keys() if mem != "ecmwf"}  # 옵션
            }
        }
        with open(os.path.join(out_set_root, "infer_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(infer_manifest, f, indent=2)
        print(f"[infer] done. saved infer_manifest.json at: {out_set_root}")

    elif MODE == "infer_aug":
        """
        학습(ERA5 cropped)을 기반으로 NoisyWrapperShiftProvider로 만든
        crop/shift 데이터만 extractor+projection해서 저장.
        """
        # --- manifest / extractor cfg 로드 (infer_set과 동일) ---
        manifest_path = os.path.join(EXTRACT_DIR, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        norm_stats = manifest["normalization"]
        ext_cfg = manifest["extractor"]

        def PREPROCESS_FN(x, idx=None):
            return preprocess_standardize_with_stats(x, norm_stats)

        extractor = MultiVarFeatureExtractor(
            num_vars=V,
            input_hw=(H, W),
            embed_dim=int(ext_cfg["embed_dim"]),
            fused_dim=int(ext_cfg["fused_dim"]),
            backbone_trainable=False,
            adapter_trainable=False,
            weights=ext_cfg["weights"],
            fusion_mode=ext_cfg["fusion_mode"],
        )
        _ = extractor(np.zeros((1, H, W, V), dtype=np.float32), training=False)

        wpath = ext_cfg.get("weights_path", None)
        if wpath is None or not os.path.exists(wpath):
            raise FileNotFoundError(f"Extractor weights not found: {wpath}")
        extractor.load_weights(wpath)
        print(f"[infer_shift] loaded extractor weights: {wpath}")

        # --- projection head 로드 (infer_set과 동일한 run_dir 규칙 사용) ---
        run_dir = os.path.join(
            CKPT_DIR,
            f"{REL['rule']}_{REL['A']}_{REL.get('B', REL['A'])}_lag{REL.get('lag', 1)}"
        )

        # =========================
        # ★ 여기서 provider만 교체
        # =========================
        era5_raw_provider = NpyFolderProvider(
            folder="out_npy_era5_850",   # 원본(181x177) 경로로 바꿔
            H=181, W=177, V=V,
            pattern=FILE_PATTERN, sort=True, mmap_mode="r"
        )

        x_provider = NoisyWrapperShiftProvider(
            base_provider=era5_provider,      # cropped(171x167)
            raw_provider=era5_raw_provider,   # raw(181x177)
            noise_std=0.02, seed=123, clip=None, n=10
        )

        T = len(x_provider)  # 대략 7300
        idxN = np.arange(T, dtype=np.int32)
        print(f"[infer_shift] T={T}")

        # --- output dirs ---
        # out_root = os.path.join(f"out_aug_features/{tag_name}", "ERA5_SHIFT")
        # out_fused_dir = os.path.join(out_root, "fused")
        # out_proj_dir = os.path.join(out_root, "proj")
        # ensure_dir(out_root); ensure_dir(out_fused_dir); ensure_dir(out_proj_dir)

        out_root = os.path.join(INFER_OUT_ROOT, "comparison")
        out_fused_dir = os.path.join(out_root, "fused")
        out_proj_dir = os.path.join(out_root, "proj")
        ensure_dir(out_root)
        ensure_dir(out_fused_dir)
        ensure_dir(out_proj_dir)

        # 1) fused 추출
        fused_path = os.path.join(out_fused_dir, "era5_crop_shift_fused.npy")
        extract_fused_features(
            extractor=extractor,
            x_provider=x_provider,
            indices=idxN,
            H=H, W=W, V=V,
            out_path=fused_path,
            batch_size=EXTRACT_BATCH,
            preprocess_fn=PREPROCESS_FN,
            dtype_out="float32",
            verbose=True
        )

        # 2) projection (D는 fused에서 확정해서 로드하는 게 가장 안전)
        D = int(np.load(fused_path, mmap_mode="r").shape[1])
        proj = load_projection_head_from_ckpt(
            run_dir,
            input_dim=D,
            proj_hidden=TRAIN["proj_hidden"],
            proj_dim=TRAIN["proj_dim"],
        )

        fused = np.load(fused_path, mmap_mode="r")
        z = proj(np.asarray(fused[:T], np.float32), training=False).numpy().astype(np.float32)
        z_path = os.path.join(out_proj_dir, "z_era5_crop_shift.npy")
        np.save(z_path, z)
        print(f"[infer_shift] saved {z_path} shape={z.shape}")

        # --- manifest ---
        infer_manifest = {
            "mode": "infer_shift",
            "tag_name": tag_name,
            "T": T,
            "provider": "NoisyWrapperShiftProvider",
            "normalization_used": norm_stats,
            "extractor_cfg": ext_cfg,
            "run_dir": run_dir,
            "outputs": {
                "fused": fused_path,
                "z": z_path
            }
        }
        with open(os.path.join(out_root, "infer_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(infer_manifest, f, indent=2)
        print(f"[infer_shift] done. saved infer_manifest.json at: {out_root}")

    else:
        raise ValueError(f"Unknown MODE: {MODE}")