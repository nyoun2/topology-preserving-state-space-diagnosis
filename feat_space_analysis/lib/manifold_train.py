import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import pickle
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from feat_space_analysis.lib.utils import ensure_dir
from feat_space_analysis.lib.feature_extract import build_projection_head
from feat_space_analysis.lib.train_total_contrastive_learning import select_enabled_views, load_views_from_manifest


def umap_projection(
    feature_path,
    mode="fit",                      # "fit" | "transform"    
    umap_mode=1,
    model_path=None,                 # 저장/로드할 pkl 경로
    n_neighbors=30, #30,
    min_dist=0.1, #0.1,
    metric="cosine",
    random_state=42,
    save_path=None,                  # 결과 좌표 npy 저장 경로
    show_plot=True,
):
    """
    UMAP projection with save/load support

    mode="fit":
        - scaler + UMAP 학습
        - model_path에 모델 저장
    mode="transform":
        - 저장된 모델 로드
        - 신규 feature transform
    """

    assert mode in ["fit", "transform"]

    # -----------------------------
    # feature 로드
    # -----------------------------
    feats = np.load(feature_path)
    print(f"[INFO] loaded features: {feats.shape}")

    if np.isnan(feats).any() or np.isinf(feats).any():
        print("[WARN] NaN/Inf detected → replacing with 0")
        feats = np.nan_to_num(feats)

    # -----------------------------
    # FIT MODE
    # -----------------------------
    if mode == "fit":
        if model_path is None:
            raise ValueError("model_path must be provided in fit mode")

        scaler = StandardScaler()
        feats_norm = scaler.fit_transform(feats)
        print("[INFO] scaler fitted")
        
        if umap_mode == 1:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
                metric=metric,
                random_state=random_state,
                verbose=False,
            )        
        elif umap_mode == 2:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
                metric=metric,
                densmap=True,
                dens_lambda=2.0,
                random_state=random_state,
            )

        umap_2d = reducer.fit_transform(feats_norm)
        print("[INFO] UMAP fitted:", umap_2d.shape)

        # 모델 저장
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "scaler": scaler,
                    "reducer": reducer,
                },
                f,
            )
        print(f"[OK] model saved: {model_path}")

    # -----------------------------
    # TRANSFORM MODE
    # -----------------------------
    else:
        if model_path is None or not os.path.exists(model_path):
            raise ValueError("valid model_path must be provided in transform mode")

        with open(model_path, "rb") as f:
            obj = pickle.load(f)

        scaler = obj["scaler"]
        reducer = obj["reducer"]
        print("[INFO] model loaded")

        feats_norm = scaler.transform(feats)
        umap_2d = reducer.transform(feats_norm)
        print("[INFO] UMAP transformed:", umap_2d.shape)

    # -----------------------------
    # 결과 저장
    # -----------------------------
    if save_path is not None:
        np.save(save_path, umap_2d)
        print(f"[OK] saved projection: {save_path}")

    # -----------------------------
    # 시각화
    # -----------------------------
    if show_plot:        
        plt.figure(figsize=(8, 6))
        plt.scatter(
            umap_2d[:, 0],
            umap_2d[:, 1],
            s=10,
            alpha=0.7,
            c=np.arange(len(umap_2d)),
            cmap="Spectral",
        )
        plt.colorbar(label="Index")
        plt.title(f"UMAP Projection ({mode})")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.show()        

    return umap_2d


def umap_to_sphere_3d(
    X,
    n_neighbors=30,
    min_dist=0.05,
    metric="euclidean",
    random_state=42,
):
    """
    X: (N, D) input
    return:
      Z3 : (N,3) raw 3D UMAP
      S  : (N,3) projected to unit sphere
    """
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        # output_metric="euclidean",  # (기본값) 필요시 명시
    )
    Z3 = reducer.fit_transform(X).astype(np.float64)

    # ---- project to unit sphere (S^2) ----
    norms = np.linalg.norm(Z3, axis=1, keepdims=True)
    eps = 1e-12
    S = Z3 / np.maximum(norms, eps)   # (N,3), ||S_i||=1

    return Z3, S, reducer


def export_best_z_all(
    fused_path: str,            # out_features/ERA5_fused.npy
    best_proj_weights: str,     # final_best_proj.weights.h5
    out_path: str,
    proj_hidden: int,
    proj_dim: int,
    dropout_rate: float = 0.0,
    batch_size: int = 1024
):
    """
    contrastive 학습이 끝난 projection head로
    '원본 fused feature'를 그대로 projection한 결과 z를 저장.

    ✔ noise 없음
    ✔ dropout 없음
    ✔ deterministic
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    F = np.load(fused_path, mmap_mode="r")  # (N,D)
    if F.ndim != 2:
        raise ValueError(f"fused must be (N,D). got {F.shape}")

    N, D = F.shape

    # projection head 구성 + build
    proj = build_projection_head(
        input_dim=D,
        hidden_dim=proj_hidden,
        proj_dim=proj_dim,
        dropout_rate=dropout_rate,   # export는 0.0 권장
    )
    
    _ = proj(tf.zeros((1, D), dtype=tf.float32), training=False)

    if not os.path.exists(best_proj_weights):
        raise FileNotFoundError(best_proj_weights)
    proj.load_weights(best_proj_weights)

    # memmap 출력
    tmp = out_path + ".mmap"
    mm = np.memmap(tmp, mode="w+", dtype="float32", shape=(N, proj_dim))

    p = 0
    while p < N:
        b = min(batch_size, N - p)
        xb = np.asarray(F[p:p+b], dtype=np.float32)
        zb = proj(xb, training=False).numpy().astype(np.float32, copy=False)
        mm[p:p+b] = zb
        p += b
        if p == N or p % (batch_size * 50) == 0:
            print(f"[export_z] {p}/{N}")

    mm.flush()
    Z = np.array(mm)
    np.save(out_path, Z)

    del mm
    try:
        os.remove(tmp)
    except OSError:
        pass

    print(f"[export_z] saved: {out_path}  shape={Z.shape}")


def export_best_z_for_manifest_views(
    manifest_path: str,
    best_proj_weights: str,
    out_dir: str,
    proj_hidden: int,
    proj_dim: int,
    dropout_rate: float = 0.0,
    batch_size: int = 1024,
    suffix: str = "_proj_z.npy",
):
    """
    manifest의 enabled views(feature_paths or views)를 읽어서
    각 fused.npy를 proj_z.npy로 일괄 변환한다.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    views = load_views_from_manifest(manifest)
    enabled_views = select_enabled_views(views)

    ensure_dir(out_dir)

    for name, meta in enabled_views.items():
        fused_path = meta["path"]

        # 예전 규칙: <basename>_fused.npy -> <basename>_proj_z.npy
        base = os.path.splitext(os.path.basename(fused_path))[0]  # "ERA5_fused"
        if base.endswith("_fused"):
            base = base[:-6]  # "ERA5"
        out_path = os.path.join(out_dir, f"{base}_proj_z.npy")

        print(f"\n[export_z_all_views] view={name}  fused={fused_path}  ->  out={out_path}")

        export_best_z_all(
            fused_path=fused_path,
            best_proj_weights=best_proj_weights,
            out_path=out_path,
            proj_hidden=proj_hidden,
            proj_dim=proj_dim,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
        )

def export_paired_z_for_set(
    fused_ecmwf_path: str,
    fused_other_path: str,
    proj: keras.Model,
    out_path_pairs: str,
    batch_size: int = 1024
):
    """
    저장: pairs (T,2,proj_dim) where [:,0]=z_ecmwf, [:,1]=z_other
    - 추론은 deterministic 권장 => augmentation 없음
    """
    ensure_dir(os.path.dirname(out_path_pairs) or ".")

    A = np.load(fused_ecmwf_path, mmap_mode="r")
    B = np.load(fused_other_path, mmap_mode="r")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"fused must be 2D. got A{A.shape}, B{B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"feature dim mismatch: {A.shape} vs {B.shape}")

    T = min(A.shape[0], B.shape[0])
    D = int(A.shape[1])
    proj_dim = int(proj.output_shape[-1])

    tmp = out_path_pairs + ".mmap"
    mm = np.memmap(tmp, mode="w+", dtype="float32", shape=(T, 2, proj_dim))

    p = 0
    while p < T:
        b = min(batch_size, T - p)
        a = np.asarray(A[p:p+b], dtype=np.float32)
        bb = np.asarray(B[p:p+b], dtype=np.float32)

        zA = proj(a, training=False).numpy().astype(np.float32, copy=False)
        zB = proj(bb, training=False).numpy().astype(np.float32, copy=False)

        mm[p:p+b, 0, :] = zA
        mm[p:p+b, 1, :] = zB

        p += b
        if p == T or p % (batch_size * 50) == 0:
            print(f"[infer-pairs] {p}/{T}")

    mm.flush()
    pairs = np.array(mm)
    np.save(out_path_pairs, pairs)
    del mm
    try:
        os.remove(tmp)
    except OSError:
        pass

    print(f"[infer] saved pairs: {out_path_pairs}  shape={pairs.shape}")


def infer_family(name: str, family_rules: dict[str, list[str]]) -> str:
    n = name.lower()
    for fam, keys in family_rules.items():
        for k in keys:
            if k in n:
                return fam
    # fallback
    if n.lower() in ["era5", "base", "gt", "truth"]:
        return "base"
    if "ecmwf" in n or "forecast" in n or "aifs" in n:
        return "reference"
    return "other"


def build_view_config(
    feature_paths: dict[str, str],
    base_view: str,
    reference_views: list[str] | None = None,
    disable_views: list[str] | None = None,
    family_rules: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    reference_views = reference_views or []
    disable_views = set(disable_views or [])
    family_rules = family_rules or {
        "appearance": ["noise", "blur"],
        "phase": ["trans", "shift", "translated"],
        "scale": ["zoom", "scale"],
    }

    if base_view not in feature_paths:
        raise ValueError(f"base_view '{base_view}' not found in feature_paths keys={list(feature_paths.keys())}")

    for rv in reference_views:
        if rv not in feature_paths:
            raise ValueError(f"reference_view '{rv}' not found in feature_paths keys={list(feature_paths.keys())}")

    vc = {}
    for name in feature_paths.keys():
        if name == base_view:
            vc[name] = {"role": "base", "enabled": True, "family": "base"}
            continue

        if name in reference_views:
            vc[name] = {"role": "reference", "enabled": True, "family": "reference", "align_to": base_view}
            continue

        fam = infer_family(name, family_rules)
        vc[name] = {"role": "aug", "enabled": (name not in disable_views), "family": fam}

    return vc