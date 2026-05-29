"""
Contrastive training (3-stage) with flexible view sets from manifest.

- manifest can keep legacy: manifest["feature_paths"] = {name: path}
  and optionally add:
    manifest["view_config"] = {
        "ERA5": {"role":"base","enabled":True},
        "ecmwf": {"role":"reference","enabled":True, "align_to":"ERA5"},
        "noise": {"role":"aug","enabled":True,"family":"appearance"},
        ...
    }

- or a newer format:
    manifest["views"] = {
        "base": {"path": "...", "role":"base", "enabled":True, "family":"base"},
        "ecmwf": {"path": "...", "role":"reference", "enabled":True, "family":"reference", "align_to":"base"},
        "blur": {"path":"...", "role":"aug", "enabled":True, "family":"appearance"},
        "translated_v2": {"path":"...", "role":"aug", "enabled":True, "family":"phase"},
        ...
    }

What this code does:
- Reads enabled views dynamically.
- Builds stage policies (warmup/main/finetune) that are view-name agnostic (family/role based).
- Samples multi-positive pairs:
    * base anchor
    * aug positives from enabled aug families (appearance/phase/scale/...) according to stage policy
    * temporal positives (base t±1) according to stage policy
    * reference positives (e.g., ecmwf) optionally included as positives
- Filters positives within candidate pool by RMSE (raw if available, else feature RMSE).
- Loss:
    * multi-positive SupCon (InfoNCE)
    * + optional align loss between base and reference (only if reference views exist & enabled & stage weight > 0)
    * + optional temporal smoothness (off by default, but supported)
"""

import os, json, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import tensorflow as tf


# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_npy_2d(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array (N,D), got {arr.shape}: {path}")
    return arr

def feature_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.mean(d * d)))

def raw_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.mean(d * d)))

def l2_normalize(x: tf.Tensor, axis=-1, eps=1e-6) -> tf.Tensor:
    return x / (tf.norm(x, axis=axis, keepdims=True) + eps)


# =========================================================
# Projection head
# =========================================================
def build_projection_head(input_dim: int, hidden_dim: int, proj_dim: int, dropout_rate: float):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(inp)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(hidden_dim, activation="gelu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(proj_dim, activation=None)(x)
    return tf.keras.Model(inp, out, name="proj_head")


# =========================================================
# SupCon (multi-positive InfoNCE)
# =========================================================
@tf.function
def supcon_loss(
    z: tf.Tensor,               # (M, d)
    labels: tf.Tensor,          # (M,)
    temperature: float = 0.1,
    sample_weight: Optional[tf.Tensor] = None,   # (M,)
) -> tf.Tensor:
    z = l2_normalize(z)
    sim = tf.matmul(z, z, transpose_b=True) / temperature  # (M,M)

    M = tf.shape(z)[0]
    labels = tf.reshape(labels, (-1, 1))  # (M,1)
    same = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)  # (M,M)

    eye = tf.eye(M, dtype=tf.float32)
    pos_mask = same - eye
    logits_mask = 1.0 - eye  # exclude self in denom

    # stabilize
    sim_max = tf.reduce_max(sim * logits_mask, axis=1, keepdims=True)
    sim = sim - sim_max

    exp_sim = tf.exp(sim) * logits_mask
    denom = tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-9
    log_prob = sim - tf.math.log(denom)

    pos_cnt = tf.reduce_sum(pos_mask, axis=1) + 1e-9
    loss_i = -tf.reduce_sum(pos_mask * log_prob, axis=1) / pos_cnt  # (M,)

    if sample_weight is not None:
        sw = tf.cast(sample_weight, tf.float32)
        return tf.reduce_sum(loss_i * sw) / (tf.reduce_sum(sw) + 1e-9)

    return tf.reduce_mean(loss_i)


# =========================================================
# Manifest parsing: views (flexible)
# =========================================================
def _infer_family_from_name(name: str) -> str:
    n = name.lower()
    if n in ["base", "era5", "gt", "truth"]:
        return "base"
    if "noise" in n:
        return "appearance"
    if "blur" in n or "smooth" in n:
        return "appearance"
    if "trans" in n or "shift" in n:
        return "phase"
    if "zoom" in n or "scale" in n:
        return "scale"
    if "ecmwf" in n or "aifs" in n or "forecast" in n:
        return "reference"
    return "other"

def load_views_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns: views dict
      views[name] = {
        "path": str,
        "enabled": bool,
        "role": "base" | "reference" | "aug",
        "family": str,
        "align_to": Optional[str]
      }
    """
    if "views" in manifest:
        views = {}
        for name, meta in manifest["views"].items():
            if "path" not in meta:
                raise ValueError(f"manifest['views'][{name}] missing 'path'")
            role = meta.get("role", "aug")
            enabled = bool(meta.get("enabled", True))
            family = meta.get("family", _infer_family_from_name(name))
            align_to = meta.get("align_to", None)
            views[name] = {"path": meta["path"], "enabled": enabled, "role": role, "family": family, "align_to": align_to}
        return views

    # legacy: feature_paths + optional view_config
    if "feature_paths" not in manifest:
        raise ValueError("manifest must include 'views' or 'feature_paths'")

    fp = manifest["feature_paths"]
    vc = manifest.get("view_config", {})

    views = {}
    for name, path in fp.items():
        meta = vc.get(name, {})
        enabled = bool(meta.get("enabled", True))
        role = meta.get("role", None)
        if role is None:
            # legacy heuristic
            role = "base" if name.lower() in ["era5", "base", "gt"] else ("reference" if "ecmwf" in name.lower() else "aug")
        family = meta.get("family", _infer_family_from_name(name))
        align_to = meta.get("align_to", None)
        views[name] = {"path": path, "enabled": enabled, "role": role, "family": family, "align_to": align_to}

    return views

def select_enabled_views(views: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    enabled = {k: v for k, v in views.items() if bool(v.get("enabled", True))}
    if len(enabled) == 0:
        raise ValueError("No enabled views found in manifest.")
    # validate base
    base_names = [k for k, v in enabled.items() if v.get("role") == "base"]
    if len(base_names) != 1:
        raise ValueError(f"Exactly one enabled base view required, found: {base_names}")
    return enabled

def get_base_name(enabled_views: Dict[str, Dict[str, Any]]) -> str:
    for k, v in enabled_views.items():
        if v["role"] == "base":
            return k
    raise RuntimeError("base view not found")

def get_reference_names(enabled_views: Dict[str, Dict[str, Any]]) -> List[str]:
    return [k for k, v in enabled_views.items() if v["role"] == "reference"]

def get_aug_names(enabled_views: Dict[str, Dict[str, Any]]) -> List[str]:
    return [k for k, v in enabled_views.items() if v["role"] == "aug"]

def group_by_family(enabled_views: Dict[str, Dict[str, Any]], names: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for n in names:
        fam = enabled_views[n].get("family", "other")
        out.setdefault(fam, []).append(n)
    return out


# =========================================================
# Stage policy (view-name agnostic)
# =========================================================
@dataclass
class StagePolicy:
    name: str
    steps: int
    lr: float
    temperature: float

    # NEW: stage curriculum control
    max_epochs: int
    patience: int
    min_delta: float  # 개선 판정 여유(옵션)

    # positives
    use_temporal: bool
    temporal_k: int                       # usually 1
    include_reference_as_positive: bool   # add reference view as positive
    pos_per_family: Dict[str, int]        # for aug families: how many views to sample per anchor

    # weights (per-view for SupCon weighting)
    weight_base: float
    weight_reference: float
    weight_per_family: Dict[str, float]   # for aug families, fallback 1.0

    # loss weights
    w_supcon: float
    w_align_reference: float
    w_temporal_smooth: float

    # rmse filtering for positive candidate pool
    use_raw_rmse: bool
    pos_rmse_keep_quantile: float

    # hard neg fraction (kept for future expansion; SupCon uses batch negatives implicitly)
    neg_hard_fraction: float


@dataclass
class TrainV2Config:
    batch_anchors: int
    proj_hidden: int
    proj_dim: int
    dropout_rate: float
    seed: int
    val_ratio: float
    patience: int

    temporal_exclude: int
    far_neg_min_dt: int
    hard_neg_pool_window: int


def build_default_stage_policies(TRAIN: Dict[str, Any]) -> List[StagePolicy]:
    # You can override any of these via TRAIN dict.
    return [
        StagePolicy(
            name="warmup",
            steps=int(TRAIN.get("warmup_steps", 200)),
            lr=float(TRAIN.get("warmup_lr", TRAIN["lr"])),
            temperature=float(TRAIN.get("warmup_temperature", TRAIN.get("temperature", 0.1))),

            max_epochs = 300,
            patience = 50,
            min_delta = 1e-6,

            use_temporal=False,
            temporal_k=0,
            include_reference_as_positive=False,

            pos_per_family=dict(TRAIN.get("warmup_pos_per_family", {"appearance": 1, "phase": 1, "scale": 1})),
            weight_base=float(TRAIN.get("warmup_weight_base", 1.0)),
            weight_reference=float(TRAIN.get("warmup_weight_reference", 1.0)),
            weight_per_family=dict(TRAIN.get("warmup_weight_per_family", {"appearance": 1.0, "phase": 0.6, "scale": 0.3})),

            w_supcon=float(TRAIN.get("warmup_w_supcon", 1.0)),
            w_align_reference=float(TRAIN.get("warmup_w_align_reference", 0.0)),
            w_temporal_smooth=float(TRAIN.get("warmup_w_temporal_smooth", 0.0)),

            use_raw_rmse=bool(TRAIN.get("use_raw_rmse", False)),
            pos_rmse_keep_quantile=float(TRAIN.get("warmup_pos_rmse_keep_quantile", TRAIN.get("pos_rmse_keep_quantile", 1.0))),
            neg_hard_fraction=float(TRAIN.get("warmup_neg_hard_fraction", TRAIN.get("neg_hard_fraction_warmup", 0.2))),
        ),
        StagePolicy(
            name="main",
            steps=int(TRAIN.get("main_steps", 400)),
            lr=float(TRAIN.get("main_lr", TRAIN["lr"])),
            temperature=float(TRAIN.get("main_temperature", TRAIN.get("temperature", 0.1))),

            max_epochs = 600,
            patience = 100,
            min_delta = 1e-6,

            use_temporal=bool(TRAIN.get("main_use_temporal", True)),
            temporal_k=int(TRAIN.get("main_temporal_k", 2)),
            include_reference_as_positive=bool(TRAIN.get("main_include_reference_as_positive", True)),

            pos_per_family=dict(TRAIN.get("main_pos_per_family", {"appearance": 1, "phase": 1, "scale": 1})),
            weight_base=float(TRAIN.get("main_weight_base", 1.0)),
            weight_reference=float(TRAIN.get("main_weight_reference", 1.0)),
            weight_per_family=dict(TRAIN.get("main_weight_per_family", {"appearance": 0.2, "phase": 0.6, "scale": 0.2})),

            w_supcon=float(TRAIN.get("main_w_supcon", 1.0)),
            w_align_reference=float(TRAIN.get("main_w_align_reference", TRAIN.get("w_align_ref_main", 0.5))),
            w_temporal_smooth=float(TRAIN.get("main_w_temporal_smooth", TRAIN.get("w_temporal_smooth_main", 0.4))),

            use_raw_rmse=bool(TRAIN.get("use_raw_rmse", False)),
            pos_rmse_keep_quantile=float(TRAIN.get("main_pos_rmse_keep_quantile", TRAIN.get("pos_rmse_keep_quantile", 0.5))),
            neg_hard_fraction=float(TRAIN.get("main_neg_hard_fraction", TRAIN.get("neg_hard_fraction_main", 0.3))),
        ),
        StagePolicy(
            name="finetune",
            steps=int(TRAIN.get("finetune_steps", 200)),
            lr=float(TRAIN.get("finetune_lr", TRAIN["lr"] * 0.3)),
            temperature=float(TRAIN.get("finetune_temperature", TRAIN.get("temperature", 0.1))),

            max_epochs = 300,
            patience = 60,
            min_delta = 1e-6,

            use_temporal=bool(TRAIN.get("finetune_use_temporal", True)),
            temporal_k=int(TRAIN.get("finetune_temporal_k", 2)),
            include_reference_as_positive=bool(TRAIN.get("finetune_include_reference_as_positive", True)),

            pos_per_family=dict(TRAIN.get("finetune_pos_per_family", {"appearance": 0, "phase": 0, "scale": 0})),
            weight_base=float(TRAIN.get("finetune_weight_base", 1.0)),
            weight_reference=float(TRAIN.get("finetune_weight_reference", 1.0)),
            weight_per_family=dict(TRAIN.get("finetune_weight_per_family", {"appearance": 0.0, "phase": 0.0, "scale": 0.0})),

            w_supcon=float(TRAIN.get("finetune_w_supcon", 1.0)),
            w_align_reference=float(TRAIN.get("finetune_w_align_reference", TRAIN.get("w_align_ref_finetune", 0.3))),
            w_temporal_smooth=float(TRAIN.get("finetune_w_temporal_smooth", TRAIN.get("w_temporal_smooth_finetune", 0.7))),

            use_raw_rmse=bool(TRAIN.get("use_raw_rmse", False)),
            pos_rmse_keep_quantile=float(TRAIN.get("finetune_pos_rmse_keep_quantile", TRAIN.get("pos_rmse_keep_quantile", 0.3))),
            neg_hard_fraction=float(TRAIN.get("finetune_neg_hard_fraction", TRAIN.get("neg_hard_fraction_finetune", 0.4))),
        ),
    ]


# =========================================================
# Sampler (flexible views)
# =========================================================
class FlexibleSampler:
    """
    features: dict[view_name] -> (N,D)
    raw: optional dict[view_name] -> (N, ...) for RMSE scoring
    enabled_views: meta dict (role/family/align_to)
    """
    def __init__(
        self,
        features: Dict[str, np.ndarray],
        raw: Optional[Dict[str, np.ndarray]],
        enabled_views: Dict[str, Dict[str, Any]],
        cfg: TrainV2Config,
        rng: np.random.Generator,
    ):
        self.F = features
        self.R = raw
        self.views = enabled_views
        self.cfg = cfg
        self.rng = rng

        self.base = get_base_name(enabled_views)
        self.refs = get_reference_names(enabled_views)
        self.augs = get_aug_names(enabled_views)
        self.aug_by_fam = group_by_family(enabled_views, self.augs)

        # validate shapes
        if self.base not in self.F:
            raise ValueError(f"Base view '{self.base}' missing in loaded features.")
        self.N = int(self.F[self.base].shape[0])
        self.D = int(self.F[self.base].shape[1])

        for name in enabled_views.keys():
            if name not in self.F:
                raise ValueError(f"Enabled view '{name}' missing in loaded features dict.")
            if self.F[name].shape != (self.N, self.D):
                raise ValueError(f"Shape mismatch for '{name}': {self.F[name].shape} vs base {(self.N,self.D)}")

        if self.R is not None:
            for name in enabled_views.keys():
                if name in self.R and self.R[name].shape[0] != self.N:
                    raise ValueError(f"Raw length mismatch for '{name}': {self.R[name].shape[0]} vs base {self.N}")

    def _rmse_score(self, view_name: str, i: int, j: int, use_raw: bool) -> float:
        # score between base[i] and view[j] (typically j==i except temporal)
        if use_raw and (self.R is not None) and (self.base in self.R) and (view_name in self.R):
            return raw_rmse(self.R[self.base][i], self.R[view_name][j])
        return feature_rmse(self.F[self.base][i], self.F[view_name][j])

    def _select_from_candidates(self, scored: List[Tuple[float, Tuple[str, int]]], keep_quantile: float, k: int) -> List[Tuple[str, int]]:
        if len(scored) == 0 or k <= 0:
            return []
        scored.sort(key=lambda x: x[0])  # small is better
        keep = max(1, int(math.ceil(len(scored) * keep_quantile)))
        kept = scored[:keep]
        # pick first k (or random among kept if you want)
        picks = [pair for _, pair in kept[:k]]
        return picks

    def pick_views_for_anchor(self, i: int, stage: StagePolicy) -> Tuple[List[Tuple[str, int, int, str]], Dict[str, Tuple[str, int]]]:
        """
        Returns:
          views: list of (view_name, idx, anchor_id, view_kind)
            view_kind in {"base_anchor","aug","reference","temporal"}
          special: dict for extra loss terms:
            {"ref": (ref_view_name, idx), "temporal": (base_view_name, idx)}
        """
        views: List[Tuple[str, int, int, str]] = []
        special: Dict[str, Tuple[str, int]] = {}

        # base anchor (always)
        views.append((self.base, i, -1, "base_anchor"))  # anchor_id filled later by caller

        # aug positives by family (enabled views only)
        for fam, names in self.aug_by_fam.items():
            k = int(stage.pos_per_family.get(fam, 0))
            if k <= 0:
                continue
            # candidate pool = same index i in each view of that family (1 per view)
            cand_scored = [(self._rmse_score(vn, i, i, stage.use_raw_rmse), (vn, i)) for vn in names]
            picks = self._select_from_candidates(cand_scored, stage.pos_rmse_keep_quantile, k)
            for vn, j in picks:
                views.append((vn, j, -1, "aug"))

        # reference view as positive (optional)
        if stage.include_reference_as_positive and len(self.refs) > 0:
            # you can pick all references or one best reference by rmse; here: choose best 1 by rmse
            cand_scored = [(self._rmse_score(rn, i, i, stage.use_raw_rmse), (rn, i)) for rn in self.refs]
            pick = self._select_from_candidates(cand_scored, stage.pos_rmse_keep_quantile, 1)
            if len(pick) > 0:
                rn, j = pick[0]
                views.append((rn, j, -1, "reference"))
                special["ref"] = (rn, j)

        # temporal positive (optional): base(t±1) candidates
        if stage.use_temporal and stage.temporal_k > 0:
            candidates = []
            if i + 1 < self.N:
                candidates.append(i + 1)
            if i - 1 >= 0:
                candidates.append(i - 1)
            if len(candidates) > 0:
                cand_scored = [(self._rmse_score(self.base, i, j, stage.use_raw_rmse), (self.base, j)) for j in candidates]
                pick = self._select_from_candidates(cand_scored, stage.pos_rmse_keep_quantile, stage.temporal_k)
                for vn, j in pick:
                    views.append((vn, j, -1, "temporal"))
                    # keep one for smoothness term (first)
                    if "temporal" not in special:
                        special["temporal"] = (vn, j)

        return views, special


# =========================================================
# Composite loss (dynamic: reference may not exist)
# =========================================================
def compute_losses(
    proj: tf.keras.Model,
    X: tf.Tensor,               # (M,D)
    labels: tf.Tensor,          # (M,)
    sample_weight: tf.Tensor,   # (M,)
    temperature: float,
    # extra term indices
    base_anchor_rows: tf.Tensor,        # (B,) rows of base anchors
    ref_rows: Optional[tf.Tensor],      # (B,) rows of reference view aligned to anchor (or None)
    temporal_rows: Optional[tf.Tensor], # (B,) rows of temporal view aligned to anchor (or None)
    stage: StagePolicy,
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    z = proj(X, training=True)  # (M,proj_dim)

    # SupCon
    L_sup = supcon_loss(z, labels, temperature=temperature, sample_weight=sample_weight)

    # gather base anchors
    z_base = tf.gather(z, base_anchor_rows)  # (B,proj_dim)

    # align ref (if provided)
    L_align = tf.constant(0.0, tf.float32)
    if (ref_rows is not None) and (stage.w_align_reference > 0):
        z_ref = tf.gather(z, ref_rows)
        L_align = tf.reduce_mean(tf.reduce_sum(tf.square(z_base - z_ref), axis=1))

    # temporal smoothness (optional)
    L_ts = tf.constant(0.0, tf.float32)
    if (temporal_rows is not None) and (stage.w_temporal_smooth > 0):
        z_tmp = tf.gather(z, temporal_rows)
        L_ts = tf.reduce_mean(tf.reduce_sum(tf.square(z_base - z_tmp), axis=1))

    total = stage.w_supcon * L_sup + stage.w_align_reference * L_align + stage.w_temporal_smooth * L_ts

    return total, {"total": total, "supcon": L_sup, "align_ref": L_align, "temporal_smooth": L_ts}


# =========================================================
# Training (3-stage) with dynamic views + dynamic stage policies
# =========================================================
def train_contrastive_multistage_flexible(
    enabled_views: Dict[str, Dict[str, Any]],
    feature_arrays: Dict[str, np.ndarray],
    raw_arrays: Optional[Dict[str, np.ndarray]],
    cfg: TrainV2Config,
    stages: List[StagePolicy],
    ckpt_dir: str,
    anchor_indices: Optional[np.ndarray] = None,
):
    ensure_dir(ckpt_dir)
    rng = np.random.default_rng(cfg.seed)

    base_name = get_base_name(enabled_views)
    N, D = feature_arrays[base_name].shape

    # anchor_indices 준비
    if anchor_indices is None:
        anchor_indices = np.arange(N, dtype=np.int32)
    else:
        anchor_indices = np.asarray(anchor_indices, dtype=np.int32)

    # (선택) 안전 체크: 범위/정렬
    anchor_indices = anchor_indices[(anchor_indices >= 0) & (anchor_indices < N)]
    anchor_indices = np.unique(anchor_indices)  # 정렬+unique

    M = int(anchor_indices.shape[0])
    if M < 10:
        raise ValueError(f"Too few anchor_indices after filtering: {M}")

    # anchor_indices 내부에서 time split (keep order)
    split = int(M * (1.0 - cfg.val_ratio))
    idx_tr = np.arange(split, dtype=np.int32)      # 0..split-1 (anchor-local)
    idx_va = np.arange(split, M, dtype=np.int32)   # split..M-1

    # 먼저 feature/raw를 anchor subset으로 줄인 뒤, train/val로 나눔
    F_all = {k: v[anchor_indices] for k, v in feature_arrays.items()}
    F_tr  = {k: v[idx_tr] for k, v in F_all.items()}
    F_va  = {k: v[idx_va] for k, v in F_all.items()}

    if raw_arrays is not None:
        R_all = {k: v[anchor_indices] for k, v in raw_arrays.items()}
        R_tr  = {k: v[idx_tr] for k, v in R_all.items()}
        R_va  = {k: v[idx_va] for k, v in R_all.items()}
    else:
        R_tr = None
        R_va = None

    sampler_tr = FlexibleSampler(F_tr, R_tr, enabled_views, cfg, rng)
    sampler_va = FlexibleSampler(F_va, R_va, enabled_views, cfg, rng)

    proj = build_projection_head(D, cfg.proj_hidden, cfg.proj_dim, cfg.dropout_rate)
    best_path = os.path.join(ckpt_dir, "_best.weights.h5")

    best_val = np.inf
    bad = 0

    def make_view_weight(stage: StagePolicy, view_name: str) -> float:
        meta = enabled_views[view_name]
        role = meta.get("role", "aug")
        if role == "base":
            return float(stage.weight_base)
        if role == "reference":
            return float(stage.weight_reference)
        fam = meta.get("family", "other")
        return float(stage.weight_per_family.get(fam, 1.0))

    def run_steps(sampler: FlexibleSampler, stage: StagePolicy, training: bool) -> Dict[str, float]:
        opt = tf.keras.optimizers.Adam(stage.lr)

        meters = {"total": [], "supcon": [], "align_ref": [], "temporal_smooth": []}

        for _ in range(stage.steps):
            B = cfg.batch_anchors
            anchors = rng.integers(0, sampler.N, size=B, dtype=np.int32)

            # build flat views list
            views_flat: List[Tuple[str, int, int, str]] = []
            specials: List[Dict[str, Tuple[str, int]]] = []

            for b, i in enumerate(anchors.tolist()):
                views, special = sampler.pick_views_for_anchor(i, stage)
                # fill anchor_id = b
                views = [(vn, j, b, kind) for (vn, j, _, kind) in views]
                views_flat.extend(views)
                specials.append(special)

            # pack X, labels, weights
            X_np = np.stack([sampler.F[vn][j] for (vn, j, _, _) in views_flat], axis=0).astype(np.float32)
            y_np = np.array([aid for (_, _, aid, _) in views_flat], dtype=np.int32)

            w_np = np.array([make_view_weight(stage, vn) for (vn, _, _, _) in views_flat], dtype=np.float32)

            X = tf.convert_to_tensor(X_np)
            y = tf.convert_to_tensor(y_np)
            w = tf.convert_to_tensor(w_np)

            # locate rows for base anchors / reference / temporal (aligned per anchor)
            base_anchor_rows = np.full((B,), -1, dtype=np.int32)
            ref_rows = np.full((B,), -1, dtype=np.int32)
            temporal_rows = np.full((B,), -1, dtype=np.int32)

            for r, (vn, j, aid, kind) in enumerate(views_flat):
                if kind == "base_anchor" and base_anchor_rows[aid] < 0:
                    base_anchor_rows[aid] = r
                if kind == "reference" and ref_rows[aid] < 0:
                    ref_rows[aid] = r
                if kind == "temporal" and temporal_rows[aid] < 0:
                    temporal_rows[aid] = r

            if np.any(base_anchor_rows < 0):
                raise RuntimeError("Some base anchors missing in batch construction.")

            base_anchor_rows_tf = tf.convert_to_tensor(base_anchor_rows)

            ref_rows_tf = None
            if stage.w_align_reference > 0 and np.any(ref_rows >= 0):
                # for anchors without reference view, fall back to base row (neutral for align)
                rr = np.where(ref_rows >= 0, ref_rows, base_anchor_rows)
                ref_rows_tf = tf.convert_to_tensor(rr)

            temporal_rows_tf = None
            if stage.w_temporal_smooth > 0 and np.any(temporal_rows >= 0):
                trr = np.where(temporal_rows >= 0, temporal_rows, base_anchor_rows)
                temporal_rows_tf = tf.convert_to_tensor(trr)

            if training:
                with tf.GradientTape() as tape:
                    total, loss_dict = compute_losses(
                        proj=proj,
                        X=X,
                        labels=y,
                        sample_weight=w,
                        temperature=stage.temperature,
                        base_anchor_rows=base_anchor_rows_tf,
                        ref_rows=ref_rows_tf,
                        temporal_rows=temporal_rows_tf,
                        stage=stage,
                    )
                grads = tape.gradient(total, proj.trainable_variables)
                opt.apply_gradients(zip(grads, proj.trainable_variables))
            else:
                # eval forward
                z = proj(X, training=False)
                L_sup = supcon_loss(z, y, temperature=stage.temperature, sample_weight=w)

                z_base = tf.gather(z, base_anchor_rows_tf)

                L_align = tf.constant(0.0, tf.float32)
                if ref_rows_tf is not None and stage.w_align_reference > 0:
                    z_ref = tf.gather(z, ref_rows_tf)
                    L_align = tf.reduce_mean(tf.reduce_sum(tf.square(z_base - z_ref), axis=1))

                L_ts = tf.constant(0.0, tf.float32)
                if temporal_rows_tf is not None and stage.w_temporal_smooth > 0:
                    z_tmp = tf.gather(z, temporal_rows_tf)
                    L_ts = tf.reduce_mean(tf.reduce_sum(tf.square(z_base - z_tmp), axis=1))

                total = stage.w_supcon * L_sup + stage.w_align_reference * L_align + stage.w_temporal_smooth * L_ts
                loss_dict = {"total": total, "supcon": L_sup, "align_ref": L_align, "temporal_smooth": L_ts}

            for k in meters.keys():
                meters[k].append(float(loss_dict[k].numpy()))

        return {k: float(np.mean(v)) for k, v in meters.items()}

    # # training: per-stage 1 "round" (you can loop rounds if needed)
    # for stage in stages:
    #     print(f"\n=== Stage: {stage.name} ===")
    #     tr = run_steps(sampler_tr, stage, training=True)
    #     va = run_steps(sampler_va, stage, training=False)
    #     print(
    #         f"[{stage.name}] train total={tr['total']:.4f} (sup={tr['supcon']:.4f}, align={tr['align_ref']:.4f}, ts={tr['temporal_smooth']:.4f}) | "
    #         f"val total={va['total']:.4f} (sup={va['supcon']:.4f}, align={va['align_ref']:.4f}, ts={va['temporal_smooth']:.4f})"
    #     )

    #     if va["total"] + 1e-6 < best_val:
    #         best_val = va["total"]
    #         bad = 0
    #         proj.save_weights(best_path)
    #     else:
    #         bad += 1
    #         if bad >= cfg.patience:
    #             print(f"Early stop triggered (patience={cfg.patience}). Best val={best_val:.4f}")
    #             break

    # 예: cfg에 이런 값들이 있다고 가정
    # cfg.patience
    # cfg.stage_epochs: dict[str,int]  # {"warmup": 10, "align": 20, "temporal": 30} 같은 식
    # 없으면 공통 cfg.max_epochs_per_stage로 통일해도 됨

    for stage in stages:
        print(f"\n########################################")
        print(f"### Stage: {stage.name}")
        print(f"########################################")

        # stage별 optimizer/lr 세팅이 있다면 stage 들어올 때 반드시 적용
        # 예: optimizer.learning_rate.assign(stage.lr)

        best_val = float("inf")
        bad = 0

        stage_best_path = os.path.join(ckpt_dir, f"_best_{stage.name}.weights.h5")

        for epoch in range(stage.max_epochs):
            print(f"\n--- [{stage.name}] epoch {epoch+1}/{stage.max_epochs} ---")

            tr = run_steps(sampler_tr, stage, training=True)
            va = run_steps(sampler_va, stage, training=False)

            print(
                f"[{stage.name}] train total={tr['total']:.4f} "
                f"(sup={tr['supcon']:.4f}, align={tr['align_ref']:.4f}, ts={tr['temporal_smooth']:.4f}) | "
                f"val total={va['total']:.4f} "
                f"(sup={va['supcon']:.4f}, align={va['align_ref']:.4f}, ts={va['temporal_smooth']:.4f})"
            )

            # 개선 판정
            if va["total"] < best_val - stage.min_delta:
                best_val = va["total"]
                bad = 0
                proj.save_weights(stage_best_path)
                print(f"[CKPT] best updated -> {best_val:.6f} (saved: {stage_best_path})")
            else:
                bad += 1
                print(f"[ES] no improve. bad={bad}/{stage.patience} (best={best_val:.6f})")
                if bad >= stage.patience:
                    print(f"Early stop stage '{stage.name}'. Best val={best_val:.6f}")
                    break

        # 다음 stage로 넘어가기 전에 해당 stage best로 복귀
        if os.path.exists(stage_best_path):
            proj.load_weights(stage_best_path)
            print(f"[LOAD] loaded stage-best weights for next stage: {stage_best_path}")
        else:
            print(f"[WARN] stage-best weights missing: {stage_best_path} (continuing without reload)")


    print(f"\nBest weights saved: {stage_best_path}")
    return stage_best_path


# =========================================================
# MODE == "train" replacement (final)
# =========================================================
def run_train_mode_flexible(EXTRACT_DIR: str, CKPT_DIR: str, TRAIN: Dict[str, Any]):
    manifest_path = os.path.join(EXTRACT_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found. run extract first: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1) load views (flexible)
    views = load_views_from_manifest(manifest)
    enabled_views = select_enabled_views(views)

    # --- NEW: valid anchor indices (exclude missing refs) ---
    anchor_indices = compute_anchor_indices_from_manifest(manifest, enabled_views)
    if anchor_indices.size < 2:
        raise ValueError(f"Too few valid anchors after applying valid_indices: {anchor_indices.size}")

    print(f"[train_all] valid anchors = {len(anchor_indices)} / N={manifest['shape']['N']}")

    # 2) load feature arrays for enabled views only
    feature_arrays: Dict[str, np.ndarray] = {}
    for name, meta in enabled_views.items():
        feature_arrays[name] = load_npy_2d(meta["path"])

    # 3) optional raw arrays (for RMSE scoring). Same structure as views.
    #    If you want raw_rmse-based filtering, add manifest["raw_views"] similarly.
    raw_arrays = None
    raw_views = manifest.get("raw_views", None)
    if raw_views is not None:
        # raw_views should be a dict like views with "path"/"enabled"
        raw_arrays = {}
        for name in enabled_views.keys():
            rv = raw_views.get(name, None)
            if rv is None:
                continue
            if not bool(rv.get("enabled", True)):
                continue
            raw_arrays[name] = np.load(rv["path"])

    # 4) run dir
    #run_dir = os.path.join(CKPT_DIR, "contrastive_multistage_flexible")
    run_dir = CKPT_DIR
    ensure_dir(run_dir)

    # 5) cfg
    base_name = get_base_name(enabled_views)
    N, D = feature_arrays[base_name].shape

    cfg = TrainV2Config(
        batch_anchors=int(TRAIN.get("batch_anchors", TRAIN.get("batch_size", 128))),
        proj_hidden=int(TRAIN.get("proj_hidden", 256)),
        proj_dim=int(TRAIN.get("proj_dim", 128)),
        dropout_rate=float(TRAIN.get("dropout_rate", 0.1)),
        seed=int(TRAIN.get("seed", 0)),
        val_ratio=float(TRAIN.get("val_ratio", 0.2)),
        patience=int(TRAIN.get("patience", 30)),

        temporal_exclude=int(TRAIN.get("temporal_exclude", 2)),
        far_neg_min_dt=int(TRAIN.get("far_neg_min_dt", 30)),
        hard_neg_pool_window=int(TRAIN.get("hard_neg_pool_window", 60)),
    )

    # 6) stage policies (name-agnostic)
    stages = build_default_stage_policies(TRAIN)

    # 7) Safety: if there is no reference view, force align weights to 0 (auto)
    if len(get_reference_names(enabled_views)) == 0:
        for st in stages:
            st.w_align_reference = 0.0
            st.include_reference_as_positive = False

    # 8) Safety: if certain families are not present, they naturally contribute 0 positives.
    #    (No action needed.)


    # 9) Train
    best_path = train_contrastive_multistage_flexible(
        enabled_views=enabled_views,
        feature_arrays=feature_arrays,
        raw_arrays=raw_arrays,
        cfg=cfg,
        stages=stages,
        ckpt_dir=run_dir,
        anchor_indices=anchor_indices, 
    )

    return best_path

def _make_valid_mask(N: int, valid_indices: list[int] | None):
    """valid_indices가 None이면 전부 True"""
    mask = np.zeros((N,), dtype=bool)
    if valid_indices is None:
        mask[:] = True
        return mask
    valid_indices = np.asarray(valid_indices, dtype=np.int32)
    valid_indices = valid_indices[(0 <= valid_indices) & (valid_indices < N)]
    mask[valid_indices] = True
    return mask

def compute_anchor_indices_from_manifest(manifest: dict, enabled_views: dict) -> np.ndarray:
    N = int(manifest["shape"]["N"])
    valid_map = manifest.get("valid_indices", {})

    # enabled reference views (manifest 포맷 상관없이 enabled_views 기준)
    ref_views = [name for name, meta in enabled_views.items() if meta.get("role") == "reference"]

    anchor_mask = np.ones((N,), dtype=bool)
    for rv in ref_views:
        rv_valid = valid_map.get(rv, None)
        if rv_valid is None:
            continue
        rv_valid = np.asarray(rv_valid, dtype=np.int32)
        rv_valid = rv_valid[(0 <= rv_valid) & (rv_valid < N)]
        m = np.zeros((N,), dtype=bool)
        m[rv_valid] = True
        anchor_mask &= m

    return np.where(anchor_mask)[0].astype(np.int32)


# =========================================================
# Example: how to enable/disable views in manifest (minimal)
# =========================================================
"""
# Legacy manifest upgrade example:

manifest["feature_paths"] = {
  "ERA5": ".../ERA5_fused.npy",
  "ecmwf": ".../ecmwf_fused.npy",
  "blur": ".../blur_fused.npy",
  "translated_v2": ".../translated_v2_fused.npy"
}

manifest["view_config"] = {
  "ERA5": {"role": "base", "enabled": True, "family":"base"},
  "ecmwf": {"role": "reference", "enabled": True, "family":"reference", "align_to":"ERA5"},
  "blur": {"role": "aug", "enabled": True, "family":"appearance"},
  "translated_v2": {"role": "aug", "enabled": True, "family":"phase"},
  # "zoom": {"enabled": False}  # even if not in feature_paths, it's just ignored
}

# Then run:
best = run_train_mode_flexible(EXTRACT_DIR, CKPT_DIR, TRAIN)
"""
