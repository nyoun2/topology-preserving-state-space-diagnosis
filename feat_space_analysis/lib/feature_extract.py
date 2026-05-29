import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers, regularizers
from keras import layers, regularizers
import json

from feat_space_analysis.lib.utils import ensure_dir
from feat_space_analysis.lib.preprocess import preprocess_identity

################################
### 클래스 (학습/샘플링 계열)
#################################

class MultiVarFeatureExtractor(keras.Model):
    """
    x (B,H,W,V) or (B,V,H,W) -> fused (B,D)
    """
    def __init__(
        self,
        num_vars=4,
        input_hw=(181, 177),
        embed_dim=256,
        fused_dim=256,
        backbone_trainable=False,
        adapter_trainable=False,
        weights="imagenet",
        fusion_mode="concat_linear",
        l2_reg=1e-6,
        #include_preprocessing=False,  # ✅ 추가
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_vars = int(num_vars)
        self.input_hw = tuple(input_hw)

        self.backbone = build_shared_backbone(
            self.input_hw, weights=weights, trainable=backbone_trainable
        )
        self.var_encoder = build_variable_encoder(self.backbone, self.input_hw, embed_dim=embed_dim, l2_reg=l2_reg)
        self.fusion = build_fusion_head(
            num_vars=self.num_vars,
            embed_dim=embed_dim,
            fused_dim=fused_dim,
            l2_reg=l2_reg,
            fusion_mode=fusion_mode
        )

        self.backbone.trainable = backbone_trainable

        if adapter_trainable and not backbone_trainable:
            self.var_encoder.trainable = True
            self.backbone.trainable = False
        elif backbone_trainable:
            self.var_encoder.trainable = True
        else:
            self.var_encoder.trainable = False

    def encode(self, x):
        if x.shape.rank != 4:
            raise ValueError("x must be rank-4: (B,H,W,V) or (B,V,H,W)")

        if x.shape[1] == self.num_vars and (x.shape[-1] != self.num_vars):
            x = tf.transpose(x, perm=[0, 2, 3, 1])

        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        V = self.num_vars

        xv = tf.transpose(x, perm=[0, 3, 1, 2])  # (B,V,H,W)
        xv = tf.expand_dims(xv, axis=-1)         # (B,V,H,W,1)
        xv = tf.reshape(xv, (B * V, H, W, 1))    # (B*V,H,W,1)

        ev = self.var_encoder(xv)                # (B*V, embed_dim)
        ev = tf.reshape(ev, (B, V, tf.shape(ev)[-1]))  # (B,V,E)

        embs = [ev[:, i, :] for i in range(V)]
        fused = self.fusion(embs)
        return fused

    def call(self, x, training=False):
        return {"fused": self.encode(x)}


class ContrastiveOnPairs(keras.Model):
    def __init__(self, proj_head, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.proj = proj_head
        self.temperature = float(temperature)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        f1, f2 = data if (isinstance(data, (tuple, list)) and len(data) == 2) else data[0]
        with tf.GradientTape() as tape:
            z1 = self.proj(f1, training=True)
            z2 = self.proj(f2, training=True)
            loss = nt_xent_loss(z1, z2, temperature=self.temperature)
            loss += tf.add_n(self.losses) if self.losses else 0.0

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        f1, f2 = data if (isinstance(data, (tuple, list)) and len(data) == 2) else data[0]
        z1 = self.proj(f1, training=False)
        z2 = self.proj(f2, training=False)
        loss = nt_xent_loss(z1, z2, temperature=self.temperature)
        loss += tf.add_n(self.losses) if self.losses else 0.0
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, f, training=False):
        return {"proj": self.proj(f, training=training)}
    

class PairSampler:
    def __init__(
        self,
        rule: str,
        A: str,
        B: str | None = None,
        lag: int = 1,
        valid_idx: np.ndarray | None = None,
    ):
        self.rule = rule
        self.A = A
        self.B = B if B is not None else A
        self.lag = int(lag)

        valid = {"within", "cross_aligned", "cross_lag"}
        if self.rule not in valid:
            raise ValueError(f"rule must be one of {sorted(valid)}")

        self.valid_idx = None
        if valid_idx is not None:
            vi = np.asarray(valid_idx, dtype=np.int32)
            if vi.ndim != 1 or vi.size == 0:
                raise ValueError("valid_idx must be 1D non-empty array")
            self.valid_idx = np.unique(vi)

    def build_pairs(self, lengths: dict[str, int]) -> np.ndarray:
        NA = int(lengths[self.A])
        NB = int(lengths[self.B])

        # base index universe
        if self.valid_idx is None:
            # 기존 전체 index 기반
            if self.rule == "within":
                max_i = NA - self.lag
                if max_i <= 0:
                    raise ValueError("Not enough samples for within-lag pairs.")
                i = np.arange(0, max_i, dtype=np.int32)
                j = i + self.lag
                return np.stack([i, j], axis=1)

            if self.rule == "cross_aligned":
                M = min(NA, NB)
                i = np.arange(0, M, dtype=np.int32)
                return np.stack([i, i], axis=1)

            if self.rule == "cross_lag":
                max_i = min(NA, NB - self.lag)
                if max_i <= 0:
                    raise ValueError("Not enough samples for cross-lag pairs.")
                i = np.arange(0, max_i, dtype=np.int32)
                j = i + self.lag
                return np.stack([i, j], axis=1)

            raise RuntimeError("unreachable")

        # valid_idx 기반 (missing 제외)
        idx = self.valid_idx

        # 범위 보호: A/B 둘 다 접근 가능해야 함
        lim = min(NA, NB)
        idx = idx[idx < lim]
        if idx.size == 0:
            raise ValueError("After clipping by dataset length, no valid indices remain.")

        if self.rule == "cross_aligned":
            # (i,i)만 사용
            return np.stack([idx, idx], axis=1).astype(np.int32, copy=False)

        # lag 규칙은 "i와 i+lag가 모두 valid"인 경우만 pair로 만든다
        valid_set = set(idx.tolist())

        pairs = []
        if self.rule == "within":
            # within은 A=A로 간주하지만, 여기서는 idx 집합 내에서만 연속쌍을 만듦
            for i in idx:
                j = i + self.lag
                if (j in valid_set) and (j < NA):
                    pairs.append((i, j))

        elif self.rule == "cross_lag":
            # A는 idx의 i, B는 i+lag (둘 다 valid이어야 함)
            for i in idx:
                j = i + self.lag
                if (j in valid_set) and (j < NB):
                    pairs.append((i, j))
        else:
            raise RuntimeError("unreachable")

        if len(pairs) == 0:
            raise ValueError(f"No valid pairs created for rule={self.rule} with lag={self.lag}")
        return np.asarray(pairs, dtype=np.int32)


class FeaturePairSequence(keras.utils.Sequence):
    def __init__(
        self,
        datasets_map: dict[str, np.ndarray],
        name_A: str,
        name_B: str,
        pairs: np.ndarray,
        batch_size: int,
        noise_std=0.01,
        dropout_rate=0.10,
        shuffle=True,
        seed=0
    ):
        self.F = datasets_map
        self.A = name_A
        self.B = name_B

        self.pairs = np.asarray(pairs, dtype=np.int32)
        self.batch_size = int(batch_size)

        self.noise_std = float(noise_std)
        self.dropout_rate = float(dropout_rate)

        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)

        if self.batch_size < 2:
            raise ValueError("contrastive learning은 batch_size>=2 권장(negatives 필요).")

        self.indices = np.arange(len(self.pairs), dtype=np.int32)
        self.on_epoch_end()

        Da = int(self.F[self.A].shape[1])
        Db = int(self.F[self.B].shape[1])
        if Da != Db:
            raise ValueError(f"Feature dim mismatch: {self.A}:{Da} vs {self.B}:{Db}")

    def __len__(self):
        return len(self.pairs) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _augment(self, f: np.ndarray) -> np.ndarray:
        x = f.astype(np.float32, copy=False)

        if self.dropout_rate > 0:
            keep = 1.0 - self.dropout_rate
            mask = (self.rng.random(x.shape) < keep).astype(np.float32) / max(keep, 1e-6)
            x = x * mask

        if self.noise_std > 0:
            x = x + self.rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)

        return x

    def __getitem__(self, i):
        batch_ids = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        ij = self.pairs[batch_ids]

        iA = ij[:, 0]
        iB = ij[:, 1]

        f1 = np.asarray(self.F[self.A][iA], dtype=np.float32)
        f2 = np.asarray(self.F[self.B][iB], dtype=np.float32)

        f1v = self._augment(f1)
        f2v = self._augment(f2)
        return (f1v, f2v)    
    

################################
### 모델 빌드/추출
#################################
def build_shared_backbone(input_hw, weights="imagenet", trainable=False):
    """
    EfficientNetB0 backbone expects (H,W,3)

    include_preprocessing=False 로 고정하는 걸 권장:
    - 내부 ImageNet 전처리에 의존하지 않고,
    - 우리가 정의한 ERA5 기반 표준화를 입력에서 수행하기 위해서.
    """
    H, W = input_hw
    core = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        pooling="avg",
        #include_preprocessing=include_preprocessing,  # ✅ 핵심
    )
    core.trainable = trainable

    inp = keras.Input(shape=(H, W, 3), name="effnet_rgb_input")
    out = core(inp)
    return keras.Model(inp, out, name="effnetb0_backbone")


def build_variable_encoder(backbone, input_hw, embed_dim=256, l2_reg=1e-6, name="var_encoder"):
    H, W = input_hw
    inp = keras.Input(shape=(H, W, 1), name=f"{name}_in")
    x = layers.Conv2D(
        3, kernel_size=1, padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f"{name}_adapter_conv1x1"
    )(inp)
    h = backbone(x)  # (B,1280)
    e = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg), name=f"{name}_proj")(h)
    e = layers.LayerNormalization(name=f"{name}_ln")(e)
    return keras.Model(inp, e, name=name)


def build_fusion_head(num_vars=4, embed_dim=256, fused_dim=256, l2_reg=1e-6, fusion_mode="concat_linear"):
    inputs = [keras.Input(shape=(embed_dim,), name=f"e_{i}") for i in range(num_vars)]

    if fusion_mode == "mean":
        f = layers.Average(name="fusion_mean")(inputs)
        if fused_dim != embed_dim:
            f = layers.Dense(fused_dim, kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense")(f)
    elif fusion_mode == "concat_linear":
        cat = layers.Concatenate(name="fusion_concat")(inputs)  # (B, V*embed_dim)
        f = layers.Dense(fused_dim, kernel_regularizer=regularizers.l2(l2_reg), name="fusion_dense")(cat)
    else:
        raise ValueError("fusion_mode must be 'mean' or 'concat_linear'")

    f = layers.LayerNormalization(name="fusion_ln")(f)
    return keras.Model(inputs, f, name="fusion_head")

def extract_fused_features(
    extractor: MultiVarFeatureExtractor,
    x_provider,
    indices: np.ndarray,
    H: int, W: int, V: int,
    out_path: str,
    batch_size: int = 16,
    preprocess_fn=preprocess_identity,
    dtype_out="float32",
    verbose=True
):
    indices = np.asarray(indices, dtype=np.int32)
    ensure_dir(os.path.dirname(out_path) or ".")

    x0 = np.asarray(x_provider(int(indices[0])), dtype=np.float32)[None, ...]
    x0 = preprocess_fn(x0, np.array([indices[0]], dtype=np.int32))
    D = int(extractor(x0, training=False)["fused"].shape[-1])

    tmp_mem = out_path + ".mmap"
    mm = np.memmap(tmp_mem, mode="w+", dtype=dtype_out, shape=(len(indices), D))

    p = 0
    while p < len(indices):
        b = min(batch_size, len(indices) - p)
        xb = np.empty((b, H, W, V), dtype=np.float32)
        idx_b = indices[p:p+b]

        for j, idx in enumerate(idx_b):
            x = np.asarray(x_provider(int(idx)), dtype=np.float32)
            if x.shape != (H, W, V):
                raise ValueError(f"x_provider shape mismatch: got {x.shape}, expected {(H,W,V)}")
            xb[j] = x

        xb = preprocess_fn(xb, idx_b)

        fused = extractor(xb, training=False)["fused"].numpy()
        mm[p:p+b] = fused.astype(dtype_out, copy=False)

        p += b
        if verbose and (p == len(indices) or p % max(1, batch_size * 20) == 0):
            print(f"[extract:{os.path.basename(out_path)}] {p}/{len(indices)}")

    mm.flush()
    F = np.array(mm)
    np.save(out_path, F)
    del mm
    try:
        os.remove(tmp_mem)
    except OSError:
        pass

    meta = {
        "out_path": out_path,
        "num_samples": int(len(indices)),
        "feature_dim": int(D),
        "dtype": str(dtype_out),
        "indices_min": int(indices.min()),
        "indices_max": int(indices.max()),
        "preprocess_fn": getattr(preprocess_fn, "__name__", "custom"),
    }
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"[extract] saved {out_path}  shape={F.shape}")

    return out_path


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


def load_projection_head_from_ckpt(best_proj_weights: str, input_dim: int, proj_hidden: int, proj_dim: int):

    # projection head 구성 + build
    proj = build_projection_head(
        input_dim=input_dim,
        hidden_dim=proj_hidden,
        proj_dim=proj_dim,
        dropout_rate=0.0,   # export는 0.0 권장
    )    
    _ = proj(tf.zeros((1, input_dim), dtype=tf.float32), training=False)

    if not os.path.exists(best_proj_weights):
        raise FileNotFoundError(best_proj_weights)
    proj.load_weights(best_proj_weights)

    # proj = build_projection_head(input_dim=input_dim, hidden_dim=proj_hidden, proj_dim=proj_dim)
    # _ = proj(tf.zeros((1, input_dim), tf.float32))

    # wpath = os.path.join(ckpt_dir, "final_best_proj.weights.h5")
    # if not os.path.exists(wpath):
    #     raise FileNotFoundError(f"proj weights not found: {wpath}")
    # proj.load_weights(wpath)
    return proj


################################
### 데이터셋/학습
#################################

def load_feature_datasets(feature_paths: dict[str, str]) -> dict[str, np.ndarray]:
    datasets = {}
    for name, path in feature_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Feature array must be 2D (N,D). got {arr.shape} for {name}")
        datasets[name] = arr
    return datasets


def train_contrastive_on_relationship(
    feature_paths: dict[str, str],
    sampler: PairSampler,
    batch_size=256,
    epochs=50,
    patience=50,
    lr=1e-3,
    temperature=0.1,
    noise_std=0.01,
    dropout_rate=0.10,
    proj_hidden=256,
    proj_dim=128,
    ckpt_dir="checkpoints_rel",
    val_ratio=0.2,
    seed=0
):
    ensure_dir(ckpt_dir)

    F = load_feature_datasets(feature_paths)
    lengths = {k: int(v.shape[0]) for k, v in F.items()}

    pairs_all = sampler.build_pairs(lengths)
    M = len(pairs_all)
    if M < batch_size * 2:
        raise ValueError(f"Too few pairs ({M}) for batch_size={batch_size}. Reduce batch_size or add data.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    pairs_all = pairs_all[perm]

    split = int(M * (1.0 - val_ratio))
    pairs_tr = pairs_all[:split]
    pairs_va = pairs_all[split:]

    seq_tr = FeaturePairSequence(
        datasets_map=F,
        name_A=sampler.A,
        name_B=sampler.B,
        pairs=pairs_tr,
        batch_size=batch_size,
        noise_std=noise_std,
        dropout_rate=dropout_rate,
        shuffle=True,
        seed=seed
    )
    seq_va = FeaturePairSequence(
        datasets_map=F,
        name_A=sampler.A,
        name_B=sampler.B,
        pairs=pairs_va,
        batch_size=batch_size,
        noise_std=noise_std,
        dropout_rate=dropout_rate,
        shuffle=False,
        seed=seed + 999
    )

    D = int(F[sampler.A].shape[1])
    proj = build_projection_head(D, hidden_dim=proj_hidden, proj_dim=proj_dim)
    model = ContrastiveOnPairs(proj_head=proj, temperature=temperature)
    model.compile(optimizer=keras.optimizers.Adam(lr))

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "best.weights.h5"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    early_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    hist = model.fit(
        seq_tr,
        validation_data=seq_va,
        epochs=epochs,
        callbacks=[ckpt_cb, early_cb],
        verbose=1
    )

    out = {
        "feature_paths": feature_paths,
        "sampler": {"rule": sampler.rule, "A": sampler.A, "B": sampler.B, "lag": sampler.lag},
        "train": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "temperature": temperature,
            "noise_std": noise_std,
            "dropout_rate": dropout_rate,
            "proj_hidden": proj_hidden,
            "proj_dim": proj_dim,
            "val_ratio": val_ratio,
            "seed": seed
        },
        "history": hist.history
    }
    with open(os.path.join(ckpt_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    model.save_weights(os.path.join(ckpt_dir, "final_best_model.weights.h5"))
    model.proj.save_weights(os.path.join(ckpt_dir, "final_best_proj.weights.h5"))

    print(f"[train] done. best weights: {os.path.join(ckpt_dir, 'best.weights.h5')}")
    print(f"[train] proj weights: {os.path.join(ckpt_dir, 'final_best_proj.weights.h5')}")
    print(f"[train] run saved: {os.path.join(ckpt_dir, 'run.json')}")





################################
### 손실/벡터 유틸(모델 관련)
#################################

def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = l2_normalize(z1)
    z2 = l2_normalize(z2)
    B = tf.shape(z1)[0]

    z = tf.concat([z1, z2], axis=0)  # (2B,D)
    sim = tf.matmul(z, z, transpose_b=True) / temperature  # (2B,2B)

    mask = tf.eye(2 * B, dtype=tf.bool)
    sim = tf.where(mask, tf.fill(tf.shape(sim), -1e9), sim)

    pos_idx = tf.concat([tf.range(B, 2 * B), tf.range(0, B)], axis=0)
    labels = tf.one_hot(pos_idx, depth=2 * B)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=sim)
    return tf.reduce_mean(loss)


def l2_normalize(x, axis=-1, eps=1e-12):
    return tf.math.l2_normalize(x, axis=axis, epsilon=eps)

def _l2(a, b, eps=1e-12):
    d = a - b
    return np.sqrt(np.sum(d * d, axis=-1) + eps)

def _cosine_distance(a, b, eps=1e-12):
    na = np.linalg.norm(a, axis=-1) + eps
    nb = np.linalg.norm(b, axis=-1) + eps
    cos = np.sum(a * b, axis=-1) / (na * nb)
    return 1.0 - cos

def _l2_normalize(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

def _zscore_pair(x0, x1, eps=1e-12):
    allx = np.concatenate([x0, x1], axis=0)
    mu = np.mean(allx, axis=0, keepdims=True)
    sd = np.std(allx, axis=0, keepdims=True) + eps
    return (x0 - mu) / sd, (x1 - mu) / sd