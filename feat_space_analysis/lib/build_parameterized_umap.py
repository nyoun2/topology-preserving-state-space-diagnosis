import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple


# =========================
# 1) Mahalanobis whitener
# =========================
def make_whitener(V: Optional[np.ndarray]):
    if V is None:
        return lambda X: X
    try:
        L = np.linalg.cholesky(V)
        W = L.T
    except np.linalg.LinAlgError:
        U, S, VT = np.linalg.svd(V)
        W = (U * np.sqrt(S)) @ VT
    return lambda X: X @ W


# =========================
# 2) kNN graph (for UMAP loss)
# =========================
def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    V: Optional[np.ndarray] = None,
):
    if metric == "mahalanobis":
        if V is None:
            raise ValueError("Mahalanobis metric requires V")
        X_work = make_whitener(V)(X)
    else:
        X_work = X

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_work)
    dist, idx = nn.kneighbors(X_work, return_distance=True)
    return idx, dist


# =========================
# 3) UMAP pos/neg pairs
# =========================
def make_umap_pos_neg_pairs(
    nn_idx: np.ndarray,
    num_neg_per_pos: int = 5,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    N, K = nn_idx.shape

    # positive: (i, neighbor j)
    src = np.repeat(np.arange(N), K)
    dst = nn_idx.reshape(-1)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    y_pos = np.ones_like(src, dtype=np.float32)

    total_pos = len(src)
    total_neg = total_pos * num_neg_per_pos

    # negative: (i, random non-neighbor j)
    all_neighbors = [set(nn_idx[i].tolist()) for i in range(N)]
    neg_i = rng.integers(0, N, size=total_neg)
    neg_j = rng.integers(0, N, size=total_neg)

    for t in range(total_neg):
        if (neg_i[t] == neg_j[t]) or (neg_j[t] in all_neighbors[neg_i[t]]):
            for _ in range(10):
                cand = rng.integers(0, N)
                if cand != neg_i[t] and cand not in all_neighbors[neg_i[t]]:
                    neg_j[t] = cand
                    break

    y_neg = np.zeros_like(neg_i, dtype=np.float32)

    i_idx = np.concatenate([src, neg_i])
    j_idx = np.concatenate([dst, neg_j])
    y = np.concatenate([y_pos, y_neg])

    return i_idx, j_idx, y


# =========================
# 4) UMAP pair loss (BCE over distance)
# =========================
def umap_pair_loss(z_i, z_j, y):
    d2 = tf.reduce_sum(tf.square(z_i - z_j), axis=-1)  # (B,)
    logits = -d2
    y = tf.cast(y, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    return tf.reduce_mean(loss)


# =========================
# 5) Temporal InfoNCE loss
# =========================
def info_nce_loss(z_anchor, z_pos, temperature: float = 0.1):
    """
    z_anchor, z_pos: (B, 2)
    각 i에 대해 (anchor_i, pos_i)를 양성,
    (anchor_i, pos_j, j!=i)를 음성으로 보는 InfoNCE
    """
    # L2 normalize
    z_a = tf.math.l2_normalize(z_anchor, axis=-1)
    z_p = tf.math.l2_normalize(z_pos, axis=-1)

    logits = tf.matmul(z_a, z_p, transpose_b=True)  # (B, B)
    logits = logits / temperature

    batch_size = tf.shape(z_a)[0]
    labels = tf.range(batch_size)  # [0, 1, ..., B-1]

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
    return tf.reduce_mean(loss)


# =========================
# 6) Base network: x -> z(2D)
# =========================
def build_base_network(input_dim, l2=1e-4, dropout=0.0):
    x_in = keras.Input(shape=(input_dim,), name="x")
    h = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x_in)
    h = layers.BatchNormalization()(h)
    if dropout > 0:
        h = layers.Dropout(dropout)(h)

    h = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(h)
    h = layers.BatchNormalization()(h)

    h = layers.Dense(64, activation="relu")(h)

    z = layers.Dense(2, activation=None, name="z2d")(h)
    return keras.Model(x_in, z, name="base_parametric_umap")


# =========================
# 7) Wrapper model: UMAP loss + InfoNCE
# =========================
class ParametricUMAPModel(keras.Model):
    def __init__(self, base_model, lambda_contrast=1.0, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature

    def train_step(self, data):
        # data: ((Xi_knn, Xj_knn, Xi_time, Xpos_time), y_knn)
        (Xi_knn, Xj_knn, Xi_time, Xpos_time), y_knn = data

        with tf.GradientTape() as tape:
            # --- UMAP pair loss ---
            zi = self.base(Xi_knn, training=True)
            zj = self.base(Xj_knn, training=True)
            loss_umap = umap_pair_loss(zi, zj, y_knn)

            # --- Temporal InfoNCE loss ---
            z_a = self.base(Xi_time, training=True)
            z_p = self.base(Xpos_time, training=True)
            loss_nce = info_nce_loss(z_a, z_p, self.temperature)

            loss = loss_umap + self.lambda_contrast * loss_nce

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))
        return {"loss": loss, "loss_umap": loss_umap, "loss_nce": loss_nce}

    def test_step(self, data):
        (Xi_knn, Xj_knn, Xi_time, Xpos_time), y_knn = data

        zi = self.base(Xi_knn, training=False)
        zj = self.base(Xj_knn, training=False)
        loss_umap = umap_pair_loss(zi, zj, y_knn)

        z_a = self.base(Xi_time, training=False)
        z_p = self.base(Xpos_time, training=False)
        loss_nce = info_nce_loss(z_a, z_p, self.temperature)

        loss = loss_umap + self.lambda_contrast * loss_nce
        return {"loss": loss, "loss_umap": loss_umap, "loss_nce": loss_nce}
    
    def call(self, inputs, training=None):
        # inputs가 (Xi_knn, Xj_knn, Xi_time, Xpos_time)로 들어올 수도 있고
        # 단일 X가 들어올 수도 있으니 방어적으로 처리
        if isinstance(inputs, (tuple, list)):
            x0 = inputs[0]
        else:
            x0 = inputs
        return self.base(x0, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_contrast": self.lambda_contrast,
            "temperature": self.temperature,
            # Functional model은 이걸로 직렬화 가능
            "base_model": keras.saving.serialize_keras_object(self.base),
        })
        return config

    @classmethod
    def from_config(cls, config):
        base_cfg = config.pop("base_model")
        base_model = keras.saving.deserialize_keras_object(base_cfg)
        return cls(base_model=base_model, **config)


# =========================
# 8) TrainConfig
# =========================
@dataclass
class TrainConfig:
    n_neighbors: int = 15
    num_neg_per_pos: int = 5
    batch_size: int = 1024
    epochs: int = 50
    lr: float = 1e-3
    metric: str = "mahalanobis"
    V: Optional[np.ndarray] = None
    patience: int = 5
    ckpt_path: str = "best_parametric_umap.weights.h5"
    seed: int = 42
    lambda_contrast: float = 1.0
    temperature: float = 0.1


# =========================
# 9) Train 함수
# =========================
def train_parametric_umap_with_contrast(
    X_train: np.ndarray,
    X_val: np.ndarray,
    cfg: TrainConfig,
    toTrain: bool,
):
    N_train, D = X_train.shape
    N_val = X_val.shape[0]

    if toTrain:

        # --- Mahalanobis V가 없으면, train에서 Σ^-1 계산 ---
        if cfg.metric == "mahalanobis" and cfg.V is None:
            cov = np.cov(X_train, rowvar=False)
            cfg.V = np.linalg.pinv(cov)

        # --- UMAP용 kNN + pos/neg 쌍 (train) ---
        nn_idx_train, _ = build_knn_graph(
            X_train, cfg.n_neighbors, metric=cfg.metric, V=cfg.V
        )
        i_idx, j_idx, y_knn = make_umap_pos_neg_pairs(
            nn_idx_train, cfg.num_neg_per_pos, cfg.seed
        )

        # temporal positive: anchor i -> pos i+1 (시간 순서라고 가정)
        # i_idx>=N_train-1 인 쌍은 시간 양성이 없으므로 제거
        mask_time = i_idx < (N_train - 1)
        i_idx = i_idx[mask_time]
        j_idx = j_idx[mask_time]
        y_knn = y_knn[mask_time]

        Xi_knn = X_train[i_idx]
        Xj_knn = X_train[j_idx]
        Xi_time = X_train[i_idx]
        Xpos_time = X_train[i_idx + 1]

        # --- Validation용 UMAP + temporal 쌍 ---
        nn_idx_val, _ = build_knn_graph(
            X_val, cfg.n_neighbors, metric=cfg.metric, V=cfg.V
        )
        i_idx_v, j_idx_v, y_knn_v = make_umap_pos_neg_pairs(
            nn_idx_val, cfg.num_neg_per_pos, cfg.seed
        )
        mask_time_v = i_idx_v < (N_val - 1)
        i_idx_v = i_idx_v[mask_time_v]
        j_idx_v = j_idx_v[mask_time_v]
        y_knn_v = y_knn_v[mask_time_v]

        Xvi_knn = X_val[i_idx_v]
        Xvj_knn = X_val[j_idx_v]
        Xvi_time = X_val[i_idx_v]
        Xvpos_time = X_val[i_idx_v + 1]

        # --- Dataset 구성 ---
        # train_ds = tf.data.Dataset.from_tensor_slices(
        #     ((Xi_knn, Xj_knn, Xi_time, Xpos_time), y_knn)
        # ).batch(cfg.batch_size).shuffle(1000, seed=cfg.seed, reshuffle_batch=True)
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((Xi_knn, Xj_knn, Xi_time, Xpos_time), y_knn)
        ).shuffle(1000, seed=cfg.seed, reshuffle_each_iteration=True) \
            .batch(cfg.batch_size)

        val_ds = tf.data.Dataset.from_tensor_slices(
            ((Xvi_knn, Xvj_knn, Xvi_time, Xvpos_time), y_knn_v)
        ).batch(cfg.batch_size)

    # --- 모델 생성 ---
    base = build_base_network(D, l2=1e-4, dropout=0.0)
    model = ParametricUMAPModel(
        base,
        lambda_contrast=cfg.lambda_contrast,
        temperature=cfg.temperature,
    )
    model.compile(optimizer=keras.optimizers.Adam(cfg.lr))

    # build 강제 (checkpoint 문제 방지)
    model.build(
        input_shape=[
            (None, D),  # Xi_knn
            (None, D),  # Xj_knn
            (None, D),  # Xi_time
            (None, D),  # Xpos_time
        ]
    )

    if toTrain:
        # --- callbacks: EarlyStopping + Checkpoint ---
        cbs = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=cfg.ckpt_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ]

    if toTrain:
        # --- 학습 ---
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs,
            callbacks=cbs,
            verbose=1,
            shuffle=False,  # dataset에서 이미 shuffle했으므로 여기선 False
        )

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.title("Training / Validation Loss")
        plt.show()

    # best weight 로드
    model.load_weights(cfg.ckpt_path)

    # 실제 2D 임베딩을 뽑는 건 base 모델
    return base


# =========================
# 10) inference + 데모
# =========================
def infer_embeddings(base_model: keras.Model, X: np.ndarray) -> np.ndarray:
    return base_model(X, training=False).numpy()


def proc_parametric_umap(input_npy = "out_contrastive/z_post_val.npy",
                         output_npy = "out_fourier/parametric_umap_z_post_mahalanobis.npy",
                         toTrain = True, tag_name="v1"):

    # rng = np.random.default_rng(0)
    # N_train, N_val, D = 4000, 1000, 64
    #
    # # 여기 X_train, X_val 자리에 실제 "시간순으로 정렬된" feature를 넣으면 됨
    # X_train = rng.normal(size=(N_train, D)).astype(np.float32)
    # X_val = rng.normal(size=(N_val, D)).astype(np.float32)

    feature_path = input_npy  #"out_contrastive/z_post_val.npy"
    X_train = np.load(feature_path)
    X_val = X_train

    cov = np.cov(X_train, rowvar=False)
    V = np.linalg.pinv(cov)

    cfg = TrainConfig(
        n_neighbors=20,
        num_neg_per_pos=3,
        batch_size=512,
        epochs=1000,
        lr=1e-3,
        metric="mahalanobis",
        V=V,  #None,          # None이면 학습 시 Σ^-1 계산
        patience=75,
        ckpt_path=f"model_all/{tag_name}/best_parametric_umap_{tag_name}.weights.h5",
        seed=42,
        lambda_contrast=1.0, #1.0,  # UMAP:InfoNCE 비중
        temperature=0.1,
    )

    base_model = train_parametric_umap_with_contrast(X_train, X_val, cfg, toTrain)

    Z_train = infer_embeddings(base_model, X_train)
    #Z_val = infer_embeddings(base_model, X_val)
    print("Z_train shape:", Z_train.shape)


    np.save(output_npy, Z_train)
    print("output saved..")

    if toTrain:
        plt.figure(figsize=(6, 6))
        plt.scatter(Z_train[:, 0], Z_train[:, 1], s=2, alpha=0.3, label="train")
        #plt.scatter(Z_val[:, 0], Z_val[:, 1], s=4, alpha=0.6, marker="x", label="val")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title("Parametric UMAP + Temporal InfoNCE (2D)")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    proc_parametric_umap()


