import os
import numpy as np
from scipy.ndimage import gaussian_filter

from feat_space_analysis.lib.preprocess import crop_shift_image, crop_dataset, rename_datetime_index
from feat_space_analysis.lib.preprocess import find_missing_mask_in_t_folder, save_selected_vars
from feat_space_analysis.lib.utils import ensure_dir
from feat_space_analysis.lib.io_paths import glob_npy


class ReferenceNpyFolderProvider:
    """
    _t####.npy 파일을 읽는 provider.
    missing_indices에 포함된 idx는 (H,W,V) shape의 0으로 채워 반환한다.

    - extract 단계에서 shape mismatch 없이 끝까지 fused를 만들기 위함
    - 학습 단계에서는 valid_idx를 따로 만들어 missing을 제외하도록 PairSampler에서 처리
    """
    def __init__(
        self,
        folder: str,
        H: int,
        W: int,
        V: int,
        n_total: int = 7308,
        missing_indices=None,
        mmap_mode: str = "r",
        prefix: str = "ecmwf",
        dtype=np.float32,
    ):
        self.folder = folder
        self.H, self.W, self.V = int(H), int(W), int(V)
        self.n_total = int(n_total)
        self.mmap_mode = mmap_mode
        self.prefix = prefix
        self.dtype = dtype

        self.missing = set(int(i) for i in (missing_indices or []))

        # 미리 mask 만들어두기 (True=valid, False=missing)
        self.valid_mask = np.ones((self.n_total,), dtype=bool)
        for i in self.missing:
            if 0 <= i < self.n_total:
                self.valid_mask[i] = False

        # missing일 때 반환할 0 텐서(재사용)
        self._zero = np.zeros((self.H, self.W, self.V), dtype=self.dtype)

    def __len__(self):
        return self.n_total

    def valid_indices(self) -> np.ndarray:
        """missing 제외한 index list (int32)"""
        return np.flatnonzero(self.valid_mask).astype(np.int32)

    def __call__(self, idx: int):
        idx = int(idx)

        if idx in self.missing:
            return self._zero  # (H,W,V)

        fp = os.path.join(self.folder, f"{self.prefix}_t{idx:04d}.npy")
        x = np.load(fp, mmap_mode=self.mmap_mode)

        # (H,W,V)로 맞추기
        if x.shape == (self.H, self.W, self.V):
            pass
        elif x.shape == (self.V, self.H, self.W):
            x = np.transpose(x, (1, 2, 0))
        else:
            raise ValueError(f"[ReferenceNpyFolderProvider] Unexpected shape {x.shape} in file {fp}")

        return np.asarray(x, dtype=self.dtype)
    

class NoisyWrapperProvider:
    """
    base_provider(idx)를 읽어서 noise를 추가한 결과를 반환.
    """
    def __init__(self, base_provider, noise_std=0.01, seed=0, clip=None):
        self.base = base_provider
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.clip = clip

    def __len__(self):
        return len(self.base)

    def __call__(self, idx: int):
        x = self.base(int(idx))
        rng = np.random.default_rng(self.seed + int(idx))
        noise = rng.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        y = x + noise
        if self.clip is not None:
            lo, hi = self.clip
            y = np.clip(y, lo, hi)
        return y.astype(np.float32, copy=False)


class NoisyWrapperShiftProvider:
    """
    - base(idx): 이미 center-crop 된 결과 (171,167,V)
    - raw(idx): crop 안 된 원본 (181,177,V)
    raw에서 랜덤 shift crop으로 (171,167,V)를 만든 뒤 noise 추가
    """
    def __init__(self, base_provider, raw_provider, noise_std=0.01, seed=0, clip=None, n=10):
        self.base = base_provider
        self.raw = raw_provider
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.clip = clip
        self.n = int(n)

        # base가 목표 shape를 제공한다고 가정
        x0 = self.base(0)
        self.Hc, self.Wc, self.V = x0.shape  # (171,167,V)

    def __len__(self):
        return len(self.base)

    def __call__(self, idx: int):
        idx = int(idx)

        # 1) 원본 로드 (181,177,V)
        x = self.raw(idx).astype(np.float32, copy=False)

        # 2) 랜덤 top/left (n=10이면 top,left는 0..10)
        rng = np.random.default_rng(self.seed + idx)
        choices = [i for i in range(self.n + 1) if i != 5] # 5는 제외하기로.
        top  = int(rng.choice(choices))
        left = int(rng.choice(choices))
        # top = int(rng.integers(0, self.n + 1))
        # left = int(rng.integers(0, self.n + 1))

        # 3) 채널별 shift crop 적용 -> (171,167,V)
        ys = []
        for v in range(self.V):
            ys.append(crop_shift_image(x[..., v], n=self.n, mode="shift", top=top, left=left))
        y = np.stack(ys, axis=2).astype(np.float32, copy=False)

        # (선택) 혹시라도 shape가 기대와 다르면 즉시 터뜨리기
        if y.shape != (self.Hc, self.Wc, self.V):
            raise ValueError(f"cropped shape mismatch: got {y.shape}, expected {(self.Hc, self.Wc, self.V)}")

        # 4) noise 추가
        noise = rng.normal(0.0, self.noise_std, size=y.shape).astype(np.float32)
        y = y + noise

        if self.clip is not None:
            lo, hi = self.clip
            y = np.clip(y, lo, hi)

        return y.astype(np.float32, copy=False)



class GaussianBlurWrapperProvider:
    # 평활화된 이미지를 data augmentation 처리하여, contrastive learning에 추가하기 위함
    # 적절한 sigma_range는 (0.2, 1.2) 정도로 부여하고, 그 결과를 확인해볼 것.
    def __init__(self, base_provider, sigma=0.6, sigma_range=None, seed=0,
                 mode="reflect", clip=None):
        self.base = base_provider
        self.sigma = sigma
        self.sigma_range = sigma_range
        self.seed = int(seed)
        self.mode = mode
        self.clip = clip

    def __len__(self):
        return len(self.base)

    def __call__(self, idx: int):
        x = self.base(int(idx)).astype(np.float32, copy=False)

        # sigma 고정 or 랜덤
        if self.sigma_range is not None:
            rng = np.random.default_rng(self.seed + int(idx))
            lo, hi = self.sigma_range
            sigma = float(rng.uniform(lo, hi))
        else:
            sigma = float(self.sigma)

        if sigma > 0:
            # (H,W,C)면 채널축은 blur 안 하도록 (0,0,0)
            if x.ndim == 3:
                y = gaussian_filter(x, sigma=(sigma, sigma, 0.0), mode=self.mode)
            else:
                y = gaussian_filter(x, sigma=sigma, mode=self.mode)
        else:
            y = x

        if self.clip is not None:
            lo, hi = self.clip
            y = np.clip(y, lo, hi)

        return y.astype(np.float32, copy=False)

class TranslatedWrapperProvider:
    """
    - base(idx): 이미 center-crop 된 결과 (171,167,V)
    - raw(idx): crop 안 된 원본 (181,177,V)
    raw에서 랜덤 shift crop으로 (171,167,V)를 만든 뒤 noise 추가
    """
    def __init__(self, base_provider, raw_provider, noise_std=0.01, seed=0, clip=None, n=10):
        self.base = base_provider
        self.raw = raw_provider
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.clip = clip
        self.n = int(n)

        # base가 목표 shape를 제공한다고 가정
        x0 = self.base(0)
        self.Hc, self.Wc, self.V = x0.shape  # (171,167,V)

    def __len__(self):
        return len(self.base)

    def __call__(self, idx: int):
        idx = int(idx)

        # 1) 원본 로드 (181,177,V)
        x = self.raw(idx).astype(np.float32, copy=False)

        # 2) 랜덤 top/left (n=10이면 top,left는 0..10)
        rng = np.random.default_rng(self.seed + idx)
        choices = [i for i in range(self.n + 1) if i != 5] # 5는 제외하기로.
        top  = int(rng.choice(choices))
        left = int(rng.choice(choices))
        # top = int(rng.integers(0, self.n + 1))
        # left = int(rng.integers(0, self.n + 1))

        # 3) 채널별 shift crop 적용 -> (171,167,V)
        ys = []
        for v in range(self.V):
            ys.append(crop_shift_image(x[..., v], n=self.n, mode="shift", top=top, left=left))
        y = np.stack(ys, axis=2).astype(np.float32, copy=False)

        # (선택) 혹시라도 shape가 기대와 다르면 즉시 터뜨리기
        if y.shape != (self.Hc, self.Wc, self.V):
            raise ValueError(f"cropped shape mismatch: got {y.shape}, expected {(self.Hc, self.Wc, self.V)}")

        # # 4) noise 추가
        # noise = rng.normal(0.0, self.noise_std, size=y.shape).astype(np.float32)
        # y = y + noise

        # if self.clip is not None:
        #     lo, hi = self.clip
        #     y = np.clip(y, lo, hi)

        return y.astype(np.float32, copy=False)


def load_feature_space(npy_path: str) -> np.ndarray:
    Z = np.load(npy_path)  # (N,2) expected
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"feature space must be (N,2), got {Z.shape}")
    return Z


def make_ecmwf_reference_data():

    data_dir = 'D:/ext_data/ecmwf'
    out_dir = 'D:/ext_data/ecmwf_4var'
    ensure_dir(out_dir)
    
    file_list_ecmwf = glob_npy(data_dir, "ecmwf*.npy")
    prefix = "ecmwf"
    save_selected_vars(file_list_ecmwf, prefix, data_dir, out_dir, 0, bSameName=True)
    # 여기까지, 4 var 파일로 재구성하여 저장됨.

    # 다음은 cropped data
    src_dir = out_dir
    out_dir = 'D:/ext_data/ecmwf_4var_crop'
    crop_dataset(src_dir, out_dir)

    # 다음은 t0000으로 이름 변경
    src_dir = out_dir
    dst_dir = 'D:/ext_data/ecmwf_4var_crop_idx'
    rename_datetime_index(src_dir, dst_dir)

    nTotal = 7308
    src_dir = 'D:/ext_data/ecmwf_4var_crop_idx' #dst_dir
    mask, missing = find_missing_mask_in_t_folder(src_dir, nTotal)
    print("missing count:", len(missing))
    print("first few missing:", missing[:20])
