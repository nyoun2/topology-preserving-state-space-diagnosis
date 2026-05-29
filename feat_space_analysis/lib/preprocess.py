import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import re
import shutil
import matplotlib.pyplot as plt
import xarray as xr
import glob

from feat_space_analysis.lib.io_paths import rearrange_folder, make_file_list_for_models
from feat_space_analysis.lib.io_paths import make_file_list_from_datafolder


class NpyFolderProvider:
    """
    idx -> npy 파일 로드 -> (H,W,V) float32 반환
    """
    def __init__(self, folder, H, W, V, pattern="*.npy", sort=True, mmap_mode="r"):
        self.folder = folder
        self.H, self.W, self.V = int(H), int(W), int(V)
        self.mmap_mode = mmap_mode

        paths = glob.glob(os.path.join(folder, pattern))
        if not paths:
            raise FileNotFoundError(f"No npy files found in: {folder} (pattern={pattern})")
        if sort:
            paths = sorted(paths)
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __call__(self, idx: int):
        p = self.paths[int(idx)]
        x = np.load(p, mmap_mode=self.mmap_mode)

        if x.shape == (self.H, self.W, self.V):
            pass
        elif x.shape == (self.V, self.H, self.W):
            x = np.transpose(x, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected shape {x.shape} in file {p}")
        return np.asarray(x, dtype=np.float32)
    
# ============================================================
# crop image, image shift
# ============================================================

def crop_shift_image(img, n, mode="center", top=0, left=0):
    """
    n을 고정한 크롭 함수
    - mode="center":
        상/하/좌/우에서 각각 n/2 픽셀 제거 (n은 짝수)
        결과 크기: (H-n, W-n)
    - mode="shift":
        결과 크기 자체는 center와 동일하게 (H-n, W-n)로 만들되,
        크롭 윈도우의 좌상단을 (top, left)로 지정한다.
        즉, 위쪽에 top 픽셀을 두고, 왼쪽에 left 픽셀을 둔 상태로 크롭.

    Parameters
    ----------
    img : np.ndarray
        입력 2D 이미지 (H, W)
    n : int
        최종적으로 H와 W에서 각각 줄어드는 총 픽셀 수
    mode : str
        "center" 또는 "shift"
    top : int
        mode="shift"일 때만 사용. 위쪽에 남길 픽셀 수 (= 윈도우 시작 y)
    left : int
        mode="shift"일 때만 사용. 왼쪽에 남길 픽셀 수 (= 윈도우 시작 x)
    Returns
    -------
    np.ndarray
        크롭된 이미지
    """
    if img.ndim != 2:
        raise ValueError("img는 (H, W) 형태의 2D 배열이어야 합니다.")

    H, W = img.shape
    out_h = H - n
    out_w = W - n

    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"n이 너무 큽니다. (H,W)=({H},{W})에서 n={n}이면 결과 크기가 음수/0이 됩니다.")

    if mode == "center":
        if n % 2 != 0:
            raise ValueError("mode='center'에서는 n이 반드시 짝수여야 합니다.")
        k = n // 2
        return img[k:k + out_h, k:k + out_w]

    if mode == "shift":
        # top/left가 유효한 범위인지 확인:
        # window가 (top, left)에서 시작해서 (out_h, out_w) 크기를 가지므로
        # top + out_h <= H, left + out_w <= W 이어야 함
        if not (0 <= top <= H - out_h):
            raise ValueError(f"top 범위 오류: top={top}, 허용 범위는 [0, {H - out_h}]")
        if not (0 <= left <= W - out_w):
            raise ValueError(f"left 범위 오류: left={left}, 허용 범위는 [0, {W - out_w}]")
        return img[top:top + out_h, left:left + out_w]

    raise ValueError("mode는 'center' 또는 'shift'만 가능합니다.")


def crop_and_save_folder(
    src_provider,
    dst_dir: str,
    n: int,
    mode: str = "center",
    top: int = 0,
    left: int = 0,
    keep_filename: bool = True,
    prefix: str = "",
    save_dtype=np.float32,
):
    """
    폴더 내 npy들을 순회하며 crop 적용 후 dst_dir에 파일로 저장.

    - 입력: (H,W,V)
    - 출력: (H-n, W-n, V)
    """
    os.makedirs(dst_dir, exist_ok=True)

    for i, src_path in enumerate(src_provider.paths):
        x = src_provider(i)  # (H,W,V) float32

        # 채널별 2D crop
        crops = []
        for v in range(x.shape[2]):
            crops.append(crop_shift_image(x[..., v], n=n, mode=mode, top=top, left=left))
        y = np.stack(crops, axis=2).astype(save_dtype, copy=False)  # (H-n, W-n, V)

        # 저장 파일명 결정
        base = os.path.basename(src_path)
        if prefix:
            name, ext = os.path.splitext(base)
            #base_out = f"{name}{suffix}{ext}"
            base_out = f"{prefix}{base}"
        else:
            base_out = base

        out_path = os.path.join(dst_dir, base_out)
        np.save(out_path, y)

        if (i + 1) % 50 == 0 or (i + 1) == len(src_provider):
            print(f"[{i+1}/{len(src_provider)}] saved -> {out_path}")


def crop_dataset(SRC_DIR, OUT_DIR, mode="center", n=10, top=5, left=5, prefix="cropped_"):
    H = 181
    W = 177
    V = 4

    #SRC_DIR = r"ext_data/out_npy_era5_850"
    FILE_PATTERN = "*.npy"

    era5_provider = NpyFolderProvider(
        folder=SRC_DIR, H=H, W=W, V=V,
        pattern=FILE_PATTERN, sort=True, mmap_mode="r"
    )

    ## ---- 설정 ----
    # n = 10                 # H,W 각각 10 줄어듦 -> (171,167,V)
    # mode = "center"         # "center" 또는 "shift"
    # top = 2                # shift일 때 위쪽에 남길 픽셀
    # left = 8               # shift일 때 왼쪽에 남길 픽셀

    #OUT_DIR = r"ext_data/out_npy_era5_850_cropped2"
    #suffix = f"_crop_{mode}_n{n}_t{top}_l{left}"
    prefix = prefix #"cropped_"

    crop_and_save_folder(
        src_provider=era5_provider,
        dst_dir=OUT_DIR,
        n=n,
        mode=mode,
        top=top,
        left=left,
        prefix=prefix,
        save_dtype=np.float32,
    )

################################
### NEW: ERA5 기반 채널 정규화
#################################
def compute_channel_mean_std(
    provider,
    indices: np.ndarray,
    H: int, W: int, V: int,
    batch_size: int = 16,
    clip_percentile: float | None = None,
):
    """
    provider로부터 (H,W,V) 샘플들을 스트리밍으로 읽어 채널별 mean/std(스칼라) 계산.
    clip_percentile을 주면 outlier 영향을 줄이기 위해 각 배치에서 퍼센타일 클립 후 통계.
    """
    indices = np.asarray(indices, dtype=np.int32)
    n = len(indices)

    # Welford online stats per channel
    count = np.zeros((V,), dtype=np.float64)
    mean = np.zeros((V,), dtype=np.float64)
    M2 = np.zeros((V,), dtype=np.float64)

    p = 0
    while p < n:
        b = min(batch_size, n - p)
        xb = np.empty((b, H, W, V), dtype=np.float32)
        idx_b = indices[p:p+b]
        for j, idx in enumerate(idx_b):
            x = np.asarray(provider(int(idx)), dtype=np.float32)
            if x.shape != (H, W, V):
                raise ValueError(f"provider shape mismatch: got {x.shape}, expected {(H,W,V)}")
            xb[j] = x

        if clip_percentile is not None:
            # 배치 단위로 채널별 클립
            # (b,H,W,V) -> 채널별로 퍼센타일 계산
            flat = xb.reshape(-1, V)  # (b*H*W, V)
            lo = np.percentile(flat, 100 - clip_percentile, axis=0)
            hi = np.percentile(flat, clip_percentile, axis=0)
            xb = np.clip(xb, lo, hi).astype(np.float32)

        # 채널별 업데이트: (b,H,W) 평균/분산을 한 번에 넣는 방식
        flat = xb.reshape(-1, V).astype(np.float64)  # (b*H*W, V)
        # Welford를 벡터화하기 어렵기 때문에, 채널별로 평균/분산만 배치로 계산하고 merge
        # merge two groups: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        bcount = flat.shape[0]
        bmean = flat.mean(axis=0)
        bvar = flat.var(axis=0)  # population variance
        bM2 = bvar * bcount

        total = count + bcount
        delta = bmean - mean
        mean = mean + delta * (bcount / np.maximum(total, 1e-12))
        M2 = M2 + bM2 + (delta * delta) * (count * bcount / np.maximum(total, 1e-12))
        count = total

        p += b
        if p == n or p % max(1, batch_size * 50) == 0:
            print(f"[normstats] {p}/{n}")

    var = M2 / np.maximum(count, 1e-12)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def preprocess_standardize_with_stats(x_batch: np.ndarray, stats: dict):
    """
    x_batch: (B,H,W,V)
    stats: {"mean": [V], "std":[V], "eps":... , "clip": optional}
    """
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, 1, -1)
    eps = float(stats.get("eps", 1e-6))
    y = (x_batch.astype(np.float32) - mean) / (std + eps)

    if "clip" in stats and stats["clip"] is not None:
        lo, hi = stats["clip"]
        y = np.clip(y, lo, hi)

    return y.astype(np.float32, copy=False)


def preprocess_identity(x_batch, idx_batch=None):
    return x_batch


def rename_datetime_index(
    src_dir: str,
    dst_dir: str,
    start_dt: datetime = datetime(2020, 1, 1, 0),
    step_hours: int = 6
):
    """
    src_dir의 ecmwf_if_YYYYMMDDHH.npy 파일을
    날짜 기준으로 정렬한 뒤, 기준시각(start_dt) 대비 6시간 간격 index로
    _t0000.npy 형태로 복사/이동한다.

    - 중간 missing은 자동으로 건너뜀
    - 파일이 있어야만 해당 index가 생성됨
    """

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    pat = re.compile(r"(\d{10})")  # YYYYMMDDHH

    files = []
    for fp in src_dir.glob("*.npy"):
        m = pat.search(fp.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), "%Y%m%d%H")
        files.append((dt, fp))

    # 날짜 기준 정렬
    files.sort(key=lambda x: x[0])

    print(f"[rename] found {len(files)} valid files")

    for dt, fp in files:
        delta = dt - start_dt
        hours = delta.total_seconds() / 3600.0

        # 6시간 그리드에 맞지 않으면 스킵
        if hours < 0 or hours % step_hours != 0:
            print(f"[skip] not aligned: {fp.name}")
            continue

        idx = int(hours // step_hours)
        out_path = dst_dir / f"ecmwf_t{idx:04d}.npy"

        if out_path.exists():
            print(f"[skip] exists: {out_path.name}")
            continue

        shutil.copy2(str(fp), str(out_path))

    print("[rename] done.")


def find_missing_mask_in_t_folder(t_dir: str, n_total: int):
    pat = re.compile(r"_t(\d{4})\.npy$")
    mask = np.zeros((n_total,), dtype=bool)

    for fp in Path(t_dir).glob("*.npy"):
        m = pat.search(fp.name)
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx < n_total:
            mask[idx] = True

    missing = np.where(~mask)[0].tolist()
    return mask, missing


# era5 grib to var npy..
def save_stacked_vars_per_time(
    variable_list,
    hPa_list,
    data_dir="data",
    out_dir="out_npy_20vars",
    start_idx=0,
    n_files=48,
    prefix="stack20",
):
    os.makedirs(out_dir, exist_ok=True)
    combos = [(var, int(hpa)) for var in variable_list for hpa in hPa_list]
    print(f"[INFO] combos (var-major): {len(combos)} (should be 20)")

    das = []
    for var, hpa in combos:
        grib_path = os.path.join(data_dir, f"sample_{var}_{hpa}.grib")
        if not os.path.isfile(grib_path):
            raise FileNotFoundError(f"GRIB not found: {grib_path}")

        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {
                    "shortName": var,
                    "typeOfLevel": "isobaricInhPa",
                    "level": hpa,
                }
            }
        )

        if var not in ds:
            raise KeyError(f"Variable '{var}' not found in {grib_path}. ds variables={list(ds.data_vars)}")

        da = ds[var]
        if "time" not in da.dims:
            raise ValueError(f"'time' dim not found in {grib_path}. dims={da.dims}")

        das.append(da)

    time_lens = [da.sizes["time"] for da in das]
    min_time = min(time_lens)
    need = start_idx + n_files
    if need > min_time:
        raise ValueError(f"[ERROR] Not enough time frames. need={need}, min_time_across_files={min_time}")

    ref0 = das[0].isel(time=start_idx).values
    ref_shape = ref0.shape
    print(f"[INFO] ref spatial shape: {ref_shape}")

    for k, da in enumerate(das[1:], start=1):
        shp = da.isel(time=start_idx).values.shape
        if shp != ref_shape:
            raise ValueError(f"Spatial shape mismatch at combo#{k}: {shp} != {ref_shape}")

    for ti in range(start_idx, start_idx + n_files):
        frames = []
        for da in das:
            arr2d = da.isel(time=ti).values.astype(np.float32)
            frames.append(arr2d)
        stack = np.stack(frames, axis=0)  # (20, y, x)
        save_path = os.path.join(out_dir, f"{prefix}_t{ti:04d}.npy")
        np.save(save_path, stack)
        if (ti - start_idx) % 10 == 0 or ti == start_idx + n_files - 1:
            print(f"[OK] saved {save_path}  shape={stack.shape}")

    print("[DONE] all files saved.")

def save_selected_vars(file_list_one, prefix, data_dir, out_dir, fig_idx, bSameName=False):
    
    ti = 0
    if fig_idx > 0:
        plt.figure(fig_idx)
    for one_file in file_list_one:
        #one_path = os.path.join(data_dir, f"{one_file}")        
        #X_one = np.load(one_path)        # (20,H,W)
        X_one = np.load(one_file)

        if fig_idx > 0:
            plt.clf()
            fig, axes = plt.subplots(2,2, figsize=(8,8))
        
        onelevel = []
        for k in range(4):
            if prefix =="ecmwf":
                onevar = X_one[k*5+0, :, :]
                #onevar = X_one[k, :, :]
            else:
                onevar = X_one[k*5+0, ::-1, :]
            if k == 1: # and prefix != "ecmwf":
                onevar = onevar * 9.80665
            onelevel.append(onevar)
            # if ti % 10 == 1 and not bSameName:
            #     print(f"{prefix} var {k} max_min: {np.max(onelevel[k])}, {np.min(onelevel[k])}")
            
            if fig_idx > 0:
                axes[k//2, k%2].imshow(onevar)

        if fig_idx > 0:
            plt.waitforbuttonpress()

        if bSameName:
            save_dir = out_dir
        else:
            #save_dir = f"{out_dir}/{prefix}"
            save_dir = os.path.join(out_dir, prefix)
        os.makedirs(save_dir, exist_ok=True)
        
        ti += 1
        stack = np.stack(onelevel, axis=0)  # (20, y, x)
        if bSameName:
            filename = os.path.basename(one_file)
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = os.path.join(save_dir, f"{prefix}_t{ti:04d}.npy")
        np.save(save_path, stack)

    print(f"total {ti} files converted")


def save_controlled_vars(file_list_one, prefix, data_dir, out_dir, fig_idx, bSameName=False):
    
    ti = 0
    if fig_idx > 0:
        plt.figure(fig_idx)
    for one_file in file_list_one:
        # one_path = os.path.join(data_dir, f"{one_file}")
        # X_one = np.load(one_path)        # (20,H,W)
        X_one = np.load(one_file)

        if fig_idx > 0:
            plt.clf()
            fig, axes = plt.subplots(2,2, figsize=(8,8))
        
        onelevel = []
        if ti == 0: # 첫 이미지는 emcwf에 해당함.
            for k in range(4):
                onevar = X_one[k*5+0, :, :]
                if k == 1:
                    onevar = onevar * 9.80665
                onelevel.append(onevar)
        else:
            for k in range(4):            
                if prefix =="ecmwf":
                    onevar = X_one[k, :, :]
                    #onevar = X_one[k, :, :]
                else:
                    onevar = X_one[k, ::-1, :]
                if k == 1: # and prefix != "ecmwf":
                    onevar = onevar * 9.80665
                onelevel.append(onevar)
            # if ti % 10 == 1 and not bSameName:
            #     print(f"{prefix} var {k} max_min: {np.max(onelevel[k])}, {np.min(onelevel[k])}")
            
            if fig_idx > 0:
                axes[k//2, k%2].imshow(onevar)

        if fig_idx > 0:
            plt.waitforbuttonpress()

        if bSameName:
            save_dir = out_dir
        else:
            #save_dir = f"{out_dir}/{prefix}"
            save_dir = os.path.join(out_dir, prefix)
        os.makedirs(save_dir, exist_ok=True)
        
        ti += 1
        stack = np.stack(onelevel, axis=0)  # (20, y, x)
        if bSameName:
            filename = os.path.basename(one_file)
            save_path = os.path.join(save_dir, filename)
        else:
            save_path = os.path.join(save_dir, f"{prefix}_t{ti:04d}.npy")
        np.save(save_path, stack)

    print(f"total {ti} files converted")


# ecmwf/aimd to 1 level npy..
def save_vars_from_ecmwf_and_aimd(
    target_date,
    data_dir="ext_data",
    out_dir="ext_data/Test_4var1lev",
    bSameLead = False,
    control_aimd=True
):
    out_dir = os.path.join(out_dir, f"{target_date}")
    os.makedirs(out_dir, exist_ok=True)

    if bSameLead:
        # 특정 lead time에 대한 비교 (n개)
        target_folder = target_date
        aimd_dir = os.path.join(data_dir, target_folder)
        file_list_ecmwf, file_list_aimd = make_file_list_from_datafolder(aimd_dir)
    else:
        # 전체 lead time을 모두 비교 (48개)        
        file_list_ecmwf, file_list_aimd = make_file_list_for_models(target_date, data_dir)

    # print("test ecmwf data")
    # print(file_list_ecmwf)

    # ## era5 일부를 랜덤하게 표시해본다면,
    # era5_start_idx = 0
    # data_dir_one = "out_npy_era5_850"
    # file_list_era5 = [
    #     f"vzuv850_t{idx:04d}.npy"
    #     for idx in range(era5_start_idx, era5_start_idx + 48)
    # ]
    # prefix = "ecmwf"
    # save_selected_vars(file_list_era5, prefix, data_dir_one, out_dir, 0)
    
    data_dir_one = os.path.join(data_dir, "ecmwf")
    prefix = "ecmwf"
    save_selected_vars(file_list_ecmwf, prefix, data_dir_one, out_dir, 0)
    last_dir = os.path.join(out_dir, prefix)
    save_dir = os.path.join(out_dir, f"{prefix}_cropped")
    crop_dataset(last_dir, save_dir, prefix=None)
    rearrange_folder(save_dir, prefix)


    len_aimd = len(file_list_aimd)
    data_dir_one = os.path.join(data_dir, f"{target_date}")
    for a in range(len_aimd):
        prefix = f"ai_{a+1:02d}"
        if control_aimd:
            save_controlled_vars(file_list_aimd[a], prefix, data_dir_one, out_dir, 0)
        else:
            save_selected_vars(file_list_aimd[a], prefix, data_dir_one, out_dir, 0)
        last_dir = os.path.join(out_dir, prefix)
        save_dir = os.path.join(out_dir, f"{prefix}_cropped")
        crop_dataset(last_dir, save_dir, prefix=None)
        rearrange_folder(save_dir, prefix)