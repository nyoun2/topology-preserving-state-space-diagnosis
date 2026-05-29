import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import shutil
import re

# ----------------------------
# Convenience: glob
# ----------------------------
def glob_npy(folder, pattern="*.npy"):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files: {os.path.join(folder, pattern)}")
    return paths

def make_file_list_for_models(target_date, data_dir, bEcmwf=True, bAimd=True):

    data_path_aimd = os.path.join(data_dir, target_date)
    data_path_ecmwf = os.path.join(data_dir, 'ecmwf')

    file_list_ecmwf = []
    if bEcmwf:
        # ecmwf 폴더 내에서 48개의 npy 파일 순차 선택
        target_t = datetime.strptime(target_date, "%Y%m%d%H")
        
        # 예측 시작점
        one_file = f"ecmwf_if_{target_date}.npy"
        one_file_path = os.path.join(data_path_ecmwf, one_file)
        file_list_ecmwf.append(one_file_path)

        nNext = 0
        while nNext < 48: # +6h 부터 +288까지
            target_t = target_t + timedelta(hours=6)
            nNext += 1

            time_str = target_t.strftime("%Y%m%d%H")
            one_file = f"ecmwf_if_{time_str}.npy"
            one_file_path = os.path.join(data_path_ecmwf, one_file)
            file_list_ecmwf.append(one_file_path)
    
    file_list_aimd = []
    if bAimd:
        # target date 폴더 내의 총 432개 npy파일 선택 x 9가지 condition 반영

        for c_model in ["fnet", "grph", "pang"]:
            for c_init in ["if", "km", "um"]:
                file_list_one_cond = []

                # 예측 시작점
                one_file = f"ecmwf_if_{target_date}.npy"                
                one_file_path = os.path.join(data_path_ecmwf, one_file)                
                file_list_one_cond.append(one_file_path)

                for c_ft in range(6, 289, 6):
                    one_file = f"{c_model}_{c_init}_{target_date}_{c_ft}.npy"
                    one_file_path = os.path.join(data_path_aimd, one_file)                
                    file_list_one_cond.append(one_file_path)

                file_list_aimd.append(file_list_one_cond)

    return file_list_ecmwf, file_list_aimd


def make_file_list_for_same_target_date(target_date, data_dir):
    
    data_path_ecmwf = os.path.join(data_dir, 'ecmwf')
    

    # ecmwf target date..
    one_file = f"ecmwf_if_{target_date}.npy"
    one_file_ecmwf = os.path.join(data_path_ecmwf, one_file)

    file_list_aimd = []    
    # target date 폴더 내의 총 432개 npy파일 선택 x 9가지 condition 반영
    target_t = datetime.strptime(target_date, "%Y%m%d%H")
    
    for c_model in ["fnet", "grph", "pang"]:
        for c_init in ["if", "km", "um"]:
            file_list_one_cond = []

            # 예측 최종 시점은 ecmwf와 동일.            
            file_list_one_cond.append(one_file_ecmwf)

            for c_ft in range(24, 289, 24):
                new_target_t = target_t - timedelta(hours=c_ft)
                new_target_date_str = new_target_t.strftime("%Y%m%d%H")

                one_file = f"{c_model}_{c_init}_{new_target_date_str}_{c_ft}.npy"
                data_path_aimd = os.path.join(data_dir, new_target_date_str)
                one_file_path = os.path.join(data_path_aimd, one_file)
                file_list_one_cond.append(one_file_path)

            file_list_aimd.append(file_list_one_cond)

    return file_list_aimd


# 동일 lead time에 해당하는 데이터만 비교
def make_file_list_from_datafolder(folder):
    # 폴더 내의 파일 검색 (aimd npy 데이터)
    # folder data_dir = "D:/data_from_server/ext_data/ecmwf"
    ext = '.npy'
    file_list = [p.name for p in Path(folder).glob(f"*{ext}")]

    tmp = file_list[0].split('_')[-1]
    lead_time = int(tmp.split('.')[0])

    date_all = []
    # file_list 내에서 date list 검출
    for one_file in file_list:
        # fnet_if_2025041100_6.npy
        one_date = one_file.split('_')[2]            
        date_all.append(one_date)

    # 날짜 값으로 count: 공통 날짜 추출
    date_counter = Counter(date_all)
    date_list = sorted(date_counter.keys())
    print(f" total files: {len(date_all)}")
    print(f" unique dates: {len(date_list)}")

    counts = set(date_counter.values())
    if len(counts) != 1:
        print(f" wrong count.. {sorted(counts)}")

    ## ecmwf file list 추출
    file_list_ecmwf = []
    for one_date in date_list:
        t = datetime.strptime(one_date, "%Y%m%d%H")
        time_new = t + timedelta(hours=lead_time)
        time_str = time_new.strftime("%Y%m%d%H")
        one_file = f"ecmwf_if_{time_str}.npy"
        file_list_ecmwf.append(one_file)

    ## aimd file list 추출
    file_list_aimd = []
    for c_model in ["fnet", "grph", "pang"]:
        for c_init in ["if", "km", "um"]:
            file_list_one_cond = []
            for one_date in date_list:
                one_file = f"{c_model}_{c_init}_{one_date}_{lead_time}.npy"
                file_list_one_cond.append(one_file)
            file_list_aimd.append(file_list_one_cond)

    return file_list_ecmwf, file_list_aimd


def list_date_folders(parent_dir: str):
    DATE_RE = re.compile(r"^\d{10}$")  # YYYYMMDDHH
    dates = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        if not DATE_RE.match(name):
            continue
        # 유효한 날짜인지 검증 (예: 2025139900 같은 거 걸러짐)
        try:
            datetime.strptime(name, "%Y%m%d%H")
        except ValueError:
            continue
        dates.append(name)

    # 문자열 정렬 == 시간 정렬 (YYYYMMDDHH 포맷이라서)
    dates.sort()
    return dates

def rearrange_folder(save_dir, prefix):

    P = Path(save_dir)
    base = P.parent

    src_old = base/f"{prefix}"
    src_new = base/f"{prefix}_cropped"
    dst = base / f"{prefix}"

    # print(src_old)
    # print(src_new)
    # print(dst)

    # 1. str 폴더 삭제
    if src_old.exists():
        shutil.rmtree(src_old)

    # 2. str_cropped → str
    src_new.rename(dst)