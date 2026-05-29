import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np

def setup_logger(log_dir="logs", name="experiment"):
    import os
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()  # 콘솔에도 같이 출력
        ]
    )
    return log_path

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def append_jsonl(filepath: str, record: dict):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_result_log(
    target_date: str,
    pos2d: list[float],
    length_list: list[int],

    dtw_scores_list: list[list[float]],
    rmse_scores_list: list[list[float]],

    best_dtw_list: list[int],   # 예: [4, 7, 5, 9]
    best_rmse_list: list[int]  # 예: [4, 7, 5, 9]
    
) -> dict:
    n = len(length_list)

    if not (
        len(dtw_scores_list) == len(rmse_scores_list) ==
        len(best_dtw_list) == len(best_rmse_list) == n
    ):
        raise ValueError("All input lists must match length_list length.")

    lengths_dict = {}
    for L, dtw_scores, rmse_scores, best_dtw, best_rmse in zip(
        length_list,
        dtw_scores_list,
        rmse_scores_list,
        best_dtw_list,
        best_rmse_list,
    ):
        lengths_dict[str(L)] = {
            "best": {
                "dtw": int(best_dtw),
                "rmse": int(best_rmse),
            },
            "scores": {
                "dtw": [float(np.round(v, 6)) for v in dtw_scores],
                "rmse": [float(np.round(v, 6)) for v in rmse_scores],
            }
        }

    return {
        "target_date": target_date,
        "pos2d": [
            np.round(float(pos2d[0]), 4),
            np.round(float(pos2d[1]), 4),
        ],
        "lengths": lengths_dict,
    }

def load_logs_jsonl(jsonl_path: str) -> list[dict]:
    """
    jsonl: 한 줄에 record 1개
    """
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_logs_json(json_path: str) -> list[dict]:
    """
    단일 json 파일:
      - {"runs":[...]} 형태 또는
      - [...records...] 형태 지원
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
        return data["runs"]
    # 혹시 단일 record 1개라면 list로 감싸기
    if isinstance(data, dict) and "target_date" in data:
        return [data]
    raise ValueError("Unknown json format. Use jsonl or {runs:[...] } or list-of-records.")


def convert_model_name(num_model):
    if num_model == 1:
        return "fnet:ifs"
    elif num_model == 2:
        return "fnet:kim"
    elif num_model == 3:
        return "fnet:um"
    elif num_model == 4:
        return "grph:ifs"
    elif num_model == 5:
        return "grph:kim"
    elif num_model == 6:
        return "grph:um"
    elif num_model == 7:
        return "pang:ifs"
    elif num_model == 8:
        return "pang:kim"
    elif num_model == 9:
        return "pang:um"