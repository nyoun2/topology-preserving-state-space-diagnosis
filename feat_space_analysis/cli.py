# feat_space_analysis/cli.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from feat_space_analysis.workflows.evaluate_models import (
    infer_and_evaluate_lead_time_trajectory,
    infer_and_evaluate_valid_time_trajectory,
)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None:
        return {}

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to use --config.\n"
            "Install it with:\n"
            "    pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a dictionary: {path}")

    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config

    for key in keys:
        if not isinstance(cur, dict):
            return default

        if key not in cur:
            return default

        cur = cur[key]

    return cur


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--tag-name",
        type=str,
        default=None,
        help="Feature/state-space experiment tag name.",
    )

    parser.add_argument(
        "--input-root-dir",
        type=str,
        default=None,
        help="Root directory containing inference input folders.",
    )

    parser.add_argument(
        "--input-subdir",
        type=str,
        default=None,
        help="Subdirectory under input_root_dir/{tag_name}.",
    )

    parser.add_argument(
        "--processed-root-dir",
        type=str,
        default=None,
        help="Root directory containing processed inference outputs.",
    )

    parser.add_argument(
        "--model-list",
        type=int,
        nargs="+",
        default=None,
        help="Model IDs to evaluate, e.g., --model-list 1 4 7.",
    )

    parser.add_argument(
        "--show-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If True, skip inference and only show existing processed results. "
            "Use --show-only or --no-show-only."
        ),
    )

    parser.add_argument(
        "--target-folder-suffix",
        type=str,
        default=None,
        help=(
            "Suffix added to target folder name. "
            "Lead-time uses ''. Valid-time currently uses '_target'."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feat-space-analysis",
        description=(
            "Reproducible lead-time and valid-time trajectory diagnosis "
            "for topology-preserving state-space evaluation of weather forecasts."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Workflow command.",
    )

    # Lead-time command
    p_lead = subparsers.add_parser(
        "lead",
        help="Run lead-time trajectory diagnosis.",
    )
    add_common_args(p_lead)

    p_lead.add_argument(
        "--feature-npy",
        type=str,
        default=None,
        help="Path to reference state-space coordinate file.",
    )

    p_lead.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to trajectory evaluation log file, json or jsonl.",
    )

    p_lead.add_argument(
        "--top-ratio",
        type=float,
        default=None,
        help="Top ratio used to construct background/model metric maps.",
    )

    # Valid-time command
    p_valid = subparsers.add_parser(
        "valid",
        help="Run valid-time trajectory diagnosis.",
    )
    add_common_args(p_valid)

    return parser


def resolve_common_args(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    tag_name = args.tag_name or get_nested(
        config,
        "experiment",
        "tag_name",
        default="CL_era5_REF-ecmwf_COND-noise-shift_v1",
    )

    input_root_dir = args.input_root_dir or get_nested(
        config,
        "paths",
        "input_root_dir",
        default="data/input",
    )

    input_subdir = args.input_subdir or get_nested(
        config,
        "paths",
        "input_subdir",
        default="Test_4var1lev",
    )

    processed_root_dir = args.processed_root_dir or get_nested(
        config,
        "paths",
        "processed_root_dir",
        default="out_test_features",
    )

    model_list = args.model_list
    if model_list is None:
        model_list = get_nested(
            config,
            "models",
            "model_list",
            default=[1, 4, 7],
        )

    show_only = args.show_only
    if show_only is None:
        show_only = bool(
            get_nested(
                config,
                "runtime",
                "show_only",
                default=True,
            )
        )

    include_dates = get_nested(
        config,
        "dates",
        "include_dates",
        default=None,
    )

    target_folder_suffix = args.target_folder_suffix
    if target_folder_suffix is None:
        target_folder_suffix = get_nested(
            config,
            "folders",
            "target_folder_suffix",
            default="",
        )

    return {
        "tag_name": tag_name,
        "input_root_dir": input_root_dir,
        "input_subdir": input_subdir,
        "processed_root_dir": processed_root_dir,
        "model_list": model_list,
        "show_only": show_only,
        "include_dates": include_dates,
        "target_folder_suffix": target_folder_suffix,
    }


def run_lead(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    common = resolve_common_args(args, config)

    feature_npy = args.feature_npy or get_nested(
        config,
        "paths",
        "feature_npy",
        default=None,
    )

    log_path = args.log_path or get_nested(
        config,
        "paths",
        "log_path",
        default=None,
    )

    top_ratio = args.top_ratio
    if top_ratio is None:
        top_ratio = get_nested(
            config,
            "trajectory",
            "top_ratio",
            default=0.15,
        )

    print("[COMMAND] lead")
    print(f"  tag_name             : {common['tag_name']}")
    print(f"  input_root_dir       : {common['input_root_dir']}")
    print(f"  input_subdir         : {common['input_subdir']}")
    print(f"  processed_root_dir   : {common['processed_root_dir']}")
    print(f"  model_list           : {common['model_list']}")
    print(f"  show_only            : {common['show_only']}")
    print(f"  include_dates        : {common['include_dates']}")
    print(f"  target_folder_suffix : {common['target_folder_suffix']}")
    print(f"  feature_npy          : {feature_npy}")
    print(f"  log_path             : {log_path}")
    print(f"  top_ratio            : {top_ratio}")

    infer_and_evaluate_lead_time_trajectory(
        show_only=common["show_only"],
        tag_name=common["tag_name"],
        input_root_dir=common["input_root_dir"],
        input_subdir=common["input_subdir"],
        processed_root_dir=common["processed_root_dir"],
        feature_npy=feature_npy,
        log_path=log_path,
        model_list=common["model_list"],
        top_ratio=top_ratio,
        include_dates=common["include_dates"],
        target_folder_suffix=common["target_folder_suffix"],
    )


def run_valid(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    common = resolve_common_args(args, config)

    print("[COMMAND] valid")
    print(f"  tag_name             : {common['tag_name']}")
    print(f"  input_root_dir       : {common['input_root_dir']}")
    print(f"  input_subdir         : {common['input_subdir']}")
    print(f"  processed_root_dir   : {common['processed_root_dir']}")
    print(f"  model_list           : {common['model_list']}")
    print(f"  show_only            : {common['show_only']}")
    print(f"  include_dates        : {common['include_dates']}")
    print(f"  target_folder_suffix : {common['target_folder_suffix']}")

    infer_and_evaluate_valid_time_trajectory(
        show_only=common["show_only"],
        tag_name=common["tag_name"],
        input_root_dir=common["input_root_dir"],
        input_subdir=common["input_subdir"],
        processed_root_dir=common["processed_root_dir"],
        model_list=common["model_list"],
        include_dates=common["include_dates"],
        target_folder_suffix=common["target_folder_suffix"],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == "lead":
        run_lead(args, config)

    elif args.command == "valid":
        run_valid(args, config)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()