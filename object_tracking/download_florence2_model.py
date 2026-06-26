import argparse
import os
import sys
from pathlib import Path


def _inject_ml_venv_site_packages():
    venv_python = os.path.expanduser(
        os.environ.get("FLORENCE2_VENV_PYTHON", "~/.venvs/ros-jazzy-ml/bin/python")
    )
    venv_root = Path(venv_python).expanduser().parent.parent
    site_packages = venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if site_packages.is_dir():
        site_packages_str = str(site_packages)
        if site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)


_inject_ml_venv_site_packages()

from huggingface_hub import snapshot_download

try:
    from object_tracking.florence2_image_segmentation import Florence2Segmentor
except ModuleNotFoundError:
    from florence2_image_segmentation import Florence2Segmentor


def default_output_dir(model_id):
    return Path(__file__).resolve().parent / "model_weights" / Path(model_id).name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and validate a complete Florence-2 snapshot for the local ROS 2 project."
    )
    parser.add_argument(
        "--model-id",
        default="microsoft/Florence-2-base-ft",
        help="Hugging Face model id to download.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory where the Florence-2 snapshot should be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist locally.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent download workers used by huggingface_hub.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(args.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "configuration_florence2.py",
        "processing_florence2.py",
        "modeling_florence2.py",
    ]

    print(f"Downloading {args.model_id} to {output_dir} ...")
    snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
        force_download=args.force,
        max_workers=max(1, int(args.max_workers)),
        allow_patterns=allow_patterns,
    )

    Florence2Segmentor.patch_local_configuration(output_dir)
    Florence2Segmentor.patch_local_processing(output_dir)
    Florence2Segmentor.patch_local_modeling(output_dir)

    if not Florence2Segmentor.is_complete_model_dir(output_dir):
        raise RuntimeError(
            "Florence-2 download finished, but the local directory is still incomplete:\n"
            f"  {output_dir}\n"
            "Please check the network / Hugging Face access and retry with --force."
        )

    print("Florence-2 snapshot is ready.")
    print(f"Local model directory: {output_dir}")
    print("Run Florence-2 in the project with:")
    print(
        "  ros2 launch object_tracking sam_node_continuous.launch.py "
        f"model_mode:=florence2 florence2_model_id:={output_dir}"
    )


if __name__ == "__main__":
    main()
