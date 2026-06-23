import os
from pathlib import Path

import torch
from torch.nn.functional import interpolate
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.utils import logging as transformers_logging
import numpy as np
import cv2

from PIL import Image as PILImage

import time

class CLIPSegmentor:
    def __init__(self):
        transformers_logging.set_verbosity_error()
        self.model_id = "CIDAS/clipseg-rd64-refined"
        self.model_source = self._resolve_model_source()
        load_kwargs = {}
        if os.path.isdir(self.model_source):
            load_kwargs["local_files_only"] = True
            print(f"Using local CLIPSeg weights from: {self.model_source}")
        else:
            print(
                f"Local CLIPSeg snapshot not found. Falling back to Hugging Face model id: "
                f"{self.model_source}"
            )

        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_source, **load_kwargs)
        self.processor = CLIPSegProcessor.from_pretrained(self.model_source, **load_kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.last_detection_score = None
        self.last_mask = None

    def runtime_info(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"CLIPSeg device={self.device}, CUDA device={gpu_name} ({total_memory_gib:.2f} GiB VRAM)"
        return f"CLIPSeg device={self.device}, CUDA unavailable"

    def _resolve_model_source(self):
        candidates = []

        env_model_dir = os.environ.get("CLIPSEG_MODEL_DIR", "").strip()
        if env_model_dir:
            candidates.append(os.path.expanduser(env_model_dir))

        try:
            share_dir = get_package_share_directory("object_tracking")
            candidates.append(os.path.join(share_dir, "model_weights", "clipseg-rd64-refined"))
        except PackageNotFoundError:
            pass

        candidates.append(str(Path(__file__).resolve().parent / "model_weights" / "clipseg-rd64-refined"))

        hf_snapshot_dir = self._find_local_hf_snapshot_dir()
        if hf_snapshot_dir:
            candidates.append(hf_snapshot_dir)

        for candidate in candidates:
            if self._is_valid_clipseg_dir(candidate):
                return candidate

        return self.model_id

    def _find_local_hf_snapshot_dir(self):
        hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        snapshots_root = os.path.join(
            hf_home,
            "hub",
            "models--CIDAS--clipseg-rd64-refined",
            "snapshots",
        )
        if not os.path.isdir(snapshots_root):
            return None

        snapshot_dirs = [
            os.path.join(snapshots_root, name)
            for name in os.listdir(snapshots_root)
            if os.path.isdir(os.path.join(snapshots_root, name))
        ]
        if not snapshot_dirs:
            return None

        snapshot_dirs.sort(key=os.path.getmtime, reverse=True)
        for snapshot_dir in snapshot_dirs:
            if self._is_valid_clipseg_dir(snapshot_dir):
                return snapshot_dir
        return None

    @staticmethod
    def _is_valid_clipseg_dir(path):
        required_files = (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        )
        return os.path.isdir(path) and all(os.path.isfile(os.path.join(path, name)) for name in required_files)

    def segment(self, image, prompt, depth_map, threshold = 0.85, min_mask_area = 200) -> np.ndarray:
        self.last_detection_score = None
        self.last_mask = None
        """
        Use CLIPSeg to generate a segmentation mask for the object described by prompt.

        Returns a binary mask (numpy array) of the same size as the image.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)

        start_time = time.time()

        # Process the image and the prompt
        inputs = self.processor(text=[prompt], images=image_pil, return_tensors="pt").to(self.device)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # Shape should be [1, 1, h, w]

        # Ensure logits are 4D
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)  # Convert from [1, H, W] to [1, 1, H, W]

        # Upsample logits to image size (image.size[::-1] gives (height, width))
        upsampled_logits = interpolate(logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False)

        # Convert logits to binary mask using sigmoid activation and threshold
        mask = upsampled_logits.sigmoid()[0][0].cpu().numpy() > threshold
        
        end_time = time.time()

        mask_area = np.sum(mask)
        segmentation_time = end_time - start_time
        if mask_area < min_mask_area and mask_area > 0:
            print(f"Object ignored due to small area: {mask_area} pixels")
            return image, None, segmentation_time, depth_map

        center_coords = self.get_center_coordinates(mask)
        self.last_detection_score = float(mask_area)
        self.last_mask = mask.astype(np.uint8)

        image_out = image.copy()
        image_out[mask > 0] = (0, 255, 0)

        return image_out, center_coords, segmentation_time, depth_map

    def get_center_coordinates(self, mask):
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))
        return (x_mean, y_mean)
