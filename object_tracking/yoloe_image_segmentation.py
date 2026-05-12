import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


class YOLOESegmentor:
    def __init__(
        self,
        model_name="yoloe-11s-seg.pt",
        text_encoder_name="mobileclip_blt.ts",
        imgsz=640,
    ):
        try:
            from ultralytics import YOLOE
            from ultralytics.utils import SETTINGS
        except ImportError as exc:
            raise ImportError(
                "YOLOE support requires the `ultralytics` package with YOLOE support installed."
            ) from exc

        self.weights_dir = self._get_weights_dir()
        self.model_path = self._resolve_weight_path(model_name)
        self.text_encoder_path = self._resolve_weight_path(text_encoder_name)

        # Force Ultralytics to look up auxiliary weights (e.g. mobileclip_blt.ts)
        # inside this package's model_weights directory instead of the workspace root.
        SETTINGS["weights_dir"] = str(self.weights_dir)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLOE(self.model_path)
        self.imgsz = imgsz
        self.current_prompt = None

    def _get_weights_dir(self):
        try:
            share_dir = Path(get_package_share_directory("object_tracking"))
            share_weights = share_dir / "model_weights"
            if share_weights.is_dir():
                return share_weights
        except PackageNotFoundError:
            pass

        return Path(__file__).resolve().parent / "model_weights"

    def _resolve_weight_path(self, weight_name):
        weight_path = Path(weight_name)
        if weight_path.is_file():
            return str(weight_path)

        candidate = self.weights_dir / weight_path.name
        if candidate.is_file():
            return str(candidate)

        raise FileNotFoundError(
            f"\n[ERROR] Required YOLOE weight file not found:\n  {candidate}\n\n"
            f"Please place `{weight_path.name}` in:\n  {self.weights_dir}\n"
        )

    def _set_prompt(self, prompt):
        if prompt == self.current_prompt:
            return

        classes = [prompt]
        self.model.set_classes(classes, self.model.get_text_pe(classes))
        self.current_prompt = prompt

    def segment(self, image, prompt, depth_map, conf=0.25, min_mask_area=200):
        """
        Run YOLOE text-prompted detection and segmentation on the whole image.
        Returns an overlay image, center coordinates, inference time and the
        original depth map, mirroring the CLIPSeg interface.
        """

        self._set_prompt(prompt)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        results = self.model.predict(
            source=image_rgb,
            conf=conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        end_time = time.time()

        segmentation_time = end_time - start_time

        if not results:
            return image, None, segmentation_time, depth_map

        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return image, None, segmentation_time, depth_map

        boxes_conf = result.boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(boxes_conf))

        mask_tensor = result.masks.data[best_idx]
        mask = mask_tensor.detach().cpu().numpy() > 0.5

        mask_area = int(np.sum(mask))
        if mask_area < min_mask_area and mask_area > 0:
            print(f"Object ignored due to small area: {mask_area} pixels")
            return image, None, segmentation_time, depth_map

        center_coords = self.get_center_coordinates(mask)

        image_out = image.copy()
        image_out[mask > 0] = (0, 255, 0)

        box = result.boxes.xyxy[best_idx].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image_out,
            f"{prompt} ({boxes_conf[best_idx]:.2f})",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

        return image_out, center_coords, segmentation_time, depth_map

    def get_center_coordinates(self, mask):
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))
        return (x_mean, y_mean)
