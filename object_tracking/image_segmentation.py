import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict

# Depth: MiDaS
import torchvision.transforms as T
from torchvision.transforms import Compose
from PIL import Image as PILImage
from pathlib import Path

class SAMSegmentor:
    def __init__(self, hfov=70, vfov=40):
        self.HFOV = hfov
        self.VFOV = vfov

        # SAM + DINO
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to("cuda")
        self.predictor = SamPredictor(self.sam)
        self.dino_model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "groundingdino_swinb_cogcoor.pth")

        # Depth estimation (MiDaS)
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to("cuda").eval()
        self.depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate_depth(self, image_bgr):
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_image = PILImage.fromarray(img)
        transformed = self.depth_transform(input_image).to("cuda")

        with torch.no_grad():
            prediction = self.depth_model(transformed.unsqueeze(0))
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    def segment(self, image_bgr, prompt):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        boxes, _, _ = predict(
            model=self.dino_model,
            image=image_rgb,
            caption=prompt,
            box_threshold=0.3,
            text_threshold=0.25
        )

        if boxes is None or len(boxes) == 0:
            return image_bgr, (None, None, None)

        input_boxes = self.predictor.transform.apply_boxes_torch(boxes, image_rgb.shape[:2]).numpy()
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=torch.tensor(input_boxes).to("cuda"),
            multimask_output=False,
        )

        mask = masks[0][0].cpu().numpy()
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return image_bgr, (None, None, None)

        center_x = np.mean(xs)
        center_y = np.mean(ys)
        h, w = image_rgb.shape[:2]

        # Нормализованные координаты
        cx_norm = (center_x - w / 2) / (w / 2)
        cy_norm = (center_y - h / 2) / (h / 2)

        # Глубина
        depth_map = self.estimate_depth(image_bgr)
        center_depth = np.median(depth_map[ys, xs])

        # Визуализация
        image_out = image_bgr.copy()
        image_out[mask > 0] = (0, 255, 0)

        return image_out, (cx_norm, cy_norm, float(center_depth))