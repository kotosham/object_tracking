import torch
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
import cv2
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict

import requests
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from PIL import Image as PILImage

class SAMSegmentor:
    def __init__(self, hfov=70, vfov=40):
        self.HFOV = hfov
        self.VFOV = vfov

        share_dir = get_package_share_directory('object_tracking')
        checkpoint_path_SAM = os.path.join(share_dir, 'model_weights', 'sam_vit_h_4b8939.pth')

        if not os.path.isfile(checkpoint_path_SAM):
            raise FileNotFoundError(f"\n[ERROR] SAM checkpoint not found at:\n  {checkpoint_path_SAM}\n\n"
                                    f"Please download it from:\n"
                                    f"  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                                    f"and place it in:\n  object_tracking/models/")

        # SAM + DINO
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path_SAM).to("cuda")
        self.predictor = SamPredictor(self.sam)
        self.dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda")

    def segment(self, image_bgr, prompt):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        text_labels = [[prompt]]
        inputs = self.dino_processor(images=image_pil, text=text_labels, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        print("received outputs from DINO")

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )

        print("dino_pricessor finished")

        result = results[0]

        # Фильтрация по порогу
        box_threshold = 0.4
        filtered = [
            (box.cpu().numpy(), score.item(), label)
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"])
            if score.item() >= box_threshold
        ]

        if not filtered:
            print("❌ Объект не найден по уверенности")
            return image_bgr, (None, None)

        # Выбери самый уверенный бокс
        box, score, label = sorted(filtered, key=lambda x: -x[1])[0]
        input_box = np.array([box])
        print(f"✔ Найден объект: {label} (score={score:.2f})")

        print("Received bounding boxes")

        input_boxes = self.predictor.transform.apply_boxes_torch(torch.tensor(input_box), image_bgr.shape[:2]).numpy()
        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=torch.tensor(input_boxes).to("cuda"),
            multimask_output=False,
        )

        print("masks acquired")

        mask = masks[0][0].cpu().numpy()
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return image_bgr, (None, None)

        center_x = np.mean(xs)
        center_y = np.mean(ys)
        h, w = image_rgb.shape[:2]

        # Нормализованные координаты
        cx_norm = (center_x - w / 2) / (w / 2)
        cy_norm = (center_y - h / 2) / (h / 2)

        # Визуализация
        image_out = image_bgr.copy()
        image_out[mask > 0] = (0, 255, 0)

        print("masked image received, returning")

        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            if score.item() >= box_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{label} ({score:.2f})"
                cv2.putText(image_out, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return image_out, (cx_norm, cy_norm) #, float(center_depth))