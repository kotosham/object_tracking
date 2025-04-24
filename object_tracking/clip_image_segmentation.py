import torch
from torch.nn.functional import interpolate
from transformers import CLIPProcessor, CLIPModel, CLIPSegProcessor, CLIPSegForImageSegmentation
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from PIL import Image as PILImage

class CLIPSegmentor:
    def __init__(self):
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def segment(self, image, prompt, threshold = 0.75) -> np.ndarray:
        """
        Use CLIPSeg to generate a segmentation mask for the object described by prompt.
        Returns a binary mask (numpy array) of the same size as the image.
        """

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)

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

        center_coords = self.get_center_coordinates(mask)

        image_out = image.copy()
        image_out[mask > 0] = (0, 255, 0)

        #print("masked image received, returning")

        return image_out, center_coords

    def get_center_coordinates(self, mask):
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))
        return (x_mean, y_mean)