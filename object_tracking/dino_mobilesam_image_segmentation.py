import torch
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from mobile_sam import sam_model_registry, SamPredictor

from PIL import Image as PILImage

import tf2_geometry_msgs
from geometry_msgs.msg import Point, PoseStamped

import math
from geometry_msgs.msg import Quaternion

import time

class GroundingDINOMobileSAMSegmentor:
    def __init__(self, hfov=70, vfov=40):
        self.HFOV = hfov
        self.VFOV = vfov
        self.dino_model_id = "IDEA-Research/grounding-dino-tiny"
        self.dino_device, self.sam_device = self._select_devices()
        self.last_detection_score = None
        self.last_detection_label = None
        self.last_mask = None

        share_dir = get_package_share_directory('object_tracking')
        checkpoint_path_SAM = os.path.join(share_dir, 'model_weights', 'mobile_sam.pt')

        if not os.path.isfile(checkpoint_path_SAM):
            raise FileNotFoundError(
                f"\n[ERROR] MobileSAM checkpoint not found at:\n  {checkpoint_path_SAM}\n\n"
                f"Please download `mobile_sam.pt` from the official MobileSAM repository\n"
                f"and place it in:\n  {os.path.join(share_dir, 'model_weights')}\n"
            )

        # MobileSAM + DINO
        self.sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path_SAM).to(self.sam_device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        self.dino_model_source = self._resolve_dino_model_source(share_dir)
        dino_load_kwargs = {}
        if os.path.isdir(self.dino_model_source):
            dino_load_kwargs["local_files_only"] = True
            print(f"Using local GroundingDINO weights from: {self.dino_model_source}")
        else:
            print(
                f"Local GroundingDINO snapshot not found. Falling back to Hugging Face model id: "
                f"{self.dino_model_source}"
            )
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_model_source, **dino_load_kwargs)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.dino_model_source,
            **dino_load_kwargs,
        ).to(self.dino_device)
        self.dino_model.eval()

    def runtime_info(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return (
                f"GroundingDINO device={self.dino_device}, "
                f"MobileSAM device={self.sam_device}, "
                f"CUDA device={gpu_name} ({total_memory_gib:.2f} GiB VRAM)"
            )
        return (
            f"GroundingDINO device={self.dino_device}, "
            f"MobileSAM device={self.sam_device}, CUDA unavailable"
        )

    def _select_devices(self):
        if not torch.cuda.is_available():
            return "cpu", "cpu"

        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        total_memory_gib = total_memory_bytes / (1024 ** 3)
        total_memory_gb = total_memory_bytes / 1e9

        # Use decimal GB here so nominal 6 GB GPUs are treated as 6 GB class
        # devices instead of being penalized by the GiB/GB conversion.
        if total_memory_gb < 6.0:
            print(
                f"CUDA device has only {total_memory_gib:.2f} GiB ({total_memory_gb:.2f} GB) VRAM. "
                "Using GroundingDINO on CPU and keeping SAM on CUDA to fit memory."
            )
            return "cpu", "cuda"

        return "cuda", "cuda"

    def _resolve_dino_model_source(self, share_dir):
        candidates = []

        env_model_dir = os.environ.get("GROUNDING_DINO_MODEL_DIR", "").strip()
        if env_model_dir:
            candidates.append(os.path.expanduser(env_model_dir))

        candidates.append(os.path.join(share_dir, "model_weights", "grounding-dino-tiny"))

        hf_snapshot_dir = self._find_local_hf_snapshot_dir()
        if hf_snapshot_dir:
            candidates.append(hf_snapshot_dir)

        for candidate in candidates:
            if self._is_valid_dino_dir(candidate):
                return candidate

        return self.dino_model_id

    def _find_local_hf_snapshot_dir(self):
        hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        snapshots_root = os.path.join(
            hf_home,
            "hub",
            "models--IDEA-Research--grounding-dino-tiny",
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
            if self._is_valid_dino_dir(snapshot_dir):
                return snapshot_dir
        return None

    @staticmethod
    def _is_valid_dino_dir(path):
        required_files = (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
        )
        return os.path.isdir(path) and all(os.path.isfile(os.path.join(path, name)) for name in required_files)

    def segment(self, image_bgr, prompt, depth_map, min_mask_area=100, box_threshold=0.50):
        self.last_detection_score = None
        self.last_detection_label = None
        self.last_mask = None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        text_labels = [[prompt]]

        start_time_DINO = time.time()

        inputs = self.dino_processor(images=image_pil, text=text_labels, return_tensors="pt").to(self.dino_device)
        with torch.inference_mode():
            outputs = self.dino_model(**inputs)

        print("received outputs from DINO")

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.25,
            target_sizes=[image_pil.size[::-1]]
        )

        print("dino_processor finished")

        end_time_DINO = time.time()

        DINO_time = end_time_DINO - start_time_DINO

        print(f"GroundingDINO's work time is {DINO_time}")

        result = results[0]

        # Фильтрация по уверенности GroundingDINO.
        box_threshold = float(box_threshold)
        filtered = [
            (box.cpu().numpy(), score.item(), label)
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"])
            if score.item() >= box_threshold
        ]

        if not filtered:
            print("Объект не найден по уверенности")
            return image_bgr, None, depth_map, 0

        # Выбери самый уверенный бокс
        box, score, label = sorted(filtered, key=lambda x: -x[1])[0]
        input_box = np.array([box])
        self.last_detection_score = float(score)
        self.last_detection_label = str(label)
        print(f"Найден объект: {label} (score={score:.2f})")

        print("Received bounding boxes")

        start_time_SAM = time.time()

        if self.sam_device == "cuda":
            torch.cuda.empty_cache()

        if self.dino_device == "cuda" and self.sam_device != "cuda":
            # Free as much VRAM as possible before the CPU SAM pass.
            del outputs
            torch.cuda.empty_cache()

        try:
            input_boxes = self.predictor.transform.apply_boxes_torch(torch.tensor(input_box), image_bgr.shape[:2]).numpy()
            if self.sam_device == "cuda":
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    self.predictor.set_image(image_rgb)
                    masks, _, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=torch.tensor(input_boxes).to(self.sam_device),
                        multimask_output=False,
                    )
            else:
                self.predictor.set_image(image_rgb)
                with torch.inference_mode():
                    masks, _, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=torch.tensor(input_boxes).to(self.sam_device),
                        multimask_output=False,
                    )
        except torch.OutOfMemoryError:
            if self.sam_device != "cuda":
                raise

            print("CUDA OOM during SAM inference, switching SAM to CPU and retrying.")
            self.sam_device = "cpu"
            self.sam = self.sam.to(self.sam_device)
            self.predictor = SamPredictor(self.sam)
            torch.cuda.empty_cache()

            input_boxes = self.predictor.transform.apply_boxes_torch(torch.tensor(input_box), image_bgr.shape[:2]).numpy()
            self.predictor.set_image(image_rgb)
            with torch.inference_mode():
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=torch.tensor(input_boxes).to(self.sam_device),
                    multimask_output=False,
                )

        end_time_SAM = time.time()

        SAM_time = end_time_SAM - start_time_SAM

        print(f"SAM's work time is {SAM_time}")

        print("masks acquired")

        mask = masks[0][0].cpu().numpy() > 0.5
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return image_bgr, None, depth_map, DINO_time + SAM_time

        mask_area = int(np.sum(mask))
        if mask_area < min_mask_area and mask_area > 0:
            print(f"Object ignored due to small area: {mask_area} pixels")
            return image_bgr, None, depth_map, DINO_time + SAM_time

        center_coords = self.get_center_coordinates(mask)
        self.last_mask = mask.astype(np.uint8)

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

        return image_out, center_coords, depth_map, DINO_time+SAM_time
    
    def get_center_coordinates(self, mask):
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))
        return (x_mean, y_mean)
    
    def get_goal_point(self, depth_image, center_coords, camera_transform, transform_base, camera_intrinsics, offset):
        fx, fy, cx, cy = camera_intrinsics
        
        x_px = int(center_coords[0])
        y_px = int(center_coords[1])
        
        depth = depth_image[y_px, x_px]

        X = (x_px - cx) * depth / fx
        Y = (y_px - cy) * depth / fy
        Z = depth

        point_camera = Point()
        point_camera.x = X
        point_camera.y = Y
        point_camera.z = float(Z)

        point_stamped = tf2_geometry_msgs.PointStamped()
        point_stamped.header.frame_id = 'depth_camera_link_optical'
        point_stamped.header.stamp = camera_transform.header.stamp
        point_stamped.point = point_camera

        point_world = tf2_geometry_msgs.do_transform_point(point_stamped, camera_transform)

        robot_x = transform_base.transform.translation.x
        robot_y = transform_base.transform.translation.y

        dx = point_world.point.x - robot_x
        dy = point_world.point.y - robot_y

        distance = np.hypot(dx, dy)

        if distance <= offset:
            goal_x = robot_x
            goal_y = robot_y

            goal = PoseStamped()
                    
            goal.header.frame_id = 'map'

            goal.pose.position.x = goal_x
            goal.pose.position.y = goal_y
            return goal

        #scale = (distance - offset) / distance
        scale = 0.8

        goal_x = robot_x + dx * scale
        goal_y = robot_y + dy * scale

        print(f'Объект в map frame: X={point_world.point.x:.2f}, Y={point_world.point.y:.2f}, Z={point_world.point.z:.2f}')
        print(f'Расстояние до цели distance = {distance:.2f}, offset = {offset:.2f}')

        goal = PoseStamped()
                    
        goal.header.frame_id = 'map'

        theta = np.arctan2(dy, dx)

        def yaw_to_quaternion(yaw):
            q = Quaternion()
            q.z = math.sin(yaw / 2.0)
            q.w = math.cos(yaw / 2.0)
            return q

        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y

        goal.pose.orientation = yaw_to_quaternion(theta)

        return goal
