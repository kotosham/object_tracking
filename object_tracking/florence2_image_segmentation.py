import os
import time
from pathlib import Path

os.environ.setdefault("HF_MODULES_CACHE", "/tmp/object_tracking_hf_modules")

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from transformers import AutoModelForCausalLM, AutoProcessor


class Florence2Segmentor:
    REQUIRED_BASE_FILES = (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "configuration_florence2.py",
        "processing_florence2.py",
        "modeling_florence2.py",
    )

    def __init__(
        self,
        model_id="microsoft/Florence-2-base-ft",
        task_prompt="<REFERRING_EXPRESSION_SEGMENTATION>",
        max_new_tokens=1024,
        num_beams=3,
    ):
        self.model_id = model_id
        self.task_prompt = task_prompt
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.last_detection_score = None
        self.last_mask = None

        self.invalid_local_candidates = []
        self.model_source = self._resolve_model_source()
        processor_kwargs = {"trust_remote_code": True}
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }
        if os.path.isdir(self.model_source):
            self.patch_local_configuration(self.model_source)
            self.patch_local_processing(self.model_source)
            self.patch_local_modeling(self.model_source)
            processor_kwargs["local_files_only"] = True
            model_kwargs["local_files_only"] = True
            print(f"Using local Florence-2 weights from: {self.model_source}")
        else:
            print(
                f"No complete local Florence-2 snapshot was found. Falling back to Hugging Face model id: "
                f"{self.model_source}"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_source, **processor_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                **model_kwargs,
            ).to(self.device)
        except AttributeError as exc:
            if "forced_bos_token_id" in str(exc) or "image_token" in str(exc):
                raise RuntimeError(self._load_failure_message()) from exc
            raise
        except OSError as exc:
            raise RuntimeError(self._load_failure_message()) from exc
        except Exception as exc:
            if not os.path.isdir(self.model_source):
                raise RuntimeError(self._remote_download_failure_message()) from exc
            raise

        self.model.eval()

    def runtime_info(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return (
                f"Florence-2 device={self.device}, dtype={self.torch_dtype}, "
                f"CUDA device={gpu_name} ({total_memory_gib:.2f} GiB VRAM)"
            )
        return f"Florence-2 device={self.device}, dtype={self.torch_dtype}, CUDA unavailable"

    @classmethod
    def is_complete_model_dir(cls, path):
        candidate = Path(path)
        if not candidate.is_dir():
            return False

        if not all((candidate / name).is_file() for name in cls.REQUIRED_BASE_FILES):
            return False

        weights_ok = any(
            (candidate / name).is_file()
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        )
        return weights_ok

    @classmethod
    def patch_local_configuration(cls, model_dir):
        config_path = Path(model_dir) / "configuration_florence2.py"
        if not config_path.is_file():
            return False

        marker = 'self.forced_bos_token_id = kwargs.get("forced_bos_token_id", None)'
        text = config_path.read_text(encoding="utf-8")
        if marker in text:
            return False

        needle = '        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True\n\n        super().__init__(\n'
        replacement = (
            '        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True\n'
            '        self.forced_bos_token_id = kwargs.get("forced_bos_token_id", None)\n\n'
            '        super().__init__(\n'
        )

        if needle not in text:
            raise RuntimeError(
                f"Could not patch Florence-2 configuration automatically:\n  {config_path}\n"
                "The expected compatibility anchor was not found."
            )

        config_path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
        return True

    @classmethod
    def patch_local_processing(cls, model_dir):
        processing_path = Path(model_dir) / "processing_florence2.py"
        if not processing_path.is_file():
            return False

        legacy = "                    tokenizer.additional_special_tokens + \\\n"
        patched = '                    list(getattr(tokenizer, "additional_special_tokens", [])) + \\\n'

        text = processing_path.read_text(encoding="utf-8")
        if patched in text:
            return False
        if legacy not in text:
            raise RuntimeError(
                f"Could not patch Florence-2 processor automatically:\n  {processing_path}\n"
                "The expected tokenizer compatibility anchor was not found."
            )

        processing_path.write_text(text.replace(legacy, patched, 1), encoding="utf-8")
        return True

    @classmethod
    def patch_local_modeling(cls, model_dir):
        modeling_path = Path(model_dir) / "modeling_florence2.py"
        if not modeling_path.is_file():
            return False

        flash_legacy = "        return self.language_model._supports_flash_attn_2\n"
        flash_patched = '        return getattr(getattr(self, "language_model", None), "_supports_flash_attn_2", False)\n'
        sdpa_legacy = "        return self.language_model._supports_sdpa\n"
        sdpa_patched = '        return getattr(getattr(self, "language_model", None), "_supports_sdpa", False)\n'

        text = modeling_path.read_text(encoding="utf-8")
        changed = False
        if flash_patched not in text:
            if flash_legacy not in text:
                raise RuntimeError(
                    f"Could not patch Florence-2 modeling automatically:\n  {modeling_path}\n"
                    "The expected Flash Attention compatibility anchor was not found."
                )
            text = text.replace(flash_legacy, flash_patched, 1)
            changed = True
        if sdpa_patched not in text:
            if sdpa_legacy not in text:
                raise RuntimeError(
                    f"Could not patch Florence-2 modeling automatically:\n  {modeling_path}\n"
                    "The expected SDPA compatibility anchor was not found."
                )
            text = text.replace(sdpa_legacy, sdpa_patched, 1)
            changed = True

        if changed:
            modeling_path.write_text(text, encoding="utf-8")
        return changed

    def _resolve_model_source(self):
        candidates = []

        env_model_dir = os.environ.get("FLORENCE2_MODEL_DIR", "").strip()
        if env_model_dir:
            candidates.append(os.path.expanduser(env_model_dir))

        try:
            share_dir = get_package_share_directory("object_tracking")
            candidates.append(os.path.join(share_dir, "model_weights", Path(self.model_id).name))
        except PackageNotFoundError:
            pass

        candidates.append(str(Path(__file__).resolve().parent / "model_weights" / Path(self.model_id).name))

        hf_snapshot_dir = self._find_local_hf_snapshot_dir()
        if hf_snapshot_dir:
            candidates.append(hf_snapshot_dir)

        for candidate in candidates:
            if self.is_complete_model_dir(candidate):
                return candidate
            if os.path.isdir(candidate):
                self.invalid_local_candidates.append(candidate)

        return self.model_id

    def _find_local_hf_snapshot_dir(self):
        hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        repo_name = self.model_id.replace("/", "--")
        snapshots_root = os.path.join(
            hf_home,
            "hub",
            f"models--{repo_name}",
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
        return snapshot_dirs[0]

    def _invalid_candidates_suffix(self):
        if not self.invalid_local_candidates:
            return ""
        formatted = "\n".join(f"  - {candidate}" for candidate in self.invalid_local_candidates)
        return f"\nIncomplete local Florence-2 directories were found at:\n{formatted}"

    def _download_hint(self):
        return (
            "Download a complete Florence-2 custom-code snapshot first, for example with:\n"
            "  ros2 run object_tracking download_florence2_model --force\n"
            "or:\n"
            "  ~/.venvs/ros-jazzy-ml/bin/python "
            "/home/user/ros2_ws/src/object_tracking/object_tracking/download_florence2_model.py --force"
        )

    def _load_failure_message(self):
        if os.path.isdir(self.model_source):
            return (
                f"Florence-2 could not be loaded from local directory:\n  {self.model_source}\n"
                "The current setup expects the official Florence custom-code snapshot plus a small compatibility "
                "patch for newer transformers. "
                f"{self._download_hint()}{self._invalid_candidates_suffix()}"
            )
        return self._remote_download_failure_message()

    def _remote_download_failure_message(self):
        return (
            "Florence-2 is not available as a complete local custom-code snapshot, and downloading it from Hugging Face "
            "failed in the current environment. "
            f"{self._download_hint()}{self._invalid_candidates_suffix()}"
        )

    def segment(self, image, prompt, depth_map, min_mask_area=100):
        self.last_detection_score = None
        self.last_mask = None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        composed_prompt = f"{self.task_prompt}{prompt}"

        start_time = time.time()
        inputs = self.processor(
            text=composed_prompt,
            images=image_pil,
            return_tensors="pt",
        )
        prepared_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    prepared_inputs[key] = value.to(self.device, self.torch_dtype)
                else:
                    prepared_inputs[key] = value.to(self.device)
            else:
                prepared_inputs[key] = value

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=prepared_inputs["input_ids"],
                pixel_values=prepared_inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                use_cache=False,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image_pil.width, image_pil.height),
        )
        segmentation_time = time.time() - start_time

        polygons_payload = parsed_answer.get(self.task_prompt)
        if polygons_payload is None and parsed_answer:
            polygons_payload = next(iter(parsed_answer.values()))
        if polygons_payload is None:
            polygons_payload = {}
        polygons = polygons_payload.get("polygons", [])
        mask = self._polygons_to_mask(polygons, image.shape[:2])
        if mask is None:
            return image, None, segmentation_time, depth_map

        mask_area = int(np.sum(mask))
        if mask_area < min_mask_area and mask_area > 0:
            print(f"Object ignored due to small area: {mask_area} pixels")
            return image, None, segmentation_time, depth_map

        center_coords = self.get_center_coordinates(mask)
        self.last_detection_score = float(mask_area)
        self.last_mask = mask.astype(np.uint8)

        image_out = image.copy()
        image_out[mask > 0] = (0, 255, 0)
        return image_out, center_coords, segmentation_time, depth_map

    def _polygons_to_mask(self, polygons, image_shape):
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon_found = False

        for polygon_group in polygons:
            if not polygon_group:
                continue
            for poly in polygon_group:
                coords = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                if coords.shape[0] < 3:
                    continue
                coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
                coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
                pts = np.round(coords).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
                polygon_found = True

        return mask if polygon_found else None

    @staticmethod
    def get_center_coordinates(mask):
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        x_mean = int(np.mean(x_indices))
        y_mean = int(np.mean(y_indices))
        return (x_mean, y_mean)
