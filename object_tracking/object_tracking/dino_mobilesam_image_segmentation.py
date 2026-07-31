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

DEFAULT_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


class GroundingDINOMobileSAMSegmentor:
    """Открытый словарь по тексту: GroundingDINO даёт рамки, MobileSAM — маски.

    model_id меняется параметром, потому что чекпоинт — это решение, которое
    принимают по замеру, а не по метрике из статьи. Замерено на этом стенде
    (11 кадров мира house, предмет заведомо в кадре, порог 0.30):
    grounding-dino-tiny находит 10 из 11, mm_grounding_dino_tiny — 6 из 11,
    хотя на LVIS второй выигрывает 41.4 против 27.4. LVIS снят на настоящих
    фотографиях, а мебель в house — некрашеные примитивы, и порядок моделей
    туда не переносится. Проверять чекпоинт надо на СВОИХ кадрах, и смена
    должна стоить одну строку в профиле.
    """

    def __init__(self, hfov=70, vfov=40, model_id=None):
        self.HFOV = hfov
        self.VFOV = vfov
        self.dino_model_id = (model_id or "").strip() or DEFAULT_DINO_MODEL_ID
        self.dino_device, self.sam_device = self._select_devices()
        self.last_detection_score = None
        self.last_detection_label = None
        self.last_mask = None

        share_dir = get_package_share_directory('object_tracking')
        checkpoint_path_SAM = self._resolve_mobile_sam_checkpoint(share_dir)

        if checkpoint_path_SAM is None:
            raise FileNotFoundError(
                f"\n[ERROR] MobileSAM checkpoint not found.\n\n"
                f"Please download `mobile_sam.pt` from the official MobileSAM repository\n"
                f"and place it in one of:\n"
                + "\n".join(f"  {d}" for d in self._model_weight_dirs(share_dir))
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
                f"GroundingDINO model={self.dino_model_id} "
                f"device={self.dino_device}, "
                f"MobileSAM device={self.sam_device}, "
                f"CUDA device={gpu_name} ({total_memory_gib:.2f} GiB VRAM)"
            )
        return (
            f"GroundingDINO model={self.dino_model_id} "
            f"device={self.dino_device}, "
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

    def _model_weight_dirs(self, share_dir):
        candidates = [
            os.path.join(share_dir, "model_weights"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "model_weights")),
        ]

        # In symlink/development workspaces the module may execute from
        # build/object_tracking/object_tracking/*.py while large weights stay in
        # src/object_tracking/object_tracking/model_weights.
        for parent in self._parents(os.path.abspath(__file__)):
            candidates.append(
                os.path.join(parent, "src", "object_tracking", "object_tracking", "model_weights")
            )

        seen = set()
        unique = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique.append(candidate)
        return unique

    @staticmethod
    def _parents(path):
        current = os.path.abspath(path)
        if os.path.isfile(current):
            current = os.path.dirname(current)
        while True:
            yield current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    def _resolve_mobile_sam_checkpoint(self, share_dir):
        for weights_dir in self._model_weight_dirs(share_dir):
            candidate = os.path.join(weights_dir, "mobile_sam.pt")
            if os.path.isfile(candidate):
                return candidate
        return None

    def _resolve_dino_model_source(self, share_dir):
        candidates = []

        env_model_dir = os.environ.get("GROUNDING_DINO_MODEL_DIR", "").strip()
        if env_model_dir:
            candidates.append(os.path.expanduser(env_model_dir))

        # Имя каталога — последний сегмент идентификатора модели, чтобы рядом
        # могли лежать несколько чекпоинтов и переключение оставалось параметром.
        local_name = self.dino_model_id.rstrip("/").split("/")[-1]
        for weights_dir in self._model_weight_dirs(share_dir):
            candidates.append(os.path.join(weights_dir, local_name))

        hf_snapshot_dir = self._find_local_hf_snapshot_dir()
        if hf_snapshot_dir:
            candidates.append(hf_snapshot_dir)

        for candidate in candidates:
            if self._is_valid_dino_dir(candidate):
                return candidate

        return self.dino_model_id

    def _find_local_hf_snapshot_dir(self):
        hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        # Кэш HF именует каталог как models--<org>--<name>; выводим его из
        # текущего идентификатора, иначе при смене чекпоинта нашли бы снапшот
        # ПРЕДЫДУЩЕЙ модели и молча запустили не то, что просили.
        cache_name = "models--" + self.dino_model_id.strip("/").replace("/", "--")
        snapshots_root = os.path.join(hf_home, "hub", cache_name, "snapshots")
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

    def segment(self, image_bgr, prompt, depth_map):
        self.last_detection_score = None
        self.last_detection_label = None
        self.last_mask = None
        self.last_bbox = None
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

        # Фильтрация по порогу
        box_threshold = 0.55
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
        self.last_bbox = tuple(int(v) for v in box.tolist())
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

    # --- DETECT_ALL по словарю -------------------------------------------
    # Домашний словарь. Короткий намеренно: один класс = один запрос к модели
    # (~0.2 с), и COCO-80 стоил бы 16 секунд на вызов. Составной запрос
    # «toilet. sofa. bed. ...» одной фразой пробовали — он разваливается:
    # при 12 классах унитаз становится «refrigerator 0.49», стол «wardrobe»,
    # телевизор «television wardrobe». Фразовая привязка размазывается по
    # токенам, и метки перестают значить что-либо.
    DEFAULT_VOCAB = (
        "toilet", "sofa", "bed", "television", "table", "chair", "sink",
        "refrigerator", "wardrobe", "bathtub", "shelf", "stove", "door",
        "person",
    )

    # На сколько победивший класс должен обойти следующий В ТОЙ ЖЕ рамке, чтобы
    # метке можно было верить. Без этого правила DETECT_ALL врал бы уверенно, и
    # это ИЗМЕРЕНО: на 11 кадрах мира house каждый из 12 классов срабатывает на
    # каждом кадре с оценкой 0.2..0.8, а argmax совпадает с истиной лишь 2 раза
    # из 11 (кадр ванны: стол 0.57, шкаф 0.56, сама ванна 0.51). Модель отвечает
    # на вопрос «здесь вообще что-то есть», а не «что именно», когда предмет —
    # некрашеный примитив без текстуры.
    #
    # Отрыв — честный признак: когда модель говорит «да» всему подряд, никто не
    # отрывается, и словарный путь возвращает пусто. Пустой ответ хуже верного,
    # но НЕСРАВНИМО лучше уверенно неверной метки: та попадает в память робота
    # на карту и переживает всю миссию.
    VOCAB_MARGIN = 0.10

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    def _best_box_per_class(self, image_pil, classes, box_threshold,
                            text_threshold):
        """{класс: (рамка, оценка)} — по одному запросу на класс.

        Батчить бессмысленно: замерено 2.2 с на 12 классов и при размере пачки
        1, и при 4, и при 12 — упирается в вычисления, а не в накладные расходы
        на вызов.
        """
        out = {}
        for cls in classes:
            inputs = self.dino_processor(
                images=image_pil, text=[[cls]], return_tensors="pt",
            ).to(self.dino_device)
            with torch.inference_mode():
                outputs = self.dino_model(**inputs)
            res = self.dino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image_pil.size[::-1]],
            )[0]
            if not len(res["scores"]):
                continue
            i = int(torch.argmax(res["scores"]))
            out[cls] = (res["boxes"][i].detach().cpu().numpy(),
                        float(res["scores"][i].item()))
        return out

    def segment_vocab(self, image_bgr, classes=None, conf=0.20,
                      min_mask_area=200):
        """DETECT_ALL: что вокруг, по словарю, с проверкой на различимость.

        Каждый класс спрашивается ОТДЕЛЬНЫМ запросом (составная фраза не
        работает, см. DEFAULT_VOCAB). Рамки всех классов группируются по
        пересечению, и внутри группы метку получает победитель — но только если
        он обошёл следующий класс на VOCAB_MARGIN. Иначе группа отбрасывается:
        значит, модель не различает, а угадывает.
        """
        from object_tracking.setofmark import Detection

        vocab = [str(c).strip() for c in (classes or self.DEFAULT_VOCAB)
                 if str(c).strip()]
        if not vocab:
            return []
        conf = float(conf)
        box_threshold = max(0.01, min(conf, 1.0))
        text_threshold = min(0.25, max(0.01, box_threshold))

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        start = time.time()
        per_class = self._best_box_per_class(image_pil, vocab, box_threshold,
                                             text_threshold)
        if not per_class:
            print(f"DETECT_ALL: ни один из {len(vocab)} классов не сработал")
            return []

        # Группируем рамки разных классов, описывающие ОДНО место. Порог
        # пересечения низкий (0.3), и это не придирка к числу: конкурирующие
        # метки на одном предмете дают заметно разные рамки — «стол» обводит
        # столешницу, «шкаф» весь силуэт. При 0.5 они попадали в РАЗНЫЕ группы,
        # у победителя не оказывалось соперника, отрыв считался от нуля, и
        # заведомо неверная метка проходила проверку. Наблюдалось ровно так:
        # кадр кровати -> «table 0.54», хотя «wardrobe 0.53» стоял рядом.
        ranked = sorted(per_class.items(), key=lambda kv: kv[1][1], reverse=True)
        groups = []                      # [[(класс, рамка, оценка), ...], ...]
        for cls, (box, score) in ranked:
            for group in groups:
                if any(self._iou(member[1], box) >= 0.3 for member in group):
                    group.append((cls, box, score))
                    break
            else:
                groups.append([(cls, box, score)])

        winners, rejected = [], []
        for group in groups:
            best_cls, best_box, best_score = group[0]
            if len(group) > 1:
                runner_up, rival = group[1][2], group[1][0]
            else:
                # Одинокая группа. Соперника по месту нет, но «уверен» и
                # «единственный, кто вообще откликнулся» — разные вещи: сравним
                # с лучшим классом ВНЕ группы. Если по кадру откликнулся только
                # один класс, сравнивать не с чем и метка принимается.
                others = [s for c, (_b, s) in per_class.items() if c != best_cls]
                runner_up = max(others) if others else 0.0
                rival = 'лучший вне группы'
            if best_score - runner_up < self.VOCAB_MARGIN:
                rejected.append('%s~%s' % (best_cls, rival))
                continue
            winners.append((best_cls, best_box, best_score))
        if rejected:
            print('DETECT_ALL: отброшено как неразличимое (отрыв < %.2f): %s'
                  % (self.VOCAB_MARGIN, ', '.join(rejected)))
        if not winners:
            print('DETECT_ALL: %d групп, ни одной различимой за %.1f с'
                  % (len(groups), time.time() - start))
            return []

        boxes_np = np.asarray([w[1] for w in winners], dtype=np.float32)
        try:
            masks = self._predict_sam_masks(image_rgb, image_bgr.shape[:2],
                                            boxes_np)
        except torch.OutOfMemoryError:
            if self.sam_device != "cuda":
                raise
            self.sam_device = "cpu"
            self.sam = self.sam.to(self.sam_device)
            self.predictor = SamPredictor(self.sam)
            torch.cuda.empty_cache()
            masks = self._predict_sam_masks(image_rgb, image_bgr.shape[:2],
                                            boxes_np)

        dets = []
        for idx, (cls, box, score) in enumerate(winners):
            x1, y1, x2, y2 = self._clamp_box(box, image_bgr.shape[:2])
            if x2 <= x1 or y2 <= y1:
                continue
            mask = masks[idx] if idx < len(masks) else None
            mask_for_det = None
            if mask is not None:
                if int(np.sum(mask)) < min_mask_area:
                    continue
                center = self.get_center_coordinates(mask)
                cx, cy = center if center else ((x1 + x2) // 2, (y1 + y2) // 2)
                mask_for_det = mask.astype(bool)
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dets.append(Detection(label=cls, confidence=score, cx=int(cx),
                                  cy=int(cy), bbox=(x1, y1, x2, y2),
                                  mask=mask_for_det))
        dets.sort(key=lambda d: d.confidence, reverse=True)
        print('DETECT_ALL: %d класс(ов) из %d за %.1f с'
              % (len(dets), len(vocab), time.time() - start))
        return dets

    def segment_all(self, image_bgr, prompt, conf=0.20, min_mask_area=200):
        """Return all GroundingDINO+MobileSAM matches as Set-of-Mark detections.

        Путь КОНКРЕТНОЙ цели: одна фраза, все её вхождения в кадре. Словарный
        путь DETECT_ALL — в segment_vocab выше; там же объяснено, почему он
        устроен иначе и почему составная фраза не годится.
        """
        from object_tracking.setofmark import Detection

        self.last_detection_score = None
        self.last_detection_label = None
        self.last_mask = None
        self.last_bbox = None

        conf = float(conf)
        box_threshold = max(0.01, min(conf, 1.0))
        text_threshold = min(0.25, max(0.01, box_threshold))

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        text_labels = [[prompt]]

        start_time_dino = time.time()
        inputs = self.dino_processor(
            images=image_pil,
            text=text_labels,
            return_tensors="pt",
        ).to(self.dino_device)
        with torch.inference_mode():
            outputs = self.dino_model(**inputs)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )
        dino_time = time.time() - start_time_dino

        result = results[0]
        filtered = [
            (box.detach().cpu().numpy(), float(score.item()), str(label))
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"])
            if float(score.item()) >= box_threshold
        ]
        if not filtered:
            print(f"GroundingDINO found no '{prompt}' candidates at conf>={box_threshold:.2f}")
            return []

        filtered.sort(key=lambda item: item[1], reverse=True)
        boxes_np = np.asarray([item[0] for item in filtered], dtype=np.float32)

        start_time_sam = time.time()
        try:
            masks = self._predict_sam_masks(image_rgb, image_bgr.shape[:2], boxes_np)
        except torch.OutOfMemoryError:
            if self.sam_device != "cuda":
                raise
            print("CUDA OOM during SAM inference, switching SAM to CPU and retrying.")
            self.sam_device = "cpu"
            self.sam = self.sam.to(self.sam_device)
            self.predictor = SamPredictor(self.sam)
            torch.cuda.empty_cache()
            masks = self._predict_sam_masks(image_rgb, image_bgr.shape[:2], boxes_np)
        sam_time = time.time() - start_time_sam
        print(f"GroundingDINO time={dino_time:.3f}s, MobileSAM time={sam_time:.3f}s")

        dets = []
        for idx, (box, score, label) in enumerate(filtered):
            x1, y1, x2, y2 = self._clamp_box(box, image_bgr.shape[:2])
            if x2 <= x1 or y2 <= y1:
                continue

            mask = masks[idx] if idx < len(masks) else None
            mask_for_det = None
            if mask is not None:
                area = int(np.sum(mask))
                if area < min_mask_area:
                    continue
                center = self.get_center_coordinates(mask)
                cx, cy = center if center else ((x1 + x2) // 2, (y1 + y2) // 2)
                mask_for_det = mask.astype(bool)
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            dets.append(
                Detection(
                    label=prompt,
                    confidence=float(score),
                    cx=int(cx),
                    cy=int(cy),
                    bbox=(x1, y1, x2, y2),
                    mask=mask_for_det,
                )
            )

        dets.sort(key=lambda d: d.confidence, reverse=True)
        if dets:
            self.last_detection_score = dets[0].confidence
            self.last_detection_label = dets[0].label
            self.last_bbox = dets[0].bbox
            self.last_mask = dets[0].mask.astype(np.uint8) if dets[0].mask is not None else None
        return dets

    def _predict_sam_masks(self, image_rgb, image_shape_hw, boxes_np):
        if self.sam_device == "cuda":
            torch.cuda.empty_cache()
        boxes_t = torch.as_tensor(boxes_np, dtype=torch.float32)
        transformed = self.predictor.transform.apply_boxes_torch(boxes_t, image_shape_hw)
        transformed = transformed.to(self.sam_device)
        self.predictor.set_image(image_rgb)
        with torch.inference_mode():
            if self.sam_device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    masks, _, _ = self.predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed,
                        multimask_output=False,
                    )
            else:
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed,
                    multimask_output=False,
                )
        return (masks[:, 0].detach().cpu().numpy() > 0.5)

    @staticmethod
    def _clamp_box(box, image_shape_hw):
        h, w = image_shape_hw
        x1, y1, x2, y2 = (int(round(float(v))) for v in box.tolist())
        return (
            max(0, min(w, x1)),
            max(0, min(h, y1)),
            max(0, min(w, x2)),
            max(0, min(h, y2)),
        )
    
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
