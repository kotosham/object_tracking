# Model Weights

Large model weights are intentionally not stored in git. Put local weights in:

```text
src/object_tracking/object_tracking/model_weights/
```

After `colcon build --symlink-install`, launch files and Python modules also see this directory through the installed package share path.

## Expected Files

| Backend | Required local files | Notes |
| --- | --- | --- |
| `dino_mobilesam` | `mobile_sam.pt` | MobileSAM checkpoint used after GroundingDINO selects a box. |
| `dino_mobilesam` | optional `grounding-dino-tiny/` directory | If absent, the code searches the Hugging Face cache and then falls back to model id `IDEA-Research/grounding-dino-tiny`. |
| `clip` | optional `clipseg-rd64-refined/` directory | If absent, the code searches the Hugging Face cache and then falls back to model id `CIDAS/clipseg-rd64-refined`. |
| `florence2` | `Florence-2-base-ft/` directory | Can be downloaded with the provided `download_florence2_model` entry point. |
| `yoloe` | `yoloe-11s-seg.pt` and `mobileclip_blt.ts` | Required by the local YOLOE wrapper before inference starts. |

The development laptop used these local files:

```text
Florence-2-base-ft/
mobile_sam.pt
mobileclip_blt.ts
yoloe-11s-seg.pt
```

## GroundingDINO

The `dino_mobilesam` backend loads GroundingDINO from the first complete source it finds:

1. `GROUNDING_DINO_MODEL_DIR`, if set;
2. `share/object_tracking/model_weights/grounding-dino-tiny`;
3. local Hugging Face cache under `~/.cache/huggingface`;
4. Hugging Face model id `IDEA-Research/grounding-dino-tiny`.

For offline runs, pre-download the Hugging Face snapshot and point `GROUNDING_DINO_MODEL_DIR` at it, or place a complete snapshot in `model_weights/grounding-dino-tiny`.

## MobileSAM

Install the Python package in the ML virtual environment:

```bash
python -m pip install \
  "git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2"
```

Then place the checkpoint here:

```text
src/object_tracking/object_tracking/model_weights/mobile_sam.pt
```

The official MobileSAM README documents loading `./weights/mobile_sam.pt` with `sam_model_registry["vit_t"]`. This project uses the same checkpoint format.

## CLIPSeg

The `clip` backend loads CLIPSeg from the first complete source it finds:

1. `CLIPSEG_MODEL_DIR`, if set;
2. `share/object_tracking/model_weights/clipseg-rd64-refined`;
3. local source checkout under `object_tracking/model_weights/clipseg-rd64-refined`;
4. local Hugging Face cache under `~/.cache/huggingface`;
5. Hugging Face model id `CIDAS/clipseg-rd64-refined`.

For offline runs, pre-download the Hugging Face snapshot and set `CLIPSEG_MODEL_DIR`.

## Florence-2

Download Florence-2 with:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run object_tracking download_florence2_model
```

Expected local directory:

```text
src/object_tracking/object_tracking/model_weights/Florence-2-base-ft/
```

The loader also accepts `FLORENCE2_MODEL_DIR` if the snapshot is stored somewhere else.

## YOLOE

Install the Python package in the ML virtual environment:

```bash
python -m pip install ultralytics
```

Then place the expected files here:

```text
src/object_tracking/object_tracking/model_weights/yoloe-11s-seg.pt
src/object_tracking/object_tracking/model_weights/mobileclip_blt.ts
```

Ultralytics documents YOLOE pretrained `.pt` segmentation weights such as `yoloe-11s-seg.pt`. The local wrapper also requires the MobileCLIP TorchScript text encoder file `mobileclip_blt.ts`, matching the default expected by Ultralytics' MobileCLIP text model.

## Verification

Check file presence:

```bash
find ~/ros2_ws/src/object_tracking/object_tracking/model_weights -maxdepth 2 -type f -printf '%P\n' | sort
```

Check core imports:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

~/.venvs/ros-jazzy-ml/bin/python -c \
  "import torch, transformers, cv2, numpy, PIL, mobile_sam; print('core CV imports OK')"
```

Check YOLOE only if needed:

```bash
~/.venvs/ros-jazzy-ml/bin/python -c \
  "from ultralytics import YOLOE; print('YOLOE import OK')"
```

## Source References

- MobileSAM official repository: https://github.com/ChaoningZhang/MobileSAM
- Ultralytics YOLOE documentation: https://docs.ultralytics.com/models/yoloe/
- Hugging Face model id for GroundingDINO: `IDEA-Research/grounding-dino-tiny`
- Hugging Face model id for CLIPSeg: `CIDAS/clipseg-rd64-refined`
- Hugging Face model id for Florence-2: `microsoft/Florence-2-base-ft`
