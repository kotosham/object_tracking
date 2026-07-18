# Object Tracking

ROS 2 package for laptop-side text-prompted object detection and segmentation in the base mobile-robot pipeline.

The package consumes RGB or RGB-D frames exported by `ar_project`, runs an open-vocabulary CV backend, and publishes a target pixel for the robot-side `target_pixel_to_goal` node:

```text
/target_prompt + RGB-D frame -> rgb_tracker_node -> /target_pixel -> ar_project/target_pixel_to_goal
```

This package does not send Nav2 goals directly in the current base architecture. Nav2 goal generation is handled in `ar_project`.

## Supported Backends

`model_mode` values:

- `dino_mobilesam` - GroundingDINO for box detection plus MobileSAM for segmentation.
- `clip` - CLIPSeg segmentation.
- `florence2` - Florence-2 referring-expression segmentation.
- `yoloe` - YOLOE text-prompted detection/segmentation.
- `auto` - legacy mode; selects DINO + MobileSAM when `use_sam:=true`, otherwise CLIPSeg.

For the diploma experiments, the main baseline model has been `dino_mobilesam`; `clip`, `yoloe`, and `florence2` are useful for comparisons.

## Main Nodes And Launch Files

- `object_tracking/rgb_tracker_node.py` - current tracker used by the base pipeline.
- `launch/sam_node_continuous.launch.py` - continuous laptop-side tracking launch.
- `launch/sam_node.launch.py` - generic launch supporting burst and continuous modes.
- `object_tracking/tracker_node.py` - older legacy node kept for compatibility.
- `object_tracking/capture_target_overlay.py` - helper for saving/debugging tracker overlays.

## Outputs

The tracker publishes:

- `/target_pixel` (`geometry_msgs/PointStamped`) - pixel target; `x/y` are image coordinates, `z` may carry depth in meters.
- `/target_mask` (`sensor_msgs/Image`) - optional binary mask.
- `/image_out` (`sensor_msgs/Image`) - debug overlay.
- `/experiment/cv_runtime` (`std_msgs/Float32`) - running average CV inference time for the current prompt.
- `/experiment/cv_model` (`std_msgs/String`) - active model mode.
- `/cmd_vel_tracker` (`geometry_msgs/Twist`) - optional step-wise search rotation command.

The robot-side `ar_project` package consumes `/target_pixel` and publishes `/goal_pose` for Nav2.

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select object_tracking
source ~/ros2_ws/install/setup.bash
```

The launch files run the tracker through an ML virtual environment by default:

```text
~/.venvs/ros-jazzy-ml/bin/python
```

Override it with `venv_python:=/path/to/python` if needed.

## Continuous Tracking Launch

Typical laptop command:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch object_tracking sam_node_continuous.launch.py \
  input_reliability:=best_effort \
  model_mode:=dino_mobilesam
```

Default continuous topics:

- input RGB: `/tracker/color/image_raw/compressed`;
- input depth: `/tracker/aligned_depth_to_color/image_raw`;
- prompt: `/target_prompt`;
- target pixel: `/target_pixel`;
- search velocity: `/cmd_vel_tracker`.

The continuous tracker subscribes to depth by default and chooses a near valid point on the segmentation mask instead of blindly using the mask center.

## Prompt Example

```bash
ros2 topic pub --once /target_prompt std_msgs/msg/String "{data: 'office chair'}"
```

Use specific prompts when the scene contains multiple similar objects, for example `black office chair` instead of `chair`.

## Step-Wise Search Rotation

Search rotation is disabled by default. When enabled, the tracker no longer spins continuously. Instead, it waits for several missed detections, publishes a short angular-velocity step, waits for the scene to settle, and analyzes again.

Example:

```bash
ros2 launch object_tracking sam_node_continuous.launch.py \
  input_reliability:=best_effort \
  model_mode:=dino_mobilesam \
  enable_search_rotation:=true \
  search_angular_speed:=0.22 \
  search_failures_before_rotation:=3 \
  search_rotation_duration_s:=0.35 \
  search_settle_time_s:=1.2 \
  search_max_rotation_steps:=8
```

Positive `search_angular_speed` rotates left/counter-clockwise in the usual ROS `base_link` convention. Use a negative value to rotate right.

## GroundingDINO Candidate Tuning

The DINO + MobileSAM backend exposes several parameters for difficult demonstrations:

- `dino_box_threshold` - minimum GroundingDINO confidence.
- `dino_mobilesam_min_mask_area` - minimum accepted mask area.
- `dino_selection_policy` - `score`, `center`, or `score_center`.
- `dino_center_weight` - off-center penalty used by `score_center`.
- `dino_max_center_distance_norm` - optional radial center filter.
- `dino_max_center_x_offset_norm` - optional horizontal center filter.
- `dino_max_center_y_offset_norm` - optional vertical center filter.

Default behavior remains `dino_selection_policy:=score`, meaning the tracker selects the highest-confidence DINO candidate.

Example for a demo where the target object is expected near the center of the camera:

```bash
ros2 launch object_tracking sam_node_continuous.launch.py \
  input_reliability:=best_effort \
  model_mode:=dino_mobilesam \
  dino_box_threshold:=0.40 \
  dino_selection_policy:=center \
  dino_max_center_x_offset_norm:=0.35 \
  dino_max_center_y_offset_norm:=0.75 \
  dino_mobilesam_min_mask_area:=700
```

For a less constrained run, omit the center filters and tune only `dino_box_threshold`.

## Timing And Synchronization Parameters

Useful continuous-mode parameters:

- `target_publish_rate` - max `/target_pixel` publication rate, default `3.0 Hz`.
- `continuous_frame_max_age` - max time a cached frame may wait before inference, default `2.0 s`.
- `continuous_rgb_stamp_max_age` - max acceptable RGB header age, default `1.0 s`.
- `depth_match_tolerance` - max RGB/depth stamp mismatch, default `0.2 s`.
- `nearest_depth_percentile` - depth percentile used to select the near point on a mask, default `5.0`.
- `nearest_depth_min_pixels` - minimum pixels in the near-depth band, default `3`.

If the logs show many stale RGB or depth-mismatch warnings over Wi-Fi, increase `continuous_rgb_stamp_max_age` and `depth_match_tolerance` for demonstration runs.

## Burst Mode

`sam_node.launch.py` can also run in burst mode:

```bash
ros2 launch object_tracking sam_node.launch.py \
  tracking_mode:=burst \
  model_mode:=dino_mobilesam
```

Burst mode waits for an explicit `/tracker/burst_complete` signal from the robot-side bridge and then publishes the best candidate from the burst. Continuous mode is the main path for the current base implementation.

## Model Weights

MobileSAM weights are expected in:

```text
object_tracking/model_weights/mobile_sam.pt
```

GroundingDINO is loaded from, in order:

1. `GROUNDING_DINO_MODEL_DIR`, if set;
2. `share/object_tracking/model_weights/grounding-dino-tiny`;
3. the local Hugging Face cache;
4. the Hugging Face model id `IDEA-Research/grounding-dino-tiny`.

Florence-2 can be downloaded with:

```bash
ros2 run object_tracking download_florence2_model
```

## Notes

- CUDA is strongly recommended for DINO + MobileSAM and Florence-2.
- The tracker may choose a semantically wrong instance when the prompt is broad and several similar objects are visible. Use more specific prompts, a higher model threshold, or the center-selection parameters for demonstrations.
- The current base pipeline is split between this package and `ar_project`; `object_tracking` publishes pixels, while `ar_project` handles depth-to-goal conversion, Nav2, SLAM, hardware, and experiment metrics.
