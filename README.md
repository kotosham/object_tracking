# Robot Object Tracker with Text Prompts and Image Segmentation

This repository contains a ROS 2 node implementation for segmenting camera images from a robot using **GroundingSAM** and **CLIP**, identifying an object based on a text prompt contraining target object description, and navigating toward it using the **Nav2** stack.

---

## Repository Structure

- `tracker_node.py` — the main ROS 2 node that performs segmentation, prompt matching, and navigation.
- `image_segmentation.py` — a module for object segmentation using GroundingSAM.
- `clip_image_segmentation.py` — a module for object segmentation using CLIP.

---

## Features

- Captures RGB frames from the robot's onboard camera.
- Identifies the object given with a text prompt and segments it using GroundingSAM or CLIP, depending on the parameter in `tracker_node.py`.
- Computes the centroid of the matched object.
- Sends navigation goals to Nav2 to approach the object.

---

## Workflow Overview

1. Capture image from robot camera.
2. Receive a text prompt describing the target object.
3. Run segmentation using either GroundingSAM or CLIP based on the text prompt.
4. Look for an object by spinning, if it has not been found.
4. Publish a navigation goal to the Nav2 stack.
5. Robot navigates autonomously to the object.

---

## Dependencies

- ROS 2 (tested on **Humble**)
- Python 3.8+
- `torch`
- `transformers`
- `opencv-python`
- `numpy`
- GroundingSAM
- OpenAI CLIP
- Nav2

> ⚠️ A CUDA-enabled GPU is highly recommended for real-time performance.

---

## Installation

To use this project you need to clone this into the `src` folder in your ROS workspace and build it along with the robot project using

```bash
colcon build --symlink-install
```

---

## Running the Node

```bash
ros2 run object_tracking tracker_node --ros-args -r /image_in:=/camera/image_raw -p use_sam:=<true or false>
```

---

## Examples

### GroundingSAM

![SAM Example](./examples/SAM_example.gif)

### CLIP

![CLIP Example](./examples/Clip_example.gif)

---

## Robot Description and Configuration

Robot model, navigation parameters, and launch files are provided in a separate repository:

[robot_repository](https://github.com/dnbabkov/ar_project)

Repository also contains three different test worlds with corresponging maps.

---

## TODO

- Add voice input for prompts
- Add natural language parsing
- Improve target navigation
- Fine tune the models for better performance