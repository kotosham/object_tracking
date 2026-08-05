# object_tracking - Edge Side of the Robust Architecture (`robust` branch)

On the `robust` branch, this repository implements the **edge/PC GPU side** of
the architecture: perception, SLAM, and the VLM planner. The robot side
(Raspberry Pi 5 and the reactive loop) lives in the
[`ar_project`](https://github.com/dnbabkov/ar_project) repository on the
`robust` branch.

> The single source of truth for architecture lives in
> `ar_project/docs/architecture/`. This file is only a short edge-side summary
> and pointer list. Contracts and modes are not duplicated here.

## Full Documentation in `ar_project/docs/`

- `docs/architecture/README.md`: 3-layer hierarchy, invariants, two modes, FMEA
  must-fix items.
- `docs/architecture/DATA_CONTRACTS.md`: Pi-PC transfer contracts, formats, QoS,
  bandwidth, and latency.
- `docs/architecture/MODES.md`: `flat`/`vlm` modes, replan timing, notes buffer.
- `docs/architecture/REPOS_INTERFACES.md`: packages, interfaces,
  REUSED/NEW/DELETED inventory, estimates.
- `docs/architecture/GAZEBO_WSL_TESTING.md`: Gazebo-on-WSL2 test plan.
- `docs/ROADMAP.md`: step-by-step implementation checklist.

## What Lives on the Edge Side

| Component | Package (`robust`) | Role |
|---|---|---|
| **Planner Orchestrator** | `planner_orchestrator` | Light async HTTP client to an external OpenAI-compatible VLM API. The model is not hosted here; GPU is not required. Implements single-in-flight requests, UUID idempotency, p99 timeout, circuit breaker, structured/enum tool calls, notes buffer, and async replan with commit-point adoption. |
| **Open-vocabulary detector** | `object_tracking/` | GroundingDINO+MobileSAM (`model_mode:=dino`) for target query and fixed context vocabulary. YOLOE remains only for legacy/comparison (`model_mode:=yoloe`). Provides Set-of-Mark candidate rendering and pixel/mask output through `DetectTarget.action`. |
| **SLAM** | RTAB-Map | offline mapping to `.db`; online localization to low-rate `MapOdomCorrection` for Pi `map_odom_relay`. This is not a TF stream. |
| **Interfaces** | `object_tracking_msgs` | `SeekObject.action`, `DetectTarget.action`, `PlanStep.msg`, `Notes.msg`, `Candidate.msg`. |
| **Transport** | bringup | `rmw_zenoh` router on this host, multicast off, QoS deadline/liveliness, chrony. |

## Hard Edge-Side Rules

- VLM and detector never write to the robot reactive path and never output
  navigation coordinates.
- No PointCloud2 or raw depth streams over Wi-Fi. Only compressed event data and
  compact metadata cross the link.
- VLM is a slow, unreliable advisor. If edge/VLM/Wi-Fi is unavailable, the robot
  continues seamlessly in `flat` mode.
- The VLM is always an external OpenAI-compatible API, not self-hosted on edge.
  Planner Orchestrator is only an HTTP client. Edge GPU serves the detector and
  SLAM only. Endpoint hosting is a deployment detail.
