# object_tracking — edge-сторона робастной архитектуры (ветка `robust`)

Этот репозиторий на ветке `robust` реализует **edge/ПК-сторону (GPU)** новой архитектуры: восприятие, SLAM и VLM-планировщик. Робот (Raspberry Pi 5) и реактивный контур живут в репозитории [`ar_project`](https://github.com/dnbabkov/ar_project) (ветка `robust`).

> **Единый источник правды по архитектуре — в `ar_project/docs/architecture/`.** Здесь — только краткая выжимка edge-стороны и указатели. Не дублируем контракты/режимы — см. оригиналы.

## Полная документация (в `ar_project/docs/`)

- `docs/architecture/README.md` — обзор 3T-иерархии, инварианты, два режима, FMEA must-fix.
- `docs/architecture/DATA_CONTRACTS.md` — контракты передачи Pi↔ПК (форматы/QoS/полоса/задержки).
- `docs/architecture/MODES.md` — режимы `flat`/`vlm`, тайминг реплана, notes-буфер.
- `docs/architecture/REPOS_INTERFACES.md` — пакеты, интерфейсы, REUSED/NEW/DELETED, оценки.
- `docs/architecture/GAZEBO_WSL_TESTING.md` — тест-план в Gazebo на WSL2.
- `docs/ROADMAP.md` — пошаговый чек-лист реализации.

## Что живёт на этой (edge) стороне

| Компонент | Пакет (robust) | Роль |
|---|---|---|
| **Planner Orchestrator** | `planner_orchestrator` (NEW) | async-клиент к Qwen3-VL-30B-A3B (OpenAI-совместимый vLLM API): single-in-flight, UUID-идемпотентность, timeout по измеренному p99, circuit-breaker, **structured/enum tool-call** (VLM выбирает только `frontier_id`/`approach_target` из реального списка, координат не порождает), streaming; notes/summary-буфер; anytime/async-реплан с adoption в commit-точке |
| **Open-vocab детектор** | `object_tracking/` (REUSED) | YOLOE (default) + GroundingDINO+MobileSAM (fallback), Set-of-Mark рендер кандидатов; CLIPSeg из грудинга исключён. Отдаёт пиксель/маску по запросу (`DetectTarget.action`) |
| **SLAM** | RTAB-Map (REUSED) | offline mapping → `.db`; online localization → **low-rate `MapOdomCorrection`** (НЕ TF-поток) для `map_odom_relay` на Pi |
| **Интерфейсы** | `object_tracking_msgs` (NEW) | `SeekObject.action`, `DetectTarget.action`, `PlanStep.msg`, `Notes.msg`, `Candidate.msg` |
| **Транспорт** | bring-up | `rmw_zenoh` systemd-роутер на этом хосте; multicast off; QoS deadline/liveliness; chrony |

## Жёсткие правила edge-стороны

- **VLM/детектор никогда не пишут в реактивный путь робота** и не выдают навигационных координат.
- **Никаких PointCloud2/сырых depth-потоков по Wi-Fi** — приходит один сжатый keyframe по событию; SLAM получает RGB-D и отдаёт компактную коррекцию.
- **VLM — ненадёжный медленный советник**: при недоступности edge/VLM/Wi-Fi робот бесшовно работает в `flat` (см. деградацию в `MODES.md`).
- **CUDA-on-WSL2 / VRAM:** FP8-сборка Qwen3-VL-30B-A3B не влезает в 24 ГБ — для sim используйте удалённый OpenAI-совместимый endpoint или 4-битную/меньшую модель (см. поправки в `GAZEBO_WSL_TESTING.md`).
