#!/usr/bin/env python3
"""Planner Orchestrator (Phase 1.6 scaffold).

Builds and runs as an empty node for now. The real content lands in Phase 4: a
lightweight async HTTP client to an external OpenAI-compatible VLM API
(Qwen3-VL; base_url + key, no local model/GPU) with single-in-flight, UUID
idempotency, p99 timeout, circuit-breaker, structured/enum tool-call (the VLM
only picks frontier_id / approach_target from the real list), streaming, a
notes/summary buffer, and anytime/async replan adopted only at a commit point.
The VLM is NEVER on the reactive path. See ar_project/docs/ROADMAP.md Phase 4.
"""
import rclpy
from rclpy.node import Node


class PlannerOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__('planner_orchestrator')
        self.get_logger().info('planner_orchestrator scaffold up (Phase 1.6).')


def main() -> None:
    rclpy.init()
    node = PlannerOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
