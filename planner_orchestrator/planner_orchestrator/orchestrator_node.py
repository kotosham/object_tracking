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

from fleet_comms.heartbeat import HeartbeatPublisher


class PlannerOrchestrator(Node):
    # Must match the fleet-wide heartbeat period (search_coordinator monitor).
    HEARTBEAT_PERIOD_S = 0.5

    def __init__(self) -> None:
        super().__init__('planner_orchestrator')

        # Phase 1.3: edge producer heartbeat. Loss of this beat (or a high
        # last_latency_ms feeding the p99 circuit-breaker, Phase 4.4) is what
        # triggers VLM->FLAT degradation (Phase 5.1). set_latency_ms()/
        # set_status() get called from the VLM client loop in Phase 4.
        self.heartbeat = HeartbeatPublisher(self, 'planner_orchestrator',
                                            period_s=self.HEARTBEAT_PERIOD_S)

        self.get_logger().info('planner_orchestrator scaffold up (Phase 1.6) with '
                               'Phase 1.3 heartbeat publisher.')


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
