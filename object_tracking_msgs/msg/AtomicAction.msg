# One ATOMIC action for the VLM Planner Orchestrator (Phase 4). The vocabulary is
# deliberately small (raw motion + perception) so the VLM does its OWN navigation
# reasoning -- a fair comparison against the FLAT policy. The VLM (or the mock)
# replans every N of these and selects ONLY from the real options passed to it
# (enum tool-call); it never emits map coordinates. The Pi executive executes each
# action through Nav2 + the safety layer -- the VLM is never on the reactive path.
uint8 TURN=0              # rotate in place by turn_yaw_rad (rad, + = CCW)
uint8 DRIVE_FORWARD=1     # drive forward_dist_m along the heading (negative = backward)
uint8 DRIVE_TO_VISIBLE=2  # approach a visible detection via Nav (mark_id from the Candidate list)
uint8 DETECT_ALL=3        # detect all objects + classes in view (broad-vocab perception)
uint8 DONE=4              # mission satisfied / target reached

uint8 action
float32 turn_yaw_rad
float32 forward_dist_m
uint32 mark_id
string arg_label          # text argument (object / room name)
string step_id            # UUID of this step
string rationale          # short justification (for logs / notes)
