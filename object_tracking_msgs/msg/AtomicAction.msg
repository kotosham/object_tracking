# One ATOMIC action for the VLM Planner Orchestrator (Phase 4) -- finer-grained
# than a FLAT skill (PlanStep). The VLM (or the mock) replans every N of these and
# selects ONLY from the real options passed to it (enum tool-call); it never emits
# map coordinates. The Pi executive executes each action through Nav2 + the safety
# layer -- the VLM is never on the reactive control path.
uint8 TURN=0              # rotate in place by turn_yaw_rad (rad, + = CCW)
uint8 DRIVE_FORWARD=1     # drive forward forward_dist_m (m), planned + collision-checked
uint8 DRIVE_TO_VISIBLE=2  # approach a visible detection (mark_id from the Candidate list)
uint8 GO_TO_FRONTIER=3    # one step toward frontier_id from the real frontier list
uint8 GET_OBSERVATION=4   # capture an observation (snapshot for the notes buffer)
uint8 DONE=5              # mission satisfied
uint8 STOP=6              # safe stop

uint8 action
float32 turn_yaw_rad
float32 forward_dist_m
uint32 mark_id
int32 frontier_id
string arg_label          # text argument (object / room name)
string step_id            # UUID of this step
string rationale          # short justification (for logs / notes)
