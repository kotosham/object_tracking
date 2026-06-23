# One plan step. Always a FLAT-solvable skill. The VLM only selects from the real
# list (enum tool-call) and never emits navigation coordinates.
uint8 EXPLORE_FRONTIER=0
uint8 GO_TO_POSE=1
uint8 APPROACH_DETECTION=2
uint8 GET_OBSERVATION=3
uint8 STOP=4

uint8 skill
int32 frontier_id               # EXPLORE_FRONTIER: id from the real frontier list
uint32 approach_target_mark     # APPROACH_DETECTION: mark_id from the Candidate list
string arg_label                # text argument (e.g. room/object name)
string step_id                  # UUID of this step
string rationale                # short justification (for logs / notes)
