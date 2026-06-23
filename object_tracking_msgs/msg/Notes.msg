# Compact notes / summarization buffer (context kept instead of storing frames).
std_msgs/Header header
uint32 mission_epoch
string summary                  # current compact summary from the model
string[] facts                  # discrete facts (visited rooms, found/excluded objects)
uint32 token_estimate           # token budget estimate for context control
