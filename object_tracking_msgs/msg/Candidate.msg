# A Set-of-Mark detection candidate produced by the open-vocab detector.
uint32 mark_id                  # numbered label on the Set-of-Mark render
string label
float32 confidence
geometry_msgs/Point pixel       # x=u, y=v, z=depth_m (as consumed by target_pixel_to_goal)
string source_frame_id
builtin_interfaces/Time stamp
sensor_msgs/RegionOfInterest bbox
