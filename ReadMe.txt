This code detects an object on an image from the robot's camera, segments it and estimates the position of the object in the 3D space, after which commands the robot to move to the found object.

To launch run this command:

```
ros2 run object_tracking tracker_node --ros-args -r /image_in:=/camera/image_raw
```