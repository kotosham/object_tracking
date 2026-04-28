import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseStamped, Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import time

import tf2_ros
import tf2_geometry_msgs

import math
from geometry_msgs.msg import Quaternion

class SAMNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.get_logger().info('Node initialized')
        self.camera_info_received = False
        self.target_reached = False
        self.target_found = False
        
        self.declare_parameter("use_sam", False)
        self.SAM = self.get_parameter("use_sam").get_parameter_value().bool_value

        self.bridge = CvBridge()
        if self.SAM:
            from object_tracking.image_segmentation import SAMSegmentor
            self.segmentor = SAMSegmentor()
            self.goal_position = None
        else:
            from object_tracking.clip_image_segmentation import CLIPSegmentor
            self.segmentor = CLIPSegmentor()

        self.current_prompt = None
        self.current_pose = None
        self.offset = 0.5
        self.offset_delta = 0.35
        self.total_seg_time = 0
        self.segmentations = 0

        self.depth_sub = self.create_subscription(Image, 
                                                  '/depth_camera/depth/image_raw', 
                                                  self.depth_callback, 
                                                  rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_sub = self.create_subscription(Image,
                                                  '/image_in', 
                                                  self.image_callback, 
                                                  rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 1)
        self.prompt_sub = self.create_subscription(String, '/target_prompt', self.prompt_callback, 1)
        
        self.search_cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/image_out', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 1)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter("search_angular_speed", 0.5)
        self.search_angular_speed = self.get_parameter('search_angular_speed').get_parameter_value().double_value

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.latest_depth = None

    def timer_callback(self):
        msg = Twist()
        if not self.target_found and not self.target_reached and not (self.current_prompt is None):
            if self.SAM:
                return
            msg.angular.z = self.search_angular_speed
            self.search_cmd_pub.publish(msg)

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.fx = msg.k[0]  # fx
            self.fy = msg.k[4]  # fy
            self.cx = msg.k[2]  # cx
            self.cy = msg.k[5]  # cy
            self.camera_info_received = True
            self.get_logger().info(f'Camera intrinsics updated: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def prompt_callback(self, msg):
        if self.current_prompt != msg.data:
            self.current_prompt = msg.data
            self.get_logger().info(f'Новый промпт получен: "{self.current_prompt}"')
            self.target_found = False
            self.target_reached = False

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Ошибка конвертации depth изображения: {e}')

    def image_callback(self, msg):
        if self.camera_info_received:
            try:
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError as e:
                self.get_logger().error(e)
                return
            
            if self.current_prompt is None:
                return
            
            if self.SAM:
                if not self.target_found:
                    if self.latest_depth is None:
                        return
                    
                    seg_img, center_coords, image_depth_map, segmentation_time = self.segmentor.segment(image, self.current_prompt, self.latest_depth)
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8'))
                    if center_coords is None:
                        if not self.target_found and not self.target_reached:
                            self.get_logger().warn('Объект не найден')
                            msg = Twist()
                            msg.angular.z = self.search_angular_speed
                            self.search_cmd_pub.publish(msg)
                            time.sleep(4.0)
                            self.latest_depth = None
                            self.get_logger().info("Ожидание после поворота окончено")
                        if self.target_found and not self.target_reached:
                            transform_base = self.tf_buffer.lookup_transform('map', 
                                                                            'base_link', 
                                                                            rclpy.time.Time(), 
                                                                            timeout=rclpy.duration.Duration(seconds=0.5))
                
                            robot_x = transform_base.transform.translation.x
                            robot_y = transform_base.transform.translation.y

                            dx = self.goal_pose.pose.position.x - robot_x
                            dy = self.goal_pose.pose.position.y - robot_y

                            distance = np.hypot(dx, dy)

                            if distance <= self.offset + self.offset_delta:
                                self.target_reached = True
                                self.get_logger().info("Объект достигнут")
                                self.get_logger().info(f'Average segmentation time is {(self.total_seg_time/self.segmentations)}')
                        return
                    else:
                        if image_depth_map is None:
                            return
                        try:
                            self.total_seg_time += segmentation_time
                            self.segmentations += 1
                            if not self.target_found and not self.target_reached:
                                camera_transform = self.tf_buffer.lookup_transform('map', 
                                                                                'depth_camera_link_optical', 
                                                                                rclpy.time.Time(), 
                                                                                timeout=rclpy.duration.Duration(seconds=0.5))
                                transform_base = self.tf_buffer.lookup_transform('map', 
                                                                            'base_link', 
                                                                            rclpy.time.Time(), 
                                                                            timeout=rclpy.duration.Duration(seconds=0.5))
                                goal_point = self.segmentor.get_goal_point(depth_image=image_depth_map,
                                                                        center_coords=center_coords,
                                                                        camera_transform=camera_transform,
                                                                        transform_base=transform_base,
                                                                        camera_intrinsics=[self.fx, self.fy, self.cx, self.cy],
                                                                        offset=self.offset)
                                goal_point.header.stamp = self.get_clock().now().to_msg()
                                self.goal_pose = goal_point
                                self.pose_pub.publish(goal_point)
                                image_depth_map = None
                                self.target_found = True
                                self.get_logger().info(f"Average GroundingDINO + SAM segmentation time is {(self.total_seg_time/self.segmentations)}")
                        except Exception as e:
                            self.get_logger().error(f'Ошибка трансформации в map: {e}')
                return

            seg_img, center_coords, segmentation_time, depth_map_used = self.segmentor.segment(image, self.current_prompt, self.latest_depth)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8'))

            if center_coords is None:
                #if not self.target_found:
                    #self.get_logger().warn('Объект не найден')
                if self.target_found and not self.target_reached:
                    transform_base = self.tf_buffer.lookup_transform('map', 
                                                                    'base_link', 
                                                                    rclpy.time.Time(), 
                                                                    timeout=rclpy.duration.Duration(seconds=0.5))
                
                    robot_x = transform_base.transform.translation.x
                    robot_y = transform_base.transform.translation.y

                    dx = self.goal_pose.pose.position.x - robot_x
                    dy = self.goal_pose.pose.position.y - robot_y

                    distance = np.hypot(dx, dy)

                    if distance <= self.offset + self.offset_delta:
                        self.target_reached = True
                        self.get_logger().info("Объект достигнут")
                        self.get_logger().info(f'Average segmentation time is {(self.total_seg_time/self.segmentations)}')

                return
            else:
                if not self.target_found:
                    self.get_logger().info('Координаты центра: (' + str(center_coords[0]) + ', ' + str(center_coords[1]) + ')')
                self.target_found = True

            self.total_seg_time += segmentation_time
            self.segmentations += 1

            if self.latest_depth is None:
                #if not self.target_found:
                #    self.get_logger().warn('Нет данных depth')
                return

            x_px = int(center_coords[0])
            y_px = int(center_coords[1])

            depth = depth_map_used[y_px, x_px]
            if np.isnan(depth) or depth <= 0.0:
                self.get_logger().warn('Некорректная глубина в центре объекта')
                return
            
            X = (x_px - self.cx) * depth / self.fx
            Y = (y_px - self.cy) * depth / self.fy
            Z = depth
    
            #self.get_logger().info(f'Объект в системе камеры: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}')

            point_camera = Point()
            point_camera.x = X
            point_camera.y = Y
            point_camera.z = float(Z)

            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = 'depth_camera_link_optical'
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point = point_camera

            if not self.target_reached:
                try:
                    transform = self.tf_buffer.lookup_transform('map', 
                                                                'depth_camera_link_optical', 
                                                                rclpy.time.Time(), 
                                                                timeout=rclpy.duration.Duration(seconds=0.5))
                    point_world = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                    transform_base = self.tf_buffer.lookup_transform('map', 
                                                                     'base_link', 
                                                                     rclpy.time.Time(), 
                                                                     timeout=rclpy.duration.Duration(seconds=0.5))
    
                    robot_x = transform_base.transform.translation.x
                    robot_y = transform_base.transform.translation.y

                    dx = point_world.point.x - robot_x
                    dy = point_world.point.y - robot_y

                    distance = np.hypot(dx, dy)
                    if 0.8 * distance > self.offset + self.offset_delta:
                        #scale = (distance - self.offset) / distance
                        scale = 0.8
                        self.get_logger().info(f'Average segmentation time is {(self.total_seg_time/self.segmentations)}')
                    else:
                        scale = 0.8
                        self.target_reached = True
                        self.get_logger().info("Объект достигнут")
                        self.get_logger().info(f'Average segmentation time is {(self.total_seg_time/self.segmentations)}')

                    goal_x = robot_x + dx * scale
                    goal_y = robot_y + dy * scale

                    self.get_logger().info(f'Объект в map frame: X={point_world.point.x:.2f}, Y={point_world.point.y:.2f}, Z={point_world.point.z:.2f}')
                    self.get_logger().info(f'Расстояние до цели distance = {distance:.2f}, offset = {self.offset:.2f} +- {self.offset_delta:.2f}')

                    self.goal_pose = PoseStamped()

                    goal = PoseStamped()
                    
                    goal.header.frame_id = 'map'
                    goal.header.stamp = self.get_clock().now().to_msg()

                    theta = np.arctan2(dy, dx)

                    def yaw_to_quaternion(yaw):
                        q = Quaternion()
                        q.z = math.sin(yaw / 2.0)
                        q.w = math.cos(yaw / 2.0)
                        return q

                    goal.pose.position.x = goal_x
                    goal.pose.position.y = goal_y

                    goal.pose.orientation = yaw_to_quaternion(theta)

                    self.goal_pose = goal
    
                    self.pose_pub.publish(goal)
                except Exception as e:
                    self.get_logger().error(f'Ошибка трансформации в map: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SAMNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
