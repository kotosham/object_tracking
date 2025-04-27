import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo 
from geometry_msgs.msg import Point, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from object_tracking.image_segmentation import SAMSegmentor
from object_tracking.clip_image_segmentation import CLIPSegmentor
import numpy as np

import tf2_ros
import tf2_geometry_msgs

class SAMNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.get_logger().info('Node initialized')
        self.camera_info_received = False

        self.bridge = CvBridge()
        self.segmentor = CLIPSegmentor()
        self.current_prompt = "a grey cube"

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 1)
        self.image_sub = self.create_subscription(Image, '/image_in', self.image_callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.prompt_sub = self.create_subscription(String, '/target_prompt', self.prompt_callback, 1)
        self.depth_sub = self.create_subscription(Image, '/depth_camera/depth/image_raw', self.depth_callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.image_pub = self.create_publisher(Image, '/image_out', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/goal_pose', 1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_depth = None

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

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Ошибка конвертации depth изображения: {e}')

    def image_callback(self, msg):
        if self.fx != None and self.fy != None and self.cx != None and self.cy != None:
            try:
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except CvBridgeError as e:
                self.get_logger().error(e)
                return

            seg_img, center_coords = self.segmentor.segment(image, self.current_prompt)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8'))

            if center_coords is None:
                self.get_logger().warn('Объект не найден')
                return
            else:
                self.get_logger().info('Координаты центра: (' + str(center_coords[0]) + ', ' + str(center_coords[1]) + ')')

            if self.latest_depth is None:
                self.get_logger().warn('Нет данных depth')
                return
            
            #img_h, img_w = image.shape[:2]
            x_px = int(center_coords[0])
            y_px = int(center_coords[1])

            depth = self.latest_depth[y_px, x_px]
            if np.isnan(depth) or depth <= 0.0:
                self.get_logger().warn('Некорректная глубина в центре объекта')
                return
            
            X = (x_px - self.cx) * depth / self.fx
            Y = (y_px - self.cy) * depth / self.fy
            Z = depth
    
            self.get_logger().info(f'Объект в системе камеры: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}')

            point_camera = Point()
            point_camera.x = X
            point_camera.y = Y
            point_camera.z = float(Z)
    
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = 'depth_camera_link_optical'
            point_stamped.header.stamp = msg.header.stamp
            point_stamped.point = point_camera

            if point_camera.z >= 0.5:

                try:
                    transform = self.tf_buffer.lookup_transform('map', 'depth_camera_link_optical', msg.header.stamp, timeout=rclpy.duration.Duration(seconds=0.5))
                    point_world = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
    
                    self.get_logger().info(f'Объект в map frame: X={point_world.point.x:.2f}, Y={point_world.point.y:.2f}, Z={point_world.point.z:.2f}')
    
                    # 5. Публикация цели в Nav2
                    goal = PoseStamped()
                    goal.header.frame_id = 'map'
                    goal.header.stamp = msg.header.stamp
                    goal.pose.position = point_world.point
                    goal.pose.orientation.w = 1.0  # Без ориентации (или повернуть к объекту)
    
                    self.pose_pub.publish(goal)
    
                except Exception as e:
                    self.get_logger().error(f'Ошибка трансформации в map: {e}')

            #if x_norm is not None:
            #    # Преобразование нормализованных координат в 3D в камере
            #    #fx = fy = (image.shape[1] / (2 * np.tan(np.deg2rad(self.segmentor.HFOV / 2))))
            #    #cx = image.shape[1] / 2
            #    #cy = image.shape[0] / 2

            #    #x_px = x_norm * cx + cx
            #    #y_px = y_norm * cy + cy

            #    #X = (x_px - cx) * depth / fx
            #    #Y = (y_px - cy) * depth / fy
            #    #Z = depth

            #    position = Point(x=X, y=Y, z=Z)
            #    self.pose_pub.publish(position)

            #    #self.get_logger().info(f'Объект в 3D: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}')
            #else:
            #    self.get_logger().warn('Объект не найден.')

def main(args=None):
    rclpy.init(args=args)
    node = SAMNode()
    rclpy.spin(node)
    rclpy.shutdown()