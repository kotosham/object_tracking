import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from image_segmentation import SAMSegmentor
import numpy as np

class SAMNode(Node):
    def __init__(self):
        super().__init__('sam_node')
        self.bridge = CvBridge()
        self.segmentor = SAMSegmentor()

        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/segmented_image', 10)
        self.pose_pub = self.create_publisher(Point, '/object_position', 10)
        self.declare_parameter('target_object', 'a red cup')

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        prompt = self.get_parameter('target_object').get_parameter_value().string_value

        seg_img, (x_norm, y_norm, depth) = self.segmentor.segment(image, prompt)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8'))

        if x_norm is not None and depth is not None:
            # Преобразование нормализованных координат в 3D в камере
            fx = fy = (image.shape[1] / (2 * np.tan(np.deg2rad(self.segmentor.HFOV / 2))))
            cx = image.shape[1] / 2
            cy = image.shape[0] / 2

            x_px = x_norm * cx + cx
            y_px = y_norm * cy + cy

            X = (x_px - cx) * depth / fx
            Y = (y_px - cy) * depth / fy
            Z = depth

            position = Point(x=X, y=Y, z=Z)
            self.pose_pub.publish(position)

            self.get_logger().info(f'Объект в 3D: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}')
        else:
            self.get_logger().warn('Объект не найден.')

def main(args=None):
    rclpy.init(args=args)
    node = SAMNode()
    rclpy.spin(node)
    rclpy.shutdown()