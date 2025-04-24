import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError
from object_tracking.image_segmentation import SAMSegmentor
from object_tracking.clip_image_segmentation import CLIPSegmentor
import numpy as np

class SAMNode(Node):
    def __init__(self):
        super().__init__('sam_node')
        self.get_logger().info('Looking for an object...')

        self.bridge = CvBridge()
        self.segmentor = CLIPSegmentor()
        self.current_prompt = "a grey box"

        self.image_sub = self.create_subscription(Image, '/image_in', self.image_callback, rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
        self.prompt_sub = self.create_subscription(String, '/target_prompt', self.prompt_callback, 1)
        self.image_pub = self.create_publisher(Image, '/image_out', 1)
        self.pose_pub = self.create_publisher(Point, '/object_position', 1)

    def prompt_callback(self, msg):
        if self.current_prompt != msg.data:
            self.current_prompt = msg.data
            self.get_logger().info(f'Новый промпт получен: "{self.current_prompt}"')

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(e)

        #self.get_logger().info('Image received')

        seg_img, center_coords = self.segmentor.segment(image, self.current_prompt)

        #if center_coords != None:
        #    #self.get_logger().info('Центр отсегментированного объекта на изображении (x, y): (' + str(center_coords[0]) + ', ' + str(center_coords[1]) + ")")
        #else:
        #    #self.get_logger().info('Объект не найден')
#
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(seg_img, encoding='bgr8'))

        #if x_norm is not None:
        #    # Преобразование нормализованных координат в 3D в камере
        #    #fx = fy = (image.shape[1] / (2 * np.tan(np.deg2rad(self.segmentor.HFOV / 2))))
        #    #cx = image.shape[1] / 2
        #    #cy = image.shape[0] / 2
#
        #    #x_px = x_norm * cx + cx
        #    #y_px = y_norm * cy + cy
#
        #    #X = (x_px - cx) * depth / fx
        #    #Y = (y_px - cy) * depth / fy
        #    #Z = depth
#
        #    position = Point(x=X, y=Y, z=Z)
        #    self.pose_pub.publish(position)
#
        #    #self.get_logger().info(f'Объект в 3D: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}')
        #else:
        #    self.get_logger().warn('Объект не найден.')

def main(args=None):
    rclpy.init(args=args)
    node = SAMNode()
    rclpy.spin(node)
    rclpy.shutdown()