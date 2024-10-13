import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('hsv_node')
        self.publisher_ = self.create_publisher(Image, 'color_detected_image', 10)
        self.subscription = self.create_subscription(
                Image,
                'image',
                self.listener_callback,
                10)
        self.subscription

        self.bridge = CvBridge()
               
    def listener_callback(self, msg):

        color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        color_detected_image = cv2.bitwise_and(color_image, color_image, mask=mask_blue)

        # overlay_height, overlay_width = self.overlay_image.shape[:2]
        # center_x, center_y = 640 // 2, 480 // 2
        # new_width, new_height = 100, 100
        # overlay_resized = cv2.resize(self.overlay_image, (new_width, new_height))

        # x1, y1 = center_x - new_width // 2, center_y - new_height // 2
        # x2, y2 = x1 + new_width, y1 + new_height

        # if x1 < 0: x1, x2 = 0, new_width
        # if y1 < 0: y1, y2 = 0, new_height
        # if x2 > 640: x1, x2 = 640 - new_width, 640
        # if y2 > 480: y1, y2 = 480 - new_height, 480

        # if overlay_resized.shape[2] == 4:
        #     alpha_overlay = overlay_resized[:, :, 3] / 255.0
        #     alpha_background = 1.0 - alpha_overlay

        #     for c in range(0, 3):
        #         color_detected_image[y1:y2, x1:x2, c] = (alpha_overlay * overlay_resized[:, :, c] +
        #                                                 alpha_background * color_detected_image[y1:y2, x1:x2, c])
        # else:
        #     color_detected_image[y1:y2, x1:x2] = overlay_resized

        msg = self.bridge.cv2_to_imgmsg(color_detected_image, encoding="bgr8")
        self.publisher_.publish(msg)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
