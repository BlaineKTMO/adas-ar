import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageTransformer(Node):
    def __init__(self):
        super().__init__('image_transformer')
        self.subscription = self.create_subscription(
            Image,
            'input_image_topic',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, 'warped_image_topic', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Define the source points (corners of the image)
        h, w = image.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

        # Define the destination points (where you want to map the source points)
        dst_points = np.float32([[100, 300], [500, 300], [100, 700], [500, 700]])

        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply the perspective transformation to the image
        warped_image = cv2.warpPerspective(image, matrix, (w, h))

        # Convert OpenCV image back to ROS Image message
        warped_image_msg = self.bridge.cv2_to_imgmsg(warped_image, 'bgr8')

        # Publish the warped image
        self.publisher.publish(warped_image_msg)

def main(args=None):
    rclpy.init(args=args)
    image_transformer = ImageTransformer()
    rclpy.spin(image_transformer)
    image_transformer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()