import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import String

from cv_bridge import CvBridge
bridge = CvBridge()

def detect_ground_plane(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        # Filter horizontal lines
        horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 10]
        
        if horizontal_lines:
            # Select the lowest horizontal line as the ground plane
            ground_line = max(horizontal_lines, key=lambda line: line[0][1])
            ground_y = ground_line[0][1]
            return ground_y
    
    # Default ground plane if no lines are detected
    return frame.shape[0] - 50

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Load overlay image
overlay_image = cv2.imread('rightarrow.png', cv2.IMREAD_UNCHANGED)

# Check if the overlay image is loaded successfully
if overlay_image is None:
    raise FileNotFoundError("Overlay image not found. Please check the file path.")

try:
    while True:
        # Wait for a coherent pair of frames: color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert color image to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Define color range for blue detection
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        # Create a mask for the blue color
        mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        # Bitwise-AND mask and original image
        color_detected_image = cv2.bitwise_and(color_image, color_image, mask=mask_blue)

        # Resize overlay image to fit the center of the screen
        overlay_height, overlay_width = overlay_image.shape[:2]
        center_x, center_y = 640 // 2, 480 // 2
        new_width, new_height = 100, 100  # Resize to 100x100 pixels
        overlay_resized = cv2.resize(overlay_image, (new_width, new_height))

        # Calculate the position to place the overlay
        x1, y1 = center_x - new_width // 2, center_y - new_height // 2
        x2, y2 = x1 + new_width, y1 + new_height

        # Ensure overlay fits within the frame
        if x1 < 0: x1, x2 = 0, new_width
        if y1 < 0: y1, y2 = 0, new_height
        if x2 > 640: x1, x2 = 640 - new_width, 640
        if y2 > 480: y1, y2 = 480 - new_height, 480

        # Overlay the resized image in the center with transparency
        if overlay_resized.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_overlay = overlay_resized[:, :, 3] / 255.0
            alpha_background = 1.0 - alpha_overlay

            for c in range(0, 3):
                color_detected_image[y1:y2, x1:x2, c] = (alpha_overlay * overlay_resized[:, :, c] +
                                                        alpha_background * color_detected_image[y1:y2, x1:x2, c])
        else:
            color_detected_image[y1:y2, x1:x2] = overlay_resized

        # Display the color-detected image with overlay
        cv2.imshow('Color Detection with Overlay', color_detected_image)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
