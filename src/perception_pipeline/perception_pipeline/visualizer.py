#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class PerceptionVisualizer(Node):
    def __init__(self):
        super().__init__('perception_visualizer')
        
        # Parameters
        self.declare_parameter('costmap_topic', '/costmap')
        self.declare_parameter('filtered_points_topic', '/filtered_points')

        costmap_topic = self.get_parameter('costmap_topic').value
        filtered_points_topic = self.get_parameter('filtered_points_topic').value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            costmap_topic,
            self.costmap_callback,
            10
        )
        
        # Image publisher for visualization
        self.visualization_pub = self.create_publisher(Image, '/perception_visualization', 10)
        
        self.get_logger().info('Perception Visualizer initialized')

    def costmap_callback(self, msg):
        # Convert costmap to image
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        
        # Create image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color mapping based on cost
        for i in range(height):
            for j in range(width):
                cost = msg.data[i * width + j]
                
                if cost == 0:  # Free space
                    image[i, j] = [0, 0, 0]  # Black
                elif cost == 100:  # Lethal obstacle
                    image[i, j] = [0, 0, 255]  # Red
                elif cost > 0:  # Inflated zone
                    # Gradient from green to cyan based on cost
                    intensity = int(255 * cost / 100)
                    image[i, j] = [255 - intensity, 255, intensity]  # BGR format
        
        # Resize for better visualization
        scale_factor = 5
        resized = cv2.resize(image, (width * scale_factor, height * scale_factor), 
                           interpolation=cv2.INTER_NEAREST)
        
        # Add grid lines
        for i in range(0, resized.shape[0], scale_factor):
            cv2.line(resized, (0, i), (resized.shape[1], i), (100, 100, 100), 1)
        for j in range(0, resized.shape[1], scale_factor):
            cv2.line(resized, (j, 0), (j, resized.shape[0]), (100, 100, 100), 1)
        
        # Add text
        cv2.putText(resized, f'Resolution: {resolution}m', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(resized, f'Size: {width}x{height}', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
        self.visualization_pub.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()