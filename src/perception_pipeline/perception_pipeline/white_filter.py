#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import struct
import numpy as np

class RefinedWhiteFilter(Node):
    def __init__(self):
        super().__init__('refined_white_filter')
        
        self.declare_parameter('input_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('output_topic', '/filtered_points')
        self.declare_parameter('min_brightness', 180)  # Increased - more selective
        self.declare_parameter('max_color_variation', 30)  # Reduced - colors must be more similar
        self.declare_parameter('min_z', -0.1)  # Remove ground plane (negative = below camera)
        self.declare_parameter('max_z', 0.5)   # Max height for lanes/potholes
        self.declare_parameter('max_distance', 3.0)  # Only consider points within 3 meters
        
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.min_brightness = self.get_parameter('min_brightness').value
        self.max_color_variation = self.get_parameter('max_color_variation').value
        self.min_z = self.get_parameter('min_z').value
        self.max_z = self.get_parameter('max_z').value
        self.max_distance = self.get_parameter('max_distance').value
        
        self.pub = self.create_publisher(PointCloud2, output_topic, 10)
        self.sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.callback,
            10
        )
        
        # Statistics
        self.total_points = 0
        self.white_points = 0
        self.callback_count = 0
        
        self.get_logger().info(f'Refined White Filter initialized')
        self.get_logger().info(f'Min brightness: {self.min_brightness}')
        self.get_logger().info(f'Max color variation: {self.max_color_variation}')
        self.get_logger().info(f'Z range: {self.min_z} to {self.max_z}')
        self.get_logger().info(f'Max distance: {self.max_distance}')

    def callback(self, msg):
        self.callback_count += 1
        
        try:
            # Read point cloud
            points_generator = point_cloud2.read_points(
                msg, 
                field_names=("x", "y", "z", "rgb"), 
                skip_nans=True
            )
            
            filtered_points = []
            
            # Process points
            for point in points_generator:
                self.total_points += 1
                
                if len(point) >= 4:
                    x, y, z, rgb_float = point
                    
                    # 1. Distance filter - only consider nearby points
                    distance = np.sqrt(x**2 + y**2 + z**2)
                    if distance > self.max_distance:
                        continue
                    
                    # 2. Height filter - remove ground and high objects
                    if z < self.min_z or z > self.max_z:
                        continue
                    
                    # 3. Extract color
                    rgb_bytes = struct.pack('f', rgb_float)
                    rgb_int = struct.unpack('I', rgb_bytes)[0]
                    
                    r = rgb_int & 0xFF
                    g = (rgb_int >> 8) & 0xFF
                    b = (rgb_int >> 16) & 0xFF
                    
                    # 4. Calculate brightness and color variation
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b
                    max_color = max(r, g, b)
                    min_color = min(r, g, b)
                    color_variation = max_color - min_color
                    
                    # 5. Additional criteria for white:
                    # a) Minimum brightness
                    # b) Low color variation (white = all colors similar)
                    # c) Not too blue-ish (lanes are often warm white)
                    blue_ratio = b / (r + 0.001)  # Avoid division by zero
                    
                    if (brightness >= self.min_brightness and 
                        color_variation <= self.max_color_variation and
                        blue_ratio < 1.2):  # Avoid blue-ish whites
                        
                        # 6. Intensity check - real white lanes are very bright
                        if r > 200 or g > 200 or b > 200:  # At least one channel very bright
                            filtered_points.append([x, y, z, rgb_int])
                            self.white_points += 1
            
            # Log statistics
            if self.callback_count % 5 == 0:
                white_percent = (self.white_points / self.total_points * 100) if self.total_points > 0 else 0
                self.get_logger().info(
                    f'Callback {self.callback_count}: '
                    f'White {len(filtered_points)}/{self.total_points} '
                    f'({white_percent:.1f}%) in this frame'
                )
            
            self.get_logger().info(f'Found {len(filtered_points)} refined white points')
            
            if filtered_points:
                # Create output point cloud
                header = msg.header
                fields = [
                    point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
                    point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
                    point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
                    point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.UINT32, count=1)
                ]
                
                filtered_pc2 = point_cloud2.create_cloud(header, fields, filtered_points)
                self.pub.publish(filtered_pc2)
                
                # Debug info
                if len(filtered_points) > 0:
                    # Sample a few points
                    sample_count = min(3, len(filtered_points))
                    for i in range(sample_count):
                        rgb_int = filtered_points[i][3]
                        r = rgb_int & 0xFF
                        g = (rgb_int >> 8) & 0xFF
                        b = (rgb_int >> 16) & 0xFF
                        brightness = 0.299 * r + 0.587 * g + 0.114 * b
                        x, y, z = filtered_points[i][:3]
                        self.get_logger().info(
                            f'Sample {i}: R={r}, G={g}, B={b}, '
                            f'Brightness={brightness:.1f}, '
                            f'Pos=({x:.2f}, {y:.2f}, {z:.2f})'
                        )
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main():
    rclpy.init()
    node = RefinedWhiteFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()