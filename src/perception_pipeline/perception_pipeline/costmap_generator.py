#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import numpy as np
from sensor_msgs_py import point_cloud2
import math

class FixedCostmapGenerator(Node):
    def __init__(self):
        super().__init__('fixed_costmap_generator')
        
        # Parameters
        self.declare_parameter('input_topic', '/filtered_points')
        self.declare_parameter('output_topic', '/costmap')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('width', 10.0)
        self.declare_parameter('height', 10.0)
        self.declare_parameter('inflation_radius', 0.3)
        
        # Get parameters
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.resolution = self.get_parameter('resolution').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        
        # Calculate grid dimensions
        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)
        
        # Publisher
        self.pub = self.create_publisher(OccupancyGrid, output_topic, 10)
        self.sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.callback,
            10
        )
        
        # Timer for periodic publishing
        self.timer = self.create_timer(0.1, self.publish_costmap)
        self.current_costmap = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        
        self.get_logger().info(f'Fixed Costmap Generator initialized')
        self.get_logger().info(f'Grid: {self.grid_width}x{self.grid_height} ({self.width}m x {self.height}m)')

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        # Map origin is at (-width/2, -height/2)
        grid_x = int((x + self.width/2) / self.resolution)
        grid_y = int((y + self.height/2) / self.resolution)
        
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return grid_x, grid_y
        return None, None

    def inflate_point(self, grid_x, grid_y, costmap):
        """Inflate a single obstacle point"""
        inflation_cells = int(self.inflation_radius / self.resolution)
        
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                dist = math.sqrt(dx**2 + dy**2) * self.resolution
                if dist <= self.inflation_radius:
                    new_x, new_y = grid_x + dx, grid_y + dy
                    if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                        cost = int(100 * (1.0 - dist / self.inflation_radius))
                        if cost > costmap[new_y, new_x]:
                            costmap[new_y, new_x] = cost

    def callback(self, msg):
        try:
            # Read point cloud data - use uvs option to get structured data
            gen = point_cloud2.read_points(
                msg, 
                field_names=("x", "y", "z", "rgb"), 
                skip_nans=True,
                uvs=[(0, 0)]  # This ensures we get proper tuples
            )
            
            # Convert generator to list and filter
            points = []
            for p in gen:
                # Check if p is iterable and has at least 3 elements
                try:
                    if hasattr(p, '__len__') and len(p) >= 3:
                        x, y, z = float(p[0]), float(p[1]), float(p[2])
                        points.append((x, y, z))
                except (TypeError, ValueError) as e:
                    self.get_logger().warn(f'Skipping point: {p}, error: {e}')
                    continue
            
            if not points:
                self.get_logger().info('No valid points received')
                return
            
            self.get_logger().info(f'Received {len(points)} valid points from point cloud')
            
            # Reset costmap
            costmap = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
            
            # Debug: Print first point
            if points:
                x, y, z = points[0]
                self.get_logger().info(f'First point: x={x:.2f}, y={y:.2f}, z={z:.2f}')
                
                # Convert to grid to debug
                grid_x, grid_y = self.world_to_grid(x, y)
                self.get_logger().info(f'First point grid: ({grid_x}, {grid_y})')
            
            # Process points
            obstacle_count = 0
            points_in_bounds = 0
            points_out_of_bounds = 0
            
            for x, y, z in points:
                # Convert to grid coordinates
                grid_x, grid_y = self.world_to_grid(x, y)
                
                if grid_x is not None and grid_y is not None:
                    points_in_bounds += 1
                    # Mark as obstacle
                    costmap[grid_y, grid_x] = 100
                    self.inflate_point(grid_x, grid_y, costmap)
                    obstacle_count += 1
                else:
                    points_out_of_bounds += 1
            
            self.current_costmap = costmap
            
            # Log statistics
            self.get_logger().info(
                f'Points in bounds: {points_in_bounds}, '
                f'out of bounds: {points_out_of_bounds}'
            )
            self.get_logger().info(f'Marked {obstacle_count} obstacle cells')
            
            # Check if costmap has any obstacles
            if np.any(costmap > 0):
                obstacle_cells = np.sum(costmap > 0)
                self.get_logger().info(f'Costmap has {obstacle_cells} obstacle cells')
                
                # Find non-zero positions for debugging
                non_zero_indices = np.where(costmap > 0)
                if len(non_zero_indices[0]) > 0:
                    for i in range(min(3, len(non_zero_indices[0]))):
                        gy, gx = non_zero_indices[0][i], non_zero_indices[1][i]
                        # Convert back to world coordinates
                        world_x = gx * self.resolution - self.width/2
                        world_y = gy * self.resolution - self.height/2
                        self.get_logger().info(f'Obstacle at grid ({gx}, {gy}) -> world ({world_x:.2f}, {world_y:.2f})')
            else:
                self.get_logger().warn('Costmap has NO obstacles!')
                
                # Debug point ranges
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    self.get_logger().info(f'X range: {min(xs):.2f} to {max(xs):.2f}')
                    self.get_logger().info(f'Y range: {min(ys):.2f} to {max(ys):.2f}')
                    self.get_logger().info(f'Map X range: [{-self.width/2:.1f}, {self.width/2:.1f}]')
                    self.get_logger().info(f'Map Y range: [{-self.height/2:.1f}, {self.height/2:.1f}]')
            
        except Exception as e:
            self.get_logger().error(f'Error processing points: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def publish_costmap(self):
        """Publish the current costmap"""
        costmap_msg = OccupancyGrid()
        costmap_msg.header.stamp = self.get_clock().now().to_msg()
        costmap_msg.header.frame_id = 'map'
        
        # Set map metadata
        costmap_msg.info.resolution = self.resolution
        costmap_msg.info.width = self.grid_width
        costmap_msg.info.height = self.grid_height
        
        # Set origin (center of map)
        costmap_msg.info.origin.position.x = -self.width/2
        costmap_msg.info.origin.position.y = -self.height/2
        costmap_msg.info.origin.position.z = 0.0
        costmap_msg.info.origin.orientation.w = 1.0
        
        # Flatten costmap and convert to list
        flat_costmap = self.current_costmap.flatten()
        costmap_msg.data = flat_costmap.tolist()
        
        # Add debug: count non-zero cells
        non_zero = np.count_nonzero(flat_costmap)
        if non_zero > 0:
            self.get_logger().info(f'Publishing costmap with {non_zero} non-zero cells')
        else:
            self.get_logger().warn('Publishing EMPTY costmap!')
        
        self.pub.publish(costmap_msg)

def main(args=None):    
    rclpy.init(args=args)
    node = FixedCostmapGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()