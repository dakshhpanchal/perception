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
        super().__init__('costmap_generator')
        
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
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y

    def inflate_point(self, grid_x, grid_y, costmap):
        """Inflate a single obstacle point"""
        inflation_cells = int(math.ceil(self.inflation_radius / self.resolution))
        
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                new_x, new_y = grid_x + dx, grid_y + dy
                
                # Check bounds
                if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                    dist = math.sqrt(dx**2 + dy**2) * self.resolution
                    if dist <= self.inflation_radius:
                        # Linear decay cost
                        cost = int(100 * (1.0 - dist / self.inflation_radius))
                        # Only update if new cost is higher
                        if cost > costmap[new_y, new_x]:
                            costmap[new_y, new_x] = cost

    def callback(self, msg):
        try:
            self.get_logger().info(f'Received point cloud with {msg.height * msg.width} points')
            
            # Reset costmap
            costmap = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
            
            # Read point cloud data
            points = []
            try:
                # Try reading points with different field configurations
                gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                for p in gen:
                    if len(p) >= 3:
                        x, y, z = float(p[0]), float(p[1]), float(p[2])
                        # Simple filter for realistic points
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            points.append((x, y, z))
            except Exception as e:
                self.get_logger().warn(f'Error reading points: {e}')
                # Alternative method: read raw data
                try:
                    data_array = np.frombuffer(msg.data, dtype=np.float32)
                    # Reshape based on point step
                    point_size = msg.point_step // 4  # Assuming float32 (4 bytes)
                    num_points = len(data_array) // point_size
                    for i in range(min(num_points, 10000)):  # Limit for performance
                        idx = i * point_size
                        if idx + 2 < len(data_array):
                            x = data_array[idx]
                            y = data_array[idx + 1]
                            z = data_array[idx + 2]
                            if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                                points.append((x, y, z))
                except Exception as e2:
                    self.get_logger().error(f'Alternative reading also failed: {e2}')
                    return
            
            if not points:
                self.get_logger().info('No valid points received')
                return
            
            self.get_logger().info(f'Processing {len(points)} valid points')
            
            # Process points
            obstacle_count = 0
            points_in_bounds = 0
            
            for x, y, z in points:
                # Convert to grid coordinates
                grid_x, grid_y = self.world_to_grid(x, y)
                
                # Mark as obstacle (100 = lethal cost)
                costmap[grid_y, grid_x] = 100
                self.inflate_point(grid_x, grid_y, costmap)
                obstacle_count += 1
                points_in_bounds += 1
            
            # Store the current costmap
            self.current_costmap = costmap
            
            # Statistics
            self.get_logger().info(f'Processed {points_in_bounds} points in bounds')
            self.get_logger().info(f'Created {obstacle_count} obstacle cells')
            
            # Debug: show costmap statistics
            if obstacle_count > 0:
                non_zero_cells = np.count_nonzero(costmap)
                max_cost = np.max(costmap)
                avg_cost = np.mean(costmap[costmap > 0]) if non_zero_cells > 0 else 0
                self.get_logger().info(f'Costmap stats: {non_zero_cells} non-zero cells, max cost: {max_cost}, avg cost: {avg_cost:.1f}')
                
                # Show sample obstacle locations
                if non_zero_cells > 0:
                    # Find some obstacle cells
                    obstacle_indices = np.where(costmap >= 100)
                    if len(obstacle_indices[0]) > 0:
                        for i in range(min(3, len(obstacle_indices[0]))):
                            gy, gx = obstacle_indices[0][i], obstacle_indices[1][i]
                            world_x = gx * self.resolution - self.width/2 + self.resolution/2
                            world_y = gy * self.resolution - self.height/2 + self.resolution/2
                            cell_cost = costmap[gy, gx]
                            self.get_logger().info(f'Obstacle at grid ({gx}, {gy}) -> world ({world_x:.2f}, {world_y:.2f}) cost: {cell_cost}')
            else:
                self.get_logger().warn('No obstacles detected in costmap!')
                
                # Debug: show point ranges
                if len(points) > 0:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    zs = [p[2] for p in points]
                    self.get_logger().info(f'Point X range: {min(xs):.2f} to {max(xs):.2f}')
                    self.get_logger().info(f'Point Y range: {min(ys):.2f} to {max(ys):.2f}')
                    self.get_logger().info(f'Point Z range: {min(zs):.2f} to {max(zs):.2f}')
                    self.get_logger().info(f'Map X range: [{-self.width/2:.1f}, {self.width/2:.1f}]')
                    self.get_logger().info(f'Map Y range: [{-self.height/2:.1f}, {self.height/2:.1f}]')
                    
                    # Check if points are within bounds
                    points_in_x = sum(1 for x in xs if -self.width/2 <= x <= self.width/2)
                    points_in_y = sum(1 for y in ys if -self.height/2 <= y <= self.height/2)
                    self.get_logger().info(f'Points within X bounds: {points_in_x}/{len(points)}')
                    self.get_logger().info(f'Points within Y bounds: {points_in_y}/{len(points)}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing points: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def publish_costmap(self):
        """Publish the current costmap"""
        if self.current_costmap is None:
            return
            
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
        costmap_msg.info.origin.orientation.x = 0.0
        costmap_msg.info.origin.orientation.y = 0.0
        costmap_msg.info.origin.orientation.z = 0.0
        costmap_msg.info.origin.orientation.w = 1.0
        
        # Flatten costmap and convert to list
        flat_costmap = self.current_costmap.flatten()
        costmap_msg.data = [int(val) for val in flat_costmap.tolist()]
        
        # Add debug info
        non_zero = np.count_nonzero(flat_costmap)
        if non_zero > 0:
            self.get_logger().info(f'Publishing costmap with {non_zero} obstacle cells')
            # Show cost distribution
            unique_costs = np.unique(flat_costmap)
            self.get_logger().info(f'Cost values present: {unique_costs}')
        else:
            self.get_logger().warn('Publishing empty costmap - no obstacles detected')
        
        self.pub.publish(costmap_msg)

def main(args=None):    
    rclpy.init(args=args)
    node = FixedCostmapGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()