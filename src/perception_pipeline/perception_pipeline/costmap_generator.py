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

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.resolution = self.get_parameter('resolution').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.inflation_radius = self.get_parameter('inflation_radius').value

        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)

        self.pub = self.create_publisher(OccupancyGrid, output_topic, 10)
        self.sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.callback,
            10
        )

        self.timer = self.create_timer(0.1, self.publish_costmap)
        self.current_costmap = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.int8
        )

        self.get_logger().info(
            f'Grid: {self.grid_width} x {self.grid_height} '
            f'({self.width}m x {self.height}m)'
        )

    def world_to_grid(self, x, y):
        grid_x = int((x + self.width / 2.0) / self.resolution)
        grid_y = int((y + self.height / 2.0) / self.resolution)

        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        return grid_x, grid_y

    def inflate_point(self, gx, gy, costmap):
        cells = int(math.ceil(self.inflation_radius / self.resolution))
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    dist = math.hypot(dx, dy) * self.resolution
                    if dist <= self.inflation_radius:
                        cost = int(100 * (1.0 - dist / self.inflation_radius))
                        if cost > costmap[ny, nx]:
                            costmap[ny, nx] = cost

    def callback(self, msg):
        costmap = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.int8
        )

        try:
            points = point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )

            for x_cam, y_cam, z_cam in points:
                # -------- CAMERA → MAP FRAME --------
                # Camera:
                #   z forward, x right, y down
                # Map (RViz):
                #   x forward, y left, z up
                x_map = z_cam
                y_map = -x_cam
                # z_map = -y_cam  (ignored for 2D costmap)
                # ------------------------------------

                # Ignore points outside map bounds early
                if not (-self.width / 2 <= x_map <= self.width / 2):
                    continue
                if not (-self.height / 2 <= y_map <= self.height / 2):
                    continue

                gx, gy = self.world_to_grid(x_map, y_map)
                costmap[gy, gx] = 100
                self.inflate_point(gx, gy, costmap)

        except Exception as e:
            self.get_logger().error(f'Point processing failed: {e}')
            return

        self.current_costmap = costmap

    def publish_costmap(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height

        msg.info.origin.position.x = -self.width / 2.0
        msg.info.origin.position.y = -self.height / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = self.current_costmap.flatten().tolist()
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FixedCostmapGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
