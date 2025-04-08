#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import math
import os
import sys

class GoalMonitorNode(Node):
    def __init__(self):
        super().__init__('goal_monitor_node')
        
        # Declare parameters with default values
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 3.0)
        self.declare_parameter('goal_threshold', 0.5)
        self.declare_parameter('metrics_file', 'metrics_dwb.yaml')
        
        # Get parameter values
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value
        self.threshold = self.get_parameter('goal_threshold').value
        self.metrics_file = self.get_parameter('metrics_file').value
        
        # Subscribe to robot odometry
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.get_logger().info(f'Goal monitor initialized. Watching for goal: x={self.goal_x}, y={self.goal_y}')
    
    def odom_callback(self, msg):
        # Get current robot position
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        
        # Calculate distance to goal
        distance = math.sqrt((current_x - self.goal_x)**2 + (current_y - self.goal_y)**2)
        
        # Check if goal is reached
        if distance <= self.threshold:
            self.get_logger().info(f'Goal reached! Current position: x={current_x}, y={current_y}')
            self.get_logger().info(f'Distance to goal: {distance} (threshold: {self.threshold})')
            
            # Print metrics file location
            from ament_index_python.packages import get_package_share_directory
            metrics_dir = os.path.join(get_package_share_directory('hunav_evaluator'), 'results')
            self.get_logger().info(f'Metrics file can be found at: {metrics_dir}')
            
            # Shutdown all nodes
            self.get_logger().info('Shutting down experiment...')
            sys.exit(0)  # Exit with success code

def main(args=None):
    rclpy.init(args=args)
    node = GoalMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()