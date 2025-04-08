#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import time

class GoalPublisherNode(Node):
    def __init__(self):
        super().__init__('goal_publisher_node')
        
        # Declare parameters with default values
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 3.0)
        
        # Get parameter values
        goal_x = self.get_parameter('goal_x').value
        goal_y = self.get_parameter('goal_y').value
        
        self.goal_publisher = self.create_publisher(
            PoseStamped, 'goal_pose', 10)
            
        # Define the goal position
        self.final_goal = PoseStamped()
        self.final_goal.header.frame_id = 'map'
        self.final_goal.pose.position.x = goal_x  # Use parameter
        self.final_goal.pose.position.y = goal_y  # Use parameter
        self.final_goal.pose.position.z = 0.0
        self.final_goal.pose.orientation.x = 0.0
        self.final_goal.pose.orientation.y = 0.0
        self.final_goal.pose.orientation.z = 0.0
        self.final_goal.pose.orientation.w = 1.0
        
        # Create a timer to publish the goal after a delay
        self.timer = self.create_timer(2.0, self.publish_goal)
        self.get_logger().info(f'Goal publisher initialized. Will publish goal in 2 seconds: x={goal_x}, y={goal_y}')
    
    def publish_goal(self):
        # Update the timestamp
        self.final_goal.header.stamp = self.get_clock().now().to_msg()
        self.goal_publisher.publish(self.final_goal)
        self.get_logger().info(f'Published goal: x={self.final_goal.pose.position.x}, y={self.final_goal.pose.position.y}')
        # Cancel the timer to publish only once
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()