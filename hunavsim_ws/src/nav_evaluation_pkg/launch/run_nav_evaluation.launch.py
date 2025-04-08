#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument, 
                           ExecuteProcess, TimerAction, RegisterEventHandler, LogInfo,
                           OpaqueFunction, GroupAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessStart


def generate_launch_description():
    # Declare launch arguments
    declare_algorithm = DeclareLaunchArgument(
        'algorithm', default_value='dwb',
        description='Navigation algorithm to use: dwb or scl'
    )
    
    declare_metrics_tag = DeclareLaunchArgument(
        'metrics_tag', default_value='',
        description='Tag for metrics files'
    )
    
    # Get algorithm choice
    algorithm = LaunchConfiguration('algorithm')
    metrics_tag = LaunchConfiguration('metrics_tag')
    
    # Create the launch description
    ld = LaunchDescription()
    
    # Add the declared arguments
    ld.add_action(declare_algorithm)
    ld.add_action(declare_metrics_tag)
    
    # Initialize a function to be called with the launch context
    def launch_setup(context):
        # Get algorithm value
        algorithm_value = context.launch_configurations['algorithm']
        metrics_tag_value = context.launch_configurations['metrics_tag']
        
        # Set up parameters for the cafe launch based on algorithm
        launch_args = {
            'configuration_file': 'agents.yaml',
            'metrics_file': PathJoinSubstitution([
                FindPackageShare('nav_evaluation_pkg'),
                'config', 
                'metrics_' + algorithm_value + '.yaml'
            ]),
            'base_world': 'empty_cafe.world',
            'use_gazebo_obs': 'True',
            'update_rate': '100.0',
            'robot_name': 'pmb2',
            'global_frame_to_publish': 'map',
            'use_navgoal_to_start': 'True',
            'navgoal_topic': 'goal_pose',
            'ignore_models': 'ground_plane cafe',
            'gzpose_x': '0.0',
            'gzpose_y': '0.0',
            'gzpose_z': '0.25',
            'gzpose_Y': '0.0',
            'navigation': 'False'  # Set to false to avoid TF error
        }
        
        # Include the PMB2 cafe launch file
        pmb2_cafe_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('hunav_gazebo_wrapper'),
                    'launch',
                    'pmb2_cafe.launch.py'
                ])
            ]),
            launch_arguments=launch_args.items()
        )
        
        # Configure specific navigation algorithm
        nav_config_node = Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': True},
                        {'autostart': True},
                        {'node_names': ['controller_server']}]
        )
        
        # Goal publisher node to set the initial and final poses
        goal_publisher = Node(
            package='nav_evaluation_pkg', 
            executable='goal_publisher_node',
            name='goal_publisher',
            output='screen',
            parameters=[{'use_sim_time': True}]
        )
        
        # Start the goal publisher after a delay to ensure the simulation is ready
        goal_timer = TimerAction(
            period=15.0,
            actions=[
                LogInfo(msg="Starting goal publisher..."),
                goal_publisher
            ]
        )
        
        # Return the actions
        return [
            pmb2_cafe_launch,
            nav_config_node,
            goal_timer
        ]
    
    # Add the launch setup function to the launch description
    ld.add_action(OpaqueFunction(function=launch_setup))
    
    return ld