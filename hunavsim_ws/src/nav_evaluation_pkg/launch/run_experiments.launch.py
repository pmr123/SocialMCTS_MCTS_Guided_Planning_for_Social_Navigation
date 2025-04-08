#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument, 
                          TimerAction, LogInfo, ExecuteProcess, RegisterEventHandler)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessIO, OnShutdown


def generate_launch_description():
    # Declare launch arguments
    declare_algorithm = DeclareLaunchArgument(
        'algorithm', default_value='dwb',
        description='Navigation algorithm to use: dwb, scl, or sfm'
    )
    
    declare_metrics_tag = DeclareLaunchArgument(
        'metrics_tag', default_value='',
        description='Tag for metrics files'
    )
    
    # Get algorithm choice
    algorithm = LaunchConfiguration('algorithm')
    algorithm_value = TextSubstitution(text='dwb')  # Default value for PathJoinSubstitution
    metrics_tag = LaunchConfiguration('metrics_tag')
    
    # Set up goal publisher node with the requested end position (x=2, y=3)
    goal_publisher = Node(
        package='nav_evaluation_pkg', 
        executable='goal_publisher_node',
        name='goal_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'goal_x': 2.0},  # Set the goal x position
            {'goal_y': 3.0}   # Set the goal y position
        ]
    )
    
    # Include the PMB2 cafe launch file with custom robot starting position
    pmb2_cafe_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('hunav_gazebo_wrapper'),
                'launch',
                'pmb2_cafe.launch.py'
            ])
        ]),
        launch_arguments={
            'configuration_file': 'agents.yaml',
            'metrics_file': PathJoinSubstitution([
                FindPackageShare('nav_evaluation_pkg'),
                'config', 
                'metrics_' + algorithm + '.yaml'
            ]),
            'base_world': 'empty_cafe.world',
            'use_gazebo_obs': 'True',
            'update_rate': '100.0',
            'robot_name': 'pmb2',
            'global_frame_to_publish': 'map',
            'use_navgoal_to_start': 'True',
            'navgoal_topic': 'goal_pose',
            'ignore_models': 'ground_plane cafe',
            'gzpose_x': '-3.0',  # Start position x
            'gzpose_y': '-6.0',  # Start position y
            'gzpose_z': '0.25',
            'gzpose_Y': '0.0',
            'navigation': 'False'  # Set to false to avoid TF error
        }.items()
    )
    
    # Goal monitor to check when the goal is reached and stop the process
    goal_monitor = Node(
        package='nav_evaluation_pkg',
        executable='goal_monitor_node',  # Ensure this node exists in your package
        name='goal_monitor',
        output='screen',
        parameters=[
            {'goal_x': 2.0},
            {'goal_y': 3.0},
            {'goal_threshold': 0.5},  # Distance threshold to consider goal reached
            {'metrics_file': PathJoinSubstitution([
                FindPackageShare('nav_evaluation_pkg'),
                'config', 
                'metrics_' + algorithm + '.yaml'
            ])}
        ]
    )
    
    # Print the metrics file location when shutting down
    metrics_location_info = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                LogInfo(msg="Metrics file can be found at: " + 
                       get_package_share_directory('hunav_evaluator') + 
                       '/results/metrics_' + algorithm + '.yaml')
            ]
        )
    )
    
    # Create the launch description
    ld = LaunchDescription()
    
    # Add the declared arguments
    ld.add_action(declare_algorithm)
    ld.add_action(declare_metrics_tag)
    
    # Log the experiment starting
    ld.add_action(LogInfo(msg="Running experiment with " + algorithm + " algorithm..."))
    ld.add_action(LogInfo(msg="Robot starting at position: x=-3.0, y=-6.0"))
    ld.add_action(LogInfo(msg="Goal position: x=2.0, y=3.0"))
    
    # Launch the PMB2 cafe environment
    ld.add_action(pmb2_cafe_launch)
    
    # Wait before starting the goal publisher
    ld.add_action(TimerAction(
        period=15.0,
        actions=[
            LogInfo(msg="Starting goal publisher..."),
            goal_publisher,
            goal_monitor
        ]
    ))
    
    # Add metrics location info handler
    ld.add_action(metrics_location_info)
    
    return ld