import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Optional
from human import Human
import os
import csv
import time

class MetricTracker:
    def __init__(self, window_size: int = 3):
        self.steps = 0
        self.reward = 0
        self.episode_steps = []
        self.episode_rewards = []
        self.goals_reached = 0
        self.total_episodes = 0
        self.window_size = window_size
        self.reward_window = deque(maxlen=window_size)
        
        # Additional metrics
        self.total_human_distance = 0.0
        self.total_wall_distance = 0.0
        self.min_human_distance = float('inf')
        self.min_wall_distance = float('inf')
        self.final_goal_distance = 0.0
        self.last_progress = 0.0
        self.stuck_steps = 0
        
        # New metrics
        self.start_time = time.time()
        self.time_to_reach_goal = 0.0
        self.path_length = 0.0
        self.prev_pos = None
        
        self.human_distances_sum = 0.0
        self.min_dist_to_people = float('inf')
        
        # Space intrusion counters
        self.intimate_space_steps = 0  # <0.45m
        self.personal_space_steps = 0  # 0.45m-1.2m
        self.social_space_steps = 0    # >1.2m
        
        self.completed = False
        self.person_collisions = 0
        
        # Movement metrics
        self.not_moving_steps = 0
        self.linear_speeds = []
        self.angular_speeds = []
        self.accelerations = []
        self.jerks = []
        self.prev_speed = 0.0
        self.prev_acceleration = 0.0
        
        # Force metrics
        self.social_force_on_agents = 0.0
        self.social_force_on_robot = 0.0
        self.obstacle_force_on_agents = 0.0
        self.obstacle_force_on_robot = 0.0
        
        # Store all metrics for CSV export
        self.all_metrics = {}
        
    def _distance_2d(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate 2D distance between two points (ignoring y/height)"""
        # Ensure we're only using x and z coordinates
        if len(pos1) > 2:
            pos1 = np.array([pos1[0], pos1[2]])
        if len(pos2) > 2:
            pos2 = np.array([pos2[0], pos2[2]])
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
    def update(self, robot_pos: np.ndarray, robot_orientation: float, 
               humans: List['Human'], goal_pos: np.ndarray, progress: float,
               linear_vel: float = 0.0, angular_vel: float = 0.0,
               social_force_agents: float = 0.0, social_force_robot: float = 0.0,
               obstacle_force_agents: float = 0.0, obstacle_force_robot: float = 0.0):
        """Update metrics with current state"""
        self.steps += 1
        self.last_progress = progress
        
        # Update path length
        if self.prev_pos is not None:
            step_distance = self._distance_2d(robot_pos, self.prev_pos)
            self.path_length += step_distance
        self.prev_pos = robot_pos.copy()
        
        # Calculate distances to humans (2D)
        human_distances = []
        for human in humans:
            dist = self._distance_2d(robot_pos, human.position)
            human_distances.append(dist)
            self.total_human_distance += dist
            self.min_human_distance = min(self.min_human_distance, dist)
            
            # Update space intrusion metrics
            if dist < 0.45:
                self.intimate_space_steps += 1
                # Count collisions (assuming collision distance is 0.3m)
                if dist < 0.3:
                    self.person_collisions += 1
            elif dist < 1.2:
                self.personal_space_steps += 1
            else:
                self.social_space_steps += 1
        
        # Update avg distance to closest person
        if human_distances:
            closest_human_dist = min(human_distances)
            self.human_distances_sum += closest_human_dist
            self.min_dist_to_people = min(self.min_dist_to_people, closest_human_dist)
            
        # Calculate distance to walls (2D)
        wall_distances = [
            abs(robot_pos[0] - 3),  # Right wall
            abs(robot_pos[0] + 3),  # Left wall
            abs(robot_pos[1] - 2),  # Top wall
            abs(robot_pos[1] + 2)   # Bottom wall
        ]
        min_wall_dist = min(wall_distances)
        self.total_wall_distance += min_wall_dist
        self.min_wall_distance = min(self.min_wall_distance, min_wall_dist)
        
        # Update final goal distance (2D)
        self.final_goal_distance = self._distance_2d(robot_pos, goal_pos)
        
        # Update movement metrics
        if abs(linear_vel) < 0.01:
            self.not_moving_steps += 1
            
        self.linear_speeds.append(abs(linear_vel))
        self.angular_speeds.append(abs(angular_vel))
        
        # Calculate acceleration and jerk
        current_acceleration = 0.0
        current_jerk = 0.0
        if self.steps > 1:
            current_acceleration = abs(linear_vel - self.prev_speed) / 0.1  # Assuming dt = 0.1s
            self.accelerations.append(current_acceleration)
            
        if self.steps > 2:
            current_jerk = abs(current_acceleration - self.prev_acceleration) / 0.1  # Assuming dt = 0.1s
            self.jerks.append(current_jerk)
            
        self.prev_speed = linear_vel
        self.prev_acceleration = current_acceleration
        
        # Update force metrics
        self.social_force_on_agents += social_force_agents
        self.social_force_on_robot += social_force_robot
        self.obstacle_force_on_agents += obstacle_force_agents
        self.obstacle_force_on_robot += obstacle_force_robot
        
        # Check if goal reached (assuming goal distance < 0.5 means reached)
        if self.final_goal_distance < 0.5 and not self.completed:
            self.completed = True
            self.time_to_reach_goal = time.time() - self.start_time
        
        # Calculate reward based on metrics
        reward = self._calculate_reward(
            min(human_distances) if human_distances else float('inf'),
            min_wall_dist,
            self.final_goal_distance,
            progress
        )
        self.reward += reward
        self.reward_window.append(reward)
        
    def _calculate_reward(self, min_human_dist: float, min_wall_dist: float, 
                         goal_dist: float, progress: float) -> float:
        """Calculate reward based on current metrics"""
        # Penalize getting too close to humans
        human_penalty = 0.0
        if min_human_dist < 0.5:
            human_penalty = -1.0
            
        # Penalize getting too close to walls
        wall_penalty = 0.0
        if min_wall_dist < 0.3:
            wall_penalty = -1.0
            
        # Penalize lack of progress
        progress_penalty = 0.0
        if progress < 0.1:
            progress_penalty = -1.0
            
        # Reward getting closer to goal
        goal_reward = -0.1 * goal_dist  # Negative reward proportional to distance
        
        return human_penalty + wall_penalty + progress_penalty + goal_reward
        
    def get_metrics(self) -> dict:
        """Get all tracked metrics"""
        # Calculate percentage values for space intrusions
        intimate_space_pct = (self.intimate_space_steps / self.steps * 100) if self.steps > 0 else 0
        personal_space_pct = (self.personal_space_steps / self.steps * 100) if self.steps > 0 else 0
        social_space_pct = (self.social_space_steps / self.steps * 100) if self.steps > 0 else 0
        
        # Calculate derived metrics
        avg_human_distance = self.total_human_distance / self.steps if self.steps > 0 else 0.0
        avg_closest_person = self.human_distances_sum / self.steps if self.steps > 0 else 0.0
        avg_wall_distance = self.total_wall_distance / self.steps if self.steps > 0 else 0.0
        
        # Calculate movement averages
        avg_linear_speed = np.mean(self.linear_speeds) if self.linear_speeds else 0.0
        avg_angular_speed = np.mean(self.angular_speeds) if self.angular_speeds else 0.0
        avg_acceleration = np.mean(self.accelerations) if self.accelerations else 0.0
        avg_jerk = np.mean(self.jerks) if self.jerks else 0.0
        
        # Calculate social work
        social_work = (self.social_force_on_robot + 
                      self.obstacle_force_on_robot + 
                      self.social_force_on_agents)
        
        # Compile all metrics
        metrics = {
            'steps': self.steps,
            'reward': self.reward,
            'time_to_reach_goal': self.time_to_reach_goal,
            'path_length': self.path_length,
            'avg_human_distance': avg_human_distance,
            'avg_closest_person': avg_closest_person,
            'min_human_distance': self.min_human_distance,
            'min_dist_to_people': self.min_dist_to_people,
            'intimate_space_intrusions': intimate_space_pct,
            'personal_space_intrusions': personal_space_pct,
            'social_space_intrusions': social_space_pct,
            'avg_wall_distance': avg_wall_distance,
            'min_wall_distance': self.min_wall_distance,
            'final_goal_distance': self.final_goal_distance,
            'progress': self.last_progress,
            'completed': self.completed,
            'person_collisions': self.person_collisions,
            'time_not_moving': self.not_moving_steps * 0.1,  # Assuming dt = 0.1s
            'avg_robot_linear_speed': avg_linear_speed,
            'avg_robot_angular_speed': avg_angular_speed,
            'avg_robot_acceleration': avg_acceleration,
            'avg_robot_jerk': avg_jerk,
            'social_force_on_agents': self.social_force_on_agents,
            'social_force_on_robot': self.social_force_on_robot,
            'obstacle_force_on_agents': self.obstacle_force_on_agents,
            'obstacle_force_on_robot': self.obstacle_force_on_robot,
            'social_work': social_work,
            'stuck': self.stuck_steps >= 30,
            'success': self.final_goal_distance < 0.5
        }
        
        # Store all metrics for CSV export
        self.all_metrics = metrics.copy()
        
        # Reset metrics for next episode
        self._reset_episode_metrics()
        
        return metrics
        
    def _reset_episode_metrics(self):
        """Reset metrics for next episode"""
        self.steps = 0
        self.reward = 0
        self.total_human_distance = 0.0
        self.total_wall_distance = 0.0
        self.min_human_distance = float('inf')
        self.min_wall_distance = float('inf')
        self.final_goal_distance = 0.0
        self.last_progress = 0.0
        self.stuck_steps = 0
        
        # Reset new metrics
        self.start_time = time.time()
        self.time_to_reach_goal = 0.0
        self.path_length = 0.0
        self.prev_pos = None
        
        self.human_distances_sum = 0.0
        self.min_dist_to_people = float('inf')
        
        self.intimate_space_steps = 0
        self.personal_space_steps = 0
        self.social_space_steps = 0
        
        self.completed = False
        self.person_collisions = 0
        
        self.not_moving_steps = 0
        self.linear_speeds = []
        self.angular_speeds = []
        self.accelerations = []
        self.jerks = []
        self.prev_speed = 0.0
        self.prev_acceleration = 0.0
        
        self.social_force_on_agents = 0.0
        self.social_force_on_robot = 0.0
        self.obstacle_force_on_agents = 0.0
        self.obstacle_force_on_robot = 0.0
        
    def _calculate_moving_average(self, data):
        """Calculate moving average with the specified window size"""
        if len(data) < self.window_size:
            return np.array(data)  # Return original data if not enough points
        return np.convolve(data, np.ones(self.window_size)/self.window_size, mode='valid')
        
    def export_metrics_to_csv(self, file_path: str = 'metrics_results.csv'):
        """Export all metrics to a CSV file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Write metrics to CSV
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header and values
            writer.writerow(['Metric', 'Value', 'Unit'])
            
            # Add each metric with its appropriate unit
            writer.writerow(['Time to reach goal', f"{self.all_metrics.get('time_to_reach_goal', 0):.2f}", 's'])
            writer.writerow(['Path length', f"{self.all_metrics.get('path_length', 0):.2f}", 'm'])
            writer.writerow(['Avg distance to closest person', f"{self.all_metrics.get('avg_closest_person', 0):.2f}", 'm'])
            writer.writerow(['Min dist to people', f"{self.all_metrics.get('min_dist_to_people', 0):.2f}", 'm'])
            writer.writerow(['Intimate space intrusions', f"{self.all_metrics.get('intimate_space_intrusions', 0):.1f}", '%'])
            writer.writerow(['Personal space intrusions', f"{self.all_metrics.get('personal_space_intrusions', 0):.1f}", '%'])
            writer.writerow(['Social+ space intrusions', f"{self.all_metrics.get('social_space_intrusions', 0):.1f}", '%'])
            writer.writerow(['Completed', 'Yes' if self.all_metrics.get('completed', False) else 'No', '-'])
            writer.writerow(['Robot and person collisions', f"{self.all_metrics.get('person_collisions', 0)}", '-'])
            writer.writerow(['Time not moving', f"{self.all_metrics.get('time_not_moving', 0):.1f}", 's'])
            writer.writerow(['Avg robot linear speed', f"{self.all_metrics.get('avg_robot_linear_speed', 0):.2f}", 'm/s'])
            writer.writerow(['Avg robot angular speed', f"{self.all_metrics.get('avg_robot_angular_speed', 0):.2f}", 'rad/s'])
            writer.writerow(['Avg robot acceleration', f"{self.all_metrics.get('avg_robot_acceleration', 0):.2f}", 'm/s²'])
            writer.writerow(['Avg robot jerk', f"{self.all_metrics.get('avg_robot_jerk', 0):.2f}", 'm/s³'])
            writer.writerow(['Social force on agents', f"{self.all_metrics.get('social_force_on_agents', 0):.2f}", 'm/s²'])
            writer.writerow(['Social force on robot', f"{self.all_metrics.get('social_force_on_robot', 0):.2f}", 'm/s²'])
            writer.writerow(['Obstacle force on agents', f"{self.all_metrics.get('obstacle_force_on_agents', 0):.2f}", 'm/s²'])
            writer.writerow(['Obstacle force on robot', f"{self.all_metrics.get('obstacle_force_on_robot', 0):.2f}", 'm/s²'])
            writer.writerow(['Social work', f"{self.all_metrics.get('social_work', 0):.2f}", 'm/s²'])
            
        print(f"Metrics exported to {file_path}")
        
    def plot_metrics(self, save_path: str = None):
        """Plot comprehensive metrics including rewards, steps, moving averages, and success rate"""
        # Check if we have any data to plot
        if not self.episode_rewards or not self.episode_steps:
            print("\nNo simulation data to plot yet.")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot episode rewards and moving average
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.5, label='Episode Reward')
        if len(self.episode_rewards) >= self.window_size:
            ma_rewards = self._calculate_moving_average(self.episode_rewards)
            ma_episodes = range(self.window_size, len(self.episode_rewards) + 1)
            ax1.plot(ma_episodes, ma_rewards, 'r-', label=f'Moving Average (window={self.window_size})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards')
        ax1.grid(True)
        ax1.legend()
        
        # Plot episode steps and moving average
        ax2.plot(episodes, self.episode_steps, 'b-', alpha=0.5, label='Episode Steps')
        if len(self.episode_steps) >= self.window_size:
            ma_steps = self._calculate_moving_average(self.episode_steps)
            ma_episodes = range(self.window_size, len(self.episode_steps) + 1)
            ax2.plot(ma_episodes, ma_steps, 'r-', label=f'Moving Average (window={self.window_size})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Steps')
        ax2.grid(True)
        ax2.legend()
        
        # Plot success rate over time
        if self.total_episodes > 0:
            cumulative_successes = np.cumsum([1 if r > 0.8 else 0 for r in self.episode_rewards])
            success_rates = [s/(i+1)*100 for i, s in enumerate(cumulative_successes)]
            ax3.plot(episodes, success_rates, 'g-', label='Success Rate')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Cumulative Success Rate')
            ax3.grid(True)
            ax3.legend()
            
            # Add final success rate text
            final_success_rate = (self.goals_reached / self.total_episodes) * 100
            ax3.text(0.02, 0.98, f'Final Success Rate: {final_success_rate:.1f}%', 
                    transform=ax3.transAxes, verticalalignment='top')
        
        # Plot reward distribution
        ax4.hist(self.episode_rewards, bins=20, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Reward Distribution')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nSimulation Summary:")
        print(f"Total Episodes: {self.total_episodes}")
        if self.total_episodes > 0:
            print(f"Goals Reached: {self.goals_reached}")
            print(f"Success Rate: {(self.goals_reached/self.total_episodes)*100:.1f}%")
            print(f"Average Steps per Episode: {np.mean(self.episode_steps):.1f}")
            print(f"Average Reward per Episode: {np.mean(self.episode_rewards):.2f}")
        else:
            print("No episodes completed yet.")
            
        # Export metrics to CSV
        self.export_metrics_to_csv() 