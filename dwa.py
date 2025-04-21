import numpy as np
from typing import List, Tuple
from logger import Logger

class DWA:
    def __init__(self, radius: float = 0.4, goal_tolerance: float = 0.5):
        # Use MiniWorld's default parameters
        self.radius = radius  # bot_radius from MiniWorld (default 0.4)
        self.goal_tolerance = goal_tolerance
        
        # Room dimensions from OneRoomS6
        self.room_size = 6  # OneRoomS6 uses size=6
        self.min_x = 0
        self.max_x = self.room_size
        self.min_z = 0
        self.max_z = self.room_size
        self.wall_margin = 0.3  # Safety margin from walls
        
        # DWA parameters using MiniWorld defaults
        self.max_speed = 0.15  # forward_step from MiniWorld
        self.min_speed = 0.0
        self.max_angular_speed = np.deg2rad(15)  # turn_step from MiniWorld (15 degrees)
        self.min_angular_speed = -np.deg2rad(15)
        self.acceleration = 0.1  # Fraction of max_speed
        self.angular_acceleration = np.deg2rad(10)  # Fraction of max_angular_speed
        self.speed_resolution = 0.01  # Finer resolution for better control
        self.angular_speed_resolution = np.deg2rad(5)  # Finer resolution for better control
        self.predict_time = 2.0  # Time to predict ahead [s]
        self.dt = 0.1  # Time step for simulation [s]
        self.trajectory_steps = 10  # Number of steps to simulate
        
        # Safety parameters
        self.safety_margin = self.radius * 1.5  # Scale with robot radius
        self.critical_margin = self.radius  # Minimum safe distance
        self.preferred_clearance = self.radius * 2.5  # Preferred clearance from obstacles
        
        # Scoring weights
        self.w_goal = 10.0      # Weight for goal distance
        self.w_heading = 1.0    # Weight for heading alignment
        self.w_human = 3.0      # Weight for human clearance
        self.w_wall = 1.0       # Weight for wall clearance
        self.w_progress = 8.0   # Weight for forward progress
        self.w_turn = -5.0      # Penalty for turning
        
        # Progress parameters
        self.min_progress = self.max_speed * 0.1  # 10% of max speed
        self.no_progress_steps = 0  # Count steps with no progress
        
        # Oscillation detection
        self.last_actions = []  # Track last few actions
        self.max_action_history = 5  # Number of actions to track
        self.oscillation_threshold = 3  # Number of alternating turns to detect oscillation
        
        # Recovery behavior
        self.stuck_time = 0
        self.last_position = None
        self.last_orientation = None
        self.stuck_threshold = 2  # Steps before considering robot stuck
        self.min_movement = self.max_speed * 0.1  # 10% of max speed
        self.last_action = None  # Track last action to prevent oscillation
        self.same_action_count = 0  # Count how many times same action was chosen
        self.max_same_actions = 2  # Maximum number of same actions before forcing change
        
        # Current velocities
        self.current_speed = 0.0
        self.current_angular_speed = 0.0
        
        # Initialize logger
        self.logger = Logger()
        
    def reset(self):
        """Reset all state variables"""
        # Reset velocities
        self.current_speed = 0.0
        self.current_angular_speed = 0.0
        
        # Reset stuck detection
        self.stuck_time = 0
        self.last_position = None
        self.last_orientation = None
        
        # Reset action history
        self.last_actions = []
        self.last_action = None
        self.same_action_count = 0
        
        # Reset progress tracking
        self.no_progress_steps = 0
        
    def _detect_oscillation(self, action: int) -> bool:
        """Detect if robot is oscillating between turning left and right"""
        self.last_actions.append(action)
        if len(self.last_actions) > self.max_action_history:
            self.last_actions.pop(0)
            
        # Check for alternating turns
        if len(self.last_actions) >= self.oscillation_threshold:
            is_oscillating = all(a in [0, 1] for a in self.last_actions)  # All actions are turns
            if is_oscillating:
                self.last_actions = []  # Clear history
                return True
        return False
        
    def _distance_2d(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate 2D Euclidean distance between two points"""
        return np.linalg.norm(pos1[:2] - pos2[:2])
        
    def _calculate_dynamic_window(self, current_speed: float, current_angular_speed: float) -> Tuple[float, float, float, float]:
        """Calculate the dynamic window based on current speeds and acceleration limits"""
        # Calculate speed bounds
        min_speed = max(self.min_speed, current_speed - self.acceleration * self.dt)
        max_speed = min(self.max_speed, current_speed + self.acceleration * self.dt)
        
        # Calculate angular speed bounds
        min_angular = max(self.min_angular_speed, current_angular_speed - self.angular_acceleration * self.dt)
        max_angular = min(self.max_angular_speed, current_angular_speed + self.angular_acceleration * self.dt)
        
        return min_speed, max_speed, min_angular, max_angular
        
    def simulate_trajectory(self, action: int, position: np.ndarray, orientation: float) -> List[np.ndarray]:
        """Simulate a trajectory for a given action"""
        trajectory = []
        current_pos = position.copy()
        current_ori = orientation
        
        # Set velocities based on action
        if action == 0:  # turn left
            linear_vel = 0.0
            angular_vel = self.max_angular_speed
        elif action == 1:  # turn right
            linear_vel = 0.0
            angular_vel = -self.max_angular_speed
        else:  # move forward
            linear_vel = self.max_speed
            angular_vel = 0.0
            
        # Simulate motion
        for _ in range(self.trajectory_steps):
            # Update orientation
            current_ori += angular_vel * self.dt
            
            # Update position
            dx = np.cos(current_ori) * linear_vel * self.dt
            dy = np.sin(current_ori) * linear_vel * self.dt
            new_pos = current_pos + np.array([dx, dy])
            
            # Check if new position is valid (only room bounds)
            if not self._is_valid_position(new_pos):
                break
                
            # Add position to trajectory
            trajectory.append(new_pos.copy())
            current_pos = new_pos
            
        return trajectory
        
    def evaluate_action(self, action: int, robot_pos: np.ndarray, robot_orientation: float, 
                       humans: List['Human'], goal_pos: np.ndarray) -> float:
        """Evaluate an action based on simulated trajectory"""
        # Simulate trajectory
        trajectory = self.simulate_trajectory(action, robot_pos, robot_orientation)
        if not trajectory:
            return float('-inf')
            
        # Get final position and orientation
        final_pos = trajectory[-1]
        final_ori = robot_orientation + self._get_angular_vel(action) * self.dt * len(trajectory)
        
        # Calculate goal score
        goal_distance = self._distance_2d(final_pos, goal_pos)
        goal_score = -self.w_goal * goal_distance
        
        # Calculate heading alignment
        goal_direction = np.arctan2(goal_pos[1] - final_pos[1], goal_pos[0] - final_pos[0])
        heading_diff = abs(self._normalize_angle(goal_direction - final_ori))
        heading_score = -self.w_heading * heading_diff
        
        # Calculate wall clearance
        wall_score = self._score_wall_clearance(trajectory)
        if np.isneginf(wall_score):
            return float('-inf')
            
        # Calculate progress towards goal
        progress = self._distance_2d(final_pos, robot_pos)
        progress_score = self.w_progress * progress
        
        # Calculate turn penalty
        turn_score = self.w_turn * abs(self._get_angular_vel(action))
        
        # Combine scores
        total_score = (goal_score + heading_score + wall_score + 
                      progress_score + turn_score)
        
        return total_score
        
    def _get_angular_vel(self, action: int) -> float:
        """Get angular velocity for an action"""
        if action == 0:  # turn left
            return self.max_angular_speed
        elif action == 1:  # turn right
            return -self.max_angular_speed
        return 0.0  # move forward
        
    def get_action(self, robot_pos: np.ndarray, robot_orientation: float, humans: List['Human'], goal_pos: np.ndarray) -> int:
        """Get the best action based on current state using dynamic window"""
        # Calculate dynamic window
        min_speed, max_speed, min_angular, max_angular = self._calculate_dynamic_window(
            self.current_speed, self.current_angular_speed
        )
        
        # Check if robot is stuck (not moving for several steps)
        if self.last_position is not None:
            if np.linalg.norm(robot_pos - self.last_position) < 0.01:  # Small movement threshold
                self.stuck_time += 1
            else:
                self.stuck_time = 0
        self.last_position = robot_pos.copy()
        
        # Calculate direct distance and angle to goal
        goal_vec = goal_pos - robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        
        # If robot is stuck, force a different action
        if self.stuck_time >= self.stuck_threshold:
            self.stuck_time = 0  # Reset stuck counter
            self.last_actions = []  # Clear action history
            
            # Check which wall we're closest to
            left_dist = abs(robot_pos[0] - self.min_x)
            right_dist = abs(robot_pos[0] - self.max_x)
            front_dist = abs(robot_pos[1] - self.max_z)
            back_dist = abs(robot_pos[1] - self.min_z)
            
            min_dist = min(left_dist, right_dist, front_dist, back_dist)
            
            # If near a wall, move away from it
            if min_dist < self.safety_margin * 1.5:  # More lenient wall avoidance
                if min_dist == left_dist:
                    return 1  # turn_right to move away from left wall
                elif min_dist == right_dist:
                    return 0  # turn_left to move away from right wall
                elif min_dist == front_dist:
                    return 0  # turn_left to move away from front wall
                else:  # back wall
                    return 1  # turn_right to move away from back wall
            
            # If not near a wall, just move forward
            return 2  # move_forward
        
        # If angle to goal is large, prioritize turning
        if abs(goal_angle - robot_orientation) > np.pi/4:
            if goal_angle > robot_orientation:
                return 0  # turn_left
            else:
                return 1  # turn_right
        
        # Evaluate all possible actions
        action_scores = []
        for action in range(3):
            score = self.evaluate_action(action, robot_pos, robot_orientation, humans, goal_pos)
            action_scores.append(score)
            
        # If all actions are unsafe, use recovery behavior
        if all(score == float('-inf') for score in action_scores):
            self.last_actions = []  # Clear action history
            
            # Check which wall we're closest to
            left_dist = abs(robot_pos[0] - self.min_x)
            right_dist = abs(robot_pos[0] - self.max_x)
            front_dist = abs(robot_pos[1] - self.max_z)
            back_dist = abs(robot_pos[1] - self.min_z)
            
            min_dist = min(left_dist, right_dist, front_dist, back_dist)
            
            # If near a wall, move away from it
            if min_dist < self.safety_margin * 1.5:  # More lenient wall avoidance
                if min_dist == left_dist:
                    return 1  # turn_right to move away from left wall
                elif min_dist == right_dist:
                    return 0  # turn_left to move away from right wall
                elif min_dist == front_dist:
                    return 0  # turn_left to move away from front wall
                else:  # back wall
                    return 1  # turn_right to move away from back wall
            
            # If not near a wall, just move forward
            return 2  # move_forward
                
        # Select the action with the highest score
        best_action = np.argmax(action_scores)
        
        # Update current velocities based on selected action
        if best_action == 0:  # Turn left
            self.current_angular_speed = min_angular
            self.current_speed = 0.0
        elif best_action == 1:  # Turn right
            self.current_angular_speed = -min_angular
            self.current_speed = 0.0
        else:  # Move forward
            self.current_angular_speed = 0.0
            self.current_speed = min_speed
            
        return best_action
        
    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to the range [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def _score_goal_distance(self, pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """Score based on distance to goal"""
        distance = np.linalg.norm(pos - goal_pos)
        # Use stronger exponential decay to make closer distances much more valuable
        score = -np.exp(distance/2.0)  # Reduced scale factor from 5.0 to 2.0
        return score
        
    def _score_heading_alignment(self, pos: np.ndarray, orientation: float, goal_pos: np.ndarray) -> float:
        """Score based on alignment with goal direction"""
        goal_direction = goal_pos - pos
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
        angle_diff = np.abs(goal_angle - orientation)
        # Normalize angle difference to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        angle_diff = abs(angle_diff)
        # Use stronger exponential decay to make small angle differences much more valuable
        score = -np.exp(angle_diff/0.2)  # Reduced scale factor from 0.5 to 0.2
        return score
        
    def _score_human_clearance(self, trajectory: List[np.ndarray], humans: List['Human']) -> float:
        """Score based on distance to humans"""
        if not humans:
            return 1.0  # No humans, return neutral score
            
        min_clearance = float('inf')
        
        for pos in trajectory:
            for human in humans:
                distance = np.linalg.norm(pos - human.position)
                clearance = distance - (self.radius + human.radius)
                if clearance < min_clearance:
                    min_clearance = clearance
                    
        # Only return -inf if we're actually hitting a human
        if min_clearance < self.radius/2:
            return float('-inf')
            
        # Otherwise return a positive score based on clearance
        score = min_clearance
        return score
        
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within room bounds"""
        x, z = pos[0], pos[1]  # Note: y in miniworld is z in our 2D representation
        return (self.min_x + self.wall_margin <= x <= self.max_x - self.wall_margin and 
                self.min_z + self.wall_margin <= z <= self.max_z - self.wall_margin)

    def _score_wall_clearance(self, trajectory: List[np.ndarray]) -> float:
        """Score based on distance to walls"""
        min_clearance = float('inf')
        
        for pos in trajectory:
            # Distance to each wall
            dist_left = abs(pos[0] - self.min_x)
            dist_right = abs(pos[0] - self.max_x)
            dist_front = abs(pos[1] - self.max_z)
            dist_back = abs(pos[1] - self.min_z)
            
            min_clearance = min(min_clearance, dist_left, dist_right, dist_front, dist_back)
            
            # If position is outside room bounds, return -inf
            if not self._is_valid_position(pos):
                return float('-inf')
                
        # Scale the score to be more forgiving near walls
        score = min_clearance * 2.0  # Double the score to make wall avoidance less strict
        return score 