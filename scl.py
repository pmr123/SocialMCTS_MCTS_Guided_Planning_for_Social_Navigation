import numpy as np
from typing import List, Tuple
from logger import Logger
from dwa import DWA

class SCL(DWA):
    def __init__(self, radius: float = 0.4, goal_tolerance: float = 0.5):
        super().__init__(radius, goal_tolerance)
        
        # Room dimensions from OneRoomS6 (inherits from DWA but explicitly stated here)
        self.room_size = 6  # OneRoomS6 uses size=6
        self.min_x = 0
        self.max_x = self.room_size
        self.min_z = 0
        self.max_z = self.room_size
        self.wall_margin = self.radius * 0.75  # Scale with robot radius
        
        # Social navigation parameters (scaled with robot radius)
        self.social_radius = self.radius * 3.75  # 1.5 meters for 0.4m radius robot
        self.personal_space = self.radius * 2.0  # 0.8 meters for 0.4m radius robot
        self.passing_preference = 0.7  # Preference for passing on the right
        self.velocity_factor = 0.5  # Factor to consider human velocity
        
        # Social cost weights
        self.w_social = 5.0  # Weight for social cost
        self.w_personal = 10.0  # Weight for personal space violation
        self.w_passing = 3.0  # Weight for passing preference
        
        # Gaussian parameters for social cost
        self.social_amplitude = 1.0
        self.social_covariance = self.radius * 1.25  # Scale with robot radius
        
    def reset(self):
        """Reset all state variables"""
        super().reset()
        
    def _calculate_social_cost(self, pos: np.ndarray, humans: List['Human']) -> float:
        """Calculate social cost at a position based on human positions and velocities"""
        # First check if position is within bounds
        if not self._is_valid_position(pos):
            return float('inf')
            
        total_cost = 0.0
        
        for human in humans:
            # Calculate distance to human
            dist = np.linalg.norm(pos - human.position)
            
            if dist < self.social_radius:
                # Calculate relative velocity
                rel_vel = np.array([human.velocity.x, human.velocity.z])
                vel_mag = np.linalg.norm(rel_vel)
                
                # Calculate direction to human
                to_human = human.position - pos
                to_human_dir = to_human / (np.linalg.norm(to_human) + 1e-6)
                
                # Calculate social cost using Gaussian
                social_cost = self.social_amplitude * np.exp(-dist**2 / (2 * self.social_covariance**2))
                
                # Adjust cost based on relative velocity
                if vel_mag > 0:
                    # If human is moving towards robot, increase cost
                    if np.dot(to_human_dir, rel_vel) > 0:
                        social_cost *= (1 + self.velocity_factor * vel_mag)
                
                # Add personal space violation cost
                if dist < self.personal_space:
                    personal_cost = self.w_personal * (1 - dist/self.personal_space)
                    social_cost += personal_cost
                
                # Add passing preference cost
                right_side = np.array([-to_human[1], to_human[0]])  # Perpendicular vector
                passing_direction = np.dot(right_side, rel_vel)
                if passing_direction > 0:  # Human moving to robot's right
                    social_cost *= self.passing_preference
                
                total_cost += social_cost
                
        return total_cost
        
    def evaluate_action(self, action: int, robot_pos: np.ndarray, robot_orientation: float, 
                       humans: List['Human'], goal_pos: np.ndarray) -> float:
        """Evaluate an action considering social costs"""
        # Get base DWA score
        base_score = super().evaluate_action(action, robot_pos, robot_orientation, humans, goal_pos)
        
        if np.isneginf(base_score):  # Check if base score is -inf
            return base_score
            
        # Simulate trajectory for this action
        trajectory = self.simulate_trajectory(action, robot_pos, robot_orientation)
        if not trajectory:  # If trajectory is empty
            return float('-inf')
            
        # Calculate social cost for the trajectory
        social_cost = 0.0
        for pos in trajectory:
            # Check if position is valid
            if not self._is_valid_position(pos):
                return float('-inf')
                
            # Calculate social cost
            pos_cost = self._calculate_social_cost(pos, humans)
            if np.isinf(pos_cost):  # Check for infinite cost
                return float('-inf')
            social_cost += pos_cost
            
        # Average the social cost
        social_cost /= len(trajectory)
        
        # Combine scores
        final_score = base_score - self.w_social * social_cost
        
        return float(final_score)  # Ensure we return a scalar
        
    def get_action(self, robot_pos: np.ndarray, robot_orientation: float, 
                  humans: List['Human'], goal_pos: np.ndarray) -> int:
        """Get the best action considering social navigation"""
        
        # Calculate dynamic window
        min_speed, max_speed, min_angular, max_angular = self._calculate_dynamic_window(
            robot_pos, robot_orientation
        )
        
        # Evaluate all possible actions
        action_scores = []
        for action in range(3):
            score = self.evaluate_action(action, robot_pos, robot_orientation, humans, goal_pos)
            action_scores.append(score)
            
        # Select the action with the highest score
        best_action = np.argmax(action_scores)
        
        return best_action 

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within room bounds"""
        x, z = pos[0], pos[1]  # Note: y in miniworld is z in our 2D representation
        return (self.min_x + self.wall_margin <= float(x) <= self.max_x - self.wall_margin and 
                self.min_z + self.wall_margin <= float(z) <= self.max_z - self.wall_margin) 