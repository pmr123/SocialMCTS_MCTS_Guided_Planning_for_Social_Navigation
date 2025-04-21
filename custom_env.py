import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional, Union

class CustomEnv:
    def __init__(self, 
                env: Union[gym.Env, Any], 
                goal_reward: float = 10.0, 
                collision_penalty: float = -5.0,
                human_collision_penalty: float = -8.0):
        """
        Custom environment wrapper with enhanced reward function
        
        Args:
            env: Base MiniWorld environment
            goal_reward: Reward for reaching the goal
            collision_penalty: Penalty for colliding with walls
            human_collision_penalty: Penalty for colliding with humans
        """
        self.env = env
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.human_collision_penalty = human_collision_penalty
        
        # Environment properties - use unwrapped to access the actual MiniWorld environment
        self.unwrapped_env = env.unwrapped
        self.agent = self.unwrapped_env.agent if hasattr(self.unwrapped_env, 'agent') else None
        self.goal_pos = self.unwrapped_env.goal_pos if hasattr(self.unwrapped_env, 'goal_pos') else None
        
        # Tracking variables
        self.last_action = None
        self.collision_count = 0
        self.human_collision_count = 0
        self.steps_taken = 0
        
    def reset(self):
        """Reset the environment"""
        obs = self.env.reset()
        
        # Reset tracking variables
        self.last_action = None
        self.collision_count = 0
        self.human_collision_count = 0
        self.steps_taken = 0
        
        # Update properties from unwrapped environment
        self.unwrapped_env = self.env.unwrapped
        self.agent = self.unwrapped_env.agent if hasattr(self.unwrapped_env, 'agent') else None
        self.goal_pos = self.unwrapped_env.goal_pos if hasattr(self.unwrapped_env, 'goal_pos') else None
        
        return obs
    
    def step(self, action):
        """
        Take a step in the environment with enhanced reward
        
        Args:
            action: Action to take (0: turn left, 1: turn right, 2: move forward)
            
        Returns:
            observation, reward, done, info
        """
        # Save the last action
        self.last_action = action
        self.steps_taken += 1
        
        # Take the action in the base environment
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Convert gym-style done to boolean
        done = done or truncated
        
        # Enhanced reward function
        enhanced_reward = reward
        
        # Track collisions
        collision = False
        human_collision = False
        
        # Check for collisions (implementation depends on environment)
        if hasattr(self.unwrapped_env, 'collision_detected') and self.unwrapped_env.collision_detected:
            collision = True
            self.collision_count += 1
            enhanced_reward += self.collision_penalty
        
        # Check for collisions with humans (implementation depends on environment)
        if hasattr(self.unwrapped_env, 'human_collision_detected') and self.unwrapped_env.human_collision_detected:
            human_collision = True
            self.human_collision_count += 1
            enhanced_reward += self.human_collision_penalty
        
        # Check for goal reached
        success = done and reward > 0
        if success:
            enhanced_reward += self.goal_reward
        
        # Update info
        info.update({
            "collision": collision,
            "human_collision": human_collision, 
            "success": success,
            "original_reward": reward,
            "enhanced_reward": enhanced_reward,
            "collision_count": self.collision_count,
            "human_collision_count": self.human_collision_count,
            "steps_taken": self.steps_taken
        })
        
        # Update properties from unwrapped environment
        self.unwrapped_env = self.env.unwrapped
        self.agent = self.unwrapped_env.agent if hasattr(self.unwrapped_env, 'agent') else None
        
        return observation, enhanced_reward, done, info
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    def copy(self):
        """Create a copy of the environment for simulation"""
        try:
            # Try to create a copy using the env's copy method
            if hasattr(self.env, 'copy'):
                env_copy = CustomEnv(
                    self.env.copy(),
                    self.goal_reward,
                    self.collision_penalty,
                    self.human_collision_penalty
                )
            else:
                # If no copy method, just use the original env (this won't be a real copy)
                print("Warning: Environment does not support copying. Using original environment.")
                env_copy = CustomEnv(
                    self.env,
                    self.goal_reward,
                    self.collision_penalty,
                    self.human_collision_penalty
                )
            
            # Copy current state
            env_copy.last_action = self.last_action
            env_copy.collision_count = self.collision_count
            env_copy.human_collision_count = self.human_collision_count
            env_copy.steps_taken = self.steps_taken
            
            return env_copy
        except Exception as e:
            print(f"Error copying environment: {e}")
            # Return self as fallback
            return self
    
    @property
    def action_space(self):
        """Get the action space from the base environment"""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Get the observation space from the base environment"""
        return self.env.observation_space 