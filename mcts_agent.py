import numpy as np
import random
import math
import torch
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import gymnasium as gym
import numpy as np
import miniworld  # Import MiniWorld to register environments
from miniworld.entity import Box
import copy as cp
import cv2

from human_goal_detector import detect_humans_and_goal, create_tagged_image
from vlm_interface import VLMInterface
from custom_env import CustomEnv

# Coordinate system conventions:
# - OpenGL convention: ground plane is x-z, height is y
# - All positions in simulation are represented as 2D arrays [x, z]
# - When 3D positions are needed (like for VLM), they are represented as [x, y, z]
# - All distance calculations are performed in 2D (x-z plane), ignoring y-axis
# - Orientations are in degrees, consistent with simulation.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # State representation
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this node
        self.children = {}  # Map of action -> MCTSNode
        self.visit_count = 0  # Number of times this node has been visited
        self.value_sum = 0.0  # Sum of values from rollouts through this node
        self.value = 0.0  # Average value (value_sum / visit_count)
        self.depth = 0 if parent is None else parent.depth + 1  # Depth in the tree
    
    def is_expanded(self):
        """Check if the node has been expanded"""
        return len(self.children) > 0
    
    def add_child(self, action, child_node):
        """Add a child node"""
        self.children[action] = child_node
    
    def update(self, value):
        """Update node statistics"""
        self.visit_count += 1
        self.value_sum += value
        self.value = self.value_sum / self.visit_count
    
    def select_child(self, exploration_weight=1.0):
        """Select child node using UCB formula"""
        # UCB formula: value + exploration_weight * sqrt(log(parent visits) / child visits)
        log_visits_parent = math.log(max(self.visit_count, 1))  # Avoid log(0)
        
        def ucb_score(child):
            # If child has never been visited, return infinity to ensure exploration
            if child.visit_count == 0:
                return float('inf')
            
            # Exploit term
            exploit = child.value
            # Explore term
            explore = exploration_weight * math.sqrt(log_visits_parent / child.visit_count)
            return exploit + explore
        
        # Select the child with the highest UCB score
        return max(self.children.items(), key=lambda item: ucb_score(item[1]))


class MCTSAgent:
    def __init__(self, 
                 action_space: List[int] = [0, 1, 2],  # 0: turn left, 1: turn right, 2: move forward
                 exploration_weight: float = 1.0,
                 num_simulations: int = 50,
                 max_depth: int = 10,
                 discount_factor: float = 0.95,
                 vlm_model_id: str = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
                 log_dir: str = "logs",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 save_vlm_logs: bool = False):
        """
        Monte Carlo Tree Search agent for navigation
        
        Args:
            action_space: List of available actions
            exploration_weight: Exploration weight for UCB formula
            num_simulations: Number of MCTS simulations to run
            max_depth: Maximum depth of the search tree
            discount_factor: Discount factor for rewards
            vlm_model_id: Model ID for VLM
            log_dir: Directory to save logs
            device: Device to run the model on
            save_vlm_logs: Whether to save VLM logs
        """
        self.action_space = action_space
        self.exploration_weight = exploration_weight
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.discount_factor = discount_factor
        self.device = device
        self.save_vlm_logs = save_vlm_logs
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize stats
        self.stats = {
            "iterations": 0,
            "goal_reached": 0,
            "collisions": 0,
            "avg_reward": 0.0,
            "total_steps": 0
        }
        
        # Initialize VLM interface
        try:
            logger.info(f"Initializing VLM interface with model {vlm_model_id}")
            self.vlm = VLMInterface(
                model_id=vlm_model_id,
                device=device,
                log_dir=os.path.join(log_dir, "vlm_logs") if save_vlm_logs else None
            )
            logger.info("VLM interface initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VLM interface: {str(e)}")
            logger.warning("Continuing without VLM, will use random actions")
            self.vlm = None
        
        # Load model if it exists
        self.load_model()
    
    def reset_stats(self):
        """Reset agent statistics"""
        self.stats = {
            "iterations": 0,
            "goal_reached": 0,
            "collisions": 0,
            "avg_reward": 0.0,
            "total_steps": 0
        }
    
    def get_state_representation(self, observation, robot_pos, robot_orientation, goal_pos):
        """
        Create a state representation for the MCTS tree
        
        Args:
            observation: RGB image observation or tuple containing observation
            robot_pos: Robot position (x, y, z)
            robot_orientation: Robot orientation in degrees
            goal_pos: Goal position (x, y, z)
            
        Returns:
            State representation for MCTS
        """
        # Handle tuple observation
        if isinstance(observation, tuple):
            observation = observation[0]  # Take the first element which should be the image
        
        # Convert observation to numpy array if it's not already
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Reshape if necessary
        if len(observation.shape) == 1:
            size = int(np.sqrt(len(observation) / 3))
            observation = observation.reshape(size, size, 3)
        
        print(f"Observation shape in get_state_representation: {observation.shape}")
        
        # Extract only x and z coordinates for 2D positions, ignoring y
        robot_pos_2d = np.array([robot_pos[0], robot_pos[2]]) if len(robot_pos) > 2 else robot_pos
        goal_pos_2d = np.array([goal_pos[0], goal_pos[2]]) if len(goal_pos) > 2 else goal_pos
        
        # Use the image itself, robot position, orientation, and goal position as the state
        return {
            "image": observation,  # Now guaranteed to be a numpy array with shape (H, W, 3)
            "robot_pos": robot_pos_2d,
            "robot_orientation": robot_orientation,
            "goal_pos": goal_pos_2d
        }
    
    def detect_objects(self, image):
        """
        Detect humans and goal in the image
        
        Args:
            image: RGB image (can be tuple, list, or numpy array)
            
        Returns:
            Detection results and tagged image
        """
        try:
            # Debug print
            print(f"Input image type: {type(image)}")
            
            # If image is a tuple, extract the first element (usually the observation)
            if isinstance(image, tuple):
                print(f"Tuple length: {len(image)}")
                print(f"Tuple contents types: {[type(x) for x in image]}")
                image = image[0]
            
            # Convert to numpy array if needed
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            print(f"Image shape before processing: {image.shape}, dtype: {image.dtype}")
            
            # Ensure correct shape and type
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected RGB image with shape (H, W, 3), got shape {image.shape}")
            
            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Add debug visualization
            debug_image = image.copy()
            cv2.imwrite('debug_input.png', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
            
            # Detect objects
            detection_results = detect_humans_and_goal(image)
            print(f"Detection results: {detection_results}")
            
            tagged_image = create_tagged_image(image, detection_results)
            cv2.imwrite('debug_tagged.png', cv2.cvtColor(tagged_image, cv2.COLOR_RGB2BGR))
            
            print(f"Tagged image shape: {tagged_image.shape}, dtype: {tagged_image.dtype}")
            return detection_results, tagged_image
            
        except Exception as e:
            logger.error(f"Error in detect_objects: {str(e)}")
            logger.error(f"Image shape: {image.shape if isinstance(image, np.ndarray) else 'not numpy array'}")
            # Return empty detection results and original image as fallback
            empty_results = {
                "red_cubes": 0,
                "blue_cuboids": 0,
                "humans_detected": 0,
                "objects": []
            }
            return empty_results, image
    
    def get_vlm_actions(self, tagged_image, robot_pos, robot_orientation, goal_pos, detection_results):
        """
        Get actions from VLM
        
        Args:
            tagged_image: Tagged RGB image
            robot_pos: Robot position (x, z) in 2D
            robot_orientation: Robot orientation in degrees
            goal_pos: Goal position (x, z) in 2D
            detection_results: Detection results from human_goal_detector
            
        Returns:
            Dict of action -> score pairs
        """
        if self.vlm is None:
            # Return random actions if VLM is not available
            actions = {}
            for i in range(2):  # Generate k=2 actions
                action = random.choice(self.action_space)
                actions[str(action)] = random.uniform(0.5, 1.0)
            detection_results["answer"] = actions
            return detection_results
        
        try:
            # Debug print
            print(f"Tagged image shape: {tagged_image.shape}, dtype: {tagged_image.dtype}")
            
            # Ensure image is in the correct format
            if tagged_image.dtype != np.uint8:
                tagged_image = (tagged_image * 255).astype(np.uint8)
            
            # Resize image if needed (check VLM's expected input size)
            expected_size = (224, 224)  # Standard size for many vision models
            if tagged_image.shape[:2] != expected_size:
                tagged_image = cv2.resize(tagged_image, expected_size)
            
            # Convert to 3D positions for VLM if needed (assuming y=0)
            robot_pos_3d = np.array([robot_pos[0], 0, robot_pos[1]]) if len(robot_pos) == 2 else robot_pos
            goal_pos_3d = np.array([goal_pos[0], 0, goal_pos[1]]) if len(goal_pos) == 2 else goal_pos
            
            # Generate actions using VLM
            results = self.vlm.generate_actions(
                image=tagged_image,
                robot_pos=robot_pos_3d,
                robot_orientation=robot_orientation,
                goal_pos=goal_pos_3d,
                detection_results=detection_results,
                k=2,
                log_output=self.save_vlm_logs
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Error in VLM processing: {str(e)}")
            logger.error(f"Tagged image info - shape: {tagged_image.shape}, dtype: {tagged_image.dtype}")
            logger.error(f"Detection results: {detection_results}")
            
            # Fallback to random actions
            actions = {}
            for i in range(2):
                action = random.choice(self.action_space)
                actions[str(action)] = random.uniform(0.5, 1.0)
            detection_results["answer"] = actions
            return detection_results
    
    def save_model(self, path=None):
        """Save the model to a file"""
        if path is None:
            path = os.path.join(self.log_dir, "mcts_agent.pkl")
        
        with open(path, 'wb') as f:
            pickle.dump({
                "stats": self.stats,
                "exploration_weight": self.exploration_weight,
                "num_simulations": self.num_simulations,
                "max_depth": self.max_depth,
                "discount_factor": self.discount_factor,
            }, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Load the model from a file"""
        if path is None:
            path = os.path.join(self.log_dir, "mcts_agent.pkl")
        
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                self.stats = data["stats"]
                self.exploration_weight = data["exploration_weight"]
                self.num_simulations = data["num_simulations"]
                self.max_depth = data["max_depth"]
                self.discount_factor = data["discount_factor"]
                
                logger.info(f"Model loaded from {path}")
                logger.info(f"Stats: {self.stats}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
        else:
            logger.info(f"No model found at {path}, using default parameters")
    
    def update_stats(self, success, collision, reward, steps):
        """Update agent statistics"""
        self.stats["iterations"] += 1
        self.stats["goal_reached"] += 1 if success else 0
        self.stats["collisions"] += 1 if collision else 0
        self.stats["total_steps"] += steps
        
        # Update average reward
        prev_avg = self.stats["avg_reward"]
        prev_count = self.stats["iterations"] - 1
        if prev_count == 0:
            self.stats["avg_reward"] = reward
        else:
            self.stats["avg_reward"] = (prev_avg * prev_count + reward) / self.stats["iterations"]
    
    def _get_goal_position(self, env) -> np.ndarray:
        """Get the position of the goal (red box) from the environment"""
        # Get all entities
        entities = env.unwrapped_env.entities
        
        # Find the red box (goal)
        goal_entity = None
        for entity in entities:
            if isinstance(entity, Box) and hasattr(entity, 'color_vec'):
                # Check if it's red (RGB values close to [1, 0, 0])
                if np.allclose(entity.color_vec, [1.0, 0.0, 0.0], atol=0.1):
                    goal_entity = entity
                    break
                    
        if goal_entity is None:
            raise ValueError("Could not find goal (red box) in environment")
            
        # Return x,z coordinates (ignoring y/height)
        return np.array([goal_entity.pos[0], goal_entity.pos[2]])
    
    def select_action(self, env: CustomEnv, observation, render=False):
        """
        Run MCTS and select the best action
        
        Args:
            env: CustomEnv environment
            observation: Current observation (RGB image)
            render: Whether to render the environment
            
        Returns:
            Selected action
        """
        # Add warning about environment copying
        if not hasattr(env, 'copy') or not callable(getattr(env, 'copy')):
            logger.warning("Environment does not support proper copying. This may affect simulation accuracy.")
        
        # Get the current state from the custom environment
        robot_pos = env.agent.pos
        robot_orientation = env.agent.dir * 90  # Convert to degrees
        goal_pos = self._get_goal_position(env)
        
        # Create root node with proper 2D coordinates
        state = self.get_state_representation(observation, robot_pos, robot_orientation, goal_pos)
        root = MCTSNode(state)
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            # Selection and expansion
            node, depth = self.select_and_expand(root, env)
            
            # Simulation/rollout
            value = self.simulate(node, env, depth)
            
            # Backpropagation
            self.backpropagate(node, value)
        
        # Select the best action based on visit count
        if not root.children:
            return random.choice(self.action_space)
        
        action, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
        return action
    
    def select_and_expand(self, root, env: CustomEnv):
        """
        Select a node to expand using UCB
        
        Args:
            root: Root node
            env: CustomEnv environment
            
        Returns:
            Selected node and depth
        """
        node = root
        depth = 0
        
        # Select until we reach a leaf node or maximum depth
        while node.is_expanded() and depth < self.max_depth:
            try:
                action, node = node.select_child(self.exploration_weight)
                depth += 1
            except Exception as e:
                logger.error(f"Error in select_child: {str(e)}")
                # If there's an error in selection, break and expand current node
                break
        
        # If the node is not expanded and within max depth, expand it
        if not node.is_expanded() and depth < self.max_depth:
            try:
                # Use level/depth 1 process
                detection_results, tagged_image = self.detect_objects(node.state["image"])
                
                # Use level/depth 2 process with VLM
                vlm_results = self.get_vlm_actions(
                    tagged_image=tagged_image, 
                    robot_pos=node.state["robot_pos"], 
                    robot_orientation=node.state["robot_orientation"], 
                    goal_pos=node.state["goal_pos"], 
                    detection_results=detection_results
                )
                
                # Extract actions and scores
                action_scores = vlm_results.get("answer", {})
                
                # If no actions from VLM, use random actions
                if not action_scores:
                    action_scores = {
                        str(action): random.uniform(0.5, 1.0) 
                        for action in self.action_space
                    }
                
                # For each possible action, create a child node
                for action_str, score in action_scores.items():
                    action = int(action_str)
                    
                    # Clone the environment to simulate the action
                    env_copy = env.copy()
                    
                    # Execute the action
                    next_obs, reward, done, info = env_copy.step(action)
                    
                    # Get the new state from the custom environment
                    next_robot_pos = np.array([env_copy.agent.pos[0], env_copy.agent.pos[2]])
                    next_robot_orientation = env_copy.agent.dir * 90
                    
                    next_state = self.get_state_representation(
                        next_obs, next_robot_pos, next_robot_orientation, node.state["goal_pos"]
                    )
                    
                    # Create child node
                    child = MCTSNode(next_state, parent=node, action=action)
                    
                    # Initialize child node with VLM score as initial value
                    child.value = float(score)
                    child.visit_count = 1  # Initialize with 1 visit to avoid division by zero
                    
                    # Add child to the current node
                    node.add_child(action, child)
                
                # If no children were created, create random action children
                if not node.children:
                    for action in self.action_space:
                        env_copy = env.copy()
                        next_obs, reward, done, info = env_copy.step(action)
                        
                        next_robot_pos = np.array([env_copy.agent.pos[0], env_copy.agent.pos[2]])
                        next_robot_orientation = env_copy.agent.dir * 90
                        
                        next_state = self.get_state_representation(
                            next_obs, next_robot_pos, next_robot_orientation, node.state["goal_pos"]
                        )
                        
                        child = MCTSNode(next_state, parent=node, action=action)
                        child.value = random.uniform(0.5, 1.0)
                        child.visit_count = 1
                        node.add_child(action, child)
                
            except Exception as e:
                logger.error(f"Error in expand: {str(e)}")
                # If expansion fails, create random action children
                for action in self.action_space:
                    env_copy = env.copy()
                    next_obs, reward, done, info = env_copy.step(action)
                    
                    next_robot_pos = np.array([env_copy.agent.pos[0], env_copy.agent.pos[2]])
                    next_robot_orientation = env_copy.agent.dir * 90
                    
                    next_state = self.get_state_representation(
                        next_obs, next_robot_pos, next_robot_orientation, node.state["goal_pos"]
                    )
                    
                    child = MCTSNode(next_state, parent=node, action=action)
                    child.value = random.uniform(0.5, 1.0)
                    child.visit_count = 1
                    node.add_child(action, child)
        
        return node, depth
    
    def simulate(self, node, env: CustomEnv, depth):
        """
        Simulate from a node until the end or until max depth is reached
        
        Args:
            node: Current node
            env: CustomEnv environment
            depth: Current depth
            
        Returns:
            Value estimation
        """
        # Clone the environment
        env_copy = env.copy()
        
        # Get current position and goal position for distance calculations
        current_pos = node.state["robot_pos"]
        goal_pos = node.state["goal_pos"]
        
        # Use default goal tolerance if not defined in environment
        goal_tolerance = getattr(env_copy, 'goal_tolerance', 1.0)  # Default to 1.0 meter
        
        # Check if already at goal
        if self._distance_2d(current_pos, goal_pos) < goal_tolerance:
            return env_copy.goal_reward  # Return goal reward directly
        
        # Run simulation
        done = False
        value = 0.0
        current_depth = depth
        
        while not done and current_depth < self.max_depth:
            # Choose a random action
            action = random.choice(self.action_space)
            
            # Take the action in the custom environment
            _, reward, done, info = env_copy.step(action)
            
            # Update value with discounted reward
            value += (self.discount_factor ** (current_depth - depth)) * reward
            current_depth += 1
        
        return value
    
    def backpropagate(self, node, value):
        """
        Backpropagate the value up the tree
        
        Args:
            node: Current node
            value: Value to backpropagate
        """
        # Update all nodes from the current node to the root
        while node is not None:
            node.update(value)
            node = node.parent
    
    def train(self, env: CustomEnv, num_episodes=100, save_interval=10, render=False):
        """
        Train the agent
        
        Args:
            env: CustomEnv environment
            num_episodes: Number of episodes to train
            save_interval: Interval to save the model
            render: Whether to render the environment
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Reset environment
            observation = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            collision = False
            
            while not done:
                # Select action using MCTS
                action = self.select_action(env, observation, render)
                
                # Take action
                next_observation, reward, done, info = env.step(action)
                
                # Update statistics
                episode_reward += reward
                steps += 1
                
                # Check for collision
                if "collision" in info and info["collision"]:
                    collision = True
                
                # Render if needed
                if render:
                    env.render()
                
                # Update observation
                observation = next_observation
            
            # Update stats
            success = "success" in info and info["success"]
            self.update_stats(success, collision, episode_reward, steps)
            
            # Log episode results
            logger.info(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}, Success = {success}, Collision = {collision}")
            
            # Save model at intervals
            if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
                self.save_model()
        
        logger.info(f"Training completed. Final stats: {self.stats}")
        self.save_model()

    def _distance_2d(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate 2D distance between two points, ignoring y-axis
        
        Args:
            pos1: First position (can be 2D or 3D)
            pos2: Second position (can be 2D or 3D)
            
        Returns:
            2D Euclidean distance
        """
        # Extract only x and z coordinates if positions are 3D
        if len(pos1) > 2:
            pos1 = np.array([pos1[0], pos1[2]])
        if len(pos2) > 2:
            pos2 = np.array([pos2[0], pos2[2]])
            
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) 