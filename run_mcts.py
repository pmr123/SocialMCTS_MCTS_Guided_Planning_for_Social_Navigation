#!/usr/bin/env python3
import os
import argparse
from matplotlib.dviread import Box
import numpy as np
import gymnasium as gym
import logging
import torch
from datetime import datetime
import json

from mcts_agent import MCTSAgent
from custom_env import CustomEnv
from human_goal_detector import detect_humans_and_goal, create_tagged_image
from metric_tracker import MetricTracker

# Import MiniWorld
try:
    import miniworld
except ImportError:
    raise ImportError("MiniWorld environment not found, please install it with: pip install gym-miniworld")

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run MCTS agent for navigation")
    
    # Environment parameters
    parser.add_argument("--env_id", type=str, default="MiniWorld-OneRoomS6-v0", 
                        help="MiniWorld environment ID")
    
    # Running parameters
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment during running")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=50,
                        help="Number of MCTS simulations per step")
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Maximum depth of the MCTS tree")
    
    # Reward parameters
    parser.add_argument("--goal_reward", type=float, default=10.0,
                        help="Reward for reaching the goal")
    parser.add_argument("--collision_penalty", type=float, default=-5.0,
                        help="Penalty for colliding with walls")
    parser.add_argument("--human_collision_penalty", type=float, default=-8.0,
                        help="Penalty for colliding with humans")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model")
    parser.add_argument("--model_id", type=str, default="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
                        help="VLM model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cuda or cpu)")
    
    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="evaluation_logs",
                        help="Directory to save logs and results")
    parser.add_argument("--log_detections", action="store_true",
                        help="Log tagged images and first VLM response")
    
    # Metrics parameters
    parser.add_argument("--csv_path", type=str, default="metrics_results.csv",
                        help="Path to save metrics results as CSV")
    
    return parser.parse_args()

def _get_goal_position(env) -> np.ndarray:
        """Get the position of the goal (red box) from the environment"""
        # Get all entities
        entities = env.miniworld_env.entities
        
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
    
def main():
    """Main function to run the trained agent"""
    args = parse_arguments()
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.env_id}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(os.path.join(log_dir, 'evaluation.log'))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Starting evaluation with arguments: {args}")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
    
    # Create base environment
    logger.info(f"Creating environment: {args.env_id}")
    base_env = gym.make(args.env_id)
    
    # Wrap with custom environment
    env = CustomEnv(
        env=base_env,
        goal_reward=args.goal_reward,
        collision_penalty=args.collision_penalty,
        human_collision_penalty=args.human_collision_penalty
    )
    
    # Create metric tracker
    metric_tracker = MetricTracker()
    
    # Create MCTS agent
    agent = MCTSAgent(
        num_simulations=args.simulations,
        max_depth=args.max_depth,
        vlm_model_id=args.model_id,
        log_dir=log_dir,
        device=args.device,
        save_vlm_logs=args.log_detections
    )
    
    # Load model if specified
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        agent.load_model(args.model_path)
    
    # Directory for saving tagged images
    detections_dir = os.path.join(log_dir, "detections")
    if args.log_detections:
        os.makedirs(detections_dir, exist_ok=True)
    
    # Run episodes
    logger.info(f"Running {args.episodes} episodes")
    
    total_reward = 0
    success_count = 0
    
    for episode in range(args.episodes):
        # Reset environment and metrics
        observation = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        # For logging first detection of each episode
        first_detection_logged = False
        
        # Run episode
        while not done:
            # Check for first detection
            if args.log_detections and not first_detection_logged:
                detection_results, tagged_image = agent.detect_objects(observation)
                
                # Check if we have at least one red cube and one blue cuboid
                if detection_results['red_cubes'] > 0 and detection_results['blue_cuboids'] > 0:
                    # Save tagged image
                    image_path = os.path.join(detections_dir, f"episode_{episode}_step_{steps}.png")
                    import cv2
                    cv2.imwrite(image_path, cv2.cvtColor(tagged_image, cv2.COLOR_RGB2BGR))
                    
                    # Save detection results
                    json_path = os.path.join(detections_dir, f"episode_{episode}_step_{steps}.json")
                    with open(json_path, 'w') as f:
                        json.dump(detection_results, f, indent=2)
                    
                    # Use VLM to get the first action prediction (for logging)
                    robot_pos = env.agent.pos
                    robot_orientation = env.agent.dir * 90
                    try:
                        goal_pos = env.goal_pos
                    except:
                        goal_pos = _get_goal_position(env)
                    
                    vlm_results = agent.get_vlm_actions(
                        tagged_image=tagged_image,
                        robot_pos=robot_pos,
                        robot_orientation=robot_orientation,
                        goal_pos=goal_pos,
                        detection_results=detection_results
                    )
                    
                    # Save VLM results
                    vlm_path = os.path.join(detections_dir, f"episode_{episode}_step_{steps}_vlm.json")
                    with open(vlm_path, 'w') as f:
                        json.dump(vlm_results, f, indent=2)
                    
                    logger.info(f"Episode {episode}, Step {steps}: Logged detection with {detection_results['red_cubes']} red cubes and {detection_results['blue_cuboids']} blue cuboids")
                    first_detection_logged = True
            
            # Select action using MCTS
            action = agent.select_action(env, observation, args.render)
            
            # Take action
            next_observation, reward, done, info = env.step(action)
            
            # Update metrics
            linear_vel = 0.0
            angular_vel = 0.0
            
            # Extract velocities if available in the environment
            if hasattr(env.env, 'agent'):
                if hasattr(env.env.agent, 'fwd_vel'):
                    linear_vel = env.env.agent.fwd_vel
                if hasattr(env.env.agent, 'ang_vel'):
                    angular_vel = env.env.agent.ang_vel
            
            # Update metric tracker with additional metrics from specifications
            metric_tracker.update(
                robot_pos=env.agent.pos,
                robot_orientation=env.agent.dir,
                humans=[],  # Should be populated with actual humans from environment
                goal_pos=env.goal_pos,
                progress=info.get('progress', 0.0),
                linear_vel=linear_vel,
                angular_vel=angular_vel,
                # Force metrics would need to come from the actual environment
                social_force_agents=0.0,
                social_force_robot=0.0,
                obstacle_force_agents=0.0,
                obstacle_force_robot=0.0
            )
            
            # Update statistics
            episode_reward += reward
            steps += 1
            
            # Render if needed
            if args.render:
                env.render()
            
            # Update observation
            observation = next_observation
        
        # Get metrics for this episode
        metrics = metric_tracker.get_metrics()
        
        # Update total statistics
        total_reward += episode_reward
        if info.get('success', False):
            success_count += 1
        
        # Log episode results
        logger.info(f"Episode {episode+1}/{args.episodes}: Reward = {episode_reward:.2f}, Steps = {steps}, Success = {info.get('success', False)}")
    
    # Log final results
    avg_reward = total_reward / args.episodes
    success_rate = success_count / args.episodes * 100
    
    logger.info(f"Evaluation completed.")
    logger.info(f"Average reward: {avg_reward:.2f}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    
    # Export metrics to CSV
    if args.csv_path:
        csv_path = os.path.join(log_dir, args.csv_path)
        metric_tracker.export_metrics_to_csv(csv_path)
        logger.info(f"Metrics exported to {csv_path}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main() 