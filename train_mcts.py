#!/usr/bin/env python3
import os
import argparse
import numpy as np
import gymnasium as gym
import logging
from datetime import datetime
import torch

from mcts_agent import MCTSAgent
from custom_env import CustomEnv

# Import MiniWorld - this environment should be installed
try:
    import miniworld
    from miniworld.envs import MiniWorldEnv
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
    parser = argparse.ArgumentParser(description="Train MCTS agent for navigation")
    
    # Environment parameters
    parser.add_argument("--env_id", type=str, default="MiniWorld-Hallway-v0", 
                        help="MiniWorld environment ID")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of episodes to train")
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # MCTS parameters
    parser.add_argument("--simulations", type=int, default=50,
                        help="Number of MCTS simulations per step")
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Maximum depth of the MCTS tree")
    parser.add_argument("--exploration", type=float, default=1.0,
                        help="Exploration weight for UCB formula")
    parser.add_argument("--discount", type=float, default=0.95,
                        help="Discount factor for rewards")
    
    # Reward parameters
    parser.add_argument("--goal_reward", type=float, default=10.0,
                        help="Reward for reaching the goal")
    parser.add_argument("--collision_penalty", type=float, default=-5.0,
                        help="Penalty for colliding with walls")
    parser.add_argument("--human_collision_penalty", type=float, default=-8.0,
                        help="Penalty for colliding with humans")
    
    # VLM parameters
    parser.add_argument("--model_id", type=str, default="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
                        help="VLM model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--save_vlm_logs", action="store_true",
                        help="Save VLM logs (prompt, response, image)")
    
    # Saving parameters
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs and model")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save model every N episodes")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.env_id}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Starting training with arguments: {args}")
    
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
    
    # Create MCTS agent
    agent = MCTSAgent(
        exploration_weight=args.exploration,
        num_simulations=args.simulations,
        max_depth=args.max_depth,
        discount_factor=args.discount,
        vlm_model_id=args.model_id,
        log_dir=log_dir,
        device=args.device,
        save_vlm_logs=args.save_vlm_logs
    )
    
    # Train the agent
    logger.info(f"Starting training for {args.episodes} episodes")
    agent.train(
        env=env,
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        render=args.render
    )
    
    # Save final model
    agent.save_model(os.path.join(log_dir, "mcts_agent_final.pkl"))
    
    # Close environment
    env.close()
    
    logger.info("Training completed.")
    logger.info(f"Final stats: {agent.stats}")

if __name__ == "__main__":
    main() 