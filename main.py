import numpy as np
import argparse
import torch
from simulation import Simulation
from rrt_star import RRTStar
from dwa import DWA
from scl import SCL
from logger import Logger
from mcts_agent import MCTSAgent
from custom_env import CustomEnv
import gymnasium as gym
import time
import os

def main():
    """Main function to run the simulation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run crowd navigation simulation')
    parser.add_argument('--num_humans', type=int, default=5, help='Number of humans in the simulation')
    parser.add_argument('--planner', type=str, choices=['rrt', 'dwa', 'scl', 'mcts'], default='dwa', 
                        help='Planner to use (rrt, dwa, scl, or mcts)')
    parser.add_argument('--render', action='store_true', help='Enable rendering of the simulation')
    
    # MCTS specific arguments
    parser.add_argument('--simulations', type=int, default=50, help='Number of MCTS simulations per step (if using mcts)')
    parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth of MCTS tree (if using mcts)')
    parser.add_argument('--exploration', type=float, default=1.0, help='Exploration weight for MCTS (if using mcts)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to MCTS model (if using mcts)')
    parser.add_argument('--save_vlm_logs', action='store_true', help='Save VLM logs (if using mcts)')
    parser.add_argument('--goal_reward', type=float, default=10.0, help='Reward for reaching goal (if using mcts)')
    parser.add_argument('--collision_penalty', type=float, default=-5.0, help='Penalty for collisions (if using mcts)')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help='Device to run on (if using mcts)')
    
    args = parser.parse_args()
    
    # Initialize logger
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, f"simulation_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    
    try:
        logger.log(f"Starting simulation with {args.num_humans} humans using {args.planner.upper()} planner", "MAIN: ")
        if args.render:
            logger.log("Rendering enabled", "MAIN: ")
        
        # Create planner based on choice
        if args.planner == 'rrt':
            planner = RRTStar(goal_tolerance=1.0)
        elif args.planner == 'dwa':
            planner = DWA(radius=0.3, goal_tolerance=1.0)
        elif args.planner == 'scl':
            planner = SCL(radius=0.3, goal_tolerance=1.0)
        elif args.planner == 'mcts':
            # For MCTS, we'll create it later after environment is initialized
            planner = None
            logger.log(f"Using MCTS with {args.simulations} simulations and depth {args.max_depth}", "MAIN: ")
        else:
            raise ValueError(f"Unknown planner type: {args.planner}")
            
        # Create simulation
        if args.planner != 'mcts':
            # Create standard simulation for traditional planners
            sim = Simulation(
                planner=planner,
                num_humans=args.num_humans,
                render=args.render,
                logger=logger
            )
            
            # Record start time
            start_time = time.time()
            
            # Run simulation
            sim.run_simulation(num_episodes=5)
            
        else:
            # Create environment
            logger.log("Creating MiniWorld environment with CustomEnv wrapper", "MAIN: ")
            render_mode = 'human' if args.render else None
            base_env = gym.make("MiniWorld-OneRoomS6-v0", render_mode=render_mode)
            
            # Create CustomEnv wrapper
            env = CustomEnv(
                env=base_env,
                goal_reward=args.goal_reward,
                collision_penalty=args.collision_penalty,
                human_collision_penalty=args.collision_penalty * 1.5  # Make human collisions slightly worse
            )
            
            # Create MCTS agent
            logger.log("Creating MCTS agent", "MAIN: ")
            agent = MCTSAgent(
                exploration_weight=args.exploration,
                num_simulations=args.simulations,
                max_depth=args.max_depth,
                discount_factor=0.95,
                log_dir=log_dir,
                device=args.device,
                save_vlm_logs=args.save_vlm_logs
            )
            
            # Load model if specified
            if args.model_path and os.path.exists(args.model_path):
                logger.log(f"Loading MCTS model from {args.model_path}", "MAIN: ")
                agent.load_model(args.model_path)
            
            # Create simulation with MCTS
            sim = Simulation(
                planner=agent,
                num_humans=args.num_humans,
                render=args.render,
                logger=logger,
                use_mcts=True,
                custom_env=env
            )
            
            # Record start time
            start_time = time.time()
            
            # Run simulation
            sim.run_simulation(num_episodes=5)
        
        # Record end time
        end_time = time.time()
        
        # Log simulation summary
        logger.log(f"Simulation completed in {end_time - start_time:.2f} seconds", "MAIN: ")
        logger.log(f"Total Episodes: {sim.metric_tracker.total_episodes}", "MAIN: ")
        logger.log(f"Goals Reached: {sim.metric_tracker.goals_reached}", "MAIN: ")
        logger.log(f"Success Rate: {sim.metric_tracker.goals_reached/max(1, sim.metric_tracker.total_episodes)*100:.1f}%", "MAIN: ")
        logger.log(f"Average Steps per Episode: {np.mean(sim.metric_tracker.episode_steps):.1f}", "MAIN: ")
        logger.log(f"Average Reward per Episode: {np.mean(sim.metric_tracker.episode_rewards):.2f}", "MAIN: ")
        
    except Exception as e:
        error_message = str(e)
        print(f"Error: {error_message}")  # Print to console first
        if not logger.file.closed:  # Only try to log if file is still open
            logger.log(f"Error during simulation: {error_message}", "MAIN: ")
    finally:
        # Always close the logger file
        if not logger.file.closed:
            logger.close()
        
if __name__ == "__main__":
    main() 