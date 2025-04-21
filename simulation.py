import gymnasium as gym
import numpy as np
import miniworld  # Import MiniWorld to register environments
from miniworld.entity import Box
from metric_tracker import MetricTracker
from human import Human
from dwa import DWA
from rrt_star import RRTStar
from scl import SCL
from mcts_agent import MCTSAgent
from custom_env import CustomEnv
from typing import List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import os
from logger import Logger
from metric_tracker import MetricTracker

class Simulation:
    def _get_goal_position(self) -> np.ndarray:
        """Get the position of the goal (red box) from the environment"""
        # Get all entities
        entities = self.miniworld_env.entities
        
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
        
    def __init__(self, planner: Union[RRTStar, DWA, SCL, MCTSAgent], num_humans: int = 2, 
                 render: bool = False, logger: Optional[Logger] = None, 
                 use_mcts: bool = False, custom_env: Optional[CustomEnv] = None):
        # Create a new logger if none is provided
        self.logger = logger if logger is not None else Logger()
        self.metric_tracker = MetricTracker()
        
        # Store planner and check if using MCTS
        self.robot = planner
        self.use_mcts = use_mcts
        self.num_humans = num_humans
        
        # Set up environment based on whether we're using MCTS or not
        if use_mcts and custom_env is not None:
            self.logger.log("Using MCTS agent with CustomEnv wrapper", "SIM: ")
            self.env = custom_env
            self.miniworld_env = custom_env.env.unwrapped
        else:
            # Pass the logger to the planner if not MCTS
            if not use_mcts and hasattr(planner, 'logger'):
                planner.logger = self.logger
                
            # Initialize environment with or without rendering
            render_mode = 'human' if render else None
            self.env = gym.make("MiniWorld-OneRoomS6-v0", render_mode=render_mode)
            self.miniworld_env = self.env.unwrapped
        
        # Set window title based on planner type
        if isinstance(planner, RRTStar):
            self.planner_name = "RRT Star Planner"
        elif isinstance(planner, SCL):
            self.planner_name = "Social Navigation (SCL) Planner"
        elif isinstance(planner, MCTSAgent):
            self.planner_name = "MCTS Agent with VLM"
        else:
            self.planner_name = "Dynamic Window Approach (DWA) Planner"
            
        # Set window title if rendering
        if render:
            import pyglet
            if len(pyglet.canvas.get_display().get_windows()) > 0:
                pyglet.canvas.get_display().get_windows()[0].set_caption(self.planner_name)
        
        # Room size for OneRoomS6 environment
        self.room_size = 6  # OneRoomS6 uses size=6
        
        # Get initial robot position and orientation from environment
        self.robot_position = np.array([self.miniworld_env.agent.pos[0], self.miniworld_env.agent.pos[2]])
        self.robot_orientation = self.miniworld_env.agent.dir
        
        # Get goal position from environment
        self.goal_position = self._get_goal_position()
        self.goal_tolerance = 1.0  # Increased from 0.5 to 1.0 for easier goal reaching
        
        # Initialize humans
        self.humans = self._initialize_humans()
        
        # Log planner type
        if isinstance(planner, RRTStar):
            planner_type = "RRT*"
        elif isinstance(planner, SCL):
            planner_type = "SCL"
        elif isinstance(planner, MCTSAgent):
            planner_type = "MCTS"
        else:
            planner_type = "DWA"
        self.logger.log(f"Using {planner_type} planner", "SIM: ")
        if render:
            self.logger.log("Rendering enabled", "SIM: ")
        
    def _initialize_humans(self) -> List[Human]:
        """Initialize human agents in the environment"""
        humans = []
        # Room bounds from OneRoomS6
        min_x, max_x = 0, self.room_size
        min_z, max_z = 0, self.room_size
        
        for i in range(self.num_humans):
            # Random position in room, respecting bounds and margins
            margin = 0.3  # Keep away from walls
            pos = np.array([
                np.random.uniform(min_x + margin, max_x - margin),
                np.random.uniform(min_z + margin, max_z - margin)
            ])
            human = Human(position=pos, logger=self.logger)
            human.reset(self.robot_position, self.goal_position)
            humans.append(human)
            self.logger.log(f"Human {i} initialized at position {pos}", "SIM: ")
        return humans

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within room bounds"""
        min_x, max_x = 0, self.room_size
        min_z, max_z = 0, self.room_size
        margin = 0.3  # Safety margin from walls
        
        x, z = pos[0], pos[1]
        return (min_x + margin <= x <= max_x - margin and 
                min_z + margin <= z <= max_z - margin)

    def run_episode(self) -> dict:
        """Run a single episode of the simulation"""
        self.logger.log("Starting episode", "SIM: ")
        
        # Reset environment
        obs = self.env.reset()
        
        # Get initial robot position and orientation from environment
        self.robot_position = np.array([self.miniworld_env.agent.pos[0], self.miniworld_env.agent.pos[2]])
        self.robot_orientation = self.miniworld_env.agent.dir
        
        # Get goal position from environment
        self.goal_position = self._get_goal_position()
        
        # Reset planner
        if hasattr(self.robot, 'reset'):
            self.robot.reset()
        
        # Reset humans
        self.humans = self._initialize_humans()
        
        # Run simulation until goal reached or max steps (100 steps from OneRoomS6)
        max_steps = 100  # OneRoomS6's default max_episode_steps
        goal_reached = False
        episode_reward = 0
        
        for step in range(max_steps):
            success, step_reward = self.step()
            episode_reward += step_reward
            
            if success:
                self.logger.log("Goal reached!", "SIM: ")
                goal_reached = True
                break
                
        # Get final metrics
        metrics = self.metric_tracker.get_metrics()
        metrics['steps'] = step + 1
        metrics['success'] = goal_reached  # Success only depends on reaching the goal
        metrics['reward'] = episode_reward  # Total episode reward
        
        # Update episode tracking
        self.metric_tracker.episode_steps.append(metrics['steps'])
        self.metric_tracker.episode_rewards.append(metrics['reward'])
        if goal_reached:  # Update goals_reached counter based on actual goal reaching
            self.metric_tracker.goals_reached += 1
        self.metric_tracker.total_episodes += 1
        
        return metrics
        
    def step(self) -> Tuple[bool, float]:
        """Perform one simulation step
        
        Returns:
            Tuple of (goal_reached, reward)
        """
        self.logger.log("Performing simulation step", "SIM: ")
        
        # Store previous position for progress check
        prev_position = self.robot_position.copy()
        
        # Get robot action based on planner type
        if self.use_mcts:
            # For MCTS, pass observation and environment to select_action
            observation = self.env.unwrapped.render()  # Get current observation
            robot_action = self.robot.select_action(self.env, observation, 
                                                   render=self.env.render_mode == 'human')
        else:
            # For traditional planners, use get_action method
            robot_action = self.robot.get_action(
                self.robot_position,
                self.robot_orientation,
                self.humans,
                self.goal_position
            )
        
        # Execute robot action
        action_names = {0: "turning left", 1: "turning right", 2: "moving forward"}
        self.logger.log(f"Robot {action_names.get(robot_action, str(robot_action))}", "SIM: ")
            
        # Step environment and handle both CustomEnv and standard Gym return values
        if self.use_mcts:
            obs, reward, done, info = self.env.step(robot_action)
        else:
            obs, reward, terminated, truncated, info = self.env.step(robot_action)
            done = terminated or truncated
        
        # Update human positions using ORCA
        for i, human in enumerate(self.humans):
            old_pos = human.position.copy()
            human.step(self.robot_position, self.goal_position, self.humans)
            # If new position is invalid, revert to old position
            if not self._is_valid_position(human.position):
                human.position = old_pos
            self.logger.log(f"Human {i} moved to {human.position}", "SIM: ")
            
        # Get new robot position and orientation from environment
        self.robot_position = np.array([self.miniworld_env.agent.pos[0], self.miniworld_env.agent.pos[2]])
        self.robot_orientation = self.miniworld_env.agent.dir
        
        # Render the environment if rendering is enabled
        if hasattr(self.env, 'render_mode') and self.env.render_mode == 'human':
            self.env.render()
        
        # Calculate progress using 2D distance
        progress = self._distance_2d(self.robot_position, prev_position)
        self.logger.log(f"Progress this step: {progress:.4f}", "SIM: ")
        
        # Update metrics with progress information
        self.metric_tracker.update(
            self.robot_position,
            self.robot_orientation,
            self.humans,
            self.goal_position,
            progress
        )
        
        # Check if goal reached using 2D distance
        goal_distance = self._distance_2d(self.robot_position, self.goal_position)
        if goal_distance <= self.goal_tolerance:
            # Verify that we actually moved to reach the goal
            if progress > 0.1:  # Must have moved at least 0.1 units
                self.logger.log("Goal reached!", "SIM: ")
                return True, reward
            else:
                self.logger.log("Robot at goal but didn't move enough", "SIM: ")
                
        return False, reward
        
    def run_simulation(self, num_episodes: int = 10):
        """Run multiple episodes of the simulation"""
        self.logger.log(f"Starting simulation with {num_episodes} episodes", "SIM: ")
        
        for episode in range(num_episodes):
            self.logger.log(f"Episode {episode + 1}/{num_episodes}", "SIM: ")
            try:
                metrics = self.run_episode()
                goal_status = "SUCCESS" if metrics['success'] else "FAILED"
                self.logger.log(
                    f"Episode {episode + 1} complete - "
                    f"Steps: {metrics['steps']}, "
                    f"Reward: {metrics['reward']:.2f}, "
                    f"Goal: {goal_status}",
                    "SIM: "
                )
            except Exception as e:
                self.logger.log(f"Error in episode {episode + 1}: {str(e)}", "SIM: ")
                import traceback
                self.logger.log(traceback.format_exc(), "SIM: ")
                continue
        
        # Create a logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Save metrics plot
        plot_file = os.path.join(log_dir, f"metrics_{self.planner_name.replace(' ', '_')}_{num_episodes}ep.png")
        self.metric_tracker.plot_metrics(save_path=plot_file)
        self.logger.log(f"Metrics plot saved to {plot_file}", "SIM: ")
        
    def _distance_2d(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate 2D distance between two points (ignoring y/height)"""
        # Ensure we're only using x and z coordinates
        if len(pos1) > 2:
            pos1 = np.array([pos1[0], pos1[2]])
        if len(pos2) > 2:
            pos2 = np.array([pos2[0], pos2[2]])
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

if __name__ == "__main__":
    # Create loggers for each algorithm
    dwa_logger = Logger(algorithm_name="dwa")
    rrt_logger = Logger(algorithm_name="rrt_star")
    scl_logger = Logger(algorithm_name="scl")
    
    try:
        # Create simulations with all planners
        dwa_sim = Simulation(planner=DWA(goal_tolerance=0.5), render=True, logger=dwa_logger)
        print("Running DWA simulation...")
        dwa_sim.run_simulation(num_episodes=5)
        
        rrt_sim = Simulation(planner=RRTStar(goal_tolerance=0.5), render=True, logger=rrt_logger)
        print("\nRunning RRT* simulation...")
        rrt_sim.run_simulation(num_episodes=5)
        
        scl_sim = Simulation(planner=SCL(goal_tolerance=0.5), render=True, logger=scl_logger)
        print("\nRunning SCL simulation...")
        scl_sim.run_simulation(num_episodes=5)
        
    finally:
        # Close all loggers
        for logger in [dwa_logger, rrt_logger, scl_logger]:
            logger.close() 