import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from miniworld.entity import Box
from logger import Logger

@dataclass
class Velocity:
    x: float
    z: float

class Human:
    def __init__(self, position: np.ndarray = None, radius: float = 0.3, max_speed: float = 0.2, logger: Logger = None):
        self.radius = radius
        self.max_speed = max_speed
        self.position = position if position is not None else np.zeros(2)  # x, z coordinates
        self.velocity = Velocity(0, 0)
        self.preferred_velocity = Velocity(0, 0)
        self.logger = logger if logger is not None else Logger()
        
        # Room dimensions from OneRoomS6
        self.room_size = 6  # OneRoomS6 uses size=6
        self.min_x = 0
        self.max_x = self.room_size
        self.min_z = 0
        self.max_z = self.room_size
        self.wall_margin = 0.3  # Safety margin from walls
        
        # Human visualization properties
        self.height = 1.7  # Average human height in meters
        self.width = 0.3   # Thin width
        self.depth = 0.3   # Thin depth
        self.color = (0, 0, 1)  # Blue color (RGB)
        
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within room bounds"""
        x, z = pos[0], pos[1]  # Note: y in miniworld is z in our 2D representation
        return (self.min_x + self.wall_margin <= x <= self.max_x - self.wall_margin and 
                self.min_z + self.wall_margin <= z <= self.max_z - self.wall_margin)
        
    def create_entity(self):
        # Create the box with the color
        entity = Box(
            color=self.color,
            size=[self.radius * 2, self.height, self.radius * 2]  # width, height, depth
        )
        
        # Set the position (x, y, z) where y is height (0 for ground level)
        entity.pos = np.array([self.position[0], 0, self.position[1]], dtype=np.float32)
        
        # Explicitly set color_vec for rendering
        entity.color_vec = np.array(self.color, dtype=np.float32)
        
        # Ensure direction is set to prevent NoneType errors during rendering
        if not hasattr(entity, 'dir') or entity.dir is None:
            entity.dir = 0.0  # Default direction (front-facing)
        
        return entity
        
    def reset(self, robot_pos=np.array([0., 0.]), goal_pos=np.array([10., 0.])):
        """Reset human position to be far from both robot and goal"""
        # Calculate the midpoint between robot and goal
        midpoint = (robot_pos + goal_pos) / 2
        
        # Calculate vector from midpoint to goal
        to_goal = goal_pos - midpoint
        goal_distance = np.linalg.norm(to_goal)
        
        # Try up to 10 times to find a valid position
        for _ in range(10):
            # Generate random position within room bounds
            x = np.random.uniform(self.min_x + self.wall_margin, self.max_x - self.wall_margin)
            z = np.random.uniform(self.min_z + self.wall_margin, self.max_z - self.wall_margin)
            pos = np.array([x, z])
            
            # Check distances
            dist_to_robot = np.linalg.norm(pos - robot_pos)
            dist_to_goal = np.linalg.norm(pos - goal_pos)
            
            # If position is far enough from both robot and goal, use it
            if dist_to_robot > goal_distance/2 and dist_to_goal > goal_distance/2:
                self.position = pos
                break
        else:
            # If we couldn't find an ideal position, place at a safe position near a wall
            self.position = np.array([
                np.random.uniform(self.min_x + self.wall_margin, self.max_x - self.wall_margin),
                self.min_z + self.wall_margin if np.random.random() > 0.5 else self.max_z - self.wall_margin
            ])
            
        # Reset velocities
        self.velocity = Velocity(0, 0)
        self.preferred_velocity = Velocity(
            np.random.uniform(-self.max_speed, self.max_speed),
            np.random.uniform(-self.max_speed, self.max_speed)
        )
        
    def _distance_2d(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate 2D distance between two points (ignoring y/height)"""
        # Ensure we're only using x and z coordinates
        if len(pos1) > 2:
            pos1 = np.array([pos1[0], pos1[2]])
        if len(pos2) > 2:
            pos2 = np.array([pos2[0], pos2[2]])
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
    def compute_orca_velocity(self, other_humans: List['Human'], robot_pos: np.ndarray, robot_orientation: float) -> Velocity:
        """Compute ORCA velocity considering both humans and robot"""
        self.logger.log(f"Computing ORCA velocity with {len(other_humans)} other humans", "HUMAN: ")
        
        # Create a simple robot agent for ORCA
        robot_agent = type('RobotAgent', (), {
            'position': robot_pos,
            'velocity': Velocity(0, 0),  # Robot's current velocity not considered
            'radius': 0.4  # MiniWorld's bot_radius
        })()
        
        # Combine all agents (humans and robot)
        agents = other_humans + [robot_agent]
        self.logger.log(f"Total agents to consider: {len(agents)}", "HUMAN: ")
        
        # Initialize ORCA constraints
        constraints = []
        
        # Add constraints for other agents (both humans and robot)
        for agent in agents:
            if agent == self:
                continue
                
            # Compute relative position and velocity
            relative_pos = agent.position - self.position
            relative_vel = np.array([agent.velocity.x - self.velocity.x, 
                                   agent.velocity.z - self.velocity.z])
            
            # Compute time to collision (only x,z distance)
            dist = self._distance_2d(agent.position, self.position)
            # Ensure we're using scalar values for radius
            agent_radius = getattr(agent, 'radius', 0.4)  # Default to bot_radius if not found
            collision_radius = self.radius + agent_radius
            self.logger.log(f"Distance to agent: {dist}, collision radius: {collision_radius}", "HUMAN: ")
            
            if dist < collision_radius:
                # Already in collision, push away
                direction = relative_pos / (dist + 1e-6)
                self.logger.log(f"Collision detected, pushing away with direction: {direction}", "HUMAN: ")
                return Velocity(
                    direction[0] * self.max_speed,
                    direction[1] * self.max_speed
                )
                
            # Add ORCA constraint
            time_to_collision = dist / (self._distance_2d(relative_vel, np.zeros(2)) + 1e-6)
            if time_to_collision > 0:
                constraints.append((relative_pos, relative_vel, time_to_collision))
                
        # Add wall constraints based on MiniWorld room dimensions
        # Left wall (min_z)
        if self.position[1] < self.min_z + self.wall_margin:
            constraints.append((np.array([0, 1]), np.zeros(2), self.wall_margin))
            
        # Right wall (max_z)
        if self.position[1] > self.max_z - self.wall_margin:
            constraints.append((np.array([0, -1]), np.zeros(2), self.wall_margin))
            
        # Front wall (max_x)
        if self.position[0] > self.max_x - self.wall_margin:
            constraints.append((np.array([-1, 0]), np.zeros(2), self.wall_margin))
            
        # Back wall (min_x)
        if self.position[0] < self.min_x + self.wall_margin:
            constraints.append((np.array([1, 0]), np.zeros(2), self.wall_margin))
            
        # If no constraints, use preferred velocity
        if not constraints:
            self.logger.log("No constraints, using preferred velocity", "HUMAN: ")
            return self.preferred_velocity
            
        # Find optimal velocity that satisfies all constraints
        preferred_direction = np.array([
            self.preferred_velocity.x,
            self.preferred_velocity.z
        ])
        
        # Normalize preferred direction
        pref_norm = np.linalg.norm(preferred_direction)
        if pref_norm > 0:
            preferred_direction = preferred_direction / pref_norm
            
        # Adjust direction based on constraints
        final_direction = preferred_direction.copy()
        
        # Adjust for wall constraints using MiniWorld bounds
        if self.position[1] < self.min_z + self.wall_margin:
            final_direction[1] = max(0, final_direction[1])
        if self.position[1] > self.max_z - self.wall_margin:
            final_direction[1] = min(0, final_direction[1])
        if self.position[0] > self.max_x - self.wall_margin:
            final_direction[0] = min(0, final_direction[0])
        if self.position[0] < self.min_x + self.wall_margin:
            final_direction[0] = max(0, final_direction[0])
            
        # Normalize final direction
        final_norm = np.linalg.norm(final_direction)
        if final_norm > 0:
            final_direction = final_direction / final_norm
            
        self.logger.log(f"Final direction: {final_direction}", "HUMAN: ")
        
        # Create velocity while ensuring it doesn't exceed max_speed
        new_velocity = Velocity(
            final_direction[0] * self.max_speed,
            final_direction[1] * self.max_speed
        )
        
        # Verify the resulting position would be valid
        new_position = np.array([
            self.position[0] + new_velocity.x,
            self.position[1] + new_velocity.z
        ])
        
        if not self._is_valid_position(new_position):
            # If resulting position would be invalid, stop movement
            return Velocity(0, 0)
            
        return new_velocity
        
    def update_position(self, humans: List['Human'], robot_pos: np.ndarray, robot_orientation: float):
        self.logger.log(f"Updating position for human at {self.position}", "HUMAN: ")
        self.logger.log(f"Current velocity: ({self.velocity.x}, {self.velocity.z})", "HUMAN: ")
        self.logger.log(f"Preferred velocity: ({self.preferred_velocity.x}, {self.preferred_velocity.z})", "HUMAN: ")
        
        # Get ORCA velocity
        orca_velocity = self.compute_orca_velocity(humans, robot_pos, robot_orientation)
        self.logger.log(f"ORCA velocity: ({orca_velocity.x}, {orca_velocity.z})", "HUMAN: ")
        
        # Calculate new position
        new_position = np.array([
            self.position[0] + orca_velocity.x,
            self.position[1] + orca_velocity.z
        ])
        
        # Only update if new position is valid
        if self._is_valid_position(new_position):
            self.position = new_position
            self.velocity = orca_velocity
        else:
            # If position would be invalid, stop movement
            self.velocity = Velocity(0, 0)
            
        self.logger.log(f"Final position: {self.position}", "HUMAN: ")
        
    def step(self, robot_position: np.ndarray, goal_position: np.ndarray, humans: List['Human']):
        """Update human position based on current state using ORCA"""
        # Get other humans (excluding self)
        other_humans = [h for h in humans if h != self]
        
        # Update position using ORCA with bounds checking
        self.update_position(other_humans, robot_position, 0.0)
        
        # Log the update
        self.logger.log(f"Human moved to {self.position}", "HUMAN: ") 