import numpy as np
from typing import List, Tuple, Optional
from logger import Logger
import math

class Node:
    def __init__(self, position: np.ndarray, orientation: float = 0.0, parent: Optional['Node'] = None):
        self.position = position
        self.orientation = orientation  # Track orientation in radians
        self.parent = parent
        self.cost = 0.0 if parent is None else parent.cost + self._action_cost(parent)
        self.children = []
        
    def _action_cost(self, parent: 'Node') -> float:
        """Calculate cost of action from parent to this node"""
        # Higher cost for turns to prefer straight paths when possible
        angle_diff = abs(self._normalize_angle(self.orientation - parent.orientation))
        if angle_diff > 0.01:  # If turning
            return 1.2 * np.linalg.norm(self.position - parent.position) + 0.2 * angle_diff
        return np.linalg.norm(self.position - parent.position)
        
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def __str__(self):
        return f"Node(pos={self.position}, cost={self.cost})"

class RRTStar:
    def __init__(
        self,
        goal_tolerance: float = 0.5,
        max_iterations: int = 1000,
        step_size: float = 0.15,  # MiniWorld's forward_step
        search_radius: float = 1.0,
        radius: float = 0.4  # MiniWorld's bot_radius
    ):
        self.goal_tolerance = goal_tolerance
        self.max_iterations = max_iterations
        self.step_size = step_size  # MiniWorld's forward_step
        self.search_radius = search_radius
        self.radius = radius  # MiniWorld's bot_radius
        self.nodes: List[Node] = []
        
        # Room dimensions from OneRoomS6
        self.room_size = 6  # OneRoomS6 uses size=6
        self.min_x = 0
        self.max_x = self.room_size
        self.min_z = 0
        self.max_z = self.room_size
        self.wall_margin = self.radius * 0.75  # Scale with robot radius
        
        # RRT* parameters
        self.max_nodes = 1000  # Maximum number of nodes in the tree
        self.goal_bias = 0.1   # Probability to sample goal directly
        
        # Movement parameters from MiniWorld
        self.forward_step = 0.15  # MiniWorld's forward_step
        self.turn_step = np.deg2rad(15)  # MiniWorld's turn_step (15 degrees)
        self.min_progress = self.forward_step * 0.1  # 10% of forward step
        
        # Safety parameters scaled with robot radius
        self.safety_margin = self.radius * 1.5
        self.collision_margin = self.radius * 2.0  # For collision checking between agents
        
        # Tree storage
        self.start_node: Optional[Node] = None
        self.goal_node: Optional[Node] = None
        
        # Path storage
        self.current_path: List[np.ndarray] = []
        self.current_path_index = 0
        
        # Initialize logger
        self.logger = Logger()
        
    def reset(self):
        """Reset the RRT* planner"""
        self.nodes = []
        self.start_node = None
        self.goal_node = None
        self.current_path = []
        self.current_path_index = 0
        self.logger.log("RRT* reset", "RRT*: ")
        
    def _distance_2d(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate 2D Euclidean distance between two points"""
        return np.linalg.norm(pos1[:2] - pos2[:2])
        
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if a position is within room bounds"""
        x, z = pos[0], pos[1]  # Note: y in miniworld is z in our 2D representation
        return (self.min_x + self.wall_margin <= x <= self.max_x - self.wall_margin and 
                self.min_z + self.wall_margin <= z <= self.max_z - self.wall_margin)
        
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from one position towards another with maximum step size"""
        direction = to_pos - from_pos
        distance = self._distance_2d(from_pos, to_pos)
        if distance <= self.forward_step:  # Use forward_step as max step size
            return to_pos
        return from_pos + (direction / distance) * self.forward_step
        
    def _find_nearest_node(self, pos: np.ndarray) -> Node:
        """Find the nearest node in the tree to a given position"""
        nearest = self.nodes[0]
        min_dist = self._distance_2d(pos, nearest.position)
        
        for node in self.nodes[1:]:
            dist = self._distance_2d(pos, node.position)
            if dist < min_dist:
                min_dist = dist
                nearest = node
                
        return nearest
        
    def _find_near_nodes(self, pos: np.ndarray) -> List[Node]:
        """Find all nodes within the neighbor radius"""
        return [node for node in self.nodes 
                if self._distance_2d(pos, node.position) <= self.search_radius]
                
    def _is_collision_free(self, pos: np.ndarray, humans: List['Human']) -> bool:
        """Check if a position is within room bounds"""
        # Only check room bounds, ignore humans
        return self._is_valid_position(pos)
        
    def _rewire(self, new_node: Node, neighbors: List[Node]):
        """Rewire the tree to optimize paths through the new node"""
        for neighbor in neighbors:
            if neighbor == new_node.parent:
                continue
                
            # Calculate potential new cost
            new_cost = new_node.cost + self._distance_2d(new_node.position, neighbor.position)
            
            # If new cost is better, update parent
            if new_cost < neighbor.cost:
                # Remove from old parent's children
                if neighbor.parent:
                    neighbor.parent.children.remove(neighbor)
                    
                # Update parent and cost
                neighbor.parent = new_node
                neighbor.cost = new_cost
                
                # Add to new parent's children
                new_node.children.append(neighbor)
                
                # Recursively update children's costs
                self._update_children_costs(neighbor)
                
    def _update_children_costs(self, node: Node):
        """Update costs of all children after rewiring"""
        for child in node.children:
            child.cost = node.cost + self._distance_2d(node.position, child.position)
            self._update_children_costs(child)
            
    def _find_best_parent(self, new_pos: np.ndarray, neighbors: List[Node]) -> Tuple[Node, float]:
        """Find the best parent node for a new position"""
        best_parent = None
        best_cost = float('inf')
        
        for neighbor in neighbors:
            # Calculate cost through this neighbor
            cost = neighbor.cost + self._distance_2d(neighbor.position, new_pos)
            
            # Check if this is the best parent so far
            if cost < best_cost:
                best_parent = neighbor
                best_cost = cost
                
        return best_parent, best_cost
        
    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """Extract path from start to goal node"""
        path = []
        current = goal_node
        while current:
            path.append(current.position)
            current = current.parent
        return list(reversed(path))
        
    def plan(self, start_pos: np.ndarray, goal_pos: np.ndarray, humans: List['Human']) -> List[np.ndarray]:
        """Plan a path from start to goal using RRT*"""
        self.reset()
        
        # Initialize with start node including orientation
        self.start_node = Node(start_pos, orientation=0.0)  # Assume initial orientation is 0
        self.goal_node = Node(goal_pos)
        self.nodes = [self.start_node]
        
        # Planning parameters
        min_step = self.forward_step * 1.5
        max_step = self.forward_step * 3
        goal_sample_rate = 0.3
        
        for i in range(self.max_iterations):
            # Sample with bias
            if np.random.random() < goal_sample_rate:
                sample_pos = goal_pos + np.random.normal(0, self.forward_step, size=2)
            else:
                sample_pos = self._get_random_sample()
            
            # Find nearest node
            nearest_node = self._find_nearest_node(sample_pos)
            if nearest_node is None:
                continue
            
            # Try different actions from nearest node
            best_node = None
            min_cost = float('inf')
            
            # Try forward movement
            forward_pos = nearest_node.position + np.array([
                np.cos(nearest_node.orientation) * self.forward_step,
                np.sin(nearest_node.orientation) * self.forward_step
            ])
            
            if self._is_valid_position(forward_pos):
                forward_node = Node(forward_pos, nearest_node.orientation, nearest_node)
                dist_to_sample = np.linalg.norm(forward_pos - sample_pos)
                if dist_to_sample < min_cost:
                    best_node = forward_node
                    min_cost = dist_to_sample
            
            # Try turning left and right
            for turn_direction in [-1, 1]:
                new_orientation = nearest_node.orientation + turn_direction * np.deg2rad(self.turn_step)
                turn_node = Node(nearest_node.position.copy(), new_orientation, nearest_node)
                dist_to_sample = np.linalg.norm(nearest_node.position - sample_pos)
                if dist_to_sample < min_cost:
                    best_node = turn_node
                    min_cost = dist_to_sample
            
            if best_node is None:
                continue
            
            # Add the best node
            self.nodes.append(best_node)
            
            # Check if we can reach goal
            if self._distance_2d(best_node.position, goal_pos) < self.forward_step * 2:
                if self._is_valid_position(goal_pos):
                    # Calculate orientation to goal
                    dx = goal_pos[0] - best_node.position[0]
                    dy = goal_pos[1] - best_node.position[1]
                    goal_orientation = np.arctan2(dy, dx)
                    goal_node = Node(goal_pos, goal_orientation, best_node)
                    self.nodes.append(goal_node)
                    return self._post_process_path(self._extract_path(goal_node))
        
        # If no path to goal found, return path to closest node
        closest_node = min(self.nodes, key=lambda n: self._distance_2d(n.position, goal_pos))
        return self._post_process_path(self._extract_path(closest_node))
        
    def _post_process_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Post-process the path to create smoother transitions"""
        if len(path) <= 2:
            return path
            
        processed_path = [path[0]]
        current_pos = path[0]
        
        for i in range(1, len(path)):
            direction = path[i] - current_pos
            distance = np.linalg.norm(direction)
            
            # Skip points that are too close
            if distance < self.forward_step:
                continue
            
            # For longer segments, add intermediate points considering orientation
            if distance > self.forward_step * 2:
                direction = direction / distance
                num_points = int(distance / (self.forward_step * 1.5))
                for j in range(1, num_points):
                    point = current_pos + direction * (j * self.forward_step * 1.5)
                    if self._is_valid_position(point):
                        processed_path.append(point)
            
            processed_path.append(path[i])
            current_pos = path[i]
        
        return processed_path
        
    def get_action(
        self,
        robot_position: np.ndarray,
        robot_orientation: float,
        humans: List['Human'],
        goal_position: np.ndarray
    ) -> int:
        """Get the next action for the robot using RRT*"""
        # Calculate distance to goal
        goal_dist = self._distance_2d(robot_position, goal_position)
        self.logger.log(f"Distance to goal: {goal_dist:.4f}", "RRT*: ")
        
        # Direct goal approach when very close
        if goal_dist < self.forward_step * 1.5:
            dx = goal_position[0] - robot_position[0]
            dy = goal_position[1] - robot_position[1]
            goal_angle = np.arctan2(dy, dx)
            angle_diff = self._normalize_angle(goal_angle - robot_orientation)
            
            # More lenient angle threshold for goal approach
            if abs(angle_diff) < np.deg2rad(self.turn_step * 2):
                forward_pos = robot_position + np.array([
                    np.cos(robot_orientation) * self.forward_step,
                    np.sin(robot_orientation) * self.forward_step
                ])
                if self._is_valid_position(forward_pos):
                    return 2  # move_forward
            return 0 if angle_diff > 0 else 1  # turn towards goal
        
        # Plan or replan path if needed
        need_replanning = (
            not self.current_path or 
            self.current_path_index >= len(self.current_path) or
            self._distance_2d(robot_position, self.current_path[self.current_path_index]) > self.forward_step * 2
        )
        
        if need_replanning:
            self.logger.log("Planning new path", "RRT*: ")
            path = self.plan(robot_position, goal_position, humans)
            if not path:
                # Simple obstacle avoidance if no path found
                forward_pos = robot_position + np.array([
                    np.cos(robot_orientation) * self.forward_step,
                    np.sin(robot_orientation) * self.forward_step
                ])
                if self._is_valid_position(forward_pos):
                    return 2  # move_forward if clear
                return 0  # turn left to explore
                
            self.current_path = path
            self.current_path_index = 0
        
        # Skip waypoints that are too close
        while (self.current_path_index < len(self.current_path) and 
               self._distance_2d(robot_position, self.current_path[self.current_path_index]) < self.forward_step):
            self.current_path_index += 1
        
        # If we've run out of waypoints, replan
        if self.current_path_index >= len(self.current_path):
            self.current_path = []
            self.current_path_index = 0
            # Turn in place to explore
            return 0
        
        # Follow current waypoint
        next_waypoint = self.current_path[self.current_path_index]
        dx = next_waypoint[0] - robot_position[0]
        dy = next_waypoint[1] - robot_position[1]
        waypoint_angle = np.arctan2(dy, dx)
        angle_diff = self._normalize_angle(waypoint_angle - robot_orientation)
        
        # Log for debugging
        waypoint_dist = self._distance_2d(robot_position, next_waypoint)
        self.logger.log(f"Distance to waypoint: {waypoint_dist:.4f}", "RRT*: ")
        self.logger.log(f"Angle difference: {np.rad2deg(angle_diff):.2f} degrees", "RRT*: ")
        
        # Handle large angle differences
        if abs(angle_diff) > np.pi/2:  # More than 90 degrees
            # Just turn, don't try to move forward
            return 0 if angle_diff > 0 else 1
        
        # More lenient forward movement when roughly aligned
        if abs(angle_diff) < np.deg2rad(self.turn_step * 2):
            forward_pos = robot_position + np.array([
                np.cos(robot_orientation) * self.forward_step,
                np.sin(robot_orientation) * self.forward_step
            ])
            if self._is_valid_position(forward_pos):
                return 2  # move_forward
        
        # Turn towards waypoint
        return 0 if angle_diff > 0 else 1
        
    def simulate_trajectory(self, action: int, position: np.ndarray, orientation: float) -> List[np.ndarray]:
        """Simulate a trajectory for a given action using fixed step sizes"""
        trajectory = []
        current_pos = position.copy()
        current_ori = orientation
        
        # Single step simulation since we can only do one action at a time
        if action == 0:  # turn left
            current_ori += np.deg2rad(self.turn_step)
        elif action == 1:  # turn right
            current_ori -= np.deg2rad(self.turn_step)
        else:  # move forward
            dx = np.cos(current_ori) * self.forward_step
            dy = np.sin(current_ori) * self.forward_step
            current_pos = current_pos + np.array([dx, dy])
            
            if not self._is_valid_position(current_pos):
                return []
                
        trajectory.append(current_pos)
        return trajectory
        
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def _get_random_sample(self) -> np.ndarray:
        """Sample a random position within the room"""
        x = np.random.uniform(self.min_x + self.wall_margin, self.max_x - self.wall_margin)
        z = np.random.uniform(self.min_z + self.wall_margin, self.max_z - self.wall_margin)
        return np.array([x, z])
        
    def _get_exploration_sample(self) -> np.ndarray:
        """Sample in unexplored regions using Voronoi bias"""
        best_pos = None
        max_clearance = -float('inf')
        
        for _ in range(10):  # Try 10 random samples
            sample = np.array([
                np.random.uniform(self.min_x + self.wall_margin, self.max_x - self.wall_margin),
                np.random.uniform(self.min_z + self.wall_margin, self.max_z - self.wall_margin)
            ])
            
            # Calculate minimum distance to existing nodes
            min_dist = min(self._distance_2d(sample, node.position) for node in self.nodes)
            
            if min_dist > max_clearance:
                max_clearance = min_dist
                best_pos = sample
                
        return best_pos
        
    def _get_focused_sample(self, current_best: Node, goal_pos: np.ndarray) -> np.ndarray:
        """Sample around the current best path"""
        if np.random.random() < 0.5:
            # Sample around current best node
            angle = np.random.uniform(-np.pi, np.pi)
            distance = np.random.uniform(0, self.forward_step * 3)
            return current_best.position + np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance
            ])
        else:
            # Sample along the direction to goal
            direction = goal_pos - current_best.position
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            distance = np.random.uniform(0, self._distance_2d(current_best.position, goal_pos))
            return current_best.position + direction * distance
            
    def _find_best_parent_with_clearance(self, pos: np.ndarray, neighbors: List[Node], humans: List['Human']) -> Tuple[Optional[Node], float]:
        """Find best parent considering path cost"""
        best_parent = None
        best_cost = float('inf')
        
        for neighbor in neighbors:
            # Check if connection is collision-free (only room bounds)
            collision_free = True
            for t in np.linspace(0, 1, num=5):
                test_pos = neighbor.position + t * (pos - neighbor.position)
                if not self._is_valid_position(test_pos):
                    collision_free = False
                    break
                    
            if not collision_free:
                continue
                
            # Calculate cost (no human clearance consideration)
            cost = neighbor.cost + self._distance_2d(neighbor.position, pos)
            
            if cost < best_cost:
                best_parent = neighbor
                best_cost = cost
                
        return best_parent, best_cost
        
    def _smooth_path(self, path: List[np.ndarray], humans: List['Human']) -> List[np.ndarray]:
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
            
        smoothed_path = [path[0]]
        current_index = 0
        
        while current_index < len(path) - 1:
            # Try to connect current point to furthest possible point
            for i in range(len(path) - 1, current_index, -1):
                # Check if direct path is possible
                can_connect = True
                test_pos = smoothed_path[-1]
                target_pos = path[i]
                
                # Check points along the line
                for t in np.linspace(0, 1, num=10):
                    interp_pos = test_pos + t * (target_pos - test_pos)
                    if not self._is_collision_free(interp_pos, humans):
                        can_connect = False
                        break
                        
                if can_connect:
                    smoothed_path.append(path[i])
                    current_index = i
                    break
            else:
                # If no connection possible, add next point
                current_index += 1
                smoothed_path.append(path[current_index])
                
        return smoothed_path 