#!/usr/bin/env python3

import os
import sys
import random
import termios
import tty
import heapq
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from colors import Colors

class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

class SnakeGame:
    def __init__(self, width: int = 30, height: int = 30):
        self.width = width
        self.height = height
        self.snake: List[Tuple[int, int]] = [(height // 2, width // 2)]
        self.direction = Direction.RIGHT
        self.previous_direction = Direction.RIGHT  # Track previous direction for change detection
        self.food: Optional[Tuple[int, int]] = None
        self.score = 0
        self.game_over = False
        self.steps_since_last_food = 0
        self._spawn_food()
    
    def _spawn_food(self):
        while True:
            food_pos = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if food_pos not in self.snake:
                self.food = food_pos
                break
    
    def reset(self):
        """Reset the game for a new episode"""
        self.snake = [(self.height // 2, self.width // 2)]
        self.direction = Direction.RIGHT
        self.previous_direction = Direction.RIGHT  # Reset previous direction
        self.score = 0
        self.game_over = False
        self.steps_since_last_food = 0
        self._spawn_food()
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get the current state representation for DQN - enhanced with spatial awareness"""
        head = self.snake[0]
        food = self.food
        
        # Current direction (one-hot encoded)
        moving_up = 1.0 if self.direction == Direction.UP else 0.0
        moving_down = 1.0 if self.direction == Direction.DOWN else 0.0
        moving_left = 1.0 if self.direction == Direction.LEFT else 0.0
        moving_right = 1.0 if self.direction == Direction.RIGHT else 0.0
        
        # Danger detection based on current heading
        danger_straight = self._is_dangerous_ahead()
        danger_left = self._is_dangerous_left()
        danger_right = self._is_dangerous_right()
        
        # Food direction relative to current heading
        food_left = self._is_food_left()
        food_right = self._is_food_right()
        food_up = self._is_food_up()
        food_down = self._is_food_down()
        
        # Distance to walls in each direction
        wall_distance_up = self._get_wall_distance('up')
        wall_distance_down = self._get_wall_distance('down')
        wall_distance_left = self._get_wall_distance('left')
        wall_distance_right = self._get_wall_distance('right')
        
        # Distance to body segments in each direction
        body_distance_up = self._get_body_distance('up')
        body_distance_down = self._get_body_distance('down')
        body_distance_left = self._get_body_distance('left')
        body_distance_right = self._get_body_distance('right')
        
        # Safe moves count (how many directions don't immediately cause death)
        safe_moves = self._count_safe_moves()
        
        # Lookahead: whether each possible move leads to a dead-end in 2-3 steps
        dead_end_up = self._is_dead_end_in_direction('up')
        dead_end_down = self._is_dead_end_in_direction('down')
        dead_end_left = self._is_dead_end_in_direction('left')
        dead_end_right = self._is_dead_end_in_direction('right')
        
        # Available space accessible from each direction
        space_up = self._get_available_space(head, 'up')
        space_down = self._get_available_space(head, 'down')
        space_left = self._get_available_space(head, 'left')
        space_right = self._get_available_space(head, 'right')
        
        # Pathfinding-based quality scores for each direction
        path_quality_up = self._get_path_quality('up')
        path_quality_down = self._get_path_quality('down')
        path_quality_left = self._get_path_quality('left')
        path_quality_right = self._get_path_quality('right')
        
        # Additional context
        snake_length = len(self.snake) / 20.0  # Normalized length
        
        state = np.array([
            danger_straight,
            danger_left,
            danger_right,
            moving_left,
            moving_right,
            moving_up,
            moving_down,
            food_left,
            food_right,
            food_up,
            food_down,
            wall_distance_up,
            wall_distance_down,
            wall_distance_left,
            wall_distance_right,
            body_distance_up,
            body_distance_down,
            body_distance_left,
            body_distance_right,
            safe_moves,
            dead_end_up,
            dead_end_down,
            dead_end_left,
            dead_end_right,
            space_up,
            space_down,
            space_left,
            space_right,
            path_quality_up,
            path_quality_down,
            path_quality_left,
            path_quality_right,
            snake_length
        ], dtype=np.float32)
        
        return state
    
    def _is_dangerous(self, pos: Tuple[int, int]) -> float:
        """Check if a position is dangerous (wall or body)"""
        row, col = pos
        # Check wall collision
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return 1.0
        # Check body collision
        if pos in self.snake:
            return 1.0
        return 0.0
    
    def _get_available_space(self, head: Tuple[int, int], direction: str) -> float:
        """Get the number of free cells available in a given direction"""
        head_row, head_col = head
        free_cells = 0
        
        if direction == 'up':
            for row in range(head_row - 1, -1, -1):
                if (row, head_col) not in self.snake and row >= 0:
                    free_cells += 1
                else:
                    break
        elif direction == 'down':
            for row in range(head_row + 1, self.height):
                if (row, head_col) not in self.snake and row < self.height:
                    free_cells += 1
                else:
                    break
        elif direction == 'left':
            for col in range(head_col - 1, -1, -1):
                if (head_row, col) not in self.snake and col >= 0:
                    free_cells += 1
                else:
                    break
        elif direction == 'right':
            for col in range(head_col + 1, self.width):
                if (head_row, col) not in self.snake and col < self.width:
                    free_cells += 1
                else:
                    break
        
        return free_cells / 10.0  # Normalize
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the game given an action"""
        # Store old state for reward calculation
        old_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        food_eaten = False
        
        # Store previous direction for change detection
        self.previous_direction = self.direction
        
        # Convert action to direction
        if action == 0 and self.direction != Direction.DOWN:  # UP
            self.direction = Direction.UP
        elif action == 1 and self.direction != Direction.UP:  # DOWN
            self.direction = Direction.DOWN
        elif action == 2 and self.direction != Direction.RIGHT:  # LEFT
            self.direction = Direction.LEFT
        elif action == 3 and self.direction != Direction.LEFT:  # RIGHT
            self.direction = Direction.RIGHT
        
        # Move snake
        head_row, head_col = self.snake[0]
        dir_row, dir_col = self.direction.value
        new_head = (head_row + dir_row, head_col + dir_col)
        
        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= self.height or 
            new_head[1] < 0 or new_head[1] >= self.width):
            self.game_over = True
        elif new_head in self.snake:
            self.game_over = True
        else:
            self.snake.insert(0, new_head)
            
            # Check food collision
            if new_head == self.food:
                self.score += 1
                self.steps_since_last_food = 0
                food_eaten = True
                self._spawn_food()
            else:
                self.snake.pop()
                self.steps_since_last_food += 1
        
        # Calculate reward
        reward = self._calculate_reward(old_distance, food_eaten)
        
        # Get new state
        new_state = self.get_state()
        
        # Check if episode should end (max steps or game over)
        done = self.game_over or self.steps_since_last_food >= 500
        
        info = {
            'score': self.score,
            'snake_length': len(self.snake),
            'steps_since_food': self.steps_since_last_food
        }
        
        return new_state, reward, done, info
    
    def _calculate_reward(self, old_distance: int, food_eaten: bool) -> float:
        """Calculate reward with clear, non-overlapping signals."""
        if self.game_over:
            return -10.0

        if food_eaten:
            return 10.0

        new_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        if new_distance < old_distance:
            return 0.3
        elif new_distance > old_distance:
            return -0.3

        return 0.0
    
    def _clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _render(self):
        self._clear_screen()
        
        # Print title
        title = "🐍 SNAKE GAME 🐍"
        print(f"{Colors.CYAN}{Colors.BOLD}{title:^{self.width * 2 + 2}}{Colors.RESET}")
        print()
        
        # Print colored header with better aspect ratio
        print(f"{Colors.CYAN}{Colors.BOLD}{'═' * (self.width * 2 + 2)}{Colors.RESET}")
        
        for row in range(self.height):
            print("║", end="")
            for col in range(self.width):
                pos = (row, col)
                if pos == self.snake[0]:
                    print(f"{Colors.GREEN}██{Colors.RESET}", end="")  # Snake head in green - 2 chars wide
                elif pos in self.snake[1:]:
                    print(f"{Colors.BLUE}▓▓{Colors.RESET}", end="")  # Snake body in blue
                elif pos == self.food:
                    print(f"{Colors.RED}◆◆{Colors.RESET}", end="")  # Food in red - 2 chars wide
                else:
                    # Create a subtle grid pattern for empty spaces
                    if (row + col) % 4 == 0:
                        print(f"{Colors.WHITE}··{Colors.RESET}", end="")
                    else:
                        print("  ", end="")  # Empty space - 2 chars wide
            print("║")
        
        print(f"{Colors.CYAN}{Colors.BOLD}{'═' * (self.width * 2 + 2)}{Colors.RESET}")
        print()
        print(f"{Colors.BLUE}Score: {Colors.BOLD}{self.score}{Colors.RESET} | Length: {len(self.snake)}")
        print(f"{Colors.MAGENTA}Controls: W=Up, A=Left, S=Down, D=Right, Q=Quit{Colors.RESET}")
        
        if self.game_over:
            print()
            print(f"{Colors.RED}{Colors.BOLD}{'=' * 40}")
            print("GAME OVER!")
            print(f"{'=' * 40}{Colors.RESET}")
            print(f"{Colors.YELLOW}Final Score: {Colors.BOLD}{self.score}{Colors.RESET}")
            print(f"{Colors.YELLOW}Snake Length: {Colors.BOLD}{len(self.snake)}{Colors.RESET}")
            print(f"\n{Colors.MAGENTA}Press any key to exit...{Colors.RESET}")
    
    def _get_key(self) -> str:
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return key
        except (termios.error, OSError):
            # Fallback for non-interactive terminals
            return input("Enter move (w/a/s/d/q): ").lower().strip()[:1] or 'q'
    
    def _move_snake(self):
        if self.game_over:
            return
        
        head_row, head_col = self.snake[0]
        dir_row, dir_col = self.direction.value
        new_head = (head_row + dir_row, head_col + dir_col)
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.height or 
            new_head[1] < 0 or new_head[1] >= self.width):
            self.game_over = True
            return
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        self.snake.insert(0, new_head)
        
        # Check food collision
        if new_head == self.food:
            self.score += 1
            self._spawn_food()
        else:
            self.snake.pop()
    
    def _handle_input(self, key: str) -> bool:
        if key == 'q':
            return False
        elif key == 'w' and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif key == 'a' and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif key == 's' and self.direction != Direction.UP:
            self.direction = Direction.DOWN
        elif key == 'd' and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT
        
        return True
    
    def play(self):
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 60)
        print("🐍 WELCOME TO SNAKE GAME! 🐍")
        print("=" * 60)
        print(f"{Colors.RESET}")
        print(f"{Colors.YELLOW}Use WASD to control the snake, Q to quit{Colors.RESET}")
        print(f"{Colors.MAGENTA}Press any key to start...{Colors.RESET}")
        self._get_key()
        
        while True:
            self._render()
            
            if self.game_over:
                self._get_key()
                break
            
            key = self._get_key()
            if not self._handle_input(key):
                break
            
            self._move_snake()

    def _is_dangerous_ahead(self) -> float:
        """Check if moving straight ahead is dangerous"""
        head = self.snake[0]
        if self.direction == Direction.UP:
            next_pos = (head[0] - 1, head[1])
        elif self.direction == Direction.DOWN:
            next_pos = (head[0] + 1, head[1])
        elif self.direction == Direction.LEFT:
            next_pos = (head[0], head[1] - 1)
        else:  # RIGHT
            next_pos = (head[0], head[1] + 1)
        
        return self._is_dangerous(next_pos)
    
    def _is_dangerous_left(self) -> float:
        """Check if turning left is dangerous"""
        head = self.snake[0]
        if self.direction == Direction.UP:
            next_pos = (head[0], head[1] - 1)
        elif self.direction == Direction.DOWN:
            next_pos = (head[0], head[1] + 1)
        elif self.direction == Direction.LEFT:
            next_pos = (head[0] + 1, head[1])
        else:  # RIGHT
            next_pos = (head[0] - 1, head[1])
        
        return self._is_dangerous(next_pos)
    
    def _is_dangerous_right(self) -> float:
        """Check if turning right is dangerous"""
        head = self.snake[0]
        if self.direction == Direction.UP:
            next_pos = (head[0], head[1] + 1)
        elif self.direction == Direction.DOWN:
            next_pos = (head[0], head[1] - 1)
        elif self.direction == Direction.LEFT:
            next_pos = (head[0] - 1, head[1])
        else:  # RIGHT
            next_pos = (head[0] + 1, head[1])
        
        return self._is_dangerous(next_pos)
    
    def _is_food_left(self) -> float:
        """Check if food is to the left of current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[1] < head[1] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[1] > head[1] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[0] > head[0] else 0.0
        else:  # RIGHT
            return 1.0 if food[0] < head[0] else 0.0
    
    def _is_food_right(self) -> float:
        """Check if food is to the right of current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[1] > head[1] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[1] < head[1] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[0] < head[0] else 0.0
        else:  # RIGHT
            return 1.0 if food[0] > head[0] else 0.0
    
    def _is_food_up(self) -> float:
        """Check if food is ahead of current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[0] < head[0] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[0] > head[0] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[1] < head[1] else 0.0
        else:  # RIGHT
            return 1.0 if food[1] > head[1] else 0.0
    
    def _is_food_down(self) -> float:
        """Check if food is behind current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[0] > head[0] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[0] < head[0] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[1] > head[1] else 0.0
        else:  # RIGHT
            return 1.0 if food[1] < head[1] else 0.0

    def _get_wall_distance(self, direction: str) -> float:
        """Get distance to wall in a specific direction"""
        head_row, head_col = self.snake[0]
        
        if direction == 'up':
            return head_row / 10.0  # Normalize by max possible distance
        elif direction == 'down':
            return (self.height - 1 - head_row) / 10.0
        elif direction == 'left':
            return head_col / 10.0
        else:  # right
            return (self.width - 1 - head_col) / 10.0
    
    def _get_body_distance(self, direction: str) -> float:
        """Get distance to nearest body segment in a specific direction"""
        head_row, head_col = self.snake[0]
        distance = 0
        
        if direction == 'up':
            for row in range(head_row - 1, -1, -1):
                if (row, head_col) in self.snake[1:]:  # Exclude head
                    return distance / 10.0  # Normalize
                distance += 1
        elif direction == 'down':
            for row in range(head_row + 1, self.height):
                if (row, head_col) in self.snake[1:]:
                    return distance / 10.0
                distance += 1
        elif direction == 'left':
            for col in range(head_col - 1, -1, -1):
                if (head_row, col) in self.snake[1:]:
                    return distance / 10.0
                distance += 1
        else:  # right
            for col in range(head_col + 1, self.width):
                if (head_row, col) in self.snake[1:]:
                    return distance / 10.0
                distance += 1
        
        return 1.0  # No body found in this direction
    
    def _count_safe_moves(self) -> float:
        """Count how many directions don't immediately cause death"""
        safe_count = 0
        
        # Check each direction
        for direction in ['up', 'down', 'left', 'right']:
            if not self._is_immediate_death(direction):
                safe_count += 1
        
        return safe_count / 4.0  # Normalize by total possible directions
    
    def _is_immediate_death(self, direction: str) -> bool:
        """Check if moving in a direction causes immediate death"""
        head_row, head_col = self.snake[0]
        
        if direction == 'up':
            next_pos = (head_row - 1, head_col)
        elif direction == 'down':
            next_pos = (head_row + 1, head_col)
        elif direction == 'left':
            next_pos = (head_row, head_col - 1)
        else:  # right
            next_pos = (head_row, head_col + 1)
        
        # Check wall collision
        if (next_pos[0] < 0 or next_pos[0] >= self.height or 
            next_pos[1] < 0 or next_pos[1] >= self.width):
            return True
        
        # Check body collision
        if next_pos in self.snake:
            return True
        
        return False
    
    def _is_dead_end_in_direction(self, direction: str) -> float:
        """Check if a direction leads to a dead end (only one safe direction)"""
        head_row, head_col = self.snake[0]
        
        # Calculate next position
        if direction == 'up':
            next_pos = (head_row - 1, head_col)
        elif direction == 'down':
            next_pos = (head_row + 1, head_col)
        elif direction == 'left':
            next_pos = (head_row, head_col - 1)
        else:  # right
            next_pos = (head_row, head_col + 1)
        
        # Check if first step is safe
        if (next_pos[0] < 0 or next_pos[0] >= self.height or 
            next_pos[1] < 0 or next_pos[1] >= self.width or
            next_pos in self.snake):
            return 1.0  # Immediate dead end
        
        # Count safe directions from this position
        safe_directions = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            check_row = next_pos[0] + dr
            check_col = next_pos[1] + dc
            
            if (check_row >= 0 and check_row < self.height and 
                check_col >= 0 and check_col < self.width and
                (check_row, check_col) not in self.snake):
                safe_directions += 1
        
        # Dead end if only 0 or 1 safe directions
        return 1.0 if safe_directions <= 1 else 0.0

    def _a_star_pathfinding(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding algorithm to find optimal path to goal"""
        if start == goal:
            return [start]
        
        open_set = []
        counter = 0  # Tiebreaker so tuples with equal f_score never compare positions
        heapq.heappush(open_set, (0, counter, start))
        came_from = {}
        in_open = {start}
        
        g_score = {start: 0}
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            in_open.discard(current)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if (neighbor[0] < 0 or neighbor[0] >= self.height or 
                    neighbor[1] < 0 or neighbor[1] >= self.width or
                    neighbor in self.snake):
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self._heuristic(neighbor, goal)
                    
                    if neighbor not in in_open:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor))
                        in_open.add(neighbor)
        
        return None  # No path found
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for A*"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from A* results"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _get_path_quality(self, direction: str) -> float:
        """Get quality score for a direction based on pathfinding analysis"""
        head = self.snake[0]
        
        # Calculate next position in the given direction
        if direction == 'up':
            next_pos = (head[0] - 1, head[1])
        elif direction == 'down':
            next_pos = (head[0] + 1, head[1])
        elif direction == 'left':
            next_pos = (head[0], head[1] - 1)
        else:  # right
            next_pos = (head[0], head[1] + 1)
        
        # Check if next position is valid
        if (next_pos[0] < 0 or next_pos[0] >= self.height or 
            next_pos[1] < 0 or next_pos[1] >= self.width or
            next_pos in self.snake):
            return 0.0  # Invalid move
        
        # Find path to food from next position
        path_to_food = self._a_star_pathfinding(next_pos, self.food)
        
        if path_to_food is None:
            return 0.0  # No path to food
        
        # Calculate path quality based on:
        # 1. Path length (shorter is better)
        # 2. Available space along the path
        # 3. Distance from walls
        
        path_length = len(path_to_food)
        max_path_length = self.width + self.height  # Theoretical maximum
        
        # Normalize path length (shorter is better)
        length_score = 1.0 - (path_length / max_path_length)
        
        # Calculate space along the path
        space_score = 0.0
        for pos in path_to_food[:min(5, len(path_to_food))]:  # Check first 5 positions
            # Count free cells around this position
            free_cells = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                check_pos = (pos[0] + dr, pos[1] + dc)
                if (check_pos[0] >= 0 and check_pos[0] < self.height and 
                    check_pos[1] >= 0 and check_pos[1] < self.width and
                    check_pos not in self.snake):
                    free_cells += 1
            space_score += free_cells / 4.0  # Normalize by max possible free cells
        
        space_score /= min(5, len(path_to_food))  # Average space score
        
        # Calculate wall distance score
        wall_distance = min(
            next_pos[0],  # Distance to top wall
            self.height - 1 - next_pos[0],  # Distance to bottom wall
            next_pos[1],  # Distance to left wall
            self.width - 1 - next_pos[1]  # Distance to right wall
        )
        wall_score = wall_distance / max(self.width, self.height)
        
        # Combine scores (weighted average)
        quality_score = (length_score * 0.4 + space_score * 0.4 + wall_score * 0.2)
        
        return quality_score

def main():
    game = SnakeGame()
    try:
        game.play()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Game interrupted. Goodbye! 👋{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}An error occurred: {e}{Colors.RESET}")
        print(f"{Colors.YELLOW}Please try again.{Colors.RESET}")

if __name__ == "__main__":
    main()