#!/usr/bin/env python3

import os
import glob
import time
import torch
import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent
from reinforce_agent import REINFORCEAgent
from colors import Colors

def print_header():
    """Print a pretty header"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 50)
    print("🎮 RUN TRAINED SNAKE MODEL 🎮")
    print("=" * 50)
    print(f"{Colors.RESET}")

def print_success(message: str):
    """Print a success message in green"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_info(message: str):
    """Print an info message in blue"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")

def print_warning(message: str):
    """Print a warning message in yellow"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")

def get_user_input():
    """Get user input with colored prompts"""
    print_header()
    
    # Get available models (both DQN and REINFORCE)
    dqn_files = glob.glob("agents/snake_dqn_episode_*.pth")
    reinforce_files = glob.glob("agents/snake_reinforce_episode_*.pth")
    model_files = dqn_files + reinforce_files
    if not model_files:
        print(f"{Colors.RED}No saved models found in agents/ directory!{Colors.RESET}")
        print(f"{Colors.YELLOW}Please train a model first using: python train.py{Colors.RESET}")
        return None, None, None, None
    
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"{Colors.CYAN}Available models:{Colors.RESET}")
    for i, model_file in enumerate(model_files):
        episode = model_file.split('_')[-1].split('.')[0]
        agent_type = "REINFORCE" if "reinforce" in model_file else "DQN"
        type_color = Colors.MAGENTA if agent_type == "REINFORCE" else Colors.CYAN
        print(f"{Colors.YELLOW}{i+1:2d}.{Colors.RESET} {type_color}[{agent_type}]{Colors.RESET} Episode {Colors.CYAN}{episode}{Colors.RESET}")
    
    # Get model selection
    while True:
        try:
            choice = input(f"\n{Colors.BLUE}Select model (1-{len(model_files)}): {Colors.RESET}")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected_model = model_files[choice_idx]
                break
            else:
                print(f"{Colors.RED}Invalid choice. Please enter a number between 1 and {len(model_files)}.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number.{Colors.RESET}")
    
    # Get run mode
    print(f"\n{Colors.CYAN}Run mode:{Colors.RESET}")
    print(f"{Colors.YELLOW}1.{Colors.RESET} Visual (see the game)")
    print(f"{Colors.YELLOW}2.{Colors.RESET} Fast (no visualization)")
    
    while True:
        try:
            mode_choice = input(f"\n{Colors.BLUE}Select mode (1-2): {Colors.RESET}")
            if mode_choice == "1":
                run_mode = "visual"
                break
            elif mode_choice == "2":
                run_mode = "fast"
                break
            else:
                print(f"{Colors.RED}Invalid choice. Please enter 1 or 2.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number.{Colors.RESET}")
    
    # Get number of episodes
    while True:
        try:
            episodes = input(f"\n{Colors.BLUE}Number of episodes to run (default 5): {Colors.RESET}")
            if episodes.strip() == "":
                episodes = 5
            else:
                episodes = int(episodes)
            if episodes > 0:
                break
            else:
                print(f"{Colors.RED}Please enter a positive number.{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number.{Colors.RESET}")
    
    # Get delay for visual mode
    delay = 0.1
    if run_mode == "visual":
        while True:
            try:
                delay_input = input(f"\n{Colors.BLUE}Delay between moves in seconds (default 0.1): {Colors.RESET}")
                if delay_input.strip() == "":
                    delay = 0.1
                else:
                    delay = float(delay_input)
                if delay >= 0:
                    break
                else:
                    print(f"{Colors.RED}Please enter a non-negative number.{Colors.RESET}")
            except ValueError:
                print(f"{Colors.RED}Please enter a valid number.{Colors.RESET}")
    
    return selected_model, run_mode, episodes, delay

def run_model(model_path: str, run_mode: str, episodes: int, delay: float):
    """Run the trained model"""
    print(f"\n{Colors.CYAN}Loading model: {Colors.BOLD}{model_path}{Colors.RESET}")
    
    env = SnakeGame()
    is_reinforce = "reinforce" in model_path

    try:
        if is_reinforce:
            agent = REINFORCEAgent()
            agent.load_model(model_path)
        else:
            agent = DQNAgent()
            agent.load_model(model_path)
            agent.epsilon = 0.0
        print_success("Model loaded successfully!")
    except Exception as e:
        print(f"{Colors.RED}Error loading model: {e}{Colors.RESET}")
        return
    
    print(f"\n{Colors.CYAN}Running {episodes} episodes in {run_mode} mode...{Colors.RESET}")
    print(f"{Colors.YELLOW}Press Ctrl+C to stop{Colors.RESET}\n")
    
    total_scores = []
    total_rewards = []
    total_lengths = []
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"{Colors.BLUE}Episode {episode + 1}/{episodes}{Colors.RESET}")
            
            while steps < 10000:  # No step limit during evaluation
                action = agent.act(state, deterministic=True) if is_reinforce else agent.act(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Update state and stats
                state = next_state
                total_reward += reward
                steps += 1
                
                # Visualize if in visual mode
                if run_mode == "visual":
                    env._render()
                    time.sleep(delay)
                
                if done:
                    break
            
            # Episode results
            score = info['score']
            total_scores.append(score)
            total_rewards.append(total_reward)
            total_lengths.append(steps)
            
            # Color code the results
            if score > 20:
                score_color = Colors.GREEN
            elif score > 10:
                score_color = Colors.YELLOW
            else:
                score_color = Colors.RED
            
            print(f"  Score: {score_color}{score}{Colors.RESET} | "
                  f"Reward: {Colors.MAGENTA}{total_reward:.2f}{Colors.RESET} | "
                  f"Steps: {Colors.CYAN}{steps}{Colors.RESET}")
            
            if run_mode == "visual":
                input(f"{Colors.YELLOW}Press Enter for next episode...{Colors.RESET}")
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Evaluation interrupted by user{Colors.RESET}")
    
    # Print summary
    if total_scores:
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*40}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*40}{Colors.RESET}")
        
        avg_score = np.mean(total_scores)
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(total_lengths)
        best_score = max(total_scores)
        
        # Color code the summary
        if avg_score > 20:
            avg_score_color = Colors.GREEN
        elif avg_score > 10:
            avg_score_color = Colors.YELLOW
        else:
            avg_score_color = Colors.RED
        
        if best_score > 30:
            best_score_color = Colors.GREEN
        elif best_score > 20:
            best_score_color = Colors.YELLOW
        else:
            best_score_color = Colors.RED
        
        print(f"{Colors.BLUE}Episodes run:{Colors.RESET} {len(total_scores)}")
        print(f"{Colors.BLUE}Average Score:{Colors.RESET} {avg_score_color}{avg_score:.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}Best Score:{Colors.RESET} {best_score_color}{best_score}{Colors.RESET}")
        print(f"{Colors.BLUE}Average Reward:{Colors.RESET} {Colors.MAGENTA}{avg_reward:.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}Average Steps:{Colors.RESET} {Colors.CYAN}{avg_length:.1f}{Colors.RESET}")

def main():
    """Main function"""
    # Get user input
    result = get_user_input()
    if result is None:
        return
    
    selected_model, run_mode, episodes, delay = result
    
    # Run the model
    run_model(selected_model, run_mode, episodes, delay)

if __name__ == "__main__":
    main() 