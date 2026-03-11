#!/usr/bin/env python3

import os
import glob
import time
from colors import Colors

def print_header():
    """Print a pretty header"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 50)
    print("📊 TRAINING MONITOR 📊")
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

def monitor_training():
    """Monitor training progress by checking for saved models"""
    print_header()
    
    # Check if agents directory exists
    if not os.path.exists("agents"):
        print(f"{Colors.RED}No agents directory found!{Colors.RESET}")
        print(f"{Colors.YELLOW}Start training first with: python train.py{Colors.RESET}")
        return
    
    dqn_files = glob.glob("agents/snake_dqn_episode_*.pth")
    reinforce_files = glob.glob("agents/snake_reinforce_episode_*.pth")
    model_files = dqn_files + reinforce_files
    
    if not model_files:
        print(f"{Colors.YELLOW}No saved models found yet.{Colors.RESET}")
        print(f"{Colors.CYAN}Training should create models every 100 episodes.{Colors.RESET}")
        return
    
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"{Colors.CYAN}Found {len(model_files)} saved models "
          f"({len(dqn_files)} DQN, {len(reinforce_files)} REINFORCE):{Colors.RESET}")
    print()
    
    latest_models = model_files[-5:]
    for i, model_file in enumerate(latest_models):
        episode = int(model_file.split('_')[-1].split('.')[0])
        file_size = os.path.getsize(model_file) / 1024
        agent_type = "REINFORCE" if "reinforce" in model_file else "DQN"
        type_color = Colors.MAGENTA if agent_type == "REINFORCE" else Colors.CYAN
        
        if episode > 5000:
            episode_color = Colors.GREEN
        elif episode > 2000:
            episode_color = Colors.YELLOW
        else:
            episode_color = Colors.RED
        
        print(f"{Colors.YELLOW}{i+1:2d}.{Colors.RESET} {type_color}[{agent_type:>9s}]{Colors.RESET} "
              f"Episode {episode_color}{episode:6d}{Colors.RESET} | "
              f"Size: {Colors.MAGENTA}{file_size:.1f} KB{Colors.RESET}")
    
    # Show latest model info
    latest_model = model_files[-1]
    latest_episode = int(latest_model.split('_')[-1].split('.')[0])
    
    print()
    print(f"{Colors.BLUE}Latest model: Episode {Colors.BOLD}{latest_episode}{Colors.RESET}")
    
    # Estimate training time
    if len(model_files) > 1:
        # Calculate episodes per hour based on time between saves
        first_model = model_files[0]
        first_episode = int(first_model.split('_')[-1].split('.')[0])
        
        # Get file modification times
        first_time = os.path.getmtime(first_model)
        latest_time = os.path.getmtime(latest_model)
        
        time_diff = latest_time - first_time  # seconds
        episode_diff = latest_episode - first_episode
        
        if time_diff > 0 and episode_diff > 0:
            episodes_per_hour = (episode_diff / time_diff) * 3600
            print(f"{Colors.CYAN}Training rate: {Colors.BOLD}{episodes_per_hour:.1f}{Colors.RESET} episodes/hour")
    
    print()
    print(f"{Colors.GREEN}To run the latest model: python run_trained_model.py{Colors.RESET}")

def main():
    """Main function"""
    monitor_training()

if __name__ == "__main__":
    main() 