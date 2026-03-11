#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
from typing import List, Tuple, Optional
from snake_game import SnakeGame
from colors import Colors

def print_header():
    """Print a pretty header for the training session"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 60)
    print("🐍 SNAKE DQN TRAINING 🐍")
    print("=" * 60)
    print(f"{Colors.RESET}")

def print_progress(episode: int, avg_reward: float, avg_score: float, avg_length: float, epsilon: float):
    """Print training progress with colors"""
    # Color code based on performance
    if avg_reward > 0:
        reward_color = Colors.GREEN
    elif avg_reward > -100:
        reward_color = Colors.YELLOW
    else:
        reward_color = Colors.RED
    
    if avg_score > 20:
        score_color = Colors.GREEN
    elif avg_score > 10:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED
    
    print(f"{Colors.BLUE}Episode {episode:6d}{Colors.RESET} | "
          f"Avg Reward: {reward_color}{avg_reward:8.2f}{Colors.RESET} | "
          f"Avg Score: {score_color}{avg_score:5.1f}{Colors.RESET} | "
          f"Avg Length: {Colors.CYAN}{avg_length:5.1f}{Colors.RESET} | "
          f"Epsilon: {Colors.MAGENTA}{epsilon:.3f}{Colors.RESET}")

def print_save_message(episode: int):
    """Print a colored save message"""
    print(f"{Colors.GREEN}{Colors.BOLD}💾 Model saved at episode {episode}{Colors.RESET}")

def print_final_stats(episode: int, episode_rewards: List[float], episode_scores: List[float]):
    """Print final training statistics with colors"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*50}")
    print("🎯 TRAINING SUMMARY")
    print(f"{'='*50}{Colors.RESET}")
    
    print(f"{Colors.BLUE}Total Episodes:{Colors.RESET} {episode}")
    
    if episode_rewards:
        final_avg_reward = np.mean(episode_rewards[-100:])
        final_avg_score = np.mean(episode_scores[-100:])
        best_score = max(episode_scores)
        
        # Color code the final stats
        if final_avg_reward > 0:
            reward_color = Colors.GREEN
        elif final_avg_reward > -100:
            reward_color = Colors.YELLOW
        else:
            reward_color = Colors.RED
        
        if final_avg_score > 20:
            score_color = Colors.GREEN
        elif final_avg_score > 10:
            score_color = Colors.YELLOW
        else:
            score_color = Colors.RED
        
        if best_score > 30:
            best_color = Colors.GREEN
        elif best_score > 20:
            best_color = Colors.YELLOW
        else:
            best_color = Colors.RED
        
        print(f"{Colors.BLUE}Final Average Reward (last 100):{Colors.RESET} {reward_color}{final_avg_reward:.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}Final Average Score (last 100):{Colors.RESET} {score_color}{final_avg_score:.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}Best Score Achieved:{Colors.RESET} {best_color}{best_score}{Colors.RESET}")

class DQN(nn.Module):
    def __init__(self, input_size: int = 33, hidden_size: int = 128, output_size: int = 4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ExperienceReplay:
    def __init__(self, capacity: int = 50000):  # Increased buffer size
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size: int = 33, action_size: int = 4, hidden_size: int = 128):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Improved hyperparameters
        self.learning_rate = 0.0005  # Lower learning rate for stability
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.99995  # Slower decay
        self.batch_size = 64  # Larger batch size
        
        # Training stats
        self.episode_count = 0
        self.target_update_count = 0
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplay(50000)  # Larger buffer
        
        # Create agents directory
        os.makedirs("agents", exist_ok=True)
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        # Warmup period - don't train until we have enough diverse experiences
        if len(self.memory) < 1000:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first to avoid the warning
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, episode: int):
        """Save the model"""
        # Ensure agents directory exists
        os.makedirs("agents", exist_ok=True)
        
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }, f"agents/snake_dqn_episode_{episode}.pth")
    
    def load_model(self, filepath: str):
        """Load a saved model with backward compatibility"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check if the saved model has a different architecture
        saved_state_dict = checkpoint['model_state_dict']
        
        # Detect the architecture of the saved model by checking layer shapes
        fc2_weight_shape = saved_state_dict['fc2.weight'].shape
        
        if fc2_weight_shape[0] == 4:  # Old model: fc2 goes directly to output (4 actions)
            # This is an old model with 2-layer architecture
            print(f"{Colors.YELLOW}Loading old 2-layer model architecture...{Colors.RESET}")
            
            # Create a temporary 2-layer network to load the weights
            class OldDQN(nn.Module):
                def __init__(self, input_size: int = 29, hidden_size: int = 128, output_size: int = 4):
                    super(OldDQN, self).__init__()
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
            
            # Load into temporary network
            temp_network = OldDQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
            temp_network.load_state_dict(saved_state_dict)
            
            # Copy weights to new network (fc1 and fc2)
            self.q_network.fc1.load_state_dict(temp_network.fc1.state_dict())
            self.q_network.fc2.load_state_dict(temp_network.fc2.state_dict())
            
            # Initialize fc3 and fc4 with default weights
            print(f"{Colors.YELLOW}Initializing new layers with default weights...{Colors.RESET}")
            
            # Copy target network weights
            self.target_network.fc1.load_state_dict(temp_network.fc1.state_dict())
            self.target_network.fc2.load_state_dict(temp_network.fc2.state_dict())
            
        else:  # New model: fc2 goes to another hidden layer
            # This is a new model with 4-layer architecture
            print(f"{Colors.GREEN}Loading new 4-layer model architecture...{Colors.RESET}")
            self.q_network.load_state_dict(saved_state_dict)
            self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        
        # Load other parameters
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        
        print(f"{Colors.GREEN}Model loaded successfully!{Colors.RESET}")

def train_dqn():
    """Main training function"""
    # Initialize environment and agent
    env = SnakeGame()
    agent = DQNAgent()
    
    # Training variables
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    print_header()
    print(f"{Colors.YELLOW}Device: {agent.device}{Colors.RESET}")
    print(f"{Colors.YELLOW}Training indefinitely - Press Ctrl+C to stop{Colors.RESET}")
    print()
    
    try:
        episode = 0
        while True:  # Train indefinitely
            episode += 1
            agent.episode_count = episode  # Update episode count for exploration boosts
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 1500:  # Max 1500 steps per episode
                # Choose action
                action = agent.act(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and stats
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train the network
                agent.replay()
                
                if done:
                    break
            
            # Update target network every 500 episodes
            agent.target_update_count += 1
            if agent.target_update_count >= 500:
                agent.update_target_network()
                agent.target_update_count = 0
            
            # Store episode stats
            episode_rewards.append(total_reward)
            episode_scores.append(info['score'])
            episode_lengths.append(steps)
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_score = np.mean(episode_scores[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print_progress(episode, avg_reward, avg_score, avg_length, agent.epsilon)
            
            # Save model every 100 episodes
            if episode % 100 == 0:
                agent.save_model(episode)
                print_save_message(episode)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Training interrupted by user{Colors.RESET}")
        # Save final model
        agent.save_model(episode)
        print_save_message(episode)
        
        # Print final stats
        print_final_stats(episode, episode_rewards, episode_scores)

if __name__ == "__main__":
    train_dqn() 