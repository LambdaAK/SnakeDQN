#!/usr/bin/env python3
"""
REINFORCE (Policy Gradient) agent for Snake.
Monte Carlo policy gradient - updates after each full episode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from typing import List, Tuple
from snake_game import SnakeGame
from colors import Colors


def print_header():
    """Print REINFORCE-specific header - clearly identifies architecture"""
    print(f"{Colors.MAGENTA}{Colors.BOLD}")
    print("=" * 60)
    print("🐍 SNAKE REINFORCE (Policy Gradient) TRAINING 🐍")
    print("=" * 60)
    print(f"{Colors.RESET}")
    print(f"{Colors.CYAN}Architecture: REINFORCE | Algorithm: Monte Carlo Policy Gradient{Colors.RESET}")


def print_progress(episode: int, avg_reward: float, avg_score: float, avg_length: float):
    """Print training progress - no epsilon (REINFORCE uses policy stochasticity)"""
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
          f"{Colors.MAGENTA}[REINFORCE]{Colors.RESET}")


def print_save_message(episode: int):
    """Print a colored save message"""
    print(f"{Colors.GREEN}{Colors.BOLD}💾 REINFORCE model saved at episode {episode}{Colors.RESET}")


def print_final_stats(episode: int, episode_rewards: List[float], episode_scores: List[float]):
    """Print final training statistics"""
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*50}")
    print("🎯 REINFORCE TRAINING SUMMARY")
    print(f"{'='*50}{Colors.RESET}")

    print(f"{Colors.BLUE}Architecture:{Colors.RESET} {Colors.MAGENTA}REINFORCE (Policy Gradient){Colors.RESET}")
    print(f"{Colors.BLUE}Total Episodes:{Colors.RESET} {episode}")

    if episode_rewards:
        final_avg_reward = np.mean(episode_rewards[-100:])
        final_avg_score = np.mean(episode_scores[-100:])
        best_score = max(episode_scores)

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


class PolicyNetwork(nn.Module):
    """Policy network: state -> action logits (softmax for probabilities)"""
    def __init__(self, input_size: int = 33, hidden_size: int = 128, output_size: int = 4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Logits, not probabilities
        return x


class REINFORCEAgent:
    """REINFORCE policy gradient agent - learns a stochastic policy."""
    AGENT_TYPE = "reinforce"

    def __init__(self, state_size: int = 33, action_size: int = 4, hidden_size: int = 128):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor for returns
        self.use_baseline = True  # Subtract mean return to reduce variance

        self.episode_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        os.makedirs("agents", exist_ok=True)

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Sample action from policy. Use deterministic=True for evaluation (greedy)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(state_tensor)
            probs = F.softmax(logits, dim=1)

        if deterministic:
            return probs.argmax(dim=1).item()
        # Sample from policy for exploration during training
        action = torch.multinomial(probs, 1).item()
        return action

    def train_step(self, trajectory: List[Tuple[np.ndarray, int, float]]):
        """
        REINFORCE update: compute discounted returns and policy gradient.
        trajectory: list of (state, action, reward) for the full episode.
        """
        if len(trajectory) < 2:
            return

        states, actions, rewards = zip(*trajectory)
        states = torch.FloatTensor(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = np.array(rewards, dtype=np.float32)

        # Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
        returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        returns = torch.FloatTensor(returns).to(self.device)

        # Baseline: subtract mean return to reduce variance
        if self.use_baseline:
            returns = returns - returns.mean()

        # Policy gradient: -log(pi(a|s)) * G_t
        logits = self.policy(states)
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = -(selected_log_probs * returns).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

    def save_model(self, episode: int):
        """Save REINFORCE model - includes agent_type for run_trained_model.py"""
        os.makedirs("agents", exist_ok=True)
        torch.save({
            'agent_type': self.AGENT_TYPE,
            'episode': episode,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
        }, f"agents/snake_reinforce_episode_{episode}.pth")

    def load_model(self, filepath: str):
        """Load a saved REINFORCE model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        print(f"{Colors.GREEN}REINFORCE model loaded successfully!{Colors.RESET}")


def train_reinforce():
    """Main REINFORCE training loop - clearly branded as REINFORCE"""
    env = SnakeGame()
    agent = REINFORCEAgent()

    episode_rewards = []
    episode_scores = []
    episode_lengths = []

    print_header()
    print(f"{Colors.YELLOW}Device: {agent.device}{Colors.RESET}")
    print(f"{Colors.YELLOW}Training indefinitely - Press Ctrl+C to stop{Colors.RESET}")
    print()

    try:
        episode = 0
        while True:
            episode += 1
            agent.episode_count = episode
            state = env.reset()
            trajectory: List[Tuple[np.ndarray, int, float]] = []
            total_reward = 0
            steps = 0

            while steps < 1500:
                action = agent.act(state, deterministic=False)
                next_state, reward, done, info = env.step(action)

                trajectory.append((state, action, reward))
                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break

            # REINFORCE: update policy after each episode
            agent.train_step(trajectory)

            episode_rewards.append(total_reward)
            episode_scores.append(info['score'])
            episode_lengths.append(steps)

            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_score = np.mean(episode_scores[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print_progress(episode, avg_reward, avg_score, avg_length)

            if episode % 100 == 0:
                agent.save_model(episode)
                print_save_message(episode)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}REINFORCE training interrupted by user{Colors.RESET}")
        agent.save_model(episode)
        print_save_message(episode)
        print_final_stats(episode, episode_rewards, episode_scores)


if __name__ == "__main__":
    train_reinforce()
