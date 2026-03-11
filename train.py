#!/usr/bin/env python3
"""
Unified training entry point for Snake AI.
Choose between DQN (value-based) and REINFORCE (policy gradient) architectures.
"""

import argparse
import sys

# Terminal colors
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_architecture_choice():
    """Display architecture options with clear labels"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}Select architecture:{Colors.RESET}")
    print(f"  {Colors.YELLOW}1.{Colors.RESET} DQN        - Deep Q-Network (value-based, experience replay)")
    print(f"  {Colors.YELLOW}2.{Colors.RESET} REINFORCE  - Policy Gradient (Monte Carlo, updates per episode)")
    print()


def get_architecture_interactive():
    """Interactive prompt for architecture selection"""
    print_architecture_choice()
    while True:
        choice = input(f"{Colors.BLUE}Enter 1 or 2: {Colors.RESET}").strip()
        if choice == "1":
            return "dqn"
        elif choice == "2":
            return "reinforce"
        print(f"{Colors.RED}Invalid. Enter 1 for DQN or 2 for REINFORCE.{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Snake AI - choose DQN or REINFORCE architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py              # Interactive: choose architecture
  python train.py --dqn        # Train DQN (value-based)
  python train.py --reinforce  # Train REINFORCE (policy gradient)
        """
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dqn", action="store_true", help="Train DQN architecture")
    group.add_argument("--reinforce", action="store_true", help="Train REINFORCE (policy gradient) architecture")
    args = parser.parse_args()

    if args.dqn:
        architecture = "dqn"
    elif args.reinforce:
        architecture = "reinforce"
    else:
        architecture = get_architecture_interactive()

    # Import and run the appropriate trainer
    if architecture == "dqn":
        print(f"\n{Colors.GREEN}{Colors.BOLD}▶ Starting DQN training...{Colors.RESET}\n")
        from dqn_agent import train_dqn
        train_dqn()
    else:
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}▶ Starting REINFORCE training...{Colors.RESET}\n")
        from reinforce_agent import train_reinforce
        train_reinforce()


if __name__ == "__main__":
    main()
