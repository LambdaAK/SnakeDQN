# Snake AI

Reinforcement-learning agents for the classic Snake game in PyTorch. Two architectures are supported: a Deep Q-Network (value-based) and REINFORCE (policy gradient).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Play manually:
```bash
python snake_game.py
```

Train an agent (interactive choice between DQN and REINFORCE):
```bash
python train.py
# or skip the prompt:
python train.py --dqn
python train.py --reinforce
```

Run a trained model from `agents/`:
```bash
python run_trained_model.py
```

Check progress of saved checkpoints:
```bash
python monitor_training.py
```

## Implementation

- **Architectures:**
  - **DQN** (`dqn_agent.py`): 4-layer MLP (33 → 128 → 128 → 64 → 4) with ReLU activations, ε-greedy exploration, a 50k-transition experience replay buffer, a periodically-synced target network, and gradient clipping.
  - **REINFORCE** (`reinforce_agent.py`): Same 4-layer MLP shape, but the head outputs action logits. Trained with Monte Carlo policy gradient (full-episode trajectories) using a mean-return baseline to reduce variance.

- **State Representation (33 dimensions):** see `state_representation.txt` for the exact layout. Includes danger flags relative to the current heading, current direction (one-hot), food direction, wall and body distances in each absolute direction, a count of immediately-safe moves, dead-end flags, available open space per direction, A*-based path-quality scores per direction, and the normalized snake length.

- **Reward Function:** see `reward_function.txt`. Five mutually exclusive cases:
  - `-10` on death (wall or self collision)
  - `+10` on eating food
  - `+0.3` if the Manhattan distance to food decreased this step
  - `-0.3` if it increased
  - `0` otherwise

- **Tests and tooling:**
  - `test_dqn.py` — sanity checks for the state vector, reward, agent action selection, and A* pathfinding.
  - `monitor_training.py` — lists saved checkpoints (DQN and REINFORCE) and estimates episodes/hour.

For more details, see the code and comments in the respective Python files.
