# Tic-Tac-Toe Reinforcement Learning

A reinforcement learning project that trains an AI agent to play Tic-Tac-Toe using Q-Learning.

## Features

- **Q-Learning Agent**: Learns optimal Tic-Tac-Toe strategy through self-play
- **Multiple Modes**: Train, play against the AI, or watch demo games
- **Comprehensive Testing**: Unit tests for environment logic
- **Flexible Training**: Configurable hyperparameters (learning rate, discount factor, exploration rate)
- **Model Persistence**: Save and load trained agents

## Project Structure

```
Tic-Tac-Toe-RL/
├── src/
│   ├── environment.py    # Tic-Tac-Toe game environment
│   ├── agents.py         # Q-Learning and Random agents
│   └── train.py          # Training and evaluation logic
├── tests/
│   └── test_environment.py  # Unit tests
├── models/               # Saved trained models
├── results/              # Training results and logs
├── main.py               # Main entry point
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yukta1103/Tic-Tac-Toe-RL.git
cd Tic-Tac-Toe-RL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train a New Agent

Train an agent for 10,000 episodes (default):
```bash
python main.py --mode train
```

Custom training parameters:
```bash
python main.py --mode train --episodes 50000 --alpha 0.1 --gamma 0.99 --epsilon 0.2
```

### Play Against the Agent

```bash
python main.py --mode play --model models/agent.pkl
```

Enter positions 0-8 to make your moves:
```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

### Watch Demo Games

Watch the agent play against a random opponent:
```bash
python main.py --mode demo --episodes 10 --model models/agent.pkl
```

## Command-Line Arguments

- `--mode`: Mode to run (`train`, `play`, `demo`)
- `--episodes`: Number of training/demo episodes (default: 10000)
- `--opponent`: Opponent type for training (default: `random`)
- `--alpha`: Learning rate (default: 0.1)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon`: Initial exploration rate (default: 0.2)
- `--epsilon-decay`: Enable epsilon decay during training (default: True)
- `--print-every`: Print statistics every N episodes (default: 1000)
- `--save-path`: Path to save trained agent (default: `models/agent.pkl`)
- `--model`: Path to load trained model (default: `models/agent.pkl`)

## Running Tests

Run the test suite:
```bash
pytest tests/test_environment.py -v
```

Or:
```bash
python tests/test_environment.py
```

## How It Works

### Environment
- **State Space**: 9 positions, each can be empty (0), X (1), or O (-1)
- **Action Space**: 9 possible positions (0-8)
- **Rewards**: 
  - Win: +1
  - Loss: -1
  - Draw: 0
  - Invalid move: -10

### Q-Learning Agent
- Uses a Q-table to store state-action values
- Epsilon-greedy exploration strategy
- Updates Q-values using the Bellman equation:
  ```
  Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
  ```

### Training Process
1. Agent plays as X (player 1) against a random opponent
2. After each move, Q-values are updated based on rewards
3. Epsilon decays over time to reduce exploration
4. Model is saved after training completes

## Expected Results

After training for 10,000+ episodes, the agent should achieve:
- **Win Rate**: 85-95% against random opponent
- **Loss Rate**: 0-5%
- **Draw Rate**: 5-15%

## Future Enhancements

- [ ] Self-play training (agent vs agent)
- [ ] Deep Q-Network (DQN) implementation
- [ ] GUI for playing against the agent
- [ ] Training visualization and learning curves
- [ ] Minimax opponent for harder training
- [ ] Multi-agent tournament mode

## License

MIT License

## Author

yukta1103
