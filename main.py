import argparse
import os
from src.environment import TicTacToeEnv
from src.agents import QLearningAgent, RandomAgent
from src.train import train_agent, evaluate_agent


def train_mode(args):
    """Train a new agent"""
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    agent = train_agent(
        episodes=args.episodes,
        opponent=args.opponent,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        print_every=args.print_every,
        save_path=args.save_path
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    stats, _ = evaluate_agent(agent, episodes=100, opponent='random')
    print(f"\nEvaluation Results (100 games):")
    print(f"  Win Rate: {100*stats['win_rate']:.1f}%")
    print(f"  Loss Rate: {100*stats['loss_rate']:.1f}%")
    print(f"  Draw Rate: {100*stats['draw_rate']:.1f}%")


def play_mode(args):
    print("=" * 60)
    print("PLAY MODE")
    print("=" * 60)
    
    # Load agent
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Train a model first using: python main.py --mode train")
        return
    
    agent = QLearningAgent.load(args.model)
    env = TicTacToeEnv()
    
    print("\nYou are O (opponent), Agent is X")
    print("Enter position 0-8 to play:")
    print("\n0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5")
    print("---------")
    print("6 | 7 | 8\n")
    
    state = env.reset()
    done = False
    
    while not done:
        print("Agent's turn (X):")
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=False)
        state, reward, done, info = env.step(action)
        env.render()
        
        if done:
            if 'winner' in info:
                print("Agent wins!")
            else:
                print("Draw!")
            break
        
        print("Your turn (O):")
        valid_actions = env.get_valid_actions()
        print(f"Valid moves: {valid_actions}")
        
        while True:
            try:
                action = int(input("Enter position (0-8): "))
                if action in valid_actions:
                    break
                else:
                    print(f"Invalid move! Choose from {valid_actions}")
            except ValueError:
                print("Please enter a number between 0-8")
        
        state, reward, done, info = env.step(action)
        env.render()
        
        if done:
            if 'winner' in info:
                print("You win!")
            else:
                print("Draw!")


def demo_mode(args):
    print("=" * 60)
    print("DEMO MODE")
    print("=" * 60)
    
    if args.model and os.path.exists(args.model):
        agent = QLearningAgent.load(args.model)
    else:
        print("No model specified, creating random agent")
        agent = RandomAgent()
    
    stats, _ = evaluate_agent(agent, episodes=args.episodes, opponent='random', render=True)
    
    print("\n" + "=" * 60)
    print(f"Results from {args.episodes} games:")
    print(f"  Win Rate: {100*stats['win_rate']:.1f}%")
    print(f"  Loss Rate: {100*stats['loss_rate']:.1f}%")
    print(f"  Draw Rate: {100*stats['draw_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Tic-Tac-Toe Reinforcement Learning')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'play', 'demo'],
                       help='Mode to run: train, play, or demo')
    
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of episodes for training/demo')
    parser.add_argument('--opponent', type=str, default='random',
                       choices=['random'],
                       help='Type of opponent to train against')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.2,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon-decay', action='store_true', default=True,
                       help='Whether to decay epsilon during training')
    parser.add_argument('--print-every', type=int, default=1000,
                       help='Print statistics every N episodes')
    parser.add_argument('--save-path', type=str, default='models/agent.pkl',
                       help='Path to save trained agent')
    
    parser.add_argument('--model', type=str, default='models/agent.pkl',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'play':
        play_mode(args)
    elif args.mode == 'demo':
        demo_mode(args)


if __name__ == '__main__':
    main()