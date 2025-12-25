"""
Train an agent to play as O (player -1, second player)
This will make the AI competitive even when the human starts first
"""
from src.environment import TicTacToeEnv
from src.agents import QLearningAgent, RandomAgent
from src.train import evaluate_agent
import numpy as np

def train_second_player_agent(
    episodes=20000,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.2,
    epsilon_decay=True,
    print_every=2000,
    save_path='models/agent_second_player.pkl'
):
    """Train an agent to play as the second player (O, player -1)"""
    
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    random_opponent = RandomAgent()
    
    print(f"Training second player agent for {episodes} episodes...")
    print(f"Hyperparameters: α={alpha}, γ={gamma}, ε={epsilon}")
    print("-" * 60)
    
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        
        while not done:
            # Random opponent goes first (plays as X, player 1)
            valid_actions = env.get_valid_actions()
            opp_action = random_opponent.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(opp_action)
            
            if done:
                # Opponent won or draw
                if 'winner' in info and info['winner'] == 1:
                    losses += 1
                elif 'draw' in info:
                    draws += 1
                break
            
            # Agent's turn (plays as O, player -1)
            state_before_agent = next_state.copy()
            valid_actions = env.get_valid_actions()
            agent_action = agent.select_action(state_before_agent, valid_actions, training=True)
            
            next_state, reward, done, info = env.step(agent_action)
            
            # Update agent's Q-values
            if done:
                # Agent won or draw
                if 'winner' in info and info['winner'] == -1:
                    agent_reward = 1.0
                    wins += 1
                elif 'draw' in info:
                    agent_reward = 0.0
                    draws += 1
                else:
                    agent_reward = -1.0
                    losses += 1
                
                agent.update(state_before_agent, agent_action, agent_reward, next_state, done, [])
            else:
                # Game continues, neutral reward
                agent.update(state_before_agent, agent_action, 0, next_state, False, env.get_valid_actions())
            
            state = next_state
        
        if epsilon_decay:
            agent.decay_epsilon()
        
        if episode % print_every == 0:
            total = wins + losses + draws
            print(f"Episode {episode}:")
            print(f"  Wins: {wins}/{total} ({100*wins/total:.1f}%)")
            print(f"  Losses: {losses}/{total} ({100*losses/total:.1f}%)")
            print(f"  Draws: {draws}/{total} ({100*draws/total:.1f}%)")
            print(f"  Q-table size: {len(agent.q_table)} states")
            print(f"  Current ε: {agent.epsilon:.4f}")
            print("-" * 60)
            wins = losses = draws = 0
    
    print("\nTraining completed!")
    agent.save(save_path)
    
    # Evaluate
    print("\nEvaluating agent...")
    # We need a custom evaluation for second player
    eval_wins = 0
    eval_losses = 0
    eval_draws = 0
    
    for _ in range(100):
        state = env.reset()
        done = False
        
        while not done:
            # Random opponent goes first
            valid_actions = env.get_valid_actions()
            opp_action = random_opponent.select_action(state, valid_actions)
            state, reward, done, info = env.step(opp_action)
            
            if done:
                if 'winner' in info:
                    eval_losses += 1
                else:
                    eval_draws += 1
                break
            
            # Agent plays
            valid_actions = env.get_valid_actions()
            agent_action = agent.select_action(state, valid_actions, training=False)
            state, reward, done, info = env.step(agent_action)
            
            if done:
                if 'winner' in info and info['winner'] == -1:
                    eval_wins += 1
                elif 'draw' in info:
                    eval_draws += 1
                else:
                    eval_losses += 1
    
    print(f"\nEvaluation Results (100 games as second player):")
    print(f"  Win Rate: {eval_wins}%")
    print(f"  Loss Rate: {eval_losses}%")
    print(f"  Draw Rate: {eval_draws}%")
    
    return agent

if __name__ == '__main__':
    train_second_player_agent()
