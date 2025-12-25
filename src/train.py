import numpy as np
from typing import Optional, Tuple
from .environment import TicTacToeEnv
from .agents import QLearningAgent, RandomAgent, Agent


class TrainingStats:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.invalid_moves = 0
        self.total_reward = 0
    
    def update(self, reward: float, info: dict):
        self.total_reward += reward
        
        if info.get('invalid_move', False):
            self.invalid_moves += 1
        elif info.get('winner') == 1:
            self.wins += 1
        elif info.get('winner') == -1:
            self.losses += 1
        elif info.get('draw', False):
            self.draws += 1
    
    def get_total(self) -> int:
        return self.wins + self.losses + self.draws
    
    def get_stats_dict(self) -> dict:
        total = self.get_total()
        if total == 0:
            return {
                'wins': 0, 'losses': 0, 'draws': 0,
                'win_rate': 0, 'loss_rate': 0, 'draw_rate': 0,
                'avg_reward': 0, 'invalid_moves': 0
            }
        
        return {
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.wins / total,
            'loss_rate': self.losses / total,
            'draw_rate': self.draws / total,
            'avg_reward': self.total_reward / (total + self.invalid_moves),
            'invalid_moves': self.invalid_moves
        }
    
    def print_stats(self, episode: int):
        stats = self.get_stats_dict()
        total = self.get_total()
        
        print(f"Episode {episode}:")
        print(f"  Wins: {stats['wins']}/{total} ({100*stats['win_rate']:.1f}%)")
        print(f"  Losses: {stats['losses']}/{total} ({100*stats['loss_rate']:.1f}%)")
        print(f"  Draws: {stats['draws']}/{total} ({100*stats['draw_rate']:.1f}%)")
        print(f"  Avg Reward: {stats['avg_reward']:.3f}")
        if stats['invalid_moves'] > 0:
            print(f"  Invalid Moves: {stats['invalid_moves']}")


def train_agent(
    episodes: int = 10000,
    opponent: str = 'random',
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    epsilon_decay: bool = True,
    print_every: int = 1000,
    save_path: Optional[str] = None
) -> QLearningAgent:
    
    env = TicTacToeEnv()
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    
    if opponent == 'random':
        opponent_agent = RandomAgent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent}")
    
    stats = TrainingStats()
    
    print(f"Training agent for {episodes} episodes...")
    print(f"Hyperparameters: α={alpha}, γ={gamma}, ε={epsilon}")
    print("-" * 60)
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        
        # For Q-learning in two-player games, we need to track:
        # - The state before agent's move
        # - The action the agent took
        # Then update after opponent responds
        prev_state = None
        prev_action = None
        
        while not done:
            # Agent's turn (Player 1 / X)
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            
            # Remember this state-action pair
            agent_state_before = state.copy()
            agent_action = action
            
            next_state, reward, done, info = env.step(action)
            
            if done:
                # Agent won immediately
                agent.update(agent_state_before, agent_action, reward, next_state, done, [])
                stats.update(reward, info)
                break
            
            # Now it's opponent's turn
            state_after_agent = next_state.copy()
            
            # Opponent's turn (Player -1 / O)
            valid_actions = env.get_valid_actions()
            opp_action = opponent_agent.select_action(state_after_agent, valid_actions)
            next_state, reward, done, info = env.step(opp_action)
            
            # Now we can update the agent's Q-value
            # The "next state" from agent's perspective is after opponent moved
            state_after_opponent = next_state.copy()
            
            if done:
                # Opponent won
                agent_reward = -1.0
                agent.update(agent_state_before, agent_action, agent_reward, state_after_opponent, done, [])
                
                # Update stats
                agent_info = info.copy()
                if 'winner' in agent_info:
                    agent_info['winner'] = -agent_info['winner']
                stats.update(agent_reward, agent_info)
                break
            else:
                # Game continues
                # Agent gets 0 reward for this move (didn't win or lose yet)
                agent.update(agent_state_before, agent_action, 0, state_after_opponent, False, env.get_valid_actions())
                state = state_after_opponent
        
        if epsilon_decay:
            agent.decay_epsilon()
        
        if episode % print_every == 0:
            stats.print_stats(episode)
            agent_stats = agent.get_stats()
            print(f"  Q-table size: {agent_stats['num_states']} states")
            print(f"  Current ε: {agent.epsilon:.4f}")
            print("-" * 60)
            stats.reset()
    
    print("\nTraining completed!")
    
    if save_path:
        agent.save(save_path)
    
    return agent


def evaluate_agent(
    agent: Agent,
    episodes: int = 100,
    opponent: str = 'random',
    render: bool = False,
    as_player: int = 1
) -> Tuple[dict, list]:

    env = TicTacToeEnv()
    
    if opponent == 'random':
        opponent_agent = RandomAgent()
    else:
        raise ValueError(f"Unknown opponent type: {opponent}")
    
    stats = TrainingStats()
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        if render:
            print(f"\n--- Game {episode + 1} ---")
            env.render()
        
        while not done:
            # Determine whose turn it is
            current_player = 1 if len(env.get_valid_actions()) % 2 != 0 else -1
            # Note: In this env, len(valid_actions) starts at 9 (odd) -> Player 1
            # After 1 move, len is 8 (even) -> Player -1
            # Logic: 9, 7, 5, 3, 1 -> Player 1's turn
            # 8, 6, 4, 2 -> Player -1's turn
            
            # Simple check: env.current_player
            # But we need to know if it's AGENT or OPPONENT turn
            
            is_agent_turn = (env.current_player == as_player)
            
            if is_agent_turn:
                valid_actions = env.get_valid_actions()
                action = agent.select_action(state, valid_actions, training=False)
                state, reward, done, info = env.step(action)
                
                # Reward interpretation:
                # Env returns 1 if Player 1 wins, -1 if Player -1 wins
                # If as_player=1, reward 1 is good (1 * 1 = 1)
                # If as_player=-1, reward -1 is good (-1 * -1 = 1)
                adjusted_reward = reward * as_player
                episode_reward += adjusted_reward
                
                if render:
                    print(f"Agent ({'X' if as_player==1 else 'O'}) plays:")
                    env.render()
            else:
                valid_actions = env.get_valid_actions()
                action = opponent_agent.select_action(state, valid_actions)
                state, reward, done, info = env.step(action)
                
                # Opponent move doesn't give direct reward to agent usually, 
                # unless game ends.
                # If opponent wins (reward for them), it's bad for agent.
                adjusted_reward = reward * as_player
                # If reward is for winner:
                # Opponent (P2) wins -> reward -1. Agent (P1) sees -1 * 1 = -1 (Loss). Correct.
                # Opponent (P1) wins -> reward 1. Agent (P2) sees 1 * -1 = -1 (Loss). Correct.
                episode_reward += adjusted_reward
                
                if render:
                    print(f"Opponent ({'O' if as_player==1 else 'X'}) plays:")
                    env.render()
            
            if done:
                # Update stats properly
                # TrainingStats assumes 'winner' field in info
                # If agent is Player 2 (-1), and winner is -1, that's a WIN for agent
                # Info always has absolute winner (1 or -1)
                
                # We need to trick TrainingStats or just manually interpret?
                # TrainingStats logic:
                # elif info.get('winner') == 1: self.wins += 1
                # elif info.get('winner') == -1: self.losses += 1
                # This assumes Agent is Player 1.
                
                # We should update info to reflect Agent perspective relative to TrainingStats expectation?
                # Or just manually increment stats.
                # Let's fix TrainingStats usage here.
                
                # Actually TrainingStats is hardcoded for Player 1 being the "Self".
                # Let's just calculate stats manually here for flexibility or create a dict.
                pass

        # Calculate result for this episode
        rewards.append(episode_reward)
        
        # Manual stats update to handle role correctly
        winner = info.get('winner', 0)
        draw = info.get('draw', False)
        
        # Create a proxy info dict for TrainingStats update if we want to reuse it,
        # otherwise we might need to modify TrainingStats or just do manual mapping.
        # Let's interpret for TrainingStats:
        # If Agent WON, we want TrainingStats to log a WIN.
        # TrainingStats logs WIN if info['winner'] == 1.
        # So if Agent (-1) won, we tell TrainingStats winner=1.
        # If Agent (-1) lost (Winner=1), we tell TrainingStats winner=-1.
        
        proxy_info = info.copy()
        if 'winner' in info:
            if info['winner'] == as_player:
                proxy_info['winner'] = 1  # Map "Agent Won" to 1 for stats
            else:
                proxy_info['winner'] = -1 # Map "Agent Lost" to -1 for stats
        
        # Reward is usually 0 unless game end in this Env step return?
        # Actually Env step returns result of THAT step.
        # If Agent moves and wins, reward is 1 (or -1).
        # We collected episode_reward.
        
        stats.update(episode_reward, proxy_info)
        
        if render:
             if 'winner' in info:
                 print(f"{'Agent' if info['winner'] == as_player else 'Opponent'} wins!")
             elif draw:
                 print("Draw!")
    
    return stats.get_stats_dict(), rewards