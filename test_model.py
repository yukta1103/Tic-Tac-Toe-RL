from src.agents import QLearningAgent
from src.train import evaluate_agent

# Load and evaluate the agent
agent = QLearningAgent.load('models/agent.pkl')
stats, _ = evaluate_agent(agent, episodes=100, opponent='random')

print(f"Win Rate: {stats['win_rate']*100:.1f}%")
print(f"Loss Rate: {stats['loss_rate']*100:.1f}%")
print(f"Draw Rate: {stats['draw_rate']*100:.1f}%")
print(f"States Learned: {agent.get_stats()['num_states']}")
print(f"Epsilon: {agent.get_stats()['epsilon']}")
