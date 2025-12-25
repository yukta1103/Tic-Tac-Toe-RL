import numpy as np
import pickle
from typing import List, Dict
from abc import ABC, abstractmethod


class Agent(ABC):
    
    @abstractmethod
    def select_action(self, state: np.ndarray, valid_actions: List[int], training: bool = False) -> int:
        pass


class RandomAgent(Agent):
    
    def select_action(self, state: np.ndarray, valid_actions: List[int], training: bool = False) -> int:
        return np.random.choice(valid_actions)


class QLearningAgent(Agent):
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.2):
        self.q_table: Dict[str, np.ndarray] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        
    def get_state_key(self, state: np.ndarray) -> str:
        return str(state.tolist())
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9)
        return self.q_table[state_key][action]
    
    def select_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in valid_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
            return np.random.choice(best_actions)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool, next_valid_actions: List[int]):
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in next_valid_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.gamma * max_next_q
        
        self.q_table[state_key][action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self, decay_rate: float = 0.9995, min_epsilon: float = 0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def save(self, filepath: str):
        data = {
            'q_table': self.q_table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'QLearningAgent':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        agent = cls(alpha=data['alpha'], gamma=data['gamma'], epsilon=data['epsilon'])
        agent.q_table = data['q_table']
        agent.initial_epsilon = data['initial_epsilon']
        print(f"Agent loaded from {filepath}")
        return agent
    
    def get_stats(self) -> Dict:
        return {
            'num_states': len(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon
        }