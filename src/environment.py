import numpy as np
from typing import Tuple, List, Dict, Optional


class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  
        self.done = False
        
    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.done = False
        return self.board.copy()
    
    def get_valid_actions(self) -> List[int]:
        return [i for i in range(9) if self.board[i] == 0]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise ValueError("Game is already over. Call reset() to start new game.")
        
        if action not in self.get_valid_actions():
            self.done = True
            return self.board.copy(), -10, True, {"invalid_move": True}
        
        self.board[action] = self.current_player
        
        if self._check_win(self.current_player):
            self.done = True
            reward = 1 if self.current_player == 1 else -1
            return self.board.copy(), reward, True, {"winner": self.current_player}
        
        if len(self.get_valid_actions()) == 0:
            self.done = True
            return self.board.copy(), 0, True, {"draw": True}
        
        self.current_player *= -1
        
        return self.board.copy(), 0, False, {}
    
    def _check_win(self, player: int) -> bool:
        board_2d = self.board.reshape(3, 3)
        
        for row in board_2d:
            if np.all(row == player):
                return True
        
        for col in board_2d.T:
            if np.all(col == player):
                return True
        
        if np.all(np.diag(board_2d) == player):
            return True
        if np.all(np.diag(np.fliplr(board_2d)) == player):
            return True
        
        return False
    
    def render(self, mode: str = 'human') -> Optional[str]:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board_2d = self.board.reshape(3, 3)
        
        output = ["\n"]
        for i, row in enumerate(board_2d):
            output.append(" " + " | ".join([symbols[cell] for cell in row]))
            if i < 2:
                output.append("-----------")
        output.append("\n")
        
        result = "\n".join(output)
        
        if mode == 'human':
            print(result)
        elif mode == 'string':
            return result
        
    def get_state(self) -> np.ndarray:
        return self.board.copy()
    
    def is_done(self) -> bool:
        return self.done