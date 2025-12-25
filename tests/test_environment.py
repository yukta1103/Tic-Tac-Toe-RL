import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import TicTacToeEnv

class TestTicTacToeEnv:
    
    def test_initialization(self):
        env = TicTacToeEnv()
        assert np.all(env.board == 0)
        assert env.current_player == 1
        assert not env.done
    
    def test_reset(self):
        env = TicTacToeEnv()
        env.board[0] = 1
        env.done = True
        
        state = env.reset()
        assert np.all(state == 0)
        assert not env.done
        assert env.current_player == 1
    
    def test_valid_actions(self):
        env = TicTacToeEnv()
        
        valid = env.get_valid_actions()
        assert len(valid) == 9
        assert valid == list(range(9))
        
        env.board[0] = 1
        valid = env.get_valid_actions()
        assert len(valid) == 8
        assert 0 not in valid
    
    def test_invalid_move(self):
        env = TicTacToeEnv()
        env.board[0] = 1
        
        state, reward, done, info = env.step(0)
        assert done
        assert reward == -10
        assert info['invalid_move']
    
    def test_step_alternates_player(self):
        env = TicTacToeEnv()
        
        assert env.current_player == 1
        env.step(0)
        assert env.current_player == -1
        env.step(1)
        assert env.current_player == 1
    
    def test_horizontal_win(self):
        env = TicTacToeEnv()
        
        # Player X wins on top row
        env.board = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
        env.current_player = 1
        
        state, reward, done, info = env.step(2)
        assert done
        assert reward == 1
        assert info['winner'] == 1
    
    def test_vertical_win(self):
        env = TicTacToeEnv()
        
        # Player O wins on left column
        env.board = np.array([-1, 0, 0, -1, 0, 0, 0, 0, 0])
        env.current_player = -1
        
        state, reward, done, info = env.step(6)
        assert done
        assert reward == -1
        assert info['winner'] == -1
    
    def test_diagonal_win(self):
        env = TicTacToeEnv()
        
        # Player X wins on main diagonal
        env.board = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])
        env.current_player = 1
        
        state, reward, done, info = env.step(8)
        assert done
        assert reward == 1
        assert info['winner'] == 1
    
    def test_draw(self):
        env = TicTacToeEnv()
        
        # Setup board for draw
        env.board = np.array([1, -1, 1, -1, 1, -1, -1, 1, 0])
        env.current_player = -1
        
        state, reward, done, info = env.step(8)
        assert done
        assert reward == 0
        assert info.get('draw', False)
    
    def test_game_continues(self):
        env = TicTacToeEnv()
        
        state, reward, done, info = env.step(0)
        assert not done
        assert reward == 0
    
    def test_render_string_mode(self):
        env = TicTacToeEnv()
        env.board = np.array([1, -1, 0, 0, 1, 0, 0, 0, -1])
        
        output = env.render(mode='string')
        assert isinstance(output, str)
        assert 'X' in output
        assert 'O' in output
        assert '.' in output
    
    def test_cannot_step_after_done(self):
        env = TicTacToeEnv()
        env.board = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
        env.done = True
        
        with pytest.raises(ValueError):
            env.step(3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])