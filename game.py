import numpy as np
# Create a tic tac toe game class.
class TicTacToe:
    """
    A class to represent a tic tac toe game.
    """
    def __init__(self):
        """
        Initialize the game.
        """
        self.board = np.zeros((3, 3))
        self.player = 1
        self.winner = None
        self.game_over = False

    def reset(self):
        """
        Reset the game.
        """
        self.board = np.zeros((3, 3))
        self.player = 1
        self.winner = None
        self.game_over = False

    def print_board(self):
        """
        Print the board.
        """
        print(self.board)

    def make_move(self, row, col):
        """
        Make a move on the board.
        row: row index
        col: column index
        """
        if self.board[row, col] == 0:
            self.board[row, col] = self.player
            self.check_winner()
            self.player = -self.player
            return True
        else:
            return False

    def check_winner(self):
        """
        Check if there is a winner.
        """
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2] != 0:
                self.winner = self.board[i, 0]
                self.game_over = True
            if self.board[0, i] == self.board[1, i] == self.board[2, i] != 0:
                self.winner = self.board[0, i]
                self.game_over = True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            self.winner = self.board[0, 0]
            self.game_over = True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            self.winner = self.board[0, 2]
            self.game_over = True
        if np.all(self.board != 0):
            self.game_over = True
    def get_random_valid_move(self):
        """
        Get a random valid move.
        """
        valid_moves = np.argwhere(self.board == 0)
        idx = np.random.choice(valid_moves.shape[0])
        return valid_moves[idx]
    def get_valid_moves(self):
        """
        Get all valid moves.
        """
        return np.argwhere(self.board == 0)
    
            