from game import TicTacToe
import numpy as np
from model import TicTacToeModel
import torch

class RandomPlayer:
    def go_once(self, tic_tac_toe: TicTacToe):
        row, col = tic_tac_toe.get_random_valid_move()
        return tic_tac_toe.make_move(row, col)
    

class GreedyPlayer:
    def go_once(self, tic_tac_toe: TicTacToe):
        """
        Make a winning move if possible.
        Block the opponent from winning if possible.
        Otherwise, make a random move.
        """
        # Get player number.
        player = tic_tac_toe.player
        # Get the board.
        board = tic_tac_toe.board
        # Check for a winning move.
        for i in range(3):
            if np.sum(board[i, :]) == 2 * player:
                col = np.argwhere(board[i, :] == 0)[0][0]
                return tic_tac_toe.make_move(i, col)
            if np.sum(board[:, i]) == 2 * player:
                row = np.argwhere(board[:, i] == 0)[0][0]
                return tic_tac_toe.make_move(row, i)
        if np.sum(np.diag(board)) == 2 * player:
            idx = np.argwhere(np.diag(board) == 0)[0][0]
            return tic_tac_toe.make_move(idx, idx)
        if np.sum(np.diag(np.fliplr(board))) == 2 * player:
            idx = np.argwhere(np.diag(np.fliplr(board)) == 0)[0][0]
            return tic_tac_toe.make_move(idx, 2 - idx)
        # Check for a blocking move.
        for i in range(3):
            if np.sum(board[i, :]) == -2 * player:
                col = np.argwhere(board[i, :] == 0)[0][0]
                return tic_tac_toe.make_move(i, col)
            if np.sum(board[:, i]) == -2 * player:
                row = np.argwhere(board[:, i] == 0)[0][0]
                return tic_tac_toe.make_move(row, i)
        if np.sum(np.diag(board)) == -2 * player:
            idx = np.argwhere(np.diag(board) == 0)[0][0]
            return tic_tac_toe.make_move(idx, idx)
        if np.sum(np.diag(np.fliplr(board))) == -2 * player:
            idx = np.argwhere(np.diag(np.fliplr(board)) == 0)[0][0]
            return tic_tac_toe.make_move(idx, 2 - idx)
        # Make a random move.
        row, col = tic_tac_toe.get_random_valid_move()
        return tic_tac_toe.make_move(row, col)
    

class AIPlayer:
    def __init__(self):
        self.model = TicTacToeModel()
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.game_results = []
    def go_once(self, tic_tac_toe: TicTacToe):
        board_state = torch.tensor(tic_tac_toe.board.reshape(1, 9), dtype=torch.float32)
        # If we are player 2, we need to flip the board.
        if tic_tac_toe.player == -1:
            board_state = -board_state
        prediction = self.model(board_state)
        best_move = torch.argmax(prediction)
        row = best_move // 3
        col = best_move % 3
        is_valid = tic_tac_toe.make_move(row, col)
        if not is_valid:
            row, col = tic_tac_toe.get_random_valid_move()
            is_valid = tic_tac_toe.make_move(row, col)
        
    def train(self, iterations: int, other):
        """
        Train the player against a random opponent.
        """
        for _ in range(iterations):
            game = TicTacToe()
            game_timeout = 25
            game_counter = 0
            while not game.game_over:
                game_counter += 1
                if game_counter > game_timeout:
                    break
                # Get the board state.
                board_state = torch.tensor(game.board.reshape(1, 9), dtype=torch.float32)
                # Get the model prediction.
                prediction = self.model(board_state)
                # Get the best move.
                best_move = torch.argmax(prediction)
                # Make the move.
                row = best_move // 3
                col = best_move % 3
                is_valid = game.make_move(row, col)
                if not is_valid:
                    # Force a random valid move.
                    row, col = game.get_random_valid_move()
                    game.make_move(row, col)

                # Determine the loss.
                target = torch.zeros(1, 9)
                if not is_valid:
                    # Invalid move
                    # Make the target be any valid move
                    
                    valid_moves = game.get_valid_moves()
                    for move in valid_moves:
                        idx = move[0] * 3 + move[1]
                        target[0, idx] = 1
                    
                elif game.winner == 1:
                    # Victory
                    # Make the target be the best move
                    target[0, best_move] = 1
                elif game.game_over:
                    # Tie
                    # Make the target be the best move
                    target[0, best_move] = 1
                elif is_valid:
                    # Other player makes a random valid move.
                    other.go_once(game)
                    if game.winner == -1:
                        # Defeat
                        # Make the target be any other move except the best move
                        valid_moves = game.get_valid_moves()
                        for move in valid_moves:
                            idx = move[0] * 3 + move[1]
                            if idx != best_move:
                                target[0, idx] = 1
                    elif game.game_over:
                        # Tie
                        # Make the target be any other move except the best move
                        valid_moves = game.get_valid_moves()
                        for move in valid_moves:
                            idx = move[0] * 3 + move[1]
                            if idx != best_move:
                                target[0, idx] = 1
                else:
                    # Continue the game
                    # Make the target be any other move except the best move
                    valid_moves = game.get_valid_moves()
                    for move in valid_moves:
                        idx = move[0] * 3 + move[1]
                        if idx != best_move:
                            target[0, idx] = 1
                        
                # Compute the loss.
                loss = self.loss(prediction, target)
                # Optimize the model.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.game_results.append(game.winner if game.winner is not None else 0)