# MIT License
#
# Copyright (c) 2018 Blanyal D'Souza
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Class for Board State and Logic."""
from copy import deepcopy
from game import Game
from random import shuffle
import numpy as np


class TicTacToeGame(Game):
    """Represents the game board and its logic.

    Attributes:
        side: An integer indicating the length of the board side.
        last_player: An integer indicating the player who played the last move.
        state: A list which stores the game state in matrix form.
    """

    def __init__(self):
        """Initializes TicTacToeGame with the initial board state."""
        super().__init__()
        self.side = 3
        self.last_player = 2
        self.state = []

        # Create a n x n matrix to represent the board

        for i in range(self.side):
            self.state.append([0 * j for j in range(self.side)])

        self.state = np.array(self.state)

    def clone(self):
        """Creates a deep clone of the game object.

        Returns:
            the cloned game object.
        """
        game_clone = TicTacToeGame()
        game_clone.state = deepcopy(self.state)
        game_clone.last_player = self.last_player
        return game_clone

    def play_move(self, move):
        """Plays a move on the game board.

        Args:
            move: A tuple in the form of (row, column).
        """
        x = move[0]
        y = move[1]

        # Switch the last player with the current player and play the move

        self.last_player = 3 - self.last_player
        self.state[x][y] = self.last_player

    def get_valid_moves(self):
        """Returns a list of valid moves to play.

        Searches the board for zeros(0). 0 represents an empty square.

        Returns:
            A list containing valid moves in the form of (row, column).
        """
        moves = []

        for x in range(self.side):
            for y in range(self.side):
                if self.state[x][y] == 0:
                    moves.append((x, y))

        shuffle(moves)
        return moves

    def check_game_over(self):
        """Checks if the game is over and return a possible winner.

        There are 3 possible scenarios.
            a) The game is over and we have a winner.
            b) The game is over but it is a draw.
            c) The game is not over.

        Returns:
            A bool representing the game over state.
            An integer representing the winner.
        """
        player_a = 1
        player_b = 2

        # Check for horizontal marks
        for x in range(self.side):
            player_a_count = 0
            player_b_count = 0
            for y in range(self.side):
                if self.state[x][y] == player_a:
                    player_a_count += 1
                elif self.state[x][y] == player_b:
                    player_b_count += 1
            if player_a_count == self.side:
                return True, player_a
            elif player_b_count == self.side:
                return True, player_b

        # Check for vertical marks
        for x in range(self.side):
            player_a_count = 0
            player_b_count = 0
            for y in range(self.side):
                if self.state[y][x] == player_a:
                    player_a_count += 1
                elif self.state[y][x] == player_b:
                    player_b_count += 1
            if player_a_count == self.side:
                return True, player_a
            elif player_b_count == self.side:
                return True, player_b

        # Check for major diagonal marks
        player_a_count = 0
        player_b_count = 0
        for x in range(self.side):
            if self.state[x][x] == player_a:
                player_a_count += 1
            elif self.state[x][x] == player_b:
                player_b_count += 1

        if player_a_count == self.side:
            return True, player_a
        elif player_b_count == self.side:
            return True, player_b

        # Check for minor diagonal marks
        player_a_count = 0
        player_b_count = 0
        for y in range(self.side - 1, -1, -1):
            x = 2 - y
            if self.state[x][y] == player_a:
                player_a_count += 1
            elif self.state[x][y] == player_b:
                player_b_count += 1

        if player_a_count == self.side:
            return True, player_a
        elif player_b_count == self.side:
            return True, player_b

        # If there are no moves left the game is over without a winner
        if len(self.get_valid_moves()) == 0:
            return True, 0

        # There are still moves left so the game is not over
        return False, None

    def print_board(self):
        """Prints the board state."""
        for x in range(self.side):
            for y in range(self.side):
                if self.state[x][y] == 0:
                    print('-    ', end='')
                elif self.state[x][y] == 1:
                    print('X    ', end='')
                elif self.state[x][y] == 2:
                    print('O    ', end='')
            print('\n')
        print('\n')
