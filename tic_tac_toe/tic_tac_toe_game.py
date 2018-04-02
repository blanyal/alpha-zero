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
import numpy as np


class TicTacToeGame(Game):
    """Represents the game board and its logic.

    Attributes:
        side: An integer indicating the length of the board side.
        player_value: An integer indicating the move value of the player.
        player_to_eval: An integer to keep track of board switching.
        state: A list which stores the game state in matrix form.
        action_size: An integer indicating the total number of board squares.
    """

    def __init__(self):
        """Initializes TicTacToeGame with the initial board state."""
        super().__init__()
        self.side = 3
        self.player_value = 1
        self.player_to_eval = 1
        self.state = []
        self.action_size = self.side * self.side

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
        game_clone.player_to_eval = self.player_to_eval
        return game_clone

    def play_action(self, action):
        """Plays an action on the game board.

        Args:
            action: A tuple in the form of (row, column).
        """
        x = action[0]
        y = action[1]

        self.state[x][y] = self.player_value

    def get_valid_moves(self):
        """Returns a list of moves along with their validity.

        Searches the board for zeros(0). 0 represents an empty square.

        Returns:
            A list containing moves in the form of (validity, row, column).
        """
        valid_moves = []

        for x in range(self.side):
            for y in range(self.side):
                if self.state[x][y] == 0:
                    valid_moves.append((1, x, y))
                else:
                    valid_moves.append((0, None, None))

        return np.array(valid_moves)

    def switch_player_state(self):
        """Change the board to the perspective of the opponent and track it.

        ie. all 1s are converted to -1s and vice versa
        """
        self.player_to_eval = -1 * self.player_to_eval  # Track board switch.
        self.state = -1 * self.state

    def check_game_over(self, player_to_eval):
        """Checks if the game is over and return a possible winner.

        There are 3 possible scenarios.
            a) The game is over and we have a winner.
            b) The game is over but it is a draw.
            c) The game is not over.

        Args:
            player_to_eval: An integer representing the board's switch status.

        Returns:
            A bool representing the game over state.
            An integer action value. (win: 1, loss: -1, draw: 0.001
        """
        if player_to_eval == 1:
            player_a = 1
            player_b = -1
        else:
            player_a = -1
            player_b = 1

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
                return True, 1
            elif player_b_count == self.side:
                return True, -1

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
                return True, 1
            elif player_b_count == self.side:
                return True, -1

        # Check for major diagonal marks
        player_a_count = 0
        player_b_count = 0
        for x in range(self.side):
            if self.state[x][x] == player_a:
                player_a_count += 1
            elif self.state[x][x] == player_b:
                player_b_count += 1

        if player_a_count == self.side:
            return True, 1
        elif player_b_count == self.side:
            return True, -1

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
            return True, 1
        elif player_b_count == self.side:
            return True, -1

        # There are still moves left so the game is not over
        valid_moves = self.get_valid_moves()

        for move in valid_moves:
            if move[0] is 1:
                return False, 0

        # If there are no moves left the game is over without a winner
        return True, 0.001

    def print_board(self, player_to_eval):
        """Prints the board state."""

        if player_to_eval is 1:
            player_a = 1
            player_b = -1
        else:
            player_a = -1
            player_b = 1

        for x in range(self.side):
            for y in range(self.side):
                if self.state[x][y] == 0:
                    print('-    ', end='')
                elif self.state[x][y] == player_a:
                    print('X    ', end='')
                elif self.state[x][y] == player_b:
                    print('O    ', end='')
            print('\n')
        print('\n')
