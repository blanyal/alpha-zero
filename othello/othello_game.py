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
from collections import Counter

import numpy as np

from game import Game


class OthelloGame(Game):
    """Represents the game board and its logic.

    Attributes:
        side: An integer indicating the length of the board side.
        current_player: An integer to keep track of the current player.
        state: A list which stores the game state in matrix form.
        action_size: An integer indicating the total number of board squares.
        directions: A dictionary containing tuples to check for valid moves.
    """

    def __init__(self):
        """Initializes TicTacToeGame with the initial board state."""
        super().__init__()
        self.side = 6
        self.current_player = 1
        self.state = []
        self.action_size = self.side * self.side

        # Create a n x n matrix to represent the board
        for i in range(self.side):
            self.state.append([0 * j for j in range(self.side)])

        self.state[(self.side // 2) - 1][(self.side // 2) - 1] = -1
        self.state[(self.side // 2)][(self.side // 2)] = -1
        self.state[(self.side // 2) - 1][(self.side // 2)] = 1
        self.state[(self.side // 2)][(self.side // 2) - 1] = 1

        self.state = np.array(self.state)

        self.directions = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 1),
            5: (1, -1),
            6: (1, 0),
            7: (1, 1)
        }

    def clone(self):
        """Creates a deep clone of the game object.

        Returns:
            the cloned game object.
        """
        game_clone = OthelloGame()
        game_clone.state = deepcopy(self.state)
        game_clone.current_player = self.current_player
        return game_clone

    def play_action(self, action):
        """Plays an action on the game board.

        Args:
            action: A tuple in the form of (row, column, direction).
        """
        x = action[1]
        y = action[2]
        d = action[3]

        self.state[x][y] = self.current_player

        count = 1

        # Flip all opponent pieces which are in the sandwich.
        while True:
            row = x + d[0] * count
            col = y + d[1] * count

            if self.state[row][col] == -self.current_player:
                self.state[row][col] = self.current_player
                count += 1
            else:
                break

        self.current_player = -self.current_player

    def get_valid_moves(self, current_player):
        """Returns a list of moves along with their validity.

        Searches the board for valid sandwich moves.

        Returns:
            A list containing moves as (validity, row, column, direction).
        """
        valid_moves = []

        pl = current_player

        side = self.side

        for x in range(self.side):
            for y in range(self.side):
                found = False

                # Search for empty squares.
                if self.state[x][y] == 0:

                    # Search in all 8 directions for a square of the opponent.
                    for i in range(len(self.directions)):
                        d = self.directions[i]

                        row = x + d[0]
                        col = y + d[1]

                        if row < side and col < side:
                            if self.state[row][col] == -pl:
                                found_valid_move = False
                                count = 2

                                # Keep searching for a sandwich condition.
                                while True:
                                    row = x + d[0] * count
                                    col = y + d[1] * count

                                    if 0 <= row < side and 0 <= col < side:
                                        if self.state[row][col] == pl:
                                            valid_moves.append((1, x, y, d))
                                            found_valid_move = True
                                            break
                                    else:
                                        break

                                    count += 1

                                if found_valid_move:
                                    found = True
                                    break

                if not found:
                    valid_moves.append((0, None, None, None))

        return np.array(valid_moves)

    def check_game_over(self, current_player):
        """Checks if the game is over and return a possible winner.

        There are 3 possible scenarios.
            a) The game is over and we have a winner.
            b) The game is over but it is a draw.
            c) The game is not over.

        Args:
            current_player: An integer representing the current player.

        Returns:
            A bool representing the game over state.
            An integer action value. (win: 1, loss: -1, draw: 0.001
        """

        player_a = current_player
        player_b = -current_player

        player_a_moves = self.get_valid_moves(player_a)
        player_b_moves = self.get_valid_moves(player_b)

        player_a_valid_count = Counter(x[0] == 1 for x in player_a_moves)
        player_b_valid_count = Counter(x[0] == 1 for x in player_b_moves)

        # Check if both players can't play any more moves.
        if player_a_valid_count[True] == 0 or player_b_valid_count[True] == 0:
            unique, piece_count = np.unique(self.state,
                                            return_counts=True)

            # Check for the player with the most number of pieces.
            if piece_count[player_a] > piece_count[player_b]:
                return True, 1
            elif piece_count[player_a] == piece_count[player_b]:
                return True, 0
            else:
                return True, -1
        else:
            return False, 0

    def print_board(self):
        """Prints the board state."""
        print("   0    1    2    3    4    5")
        for x in range(self.side):
            print(x, end='')
            for y in range(self.side):
                if self.state[x][y] == 0:
                    print('  -  ', end='')
                elif self.state[x][y] == 1:
                    print('  X  ', end='')
                elif self.state[x][y] == -1:
                    print('  O  ', end='')
            print('\n')
        print('\n')
