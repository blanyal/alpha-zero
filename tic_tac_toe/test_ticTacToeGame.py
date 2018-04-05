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
"""Class to run unit tests for the TicTacToeGame class."""
from unittest import TestCase

from tic_tac_toe.tic_tac_toe_game import TicTacToeGame


class TestTicTacToeGame(TestCase):
    """Class to run unit tests for the TicTacToeGame class."""

    def test_check_game_over_1(self):
        """Test case for the check_game_over function.

        Test for game over with a win.
        """
        game = TicTacToeGame()
        game.state = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
        game_over, value = game.check_game_over(1)

        self.assertEqual(game_over, True)
        self.assertEqual(value, 1)

    def test_check_game_over_2(self):
        """Test case for the check_game_over function.

        Test for game over with a loss.
        """
        game = TicTacToeGame()
        game.state = [[1, 0, -1], [1, 1, 0], [1, -1, -1]]
        game_over, value = game.check_game_over(-1)

        self.assertEqual(game_over, True)
        self.assertEqual(value, -1)

    def test_check_game_over_3(self):
        """Test case for the check_game_over function.

        Test for game over with a draw.
        """
        game = TicTacToeGame()
        game.state = [[1, 1, -1], [-1, -1, 1], [1, 1, -1]]
        game_over, value = game.check_game_over(1)

        self.assertEqual(game_over, True)
        self.assertEqual(value, 0.001)

    def test_check_game_over_4(self):
        """Test case for the check_game_over function.

        Test for game not over.
        """
        game = TicTacToeGame()
        game.state = [[-1, 0, 0], [0, -1, 0], [0, 1, 0]]
        game_over, value = game.check_game_over(-1)

        self.assertEqual(game_over, False)
        self.assertEqual(value, 0)
