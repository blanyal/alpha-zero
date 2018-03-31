from unittest import TestCase
from tic_tac_toe.tic_tac_toe_game import TicTacToeGame


class TestTicTacToeGame(TestCase):

    def test_check_game_over_1(self):
        game = TicTacToeGame()
        game.state = [[1, 0, 2], [1, 0, 2], [1, 0, 2]]
        game_over, winner = game.check_game_over()

        self.assertEqual(game_over, True)
        self.assertEqual(winner, 1)

    def test_check_game_over_2(self):
        game = TicTacToeGame()
        game.state = [[1, 0, 2], [1, 1, 0], [1, 2, 2]]
        game_over, winner = game.check_game_over()

        self.assertEqual(game_over, True)
        self.assertEqual(winner, 1)

    def test_check_game_over_3(self):
        game = TicTacToeGame()
        game.state = [[1, 1, 2], [2, 2, 1], [1, 1, 2]]
        game_over, winner = game.check_game_over()

        self.assertEqual(game_over, True)
        self.assertEqual(winner, 0)

    def test_check_game_over_4(self):
        game = TicTacToeGame()
        game.state = [[2, 0, 0], [0, 2, 0], [0, 1, 0]]
        game_over, winner = game.check_game_over()

        self.assertEqual(game_over, False)
        self.assertEqual(winner, None)
