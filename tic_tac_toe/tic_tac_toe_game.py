from copy import deepcopy
from game import Game


class TicTacToeGame(Game):
    def __init__(self):
        super().__init__()
        self.side = 3
        self.current_player = 1
        self.state = []

        for i in range(self.side):
            self.state.append([0 * j for j in range(self.side)])

    def clone(self):
        board_clone = TicTacToeGame()
        board_clone.state = deepcopy(self.state)
        board_clone.current_player = self.current_player
        return board_clone
