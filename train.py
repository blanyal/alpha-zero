from config import CFG
from mcts import MonteCarloTreeSearch


class Train:
    def __init__(self, game, net):
        self.game = game
        self.net = net

    def start(self):
        for i in range(CFG.num_iterations):
            for j in range(CFG.num_games):
                self.play_game()

    def play_game(self):
        pass
