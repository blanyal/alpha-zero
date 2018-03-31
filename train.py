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
        mcts = MonteCarloTreeSearch(self.net)

        game_over = False
        winner = 0

        while not game_over:
            move = mcts.search(self.game)
            self.game.play_move(move)
            self.game.print_board()
            game_over, winner = self.game.check_game_over()

        print(winner)
