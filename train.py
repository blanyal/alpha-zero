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
"""Class to train the Neural Network."""
from config import CFG
from mcts import MonteCarloTreeSearch, TreeNode


class Train(object):
    """Class with functions to train the Neural Network using MCTS.

    Attributes:
        game: An object containing the game state.
        net: An object containing the neural network.
    """

    def __init__(self, game, net):
        """Initializes Train with the board state and neural network."""
        self.game = game
        self.net = net

    def start(self):
        """Main training loop."""
        for i in range(CFG.num_iterations):
            print("Iteration", i + 1)

            for j in range(CFG.num_games):
                game = self.game.clone()  # Create a fresh clone for each game.
                self.play_game(game)

    def play_game(self, game):
        """Loop for each self-play game.

        Runs MCTS for each game state and plays a move based on the MCTS output.
        Stops when the game is over and prints out a winner.

        Args:
            game: An object containing the game state.
        """
        print("Start Self Play Game")

        mcts = MonteCarloTreeSearch(self.net)

        game_over = False
        value = 0
        move_count = 0

        node = TreeNode()

        # Keep playing until the game is in a terminal state.
        while not game_over:
            # MCTS simulations to get the best child node.
            if move_count < CFG.temperature_thresh:
                best_child = mcts.search(game, node, CFG.temperature_init)
            else:
                best_child = mcts.search(game, node, CFG.temperature_final)

            action = best_child.action
            game.play_action(action)  # Play the child node's action.
            move_count += 1

            game.print_board(game.player_to_eval)

            game_over, value = game.check_game_over(game.player_to_eval)

            game.switch_player_state()  # Switch the board,

            best_child.parent = None
            node = best_child  # Make the child node the root node.

        if value is 1:
            print("win")
        elif value is -1:
            print("loss")
        else:
            print("draw")
