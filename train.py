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
from neural_net import NeuralNetworkWrapper
from evaluate import Evaluate


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
        self.eval_net = NeuralNetworkWrapper(game)

    def start(self):
        """Main training loop."""
        for i in range(CFG.num_iterations):
            print("Iteration", i + 1)

            training_data = []  # list to store self play states, pis and vs

            for j in range(CFG.num_games):
                print("Start Training Self-Play Game", j + 1)
                game = self.game.clone()  # Create a fresh clone for each game.
                self.play_game(game, training_data)

            # Save the current neural network model.
            self.net.save_model()

            # Load the recently saved model into the evaluator network.
            self.eval_net.load_model()

            # Train the network using self play values.
            self.net.train(training_data)

            # Initialize MonteCarloTreeSearch objects for both networks.
            current_mcts = MonteCarloTreeSearch(self.net)
            eval_mcts = MonteCarloTreeSearch(self.eval_net)

            evaluator = Evaluate(current_mcts=current_mcts, eval_mcts=eval_mcts,
                                 game=self.game)
            wins, losses = evaluator.evaluate()

            print("wins:", wins)
            print("losses:", losses)

            num_games = wins + losses

            if num_games == 0:
                win_rate = 0
            else:
                win_rate = wins / num_games

            print("win rate:", win_rate)

            if win_rate > CFG.eval_win_rate:
                # Save current model as the best model.
                print("New model saved as best model.")
                self.net.save_model("best_model")
            else:
                print("New model discard and previous model loaded.")
                # Discard current model and use previous best model.
                self.net.load_model()

    def play_game(self, game, training_data):
        """Loop for each self-play game.

        Runs MCTS for each game state and plays a move based on the MCTS output.
        Stops when the game is over and prints out a winner.

        Args:
            game: An object containing the game state.
            training_data: A list to store self play states, pis and vs.
        """
        mcts = MonteCarloTreeSearch(self.net)

        game_over = False
        value = 0
        move_count = 0
        self_play_data = []

        node = TreeNode()

        # Keep playing until the game is in a terminal state.
        while not game_over:
            # MCTS simulations to get the best child node.
            if move_count < CFG.temp_thresh:
                best_child = mcts.search(game, node, CFG.temp_init)
            else:
                best_child = mcts.search(game, node, CFG.temp_final)

            # Store state, pi and v for training.
            self_play_data.append([game.state,
                                   best_child.parent.child_psas,
                                   0,
                                   best_child.parent.player_to_eval])

            action = best_child.action
            game.play_action(action)  # Play the child node's action.
            move_count += 1

            game_over, value = game.check_game_over(game.player_to_eval)

            game.switch_player_state()  # Switch the board.

            best_child.parent = None
            node = best_child  # Make the child node the root node.

        # Update v as the value of the game result.
        for game_state in self_play_data:
            game_state[2] = value * game_state[3]
            game_state.pop()
            training_data.append(game_state)

        # game.print_board(game.player_to_eval)

        if value == 1 * game.player_to_eval * -1:
            print("win")
        elif value == -1 * game.player_to_eval * -1:
            print("loss")
        else:
            print("draw")
