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
"""File which starts the program"""
import argparse
import os

from tic_tac_toe.tic_tac_toe_game import TicTacToeGame
from othello.othello_game import OthelloGame
from connect_four.connect_four_game import ConnectFourGame
from neural_net import NeuralNetworkWrapper
from train import Train
from human_play import HumanPlay
from config import CFG

# Code to read command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--num_iterations",
                    help="Number of iterations.",
                    dest="num_iterations",
                    type=int,
                    default=CFG.num_iterations)

parser.add_argument("--num_games",
                    help="Number of self play games played for each iteration.",
                    dest="num_games",
                    type=int,
                    default=CFG.num_games)

parser.add_argument("--num_mcts_sims",
                    help="Number of MCTS simulations per game.",
                    dest="num_mcts_sims",
                    type=int,
                    default=CFG.num_mcts_sims)

parser.add_argument("--c_puct",
                    help="The level of exploration used in MCTS.",
                    dest="c_puct",
                    type=float,
                    default=CFG.c_puct)

parser.add_argument("--l2_val",
                    help="The level of L2 regularization used for training.",
                    dest="l2_val",
                    type=float,
                    default=CFG.l2_val)

parser.add_argument("--momentum",
                    help="Momentum Parameter for the momentum optimizer.",
                    dest="momentum",
                    type=float,
                    default=CFG.momentum)

parser.add_argument("--learning_rate",
                    help="Learning Rate for the momentum optimizer.",
                    dest="learning_rate",
                    type=float,
                    default=CFG.learning_rate)

parser.add_argument("--t_policy_val",
                    help="Value for policy prediction.",
                    dest="t_policy_val",
                    type=float,
                    default=CFG.t_policy_val)

parser.add_argument("--temp_init",
                    help="Initial Temperature value to control exploration.",
                    dest="temp_init",
                    type=float,
                    default=CFG.temp_init)

parser.add_argument("--temp_final",
                    help="Final Temperature value to control exploration.",
                    dest="temp_final",
                    type=float,
                    default=CFG.temp_final)

parser.add_argument("--temp_thresh",
                    help="Threshold where temperature init changes to final.",
                    dest="temp_thresh",
                    type=int,
                    default=CFG.temp_thresh)

parser.add_argument("--epochs",
                    help="Number of epochs during training.",
                    dest="epochs",
                    type=int,
                    default=CFG.epochs)

parser.add_argument("--batch_size",
                    help="Batch size for training.",
                    dest="batch_size",
                    type=int,
                    default=CFG.batch_size)

parser.add_argument("--dirichlet_alpha",
                    help="Alpha value for Dirichlet noise.",
                    dest="dirichlet_alpha",
                    type=float,
                    default=CFG.dirichlet_alpha)

parser.add_argument("--epsilon",
                    help="Number of iterations.",
                    dest="epsilon",
                    type=float,
                    default=CFG.epsilon)

parser.add_argument("--model_directory",
                    help="Name of the directory to store models.",
                    dest="model_directory",
                    type=str,
                    default=CFG.model_directory)

parser.add_argument("--num_eval_games",
                    help="Number of self-play games to play for evaluation.",
                    dest="num_eval_games",
                    type=int,
                    default=CFG.num_eval_games)

parser.add_argument("--eval_win_rate",
                    help="Win rate needed to be the best model.",
                    dest="eval_win_rate",
                    type=float,
                    default=CFG.eval_win_rate)

parser.add_argument("--load_model",
                    help="Bool to initialize the network with the best model.",
                    dest="load_model",
                    type=int,
                    default=CFG.load_model)

parser.add_argument("--human_play",
                    help="Bool to play as a Human vs the AI.",
                    dest="human_play",
                    type=int,
                    default=CFG.human_play)

parser.add_argument("--resnet_blocks",
                    help="Number of residual blocks in the resnet.",
                    dest="resnet_blocks",
                    type=int,
                    default=CFG.resnet_blocks)

parser.add_argument("--record_loss",
                    help="Binary to record policy and value loss to a file.",
                    dest="record_loss",
                    type=int,
                    default=CFG.record_loss)

parser.add_argument("--loss_file",
                    help="Name of the file to record loss.",
                    dest="loss_file",
                    type=str,
                    default=CFG.loss_file)

parser.add_argument("--game",
                    help="Name of the file to record loss.",
                    dest="game",
                    type=int,
                    default=CFG.game)

if __name__ == '__main__':
    """Initializes game state, neural network and the training loop"""
    arguments = parser.parse_args()

    # Replace CFG values with the values from the command line.
    CFG.num_iterations = arguments.num_iterations
    CFG.num_games = arguments.num_games
    CFG.num_mcts_sims = arguments.num_mcts_sims
    CFG.c_puct = arguments.c_puct
    CFG.l2_val = arguments.l2_val
    CFG.momentum = arguments.momentum
    CFG.learning_rate = arguments.learning_rate
    CFG.t_policy_val = arguments.t_policy_val
    CFG.temp_init = arguments.temp_init
    CFG.temp_final = arguments.temp_final
    CFG.temp_thresh = arguments.temp_thresh
    CFG.epochs = arguments.epochs
    CFG.batch_size = arguments.batch_size
    CFG.dirichlet_alpha = arguments.dirichlet_alpha
    CFG.epsilon = arguments.epsilon
    CFG.model_directory = arguments.model_directory
    CFG.num_eval_games = arguments.num_eval_games
    CFG.eval_win_rate = arguments.eval_win_rate
    CFG.load_model = arguments.load_model
    CFG.human_play = arguments.human_play
    CFG.resnet_blocks = arguments.resnet_blocks
    CFG.record_loss = arguments.record_loss
    CFG.loss_file = arguments.loss_file
    CFG.game = arguments.game

    # Initialize the game object with the chosen game.
    game = object
    if CFG.game == 0:
        game = TicTacToeGame()
    elif CFG.game == 1:
        game = OthelloGame()
    elif CFG.game == 2:
        game = ConnectFourGame()

    net = NeuralNetworkWrapper(game)

    # Initialize the network with the best model.
    if CFG.load_model:
        file_path = CFG.model_directory + "best_model.meta"
        if os.path.exists(file_path):
            net.load_model("best_model")
        else:
            print("Trained model doesn't exist. Starting from scratch.")
    else:
        print("Trained model not loaded. Starting from scratch.")

    # Play vs the AI as a human instead of training.
    if CFG.human_play:
        human_play = HumanPlay(game, net)
        human_play.play()
    else:
        train = Train(game, net)
        train.start()
