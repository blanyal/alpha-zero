from config import CFG
from tic_tac_toe.tic_tac_toe_game import TicTacToeGame
from neural_net import NeuralNetwork
from train import Train

if __name__ == '__main__':
    game = TicTacToeGame()
    net = NeuralNetwork(game)
    train = Train(game, net)

    train.start()
