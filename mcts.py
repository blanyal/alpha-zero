from config import CFG
from random import choice
import numpy as np


class TreeNode:

    def __init__(self, game, parent=None, move=None):
        self.n = 0
        self.q = 0
        self.game = game
        self.move = move
        self.children = []
        self.parent = parent
        self.untried_moves = game.get_valid_moves()

    def is_not_leaf(self):
        if len(self.children) > 0:
            return True
        return False

    def is_expanded(self):
        if len(self.untried_moves) == 0:
            return True
        return False

    def select_child(self):
        uct_list = [
            (c.q / c.n) + CFG.c_puct * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]

        max_index = np.argmax(uct_list)

        return self.children[max_index]

    def add_child_node(self, game, parent, move):
        child_node = TreeNode(game, parent, move)
        self.children.append(child_node)
        return child_node

    def back_prop(self, game_over, winner):
        self.n += 1

        if self.game.current_player == winner and game_over is True:
            self.q += 1


class MonteCarloTreeSearch:

    def __init__(self, net):
        self.root = None
        self.game = None
        self.net = net

    def search(self, game):
        self.root = TreeNode(game)
        self.game = game

        for i in range(CFG.num_mcts_sims):
            node = self.root
            game = self.game.clone()

            while node.is_not_leaf() and node.is_expanded():
                node = node.select_child()
                game.play_move(node.move)

            if not node.is_expanded():
                move = node.untried_moves.pop()
                game.play_move(move)
                node = node.add_child_node(game, node, move)

            while not game.check_game_over()[0]:
                moves = game.get_valid_moves()
                game.play_move(choice(moves))

            while node is not None:
                game_over, winner = game.check_game_over()
                node.back_prop(game_over, winner)
                node = node.parent

        return sorted(self.root.children, key=lambda c: c.q)[-1].move
