from config import CFG
from random import choice
import numpy as np
import math


class TreeNode:

    def __init__(self, game, parent=None, move=None):
        self.n = 0
        self.q = 0
        self.move = move
        self.children = []
        self.parent = parent
        self.untried_moves = game.get_valid_moves()
        self.last_player = game.last_player

    def is_not_leaf(self):
        if len(self.children) > 0:
            return True
        return False

    def is_expanded(self):
        if len(self.untried_moves) == 0:
            return True
        return False

    def select_child(self, c_puct_arg=None):
        if c_puct_arg is None:
            c_puct = CFG.c_puct
        else:
            c_puct = c_puct_arg

        highest_uct = 0
        highest_index = 0

        for idx, child in enumerate(self.children):
            uct = child.q / child.n + c_puct * math.sqrt(2 * math.log(self.n) / child.n)
            if uct > highest_uct:
                highest_uct = uct
                highest_index = idx

        return self.children[highest_index]

    def add_child_node(self, game, parent, move):
        child_node = TreeNode(game, parent, move)
        self.children.append(child_node)
        return child_node

    def back_prop(self, last_player, winner):
        self.n += 1

        if winner == 0:
            self.q += 0.5

        if last_player == winner:
            self.q += 1
        else:
            self.q += 0


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

            while len(game.get_valid_moves()) > 0:
                moves = game.get_valid_moves()
                game.play_move(choice(moves))

            while node is not None:
                game_over, winner = game.check_game_over()
                node.back_prop(node.last_player, winner)
                node = node.parent

        best_child_node = self.root.select_child(0.0)
        return best_child_node.move
