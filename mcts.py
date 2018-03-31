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
"""Classes for Monte Carlo Tree Search."""
from config import CFG
from random import choice
import math


class TreeNode(object):
    """Represents a board state and stores statistics for actions at that state.

    Attributes:
        Nsa: An integer for visit count.
        Wsa: A float for the total action value.
        Qsa: A float for the mean action value.
        Psa: A float for the prior probability of reaching this node.
        move: A tuple(row, column) of the prior move of reaching this node.
        children: A list which stores child nodes.
        parent: A TreeNode representing the parent node.
        untried_moves: A list containing all valid untried moves.
        last_player: An integer indicating the player who played the last move.
    """

    def __init__(self, game, parent=None, move=None):
        """Initializes TreeNode with the initial statistics and data."""
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = 0.0
        self.move = move
        self.children = []
        self.parent = parent
        self.untried_moves = game.get_valid_moves()
        self.last_player = game.last_player

    def is_not_leaf(self):
        """Checks if a TreeNode is a leaf.

        Returns:
            A boolean value indicating if a TreeNode is a leaf.
        """
        if len(self.children) > 0:
            return True
        return False

    def is_expanded(self):
        """Checks if a TreeNode is fully expanded.

        Returns:
            A boolean value indicating if a TreeNode is a fully expanded.
        """
        if len(self.untried_moves) == 0:
            return True
        return False

    def select_child(self):
        """Selects a child node based on the AlphaZero PUCT formula.

        Returns:
            A child TreeNode which is the most promising according to PUCT.
        """
        c_puct = CFG.c_puct

        highest_uct = 0
        highest_index = 0

        # Select the child with the highest Q + U value
        for idx, child in enumerate(self.children):
            uct = (child.Wsa / child.Nsa) + child.Psa * c_puct * (
                    math.sqrt(self.Nsa) / 1 + child.Nsa)
            if uct > highest_uct:
                highest_uct = uct
                highest_index = idx

        return self.children[highest_index]

    def add_child_node(self, game, parent, move):
        """Creates and adds a child TreeNode to the current node.

        Args:
            game: An object containing the game state.
            parent: A TreeNode which is the parent of the current node.
            move: A tuple(row, column) of the prior move of reaching this node.

        Returns:
            The newly created child TreeNode.
        """
        child_node = TreeNode(game, parent, move)
        self.children.append(child_node)
        return child_node

    def back_prop(self, last_player, winner):
        """Update the current nodes statistics based on the game outcome.

        Args:
            last_player: An integer for the player who played the last move.
            winner: An integer indicating the player who won the game.
        """
        self.Nsa += 1

        # For draws
        if winner == 0:
            self.Wsa += 0.5

        # For wins or losses
        if last_player == winner:
            self.Wsa += 1
        else:
            self.Wsa += 0


class MonteCarloTreeSearch(object):
    """Represents a Monte Carlo Tree Search Algorithm.

    Attributes:
        root: A TreeNode representing the board state and its statistics.
        game: An object containing the game state.
        net: An object containing the neural network.
    """

    def __init__(self, net):
        """Initializes TreeNode with the TreeNode, board and neural network."""
        self.root = None
        self.game = None
        self.net = net

    def search(self, game):
        """MCTS loop to get the best move which can be played at a given state.

        Args:
            game: An object containing the game state.

        Returns:
            A tuple(row, column) of the best move to play at this state.
        """
        self.root = TreeNode(game)
        self.game = game

        for i in range(CFG.num_mcts_sims):
            node = self.root
            game = self.game.clone()  # Create a fresh clone for each loop.

            # Loop when node has been fully expanded but is not a leaf.
            while node.is_not_leaf() and node.is_expanded():
                node = node.select_child()
                game.play_move(node.move)

            # Play a move and add a child TreeNode if node isn't fully expanded.
            if not node.is_expanded():
                move = node.untried_moves.pop()
                game.play_move(move)
                node = node.add_child_node(game, node, move)

            # Loop until all moves are exhausted.
            while len(game.get_valid_moves()) > 0:
                moves = game.get_valid_moves()
                game.play_move(choice(moves))

            # Back propagate node statistics until the root node.
            while node is not None:
                game_over, winner = game.check_game_over()
                node.back_prop(node.last_player, winner)
                node = node.parent

        highest_nsa = 0
        highest_index = 0

        # Select the child's move with the highest visit count.
        for idx, child in enumerate(self.root.children):
            if child.Nsa > highest_nsa:
                highest_nsa = child.Nsa
                highest_index = idx

        return self.root.children[highest_index].move
