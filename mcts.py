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
import math


class TreeNode(object):
    """Represents a board state and stores statistics for actions at that state.

    Attributes:
        Nsa: An integer for visit count.
        Wsa: A float for the total action value.
        Qsa: A float for the mean action value.
        Psa: A float for the prior probability of reaching this node.
        action: A tuple(row, column) of the prior move of reaching this node.
        children: A list which stores child nodes.
        parent: A TreeNode representing the parent node.
    """

    def __init__(self, parent=None, action=None, psa=0.0):
        """Initializes TreeNode with the initial statistics and data."""
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = psa
        self.action = action
        self.children = []
        self.parent = parent

    def is_not_leaf(self):
        """Checks if a TreeNode is a leaf.

        Returns:
            A boolean value indicating if a TreeNode is a leaf.
        """
        if len(self.children) > 0:
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
            uct = child.Qsa + child.Psa * c_puct * (
                    math.sqrt(self.Nsa) / 1 + child.Nsa)
            if uct > highest_uct:
                highest_uct = uct
                highest_index = idx

        return self.children[highest_index]

    def expand_node(self, game, psa_vector):
        """Expands the current node by adding valid moves as children.

        Args:
            game: An object containing the game state.
            psa_vector: A list containing move probabilities for each move.
        """
        valid_moves = game.get_valid_moves()
        for idx, move in enumerate(valid_moves):
            if move[0] is not 0:
                action = (move[1], move[2])
                self.add_child_node(parent=self, action=action,
                                    psa=psa_vector[idx])

    def add_child_node(self, parent, action, psa=0.0):
        """Creates and adds a child TreeNode to the current node.

        Args:
            parent: A TreeNode which is the parent of this node.
            action: A tuple(row, column) of the prior move to reach this node.
            psa: A float representing the raw move probability for this node.

        Returns:
            The newly created child TreeNode.
        """

        child_node = TreeNode(parent=parent, action=action,
                              psa=psa)
        self.children.append(child_node)
        return child_node

    def back_prop(self, wsa, v):
        """Update the current node's statistics based on the game outcome.

        Args:
            wsa: A float representing the action value for this state.
            v: A float representing the network value of this state.
        """
        self.Nsa += 1
        self.Wsa = wsa + v
        self.Qsa = self.Wsa / self.Nsa


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

    def search(self, game, node, temperature):
        """MCTS loop to get the best move which can be played at a given state.

        Args:
            game: An object containing the game state.
            node: A TreeNode representing the board state and its statistics.
            temperature: A float to control the level of exploration.

        Returns:
            A child node representing the best move to play at this state.
        """
        self.root = node
        self.game = game

        for i in range(CFG.num_mcts_sims):
            node = self.root
            game = self.game.clone()  # Create a fresh clone for each loop.

            # Loop when node is not a leaf
            while node.is_not_leaf():
                node = node.select_child()
                game.play_action(node.action)

                if node.is_not_leaf():
                    # Switch the board to let the opponent play.
                    game.switch_player_state()

            # Get move probabilities and values from the network for this state.
            psa_vector, v = self.net.predict(game.state)

            # Try expanding the current node.
            node.expand_node(game=game, psa_vector=psa_vector)

            # Back propagate node statistics up to the root node.
            while node is not None:
                game_over, wsa = game.check_game_over(game.player_to_eval)
                node.back_prop(wsa, v)
                node = node.parent

        highest_nsa = 0
        highest_index = 0

        # Select the child's move stochastically using a temperature parameter.
        for idx, child in enumerate(self.root.children):
            if child.Nsa ** (1.0 / temperature) > highest_nsa:
                highest_nsa = child.Nsa
                highest_index = idx

        return self.root.children[highest_index]
