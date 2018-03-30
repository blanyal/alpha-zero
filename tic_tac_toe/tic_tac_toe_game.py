from copy import deepcopy
from game import Game
from random import shuffle


class TicTacToeGame(Game):
    def __init__(self):
        super().__init__()
        self.side = 3
        self.current_player = 1
        self.state = []

        for i in range(self.side):
            self.state.append([0 * j for j in range(self.side)])

    def clone(self):
        board_clone = TicTacToeGame()
        board_clone.state = deepcopy(self.state)
        board_clone.current_player = self.current_player
        return board_clone

    def play_move(self, move):
        x = move[0]
        y = move[1]

        self.state[x][y] = self.current_player
        self.current_player = 3 - self.current_player

    def get_valid_moves(self):
        moves = []

        for x in range(self.side):
            for y in range(self.side):

                if self.state[x][y] == 0:
                    moves.append((x, y))

        shuffle(moves)
        return moves

    def check_game_over(self):
        player_a = self.current_player
        player_b = 3 - self.current_player

        for x in range(self.side):
            player_a_count = 0
            player_b_count = 0
            for y in range(self.side):
                if self.state[x][y] == player_a:
                    player_a_count += 1
                elif self.state[x][y] == player_b:
                    player_b_count += 1
            if player_a_count == self.side:
                return True, player_a
            elif player_b_count == self.side:
                return True, player_b

        for x in range(self.side):
            player_a_count = 0
            player_b_count = 0
            for y in range(self.side):
                if self.state[y][x] == player_a:
                    player_a_count += 1
                elif self.state[y][x] == player_b:
                    player_b_count += 1
            if player_a_count == self.side:
                return True, player_a
            elif player_b_count == self.side:
                return True, player_b

        player_a_count = 0
        player_b_count = 0
        for x in range(self.side):
            if self.state[x][x] == player_a:
                player_a_count += 1
            elif self.state[x][x] == player_b:
                player_b_count += 1

        if player_a_count == self.side:
            return True, player_a
        elif player_b_count == self.side:
            return True, player_b

        player_a_count = 0
        player_b_count = 0
        for x in range(self.side - 1, -1, -1):
            if self.state[x][x] == player_a:
                player_a_count += 1
            elif self.state[x][x] == player_b:
                player_b_count += 1

        if player_a_count == self.side:
            return True, player_a
        elif player_b_count == self.side:
            return True, player_b

        if len(self.get_valid_moves()) == 0:
            return True, 0

        return False, None

    def print_board(self):
        for x in range(self.side):
            for y in range(self.side):
                if self.state[x][y] == 0:
                    print('-', end='')
                elif self.state[x][y] == 1:
                    print('X', end='')
                elif self.state[x][y] == 2:
                    print('O', end='')
            print('n')
