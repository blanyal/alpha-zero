import sys

sys.path.append('..')


class Board:

    def __init__(self):
        self.current_player = 1
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]

    def play_move(self, move, player):
        (x, y) = move

        assert 2 >= x >= 0 and 2 >= y >= 0 and move == int(x) and move == int(y), 'Invalid Move'
        assert self.board[x][y] == 0, 'Move already played'

        self.board[x][y] = player

    def get_possible_moves(self):
        moves = []

        for x in range(3):
            for y in range(3):
                if self.board[x][y] == 0:
                    moves.append((x, y))

        return moves

    def print_board(self):
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == 0:
                    print('-', end='')
                elif self.board[x][y] == 1:
                    print('X', end='')
                elif self.board[x][y] == 2:
                    print('O', end='')

            print('n')
