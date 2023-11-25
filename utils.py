import chess
import chess.svg


from IPython.display import display, HTML

def display_board(board):
    display(HTML(chess.svg.board(board=board)))
games = open('data/ELO_2000.txt', 'r').read()
games = games.splitlines()
g = games[0]
board = chess.Board()
moves = g.split(' ')
print(moves)
for move in moves:
    # print(move)
    if '.' in move or move == '1-0' or move == '0-1' or move == '1/2-1/2':
        continue
    else:
        board.push_san(move)
        print(board)
        print('\n')
        display_board(board)