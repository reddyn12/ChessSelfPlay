import os
import sys
import chess
import chess.svg
from tqdm import tqdm
games = open('data/ELO_2000.txt', 'r').read()
games = games.splitlines()
print(len(games))
# games = games[:200000]


#GAME: 113239 - PROBLEMATIC

# from utils import display_board
newGames = []
# g = games[100]
# games = games[:200000]
for g in tqdm(games):
    # print("GAME:", i)


    board = chess.Board()
    moves = g.split(' ')
    # # newGames.append(len(moves))
    # print(moves)
    newMoves = []
    valid = True
    for move in moves:
        special = False
        # print("TOKEN:", move)
        if '.' in move or move == '1-0' or move == '0-1' or move == '1/2-1/2' or move == '*':
            special = True
        else:
            
            try:
                m = board.push_san(move)
            except Exception as e:
                print("EXCEPTION:", e)
                print("GAME:", g)
                print("MOVE:", move)
                print("MOVES:", moves)
                print("NEW MOVES:", newMoves)
                print("BOARD:", board)
                with open('data/PROBLEM_GAMES.txt', 'a') as file:
                    file.write(g)
                    file.write('\n')
                valid = False
                break
                # sys.exit()
            
            # print('MOVE UCI',m.uci())
            # print('MOVE SAN',m)

            # print(board)
            # print(board.piece_at(chess.parse_square('e2')))
            # print('\n')
                # display_board(board)
        if special:
            newMoves.append(move)
        else:
            newMoves.append(m.uci())
    if valid:
        newGames.append(' '.join(newMoves))
newGameText = '\n'.join(newGames)
#output to file
print("Writing to file")
f = open('data/ELO_2000_UCI.txt', 'w')
f.write(newGameText)
f.close()

# import jax.numpy as jnp
# print("Mean Length COMPUTING")

# newGames = jnp.array(newGames, dtype=jnp.int32)

# import matplotlib.pyplot as plt

# plt.hist(newGames, bins=10)
# plt.xlabel('Number of Moves')
# plt.ylabel('Frequency')
# plt.title('Distribution of Numbers in newGames')
# plt.show()

# print("Mean Length", jnp.mean(newGames))