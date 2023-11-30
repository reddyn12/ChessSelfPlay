import os
import sys

games = open('data/ELO_2000.txt', 'r').read()
games = games.splitlines()
print(len(games))
games = games[:200000]