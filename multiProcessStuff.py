import time
import chess
import chess.svg
import jax
# import multiprocessing
from multiprocessing import Pool, freeze_support, RLock
from tqdm import tqdm
# from utils import display_board

games = open('data/ELO_2000.txt', 'r').read()
games = games.splitlines()
print(len(games))
games = games[:200000]

newGames = []
# g = games[100]
pool = Pool(processes=2)
pbar = tqdm(total=len(games))
def process_game(g):
    # time.sleep(0.01)
    # pbar.update(1)
    # print("New Game Compute:", time.time())
    board = chess.Board()
    moves = g.split(' ')
    newMoves = []
    for move in moves:
        special = False
        if '.' in move or move == '1-0' or move == '0-1' or move == '1/2-1/2'or move == '*':
            special = True
        else:
            m = board.push_san(move)
        if special:
            newMoves.append(move)
        else:
            newMoves.append(m.uci())
    return ' '.join(newMoves)
def splitGames(games, num_workers):
    # with multiprocessing.Pool(num_workers) as p:
    #     results = p.map(process_game, games)
    # return results
    jaxArr = jax.numpy.array(games, dtype=object)
    return jax.pmap(process_game)(jaxArr)
if __name__ == '__main__':
    freeze_support() # For Windows support

    num_processes = 4
    num_jobs = 30
    random_seed = 0
    # random.seed(random_seed) 

    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

    # argument_list = [random.randint(0, 100) for _ in range(num_jobs)]

    jobs = [pool.apply_async(process_game, args=(i,n,)) for i, n in enumerate(newGames)]
    pool.close()
    newGames = [job.get() for job in jobs]

    # Important to print these blanks
    print("\n" * (len(newGames) + 1))

#     with multiprocessing.Pool(processes=4) as pool:
#         newGames = pool.map(process_game, tqdm(games))
    # with multiprocessing.Pool() as pool:
    #     for result in tqdm(pool.imap_unordered(process_game, games), total=len(games)):
    #         newGames.append(result)
# newGames = splitGames(games, 4)


# newGames = list(tqdm(pool.imap_unordered(process_game, games), total=len(games)))

# newGames = pool.imap_unordered(process_game, games)
# for result in pool.imap_unordered(process_game, games):
#     newGames.append(result)
#     pbar.update(1)
# pool.close()
# pool.join()
# pbar.close()
    newGameText = '\n'.join(newGames)
    #output to file
    f = open('data/ELO_2000_UCI.txt', 'w')
    f.write(newGameText)
    f.close()