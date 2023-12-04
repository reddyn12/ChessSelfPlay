import time
from model import Tranformer, GPTConfig #, ChessGPT, cross_entropy_loss
import os
import jax
import jax.numpy as jnp
import tokenizer
import sys
import optax
from tqdm import tqdm
import pickle
from utils import saveWeights, loadWeights
import numpy as np

INT_DTYPE = jnp.int16
FLOAT_DTYPE = jnp.float16
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
PAD_TOKEN = vocab['<PAD>']
nBatches = 10000
BATCH_SIZE = 128
BLOCK_SIZE = 400
# BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 123
VOCAB_SIZE = len(vocabDecode)

print("Loading Vocab")
gamePath = 'data/ELO_2000_UCI.txt'

print("Opening Games File")
file = open(gamePath, 'r')
print("Reading Games File")

# PROBLEMO -------------------------- HUGE --------------------- PROBLEMO 
# games = file.read(200000000)
games = file.read()
print("Closing Games File")
file.close()
print('Spliting Games')
games = games.splitlines()
print("FNIISHED Spliting Games File")
print('Length of GAMES:',len(games))
# sys.exit()
 
# games = games[100000:130000]
games = games[:13000]

tokenizedGames = []
print("Tokenizing Games")
for g in tqdm(games):
    # g = g[:min((len(g), 500))]
    arr = jnp.array(tokenizer.tokenizeLine(g, vocab, BLOCK_SIZE, pad=True), dtype=jnp.int16)
    tokenizedGames.append(arr)


    # tokenizedGames = jnp.vstack((tokenizedGames, arr))
# print(tokenizedGames[180:210])
# sys.exit()   


print("Converting to jnp array")
# JtokenizedGames = tokenizer.pad_sequences(tokenizedGames, vocab['<PAD>'])
JtokenizedGames = jnp.vstack(tokenizedGames)
print("FINISHED converting to jnp array")

config = GPTConfig()
config.vocab_size = VOCAB_SIZE
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)
d = jnp.empty((BATCH_SIZE, BLOCK_SIZE), dtype=INT_DTYPE)
d_size_gb = d.size * d.itemsize / 1024**3
print('JNP Batch GB size',d_size_gb)
# dnp = np.ones((BATCH_SIZE, BLOCK_SIZE), dtype=np.int16)
# input('Cont?')
params = chessModel.init(jax.random.PRNGKey(0), d)
print('Casting to PARAMS float16')
params = jax.tree_map(lambda x: x.astype(jnp.float16), params)
print('FINISHED Casting PARAMS to float16')
print('Making ADAM Optimizer')
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)
print('FINISHED Making ADAM Optimizer')




print('SLEEPING')
time.sleep(1000)