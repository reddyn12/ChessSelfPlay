

# %%

import os
import sys
import token
import jax
import jax.numpy as jnp
import tinygrad
import tokenizer
import random
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1

vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
vocabSize = len(vocabDecode)
games = open('data/ELO_2000_UCI.txt', 'r').read()
games = games.splitlines()


games = games[:10]


tokenizedGames = []

for g in games:
    arr = jnp.array(tokenizer.tokenizeLine(g, vocab), dtype=jnp.int32)
    # arr = tokenizer.tokenizeLine(g, vocab)


    # arr = arr[:random.randint(1, len(arr)-1)]
    tokenizedGames.append(arr)

paddedGames = tokenizer.pad_sequences(tokenizedGames, vocab['<PAD>'])

# %%
print("Converting to jnp array")
JtokenizedGames = jnp.array(paddedGames, dtype=jnp.int32)
print("FINISHED converting to jnp array")


# %%
# FASTEST JAX APPEND
JtokenizedGames = jnp.vstack(paddedGames, dtype=jnp.int32)

# %%
from model import Tranformer, GPTConfig, ChessGPT

config = GPTConfig()
config.n_vocab = vocabSize
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)
params = chessModel.init(jax.random.PRNGKey(0), JtokenizedGames[:2])

outputTest1 = chessModel.apply(params, JtokenizedGames[0:1])
outputTest2 = chessModel.apply(params, JtokenizedGames[1:2])
# vars = chessModel.init(jax.random.PRNGKey(0), tokenizedGames[-1])
# %%
outputTest1
# %%
outputTest2
# %%
outputTest3 = chessModel.apply(params, JtokenizedGames[0:1])
outputTest3
# %%
print(JtokenizedGames.shape)
tot = chessModel.apply(params, JtokenizedGames)
print(tot.shape)
print(outputTest1.shape)
# %%
for k in params.keys():
    print(k)
params['params']['blocks_1'].keys()

# %%
