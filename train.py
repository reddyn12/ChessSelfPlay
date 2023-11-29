

# %%

import os
import sys
import jax
import jax.numpy as jnp
import tinygrad
import tokenizer

CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1

vocab, vocabSize = tokenizer.makeVocabUCI_SMALL()
games = open('data/ELO_2000_UCI.txt', 'r').read()
games = games.splitlines()


games = games[:10]


tokenizedGames = []

for g in games:
    arr = jnp.array(tokenizer.tokenizeLine(g, vocab), dtype=jnp.int32)
    # print(len(arr))
    
    tokenizedGames.append(arr)



print("Converting to jnp array")
tokenizedGames = jnp.array(tokenizedGames, dtype=jnp.int32)
print("FINISHED converting to jnp array")
# jnp.as


# ans = jnp.array(tokenizedGames)
# print(ans.shape)
# print(len(tokenizedGames))
# mean_length = jnp.mean(jnp.array([len(arr) for arr in tokenizedGames]))
# print(mean_length)



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
params = chessModel.init(jax.random.PRNGKey(0), tokenizedGames[-1])
outputTest = chessModel.apply(params, tokenizedGames[0])
# vars = chessModel.init(jax.random.PRNGKey(0), tokenizedGames[-1])
# %%
outputTest.shape
# %%
tokenizedGames[0].dtype
# type(tokenizedGames)
# %%
