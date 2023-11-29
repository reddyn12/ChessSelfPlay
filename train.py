

# %%
from model import Tranformer, GPTConfig, ChessGPT
import os
import sys
import token
import jax
import optax
import jax.numpy as jnp
import tinygrad
import tokenizer
import random
randKEY = jax.random.PRNGKey(seed=123)
# jax.random.
# print("rand key", randKEY)
# sys.exit()
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
def makeTargets(x):
    data = x[:, :-1]
    target = x[:, -1:]
    return data, target
def getBatch(games, size = 10):
    # k = jax.random.PRNGKey(0)
    global randKEY
    randKEY, k = jax.random.split(randKEY)
    idx = jax.random.randint(k, (size,), 0, len(games))
    batch = jnp.take(games, idx, axis=0)
    min_length = jnp.min(jnp.sum(batch != 0, axis = 1))
    print("Min Length", min_length)
    randKEY, k = jax.random.split(randKEY)
    randInd = jax.random.randint(k, (1,), 2, min_length)
    print("Get Batch RAndom",randInd)
    batch = batch[:, :randInd[0]]
    return batch



vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
vocabSize = len(vocabDecode)
games = open('data/ELO_2000_UCI.txt', 'r').read()
games = games.splitlines()

 
games = games[:30]


tokenizedGames = []

for g in games:
    arr = jnp.array(tokenizer.tokenizeLine(g, vocab), dtype=jnp.int32)
    # arr = tokenizer.tokenizeLine(g, vocab)


    # arr = arr[:random.randint(1, len(arr)-1)]
    tokenizedGames.append(arr)

paddedGames = tokenizer.pad_sequences(tokenizedGames, vocab['<PAD>'])


print("Converting to jnp array")
JtokenizedGames = jnp.array(paddedGames, dtype=jnp.int32)
print("FINISHED converting to jnp array")


# # %%
# # FASTEST JAX APPEND
# JtokenizedGames = jnp.vstack(paddedGames, dtype=jnp.int32)



config = GPTConfig()
config.vocab_size = vocabSize
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)
randKEY, k = jax.random.split(randKEY)
params = chessModel.init(k, JtokenizedGames[:2])
# p1 = chessModel.init(k, JtokenizedGames[5:7])
# print("Params", params['params']['wte'])
# print("Params", p1['params']['wte'])


# %%
# testJTokenArr = jnp.array(tokenizedGames, dtype=jnp.int32)
b = getBatch(JtokenizedGames, 10)
# b = getBatch(tokenizedGames, 10)
d,t = makeTargets(b)
# %%
d.shape
# %%
ans, loss = chessModel.apply(params, d, t)
ansTokens = jnp.argmax(ans, axis=-1)
ans.shape
# %%
ansTokens.shape
# %%
ans
# %%
d
# %%
vocabDecode[2423]
# %%
