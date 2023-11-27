

# %%

import os
import sys
import jax
import jax.numpy as jnp
import tinygrad
import tokenizer

CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1

vocab = tokenizer.makeVocabUCI_SMALL()
games = open('data/ELO_2000_UCI.txt', 'r').read()
games = games.splitlines()

tokenizedGames = []

for g in games:
    arr = tokenizer.tokenizeLine(g, vocab)
    print(len(arr))
    tokenizedGames.append(arr)

# %%
# ans = jnp.array(tokenizedGames)
# print(ans.shape)
print(len(tokenizedGames))
mean_length = jnp.mean(jnp.array([len(arr) for arr in tokenizedGames]))
print(mean_length)



# %%
