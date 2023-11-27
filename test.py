import jax
import jax.numpy as jnp

# chack if jax is using GPU
print(jax.devices())


import tokenizer

v1 = tokenizer.makeVocabUCI()
v2 = tokenizer.makeVocabUCI_SMALL()

print(v2)

print(v1)

print("LENGth of v1: ", len(v1), "LENGth of v2: ", len(v2))