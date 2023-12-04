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

nBatches = 10000
BATCH_SIZE = 128
BLOCK_SIZE = 400
# BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 123
VOCAB_SIZE = len(vocabDecode)


config = GPTConfig()
config.vocab_size = VOCAB_SIZE
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)
d = jnp.ones((BATCH_SIZE, BLOCK_SIZE), dtype=jnp.int32)
params = chessModel.init(jax.random.PRNGKey(RAND_SEED), d)




print('SLEEPING')
time.sleep(1000)