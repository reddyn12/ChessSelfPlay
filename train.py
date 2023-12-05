import time
# from model import Tranformer, GPTConfig
#  #, ChessGPT, cross_entropy_loss
import os
import jax
import model
import jax.numpy as jnp
import tokenizer
import sys
import optax
from tqdm import tqdm
import pickle
from utils import saveWeights, loadWeights
import numpy as np


deviceCnt = jax.device_count()
print('Device Count', deviceCnt)
INT_DTYPE = jnp.int16
FLOAT_DTYPE = jnp.float16
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
PAD_TOKEN = vocab['<PAD>']
nBatches = 10000
BATCH_SIZE = 128*4 #* deviceCnt
BATCH_ACC = 16
# BLOCK_SIZE = 400
BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 123
VOCAB_SIZE = len(vocabDecode)
randKEY = jax.random.PRNGKey(seed=RAND_SEED)
BATCH_SIZE_CUM = None

config = model.GPTConfig()
config.vocab_size = VOCAB_SIZE
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

hyperconfig = model.HyperConfig()
hyperconfig.BATCH_SIZE = BATCH_SIZE
hyperconfig.BATCH_ACC = BATCH_ACC
hyperconfig.BLOCK_SIZE = BLOCK_SIZE
hyperconfig.CONTEXT_LENGTH = CONTEXT_LENGTH
hyperconfig.RAND_SEED = RAND_SEED
hyperconfig.VOCAB_SIZE = VOCAB_SIZE
hyperconfig.deviceCnt = deviceCnt
hyperconfig.nBatches = nBatches
hyperconfig.BATCH_SIZE_CUM = BATCH_SIZE_CUM
hyperconfig.INT_DTYPE = INT_DTYPE
hyperconfig.FLOAT_DTYPE = FLOAT_DTYPE

# BATCH_SIZE = 64
# nBatches = 10

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

print("Converting to jnp array")
JtokenizedGames = jnp.vstack(tokenizedGames)
print("FINISHED converting to jnp array")
print('Making model State')
state = model.create_train_state(randKEY, config, hyperconfig)
print('Finished making model State')

@jax.jit
def getBatchSplit(randKey:jax.dtypes.prng_key):
    randKey, k = jax.random.split(randKey)
    idx = jax.random.randint(k, (BATCH_SIZE,), 0, len(JtokenizedGames))
    batch = jnp.take(JtokenizedGames, idx, axis=0)
    d,t, idxs, randKey = splitGames(batch,randKey)
    return d,t, idxs, randKey
@jax.jit
def splitGame(x:jnp.array, randKey:jax.dtypes.prng_key):
    ind = jnp.argmax(jnp.equal(x, PAD_TOKEN), axis=0)
    idx = jax.random.randint(randKey, (1,), 2, ind)[0]
    maskY = jnp.where(jnp.arange(x.shape[0]) <= idx, 1, 0)
    maskX = jnp.where(jnp.arange(x.shape[0]) < idx, 1, 0)
    return x*maskX, x*maskY, idx
@jax.jit
def splitGames(batch:jnp.array, randKey:jax.dtypes.prng_key):
    randKeys = jax.random.split(randKey, BATCH_SIZE)
    randKey, k = jax.random.split(randKey)
    d,t,idxs = jax.vmap(splitGame)(batch,randKeys)
    return d,t, idxs, randKey
def trainStep(rng, state):
    d,t,idxs, rng = getBatchSplit(rng)
    grads, loss, accuracy = model.apply_model(state, d,t,idxs)
    state = model.update_model(state, grads)
    return state, loss, accuracy


    

for currStep in tqdm(range(nBatches)):
    randKEY, rng = jax.random.split(randKEY)
    state, loss, accuracy = trainStep(rng, state)
    if currStep%100==0:
        print('Step:',currStep, 'Loss:', loss, 'Accuracy:', accuracy)
    if currStep%100==20:
        saveWeights('model_weights.pkl', state.params)
        # print('Saved Weights')



print('Finished Training')
saveWeights('model_weights.pkl', state.params)
d,t,idxs, randKey = getBatchSplit(randKEY)
g,l,a = model.apply_model(state, d,t,idxs)
print('loss: ', l, 'accuracy: ', a)
