
import functools
from functools import reduce
import time
# from model import Tranformer, GPTConfig
#  #, ChessGPT, cross_entropy_loss
import os
from turtle import st
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
from flax import jax_utils
from flax.training import train_state
meanFn = functools.partial(jnp.mean, axis=0)
@jax.jit
def mean_dict(dict):
    return jax.tree_map(meanFn, dict)

deviceCnt = jax.device_count()
print('Device Count', deviceCnt)
INT_DTYPE = jnp.int16
FLOAT_DTYPE = jnp.float32
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
PAD_TOKEN = vocab['<PAD>']
nBatches = 100000
# DROP TRailing 4 if on A100
BATCH_SIZE = 128//4//1 #* deviceCnt
# BATCH_ACC = 16//4
BATCH_ACC = 32//2#*4
# BLOCK_SIZE = 400
BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 1235
VOCAB_SIZE = len(vocabDecode)
randKEY = jax.random.PRNGKey(seed=RAND_SEED)
BATCH_SIZE_CUM = None

config = model.GPTConfig()
config.vocab_size = VOCAB_SIZE
config.n_layer = 12
config.n_head = 12//2
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

savedTokenGamesFile = 'tokenizedGames.npy'
modelSavedFile = 'model_weights_PARALLEL.pkl'

# savedTokenGames = None
savedTokenGames = 'tokenizedGames.npy'
# modelSaved = None
modelSaved = 'model_weights_PARALLEL.pkl'
if savedTokenGames is None:
    print("Loading Games")
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

    tokenizedGames = []
    print("Tokenizing Games")
    for g in tqdm(games):
        # g = g[:min((len(g), 500))]
        arr = jnp.array(tokenizer.tokenizeLine(g, vocab, BLOCK_SIZE, pad=True), dtype=jnp.int16)
        tokenizedGames.append(arr)

    print("Converting to jnp array")
    JtokenizedGames = jnp.vstack(tokenizedGames)
    print("FINISHED converting to jnp array")

    print('Saving Tokenized Games')
    jnp.save('tokenizedGames.npy', JtokenizedGames)
    print('Finished Saving Tokenized Games')
else:
    print('Loading Tokenized Games FROM SAVE')
    JtokenizedGames = jnp.load(savedTokenGames)
    # may need to add to the if loop also
    JtokenizedGames = JtokenizedGames[:1000000]

    print('Finished Loading Tokenized Games FROM SAVE')
    # sys.exit()
randKEY, rngs = jax.random.split(randKEY)
rngs = jax.random.split(rngs, deviceCnt)
if modelSaved is None:
    print('MAKING model State')
    # stateTest = model.create_train_statePMAP(rngs, config, hyperconfig)
    state = model.create_train_state(rngs[0], config, hyperconfig)
    state = jax_utils.replicate(state)
    print('Finished making model State')
    # print(type(stateTest))
    # print('MODELSAVED NONE',stateTest.params['wpe']['embedding'].shape)
    print('V2-MODELSAVED NONE',state.params['wpe']['embedding'].shape)
    modelSaved = modelSavedFile
    # sys.exit()
else:
    print('LOADING model State')
    # state = model.loadTrainStatePMAP(rngs, modelSaved, config)
    state = model.loadTrainState(modelSaved, config)
    # state = state*deviceCnt
    state = jax_utils.replicate(state)
    # state = jax_utils.replicate(state)
    print('Finished Loading model State')
    # print(type(state.params))
    # print(state.params['wpe']['embedding'].shape)
    # sys.exit()

# @jax.jit
def getBatchSplit(randKey:jax.dtypes.prng_key):
    randKey, k = jax.random.split(randKey)
    idx = jax.random.randint(k, (BATCH_SIZE,), 0, len(JtokenizedGames))
    batch = jnp.take(JtokenizedGames, idx, axis=0)
    d,t, idxs, randKey = splitGames(batch,randKey)
    return d,t, idxs, randKey
# @jax.jit
def splitGame(x:jnp.array, randKey:jax.dtypes.prng_key):
    ind = jnp.argmax(jnp.equal(x, PAD_TOKEN), axis=0)
    idx = jax.random.randint(randKey, (1,), 2, ind)[0]
    maskY = jnp.where(jnp.arange(x.shape[0]) <= idx, 1, 0)
    maskX = jnp.where(jnp.arange(x.shape[0]) < idx, 1, 0)
    return x*maskX, x*maskY, idx
# @jax.jit
def splitGames(batch:jnp.array, randKey:jax.dtypes.prng_key):
    randKeys = jax.random.split(randKey, batch.shape[0])
    randKey, k = jax.random.split(randKey)
    d,t,idxs = jax.vmap(splitGame)(batch,randKeys)
    # print('SPLITGAMES IDX CHECK:',idxs)
    return d,t, idxs, randKey

# randKEY, rngs = jax.random.split(randKEY)
d,t, idxs, randKEY = getBatchSplit(randKEY)
print('IDX CHECK:',idxs)
print(d[0])
print(t[0])