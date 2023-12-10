# https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html

import functools
from functools import reduce
import re
import stat
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
BATCH_ACC = 16//1
# BLOCK_SIZE = 400
BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 12313
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
modelSaved = None
# modelSaved = 'model_weights_PARALLEL.pkl'
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
    # sys.exit()
    # games = games[100000:130000]

    # games = games[:13000]


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
    state = jax_utils.replicate(state)
    # state = jax_utils.replicate(state)
    print('Finished Loading model State')
    # print(type(state.params))
    # print(state.params['wpe']['embedding'].shape)
    # sys.exit()

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
    randKeys = jax.random.split(randKey, batch.shape[0])
    randKey, k = jax.random.split(randKey)
    d,t,idxs = jax.vmap(splitGame)(batch,randKeys)
    return d,t, idxs, randKey
# def custAppend(x, y):
#     return jnp.append(x, y, axis=0)
# def stack_dicts_helper(d1, d2):
#     print(type(d1), type(d2))
#     # return jax.tree_map(lambda x, y: jnp.append(x, y), d1, d2), None
#     return jax.tree_map(lambda x, y: jnp.vstack((x, y)), d1, d2[0]) , d2[1:]
@jax.jit
def stack_dicts(dicts):
    #FUNCTOOLS IS GOD
    # print('Starting Stack Dicts')
    dicts = jax.tree_map(lambda x: jax.tree_map(lambda x: jnp.array([x]), x), dicts)
    return jax.tree_map(lambda *x: jnp.vstack(x), *dicts)
     
    dicts[0] = jax.tree_map(lambda x: jnp.array([x]), dicts[0])
    # for i in range(len(dicts)):
    #     dicts[i] = jax.tree_map(lambda x: jnp.array([x]), dicts[i])

    # seq_dicts = jax.tree_leaves(dicts)
    print('Starting Reduce')
    stacked_dicts, _ = jax.lax.scan(stack_dicts_helper, dicts[0], dicts[1:])
    return stacked_dicts
    # return jax.lax.reduce(stack_dicts_helper, dicts)
# def stack_dicts(dicts):
#     def body_fun(carry, x):
#         return jax.tree_map(lambda a, b: jnp.vstack((a, b)), carry, x), None
#     print('Starting Stack Dicts')
#     dicts = jax.tree_map(lambda x: jax.tree_map(lambda x: jnp.array([x]), x), dicts)
#     print('Starting Reduce')
#     result, _ = jax.lax.scan(body_fun, dicts[0], dicts[1:])
#     return result
meanFn = functools.partial(jnp.mean, axis=0)
@jax.jit
def mean_list_dicts(dicts):
    # print('MEAN PRE',dicts[0]['wpe']['embedding'].shape)
    d = stack_dicts(dicts)
    # print('MEAN POST', d['wpe']['embedding'].shape)
    return jax.tree_map(meanFn, d)
@jax.jit
def mean_dict(dict):
    return jax.tree_map(meanFn, dict)
# @jax.jit
@jax.pmap
def forward(rng, state):
    d,t,idxs, rng = getBatchSplit(rng)
    grads, loss, accuracy = model.apply_model(state, d,t,idxs)
    return grads, loss, accuracy

@functools.partial(jax.pmap, static_broadcasted_argnums=(1))
def trainStepACC(rng, state):
    # rng, k = jax.random.split(rng)
    # k = jax.random.split(k, BATCH_ACC)
    # inp = list(zip(rng, [state]*BATCH_ACC))
    # print(inp[0])
    # g,l,a = jax.vmap(forward, in_axes=(0,None))(k, state)
    
# jax.pmap()

    # return loss, grads, accuracy
    g = [None] * BATCH_ACC
    l = jnp.zeros(BATCH_ACC, dtype=jnp.float32)
    a = jnp.zeros(BATCH_ACC, dtype=jnp.float32)
    for j in range(BATCH_ACC):

    # for j in tqdm(range(BATCH_ACC), desc='BATCH_ACC'):
        rng, k = jax.random.split(rng)
        grads, loss, accuracy = forward(k, state)
        # print(grads['wpe']['embedding'].shape)
        g[j] = grads
        l = l.at[j].set(loss)
        a = a.at[j].set(accuracy)
        # l = jax.ops.index_update(l, j, loss)
        # a = jax.ops.index_update(a, j, accuracy)
    #     # print(grads)
    #     # print(grads.keys())
    #     print(grads['wpe']['embedding'].shape)
    #     print()
    #     sys.exit()
    #     state = model.update_model(state, grads)
    # l = jnp.stack(l)
    # a = jnp.stack(a)
    
    # return g, l, a
    loss = jnp.mean(l)
    accuracy = jnp.mean(a)
    # print('PRE TREEMAP grad', g[1]['wpe']['embedding'].shape)
    # print('GETTING GRAD MEAN')
    grad = mean_list_dicts(g)
    # grad = jax.tree_map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *g)
    # grad = {}
    # for key in g[0].keys():
    #     grad[key] = jnp.mean(jnp.stack([g[i][key] for i in range(len(g))]), axis=0)
    
    # print('grad', grad.keys())
    # print('POST TREE MAP grad', grad['wpe']['embedding'].shape)
    # sys.exit()
    return grad, loss, accuracy
def trainStep(rng, state):
    # state = train_state.TrainState(*state_tuple)
    
    # grads, loss, accuracy = trainStepACC(rng, state)
    grads, loss, accuracy = forward(rng, state)
    # grads = mean_dict(grads)
    state = model.update_modelPMAP(state, grads)
    
    # loss = jnp.mean(loss)
    # accuracy = jnp.mean(accuracy)
    # state, loss, accuracy = trainStepSub(rng, state)
    # state_tuple = tuple(state.as_dict().values())
    # return state, loss, accuracy
    return state, loss, accuracy
# trainStepPmap = jax.pmap(trainStepACC)

# jax_utils.s
print('Starting Training')
for currStep in tqdm(range(nBatches)):
    # state = jax_utils.replicate(state)
    randKEY, rng = jax.random.split(randKEY)
    rngs = jax.random.split(rng, deviceCnt)
    state, losses, accuracys = trainStep(rngs, state)
    print('TRAING STEP:',state.params['wpe']['embedding'].shape)
    sys.exit()

    # print(grads['wpe']['embedding'])
    # print(type(grads))
    # print(grads['wpe']['embedding'].shape)
    # print(type(state))
    # print('yuh')
    # print(losses)
    # print(accuracys)
    # sys.exit()

    if currStep%20==0:
        # print('INFO !!! INFO')
        # print('GAMES TRAINED:',currStep*BATCH_ACC*BATCH_SIZE,'Step:',currStep*BATCH_ACC,'subset',currStep, 'Loss:', loss, 'Accuracy:', accuracy)
        loss = jnp.mean(losses)
        accuracy = jnp.mean(accuracys)
        print('GAMES TRAINED:',currStep*BATCH_SIZE*deviceCnt,'CURRENT_STEP:',currStep, 'Loss:', loss, 'Accuracy:', accuracy)
        print('LOSESS:', losses)
        print('ACCURACYS:', accuracys)
    if currStep%100==20:
        print('Saving Weights')
        saveWeights(modelSaved, state.params)
        # print('Saved Weights')



print('Finished Training')
saveWeights(modelSaved, state.params)
# randKEY, rng = jax.random.split(randKEY)
d,t,idxs, randKEY = getBatchSplit(randKEY)
g,l,a = model.apply_model(state, d,t,idxs)
print('loss: ', l, 'accuracy: ', a)
