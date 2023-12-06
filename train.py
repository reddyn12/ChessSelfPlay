import functools
from functools import reduce
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

deviceCnt = jax.device_count()
print('Device Count', deviceCnt)
INT_DTYPE = jnp.int16
FLOAT_DTYPE = jnp.float32
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
PAD_TOKEN = vocab['<PAD>']
nBatches = 10000
# DROP TRailing 4 if on A100
BATCH_SIZE = 128//4//4 #* deviceCnt
BATCH_ACC = 16//4
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
    randKeys = jax.random.split(randKey, batch.shape[0])
    randKey, k = jax.random.split(randKey)
    d,t,idxs = jax.vmap(splitGame)(batch,randKeys)
    return d,t, idxs, randKey
def custAppend(x, y):
    return jnp.append(x, y, axis=0)
def stack_dicts_helper(d1, d2):
    print(type(d1), type(d2))
    # return jax.tree_map(lambda x, y: jnp.append(x, y), d1, d2), None
    return jax.tree_map(lambda x, y: jnp.vstack((x, y)), d1, d2[0]) , d2[1:]
def stack_dicts(dicts):
    #FUNCTOOLS IS GOD
    print('Starting Stack Dicts')
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
def mean_list_dicts(dicts):
    print('MEAN PRE',dicts[0]['wpe']['embedding'].shape)
    d = stack_dicts(dicts)
    print('MEAN POST', d['wpe']['embedding'].shape)
    return jax.tree_map(meanFn, d)

def forward(rng, state):
    d,t,idxs, rng = getBatchSplit(rng)
    grads, loss, accuracy = model.apply_model(state, d,t,idxs)
    return grads, loss, accuracy

# @functools.partial(jax.pmap, static_broadcasted_argnums=(1))
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
        rng, k = jax.random.split(rng)
        grads, loss, accuracy = forward(k, state)
        print(grads['wpe']['embedding'].shape)
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
    print('GETTING GRAD MEAN')
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
    
    state, loss, accuracy = trainStepACC(rng, state)
    # state, loss, accuracy = trainStepSub(rng, state)
    # state_tuple = tuple(state.as_dict().values())
    return state, loss, accuracy
# trainStepPmap = jax.pmap(trainStepACC)

    
print('Starting Training')
for currStep in tqdm(range(nBatches)):
    randKEY, rng = jax.random.split(randKEY)
    rngs = jax.random.split(rng, deviceCnt)
    # jnp.arange(4)
    # print('state', state)
    # rngs = rngs[:,1]
    # states,losses,accuracys = jax.pmap(lambda rng, state: trainStep(rng, state))(rngs, [state]*deviceCnt)
    # states = [state]*deviceCnt
    # rng_state_tuples = tuple(list(zip(rngs, states)))
    # print(len(rng_state_tuples))
    # print(rng_state_tuples[0][1])
    # print(rng_state_tuples[0][0])

    # sys.exit()
    # states,losses,accuracys = jax.pmap(trainStep)(rng_state_tuples)
    # states,losses,accuracys = trainStepPmap(rngs, state)
    # states = jax_utils.replicate(state)
    # state_tuple = tuple(state.values())
    # state_tuple = jax_utils.replicate(state_tuple)
    # print(rngs)
    # print(type(state))
    # print(type(state.params))
    # print(type(state.opt_state))
    # print('len of OPT STATE', len(state.opt_state))
    # print(state.opt_state[1])

    # print(dir(state.opt_state[0]))
    # sys.exit()

    # states,losses,accuracys = trainStepPmap(rngs, state)
    # states = [train_state.TrainState(*state_tup) for state_tup in states_tups]
    # states, losses, accuracys = jax.pmap(lambda rng: trainStep(rng, state))(rngs)
    # state = model.average_train_state(states)

    # state, loss, accuracy = trainStep(rng, state)
    print('Replicating States')
    # states = jax_utils.replicate(state)
    # inps = [[rngs[0],state],[rngs[1],state],[rngs[2],state],[rngs[3],state]]
    print('FINISHED Replicated States')
    print('Starting PMAP Step')
    # temp = trainStepPmap(rngs, states)

    # len(states) doesnt work... * not work as expected
    # states = [state]*deviceCnt
    # print(rngs)
    # print(len(states))
    grads, loss, accuracy = trainStepACC(rng, state)
    # inps = list(zip(rngs, states))
    # temp = jax.pmap(trainStepACC, in_axes=(0, None))(rngs, state)
    # temp = trainStepPmap(inps)
    print('Finished PMAP Step')
    # 
    # print()
    # print(len(temp))
    
    
    # print(grads[0].keys())
    # g = mean_list_dicts(grads)
    # print(g.keys()
    print(grads['wpe']['embedding'].shape)
    sys.exit()
    # state, loss, accuracy = trainStep(rng)

    if currStep%20==0:
        print('INFO !!! INFO')
        print('GAMES TRAINED:',currStep*BATCH_ACC*BATCH_SIZE,'Step:',currStep*BATCH_ACC,'subset',currStep, 'Loss:', loss, 'Accuracy:', accuracy)
        # loss = jnp.mean(losses)
        # accuracy = jnp.mean(accuracys)
        # print('GAMES TRAINED:',currStep*BATCH_ACC*BATCH_SIZE*deviceCnt,'CURRENT_STEP:',currStep, 'Loss:', loss, 'Accuracy:', accuracy)

    if currStep%100==20:
        saveWeights('model_weights.pkl', state.params)
        # print('Saved Weights')



print('Finished Training')
saveWeights('model_weights.pkl', state.params)
d,t,idxs, randKey = getBatchSplit(randKEY)
g,l,a = model.apply_model(state, d,t,idxs)
print('loss: ', l, 'accuracy: ', a)
