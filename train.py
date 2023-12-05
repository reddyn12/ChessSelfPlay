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
deviceCnt = jax.device_count()
print('Device Count', deviceCnt)
INT_DTYPE = jnp.int16
FLOAT_DTYPE = jnp.float16
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
PAD_TOKEN = vocab['<PAD>']
nBatches = 10000
BATCH_SIZE = 128*4 #* deviceCnt

# BLOCK_SIZE = 400
BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 123
VOCAB_SIZE = len(vocabDecode)
randKEY = jax.random.PRNGKey(seed=RAND_SEED)

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

# IDXS is ind for last token in target set.. so the y_actual
@jax.jit
def getBatchSplit(randKey:jax.dtypes.prng_key):
    randKey, k = jax.random.split(randKey)
    idx = jax.random.randint(k, (BATCH_SIZE,), 0, len(JtokenizedGames))
    batch = jnp.take(JtokenizedGames, idx, axis=0)
    d,t, idxs, randKey = splitGames(batch,randKey)
    return d,t, idxs, randKey

@jax.jit
def getBatch(randKey:jax.dtypes.prng_key):

    randKey, k = jax.random.split(randKey)
    idx = jax.random.randint(k, (BATCH_SIZE,), 0, len(JtokenizedGames))
    batch = jnp.take(JtokenizedGames, idx, axis=0)
    return batch, randKey

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
@jax.jit
def getLoss(params, d, t):
    loss = optax.softmax_cross_entropy(d, t)
    loss = jnp.mean(loss)
    return loss
    logits = chessModel.apply(params, d)
    logits = logits[:, idxs-1, :]
    t = t[:, idxs]
    t_one_hot = jax.nn.one_hot(t, config.vocab_size)
    loss = optax.softmax_cross_entropy(logits, t_one_hot)
    loss = jnp.mean(loss)
    return loss
@jax.jit
def forwardClips(params, d,t,idxs):
    logits = chessModel.apply(params, d)
    logits = logits[:, idxs-1, :]
    tt = t[:, idxs]
    tt = jax.nn.one_hot(tt, config.vocab_size)
    return logits, tt
@jax.jit
def getLossOLD(params, d, t):
    maskD = jnp.equal(d, PAD_TOKEN)
    maskD = 1 - maskD
    maskT = jnp.equal(t, PAD_TOKEN)
    maskT = 1 - maskT
    logits = chessModel.apply(params, d)
    logits = logits * maskD[:, :, None]
    t_one_hot = jax.nn.one_hot(t, config.vocab_size)
    t_one_hot = t_one_hot * maskT[:, :, None]
    loss = optax.softmax_cross_entropy(logits, t_one_hot)
    loss = jnp.mean(loss)
    return loss

    # # Calculate the mask
    # mask = jnp.equal(d, PAD_TOKEN)
    # mask = 1 - mask  # Invert the mask
    # # Apply the mask to logits
    # logits = chessModel.apply(params, d)
    # logits = logits * mask[:, :, None]  # None is used to match the shape of logits
    # t_one_hot = jax.nn.one_hot(t, config.vocab_size)
    # loss = optax.softmax_cross_entropy(logits, t_one_hot)
    # loss = jnp.mean(loss * mask)  # Apply the mask to the loss
    # return loss


config = GPTConfig()
config.vocab_size = VOCAB_SIZE
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)
d = jnp.empty((BATCH_SIZE, BLOCK_SIZE), dtype=INT_DTYPE)
d_size_gb = d.size * d.itemsize / 1024**3
print('JNP Batch GB size',d_size_gb)
# dnp = snp.ones((BATCH_SIZE, BLOCK_SIZE), dtype=np.int16)
# input('Cont?')
print('Initializing PARAMS')
params = chessModel.init(jax.random.PRNGKey(0), d)
# print('Casting to PARAMS float16')
# params = jax.tree_map(lambda x: x.astype(jnp.float16), params)
# print('FINISHED Casting PARAMS to float16')
print('Making ADAM Optimizer')
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)
print('FINISHED Making ADAM Optimizer')
lossGrad = jax.jit(jax.grad(getLoss))

@jax.jit
def updateParams(params, d, t, idxs, opt_state):
    logits, tt = forwardClips(params, d, t, idxs)
    loss = getLoss(params, logits, tt)
    grads = lossGrad(params, logits, tt)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
def update(randKey:jax.dtypes.prng_key,params=params, opt_state=opt_state):
    # randKey, k = jax.random.split(randKey)
    d,t,idxs, randKey = getBatchSplit(randKey)
    # logits, tt = forwardClips(params, d, t, idxs)
    # loss = getLoss(params, logits, tt)
    # grads = lossGrad(params, logits, tt)
    params, opt_state, loss = updateParams(params, d, t, idxs, opt_state)
    
    return params, opt_state, loss
updatePmap = jax.pmap(update)
# updatePmap = jax.pmap(update, axis_name='batch', donate_argnums=(0,1,2,3))

def pmean_nested_dict(nested_dict):
    return {k: pmean_nested_dict(v) if isinstance(v, dict) else jnp.mean(v) for k, v in nested_dict.items()}

for i in tqdm(range(nBatches)):
    # randKeys = jax.random.split(randKEY, deviceCnt)
    # randKEY, k = jax.random.split(randKEY)
    # randKEY, k = jax.random.split(randKEY)

    randKEY, k = jax.random.split(randKEY)
    pmapBatch = jax.random.split(k,deviceCnt)
    # print('RANDKEY', randKEY)
    # print('PMAPBATCH', pmapBatch)
    # sys.exit()
    # d,t,idxs, randKEY_Disc = jax.vmap(getBatchSplit)(randKeys)

    # # pmapBatch = []
    # # for j in range(deviceCnt):
    # #     d,t,idxs, randKEY= getBatchSplit(randKEY)
    # #     pmapBatch.append((params,d,t,idxs, opt_state))
    
    # d,t,idxs, randKEY = getBatchSplit(randKEY)
    
    # # logits,tt = forwardClips(params, d,t,idxs)
    # # # print('LOGITS',logits.shape, 'TT', tt.shape)
    # # loss = getLoss(params, logits, tt)
    # # grads = lossGrad(params, logits, tt)
    # # updates, opt_state = optimizer.update(grads, opt_state)
    # # params = optax.apply_updates(params, updates)
    
    # losses, grads  = updatePmap(pmapBatch)
    paramsTemp, opt_stateTemp, losses = updatePmap(pmapBatch)
    # params, opt_state, loss = update(randKEY)
    # params, opt_state, loss = update(randKEY
    # params, opt_state, losses = update(params, d, t, idxs, opt_state)
    # # params, opt_state, losses = updatePmap(pmapBatch)
    # # params, opt_state, losses = updatePmap(params, d, t, idxs,opt_state)
    # loss = jnp.mean(losses)
    
    # print(grads.keys())
    # grad = pmean_nested_dict(grads)

    # print(opt_stateTemp)
    params = pmean_nested_dict(paramsTemp)
    # print(params['params'])
    print(params['params']['bloacks_0'])
    opt_state = opt_stateTemp[0]
    loss = jnp.mean(losses)

    # updates, opt_state = optimizer.update(grad, opt_state)
    # params = optax.apply_updates(params, updates)

    print(i, " | Loss", loss, losses, randKEY)
    # print(d[0, :100])
    # print(t[0, :100])

    if i%100==20:
        saveWeights('model_weights.pkl', params)
        
# # %%
losses = []
for i in range(3):
    d,t,idxs, randKEY = getBatchSplit(randKEY)
    logits,tt = forwardClips(params, d,t,idxs)
    # print('LOGITS',logits.shape, 'TT', tt.shape)
    loss = getLoss(params, logits, tt)
    print(loss)
    losses.append(loss)
print('MEAN', np.mean(losses))


print('SLEEPING')
time.sleep(1000)