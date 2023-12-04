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
PAD_TOKEN = vocab['<PAD>']
nBatches = 10000
BATCH_SIZE = 128
BATCH_SIZE = 64
BLOCK_SIZE = 400
# BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 123
VOCAB_SIZE = len(vocabDecode)
randKEY = jax.random.PRNGKey(seed=RAND_SEED)

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
@jax.jit
def getBatch(randKey:jax.random.PRNGKey):
    # k = jax.random.PRNGKey(0)
    # global randKEY
    # global JtokenizedGames
    randKey, k = jax.random.split(randKey)
    idx = jax.random.randint(k, (BATCH_SIZE,), 0, len(JtokenizedGames))
    # idx = np.random.randint(0, len(JtokenizedGames), (BATCH_SIZE,))
    batch = jnp.take(JtokenizedGames, idx, axis=0)
    return batch, randKey

@jax.jit
def splitGame(x:jnp.array, randKey:jax.random.PRNGKey):
    # global randKEY
    ind = jnp.argmax(jnp.equal(x, PAD_TOKEN), axis=0)
    # randKey, k = jax.random.split(randKey)
    idx = jax.random.randint(randKey, (1,), 2, ind)[0]

    # idx = np.random.randint(2, ind)
    # print(ind, 'with split at', idx)
    maskY = jnp.where(jnp.arange(x.shape[0]) <= idx, 1, 0)
    # print(maskY)
    maskX = jnp.where(jnp.arange(x.shape[0]) < idx, 1, 0)
    # print(maskX)
    return x*maskX, x*maskY
@jax.jit
def splitGames(batch:jnp.array, randKey:jax.random.PRNGKey):
    randKeys = jax.random.split(randKey, BATCH_SIZE)
    randKey, k = jax.random.split(randKey)
    d,t = jax.vmap(splitGame)(batch,randKeys)
    return d,t, randKey
@jax.jit
def getLoss(params, d, t):
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
# dnp = np.ones((BATCH_SIZE, BLOCK_SIZE), dtype=np.int16)
# input('Cont?')
print('Initializing PARAMS')
params = chessModel.init(jax.random.PRNGKey(0), d)
print('Casting to PARAMS float16')
params = jax.tree_map(lambda x: x.astype(jnp.float16), params)
print('FINISHED Casting PARAMS to float16')
print('Making ADAM Optimizer')
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)
print('FINISHED Making ADAM Optimizer')


losses = []
for i in tqdm(range(nBatches)):
    b, randKEY = getBatch(randKEY)
    # d,t = makeTargets(b)
    d,t, randKEY = splitGames(b,randKEY)
    loss, grads = jax.value_and_grad(getLoss)(params, d, t)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    print(i, " | Loss", loss, randKEY)

    if i%100==20:
        saveWeights('model_weights.pkl', params)
        
# # %%
for i in range(3):
    b = getBatch()
    # d,t = makeTargets(b)
    d,t = splitGames(b)
    loss = getLoss(params, d, t)
    print(loss)


print('SLEEPING')
time.sleep(1000)