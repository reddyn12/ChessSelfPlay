

# # %%
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




print(jax.devices())
# print(jax.local_devices())

# jax.config.update("jax_default_float_precision", "float16")
# print(jax.config.values['jax_default_float_precision'])
# # jax.random.key()
# # jax.random.PRNGKey()
# temp = jnp.ones((1,1))
# print(temp.dtype)
# sys.exit()
nBatches = 10000
BATCH_SIZE = 128
BLOCK_SIZE = 400
# BLOCK_SIZE = 512
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
RAND_SEED = 123
# randKEY = jax.random.PRNGKey(RAND_SEED)
# test = jax.random.randint(randKEY, (10,), 0, 100)
# del randKEY

@jax.jit
def getBatch():
    # k = jax.random.PRNGKey(0)
    # global randKEY
    # global JtokenizedGames
    # randKEY, k = jax.random.split(randKEY)
    # idx = jax.random.randint(jax.random.PRNGKey(RAND_SEED), (BATCH_SIZE,), 0, len(JtokenizedGames))

    idx = np.random.randint(0, len(JtokenizedGames), (BATCH_SIZE,))
    batch = jnp.take(JtokenizedGames, idx, axis=0)
    return batch

@jax.jit
def splitGame(x:jnp.array):
    # global randKEY
    ind = jnp.argmax(jnp.equal(x, PAD_TOKEN), axis=0)
    # randKEY, k = jax.random.split(randKEY)
    # idx = jax.random.randint(jax.random.PRNGKey(RAND_SEED), (1,), 2, ind)[0]

    idx = np.random.randint(2, ind)
    # print(ind, 'with split at', idx)
    maskY = jnp.where(jnp.arange(x.shape[0]) <= idx, 1, 0)
    # print(maskY)
    maskX = jnp.where(jnp.arange(x.shape[0]) < idx, 1, 0)
    # print(maskX)
    return x*maskX, x*maskY
@jax.jit
def splitGames(batch:jnp.array):
    d,t = jax.vmap(splitGame)(batch)
    return d,t

@jax.jit
def getLoss(params, d, t):
    # Calculate the mask
    mask = jnp.equal(d, PAD_TOKEN)
    mask = 1 - mask  # Invert the mask

    # Apply the mask to logits
    logits = chessModel.apply(params, d)
    logits = logits * mask[:, :, None]  # None is used to match the shape of logits
    t_one_hot = jax.nn.one_hot(t, config.vocab_size)
    loss = optax.softmax_cross_entropy(logits, t_one_hot)
    loss = jnp.mean(loss * mask)  # Apply the mask to the loss
    return loss

    # # Create a mask for pad tokens
    # # SWITCHED t WITH d vvvvvv
    # pad_mask = jnp.where(d == vocab['<PAD>'], 0, 1)
    
    # logits = chessModel.apply(params, d)
    # logits = logits[:, -1, :]
    
    # t_one_hot = jax.nn.one_hot(t, config.vocab_size)
    
    # loss = optax.softmax_cross_entropy(logits, t_one_hot)
    # loss = jnp.mean(loss * pad_mask)  # Apply the mask to the loss
    # return loss


# @jax.jit
# def makeTargets(x):
#     data = x[:, :-1]
#     target = x[:, -1]
#     return data, target


print("Loading Vocab")
vocabPath = 'data/ELO_2000_UCI.txt'
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
PAD_TOKEN = vocab['<PAD>']
vocabSize = len(vocabDecode)
print("Opening Games File")
file = open('data/ELO_2000_UCI.txt', 'r')
print("Reading Games File")

games = file.read(10000)
print("Closing Games File")
file.close()
print('Spliting Games')
games = games.splitlines()
print("FNIISHED Spliting Games File")
print('Length of GAMES:',len(games))
sys.exit()
 
# games = games[100000:130000]
games = games[:13000]

tokenizedGames = []
print("Tokenizing Games")
for g in tqdm(games):
    # g = g[:min((len(g), 500))]
    arr = jnp.array(tokenizer.tokenizeLine(g, vocab, BLOCK_SIZE, pad=True), dtype=jnp.int16)
    tokenizedGames.append(arr)


    # tokenizedGames = jnp.vstack((tokenizedGames, arr))
# print(tokenizedGames[180:210])
# sys.exit()   


print("Converting to jnp array")
# JtokenizedGames = tokenizer.pad_sequences(tokenizedGames, vocab['<PAD>'])
JtokenizedGames = jnp.vstack(tokenizedGames)
print("FINISHED converting to jnp array")

# import time
# print("Sleeping for 20 seconds")
# time.sleep(20)

config = GPTConfig()
config.vocab_size = vocabSize
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)
# randKEY, k = jax.random.split(randKEY)

b = getBatch()
d,t = splitGames(b)
params = chessModel.init(jax.random.PRNGKey(RAND_SEED), d)

print('Casting to float16')
params = jax.tree_map(lambda x: x.astype(jnp.float16), params)
print('FINISHED Casting to float16')

# p1 = chessModel.init(k, JtokenizedGames[5:7])
# print("Params", params['params']['wte'])
# print("Params", p1['params']['wte'])



# b = getBatch()
# d,t = splitGames(b)
# # dd,tt = makeTargets(b)
# print(d.shape, t.shape)
# print(d)
# # print()
# print(d[0])
# print(t[0])
# # print(b[0])
# print(getLoss(params, d, t))
# sys.exit()

# # %%
losses = []
# Create the Adam optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

print("Starting Training")
# Training loop
# 96 min 1000 batches with size batch 100 -CPU

losses = []
for i in tqdm(range(nBatches)):
    b = getBatch()
    # d,t = makeTargets(b)
    d,t = splitGames(b)
    loss, grads = jax.value_and_grad(getLoss)(params, d, t)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    print(i, " | Loss", loss)

    if i%100==20:
        saveWeights('model_weights.pkl', params)
        
# # %%
for i in range(3):
    b = getBatch()
    # d,t = makeTargets(b)
    d,t = splitGames(b)
    loss = getLoss(params, d, t)
    print(loss)
