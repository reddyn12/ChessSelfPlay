

# %%
from turtle import up
from model import Tranformer, GPTConfig, ChessGPT, cross_entropy_loss
import os
import jax
import optax
import jax.numpy as jnp
import tokenizer
import sys
import jax
import optax
from tqdm import tqdm

# %%
randKEY = jax.random.PRNGKey(seed=123)
# jax.random.
# print("rand key", randKEY)
# sys.exit()
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
def makeTargets(x):
    data = x[:, :-1]
    target = x[:, -1:]
    return data, target
def getBatch(games, size = 10, clip = None):
    # k = jax.random.PRNGKey(0)
    global randKEY

    randKEY, k = jax.random.split(randKEY)
    idx = jax.random.randint(k, (size,), 0, len(games))
    batch = jnp.take(games, idx, axis=0)
    min_length = jnp.min(jnp.sum(batch != 0, axis = 1))
    print("Min Length", min_length, games.shape, batch.shape)
    randKEY, k = jax.random.split(randKEY)
    randInd = jax.random.randint(k, (1,), 2, min_length)
    print("Get Batch RAndom",randInd)
    batch = batch[:, :randInd[0]]
    return batch



vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()
vocabSize = len(vocabDecode)
games = open('data/ELO_2000_UCI.txt', 'r').read()
games = games.splitlines()

 
# games = games[:30]


tokenizedGames = []

for g in games:
    arr = jnp.array(tokenizer.tokenizeLine(g, vocab), dtype=jnp.int32)
    # arr = tokenizer.tokenizeLine(g, vocab)


    # arr = arr[:random.randint(1, len(arr)-1)]
    tokenizedGames.append(arr)
    # tokenizedGames = jnp.vstack((tokenizedGames, arr))
    
print("Converting to jnp array")
JtokenizedGames = tokenizer.pad_sequences(tokenizedGames, vocab['<PAD>'])

print("FINISHED converting to jnp array")


# # %%
# # FASTEST JAX APPEND
# JtokenizedGames = jnp.vstack(paddedGames, dtype=jnp.int32)



config = GPTConfig()
config.vocab_size = vocabSize
config.n_layer = 12
config.n_head = 12
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)
randKEY, k = jax.random.split(randKEY)
params = chessModel.init(k, JtokenizedGames[:2])
# p1 = chessModel.init(k, JtokenizedGames[5:7])
# print("Params", params['params']['wte'])
# print("Params", p1['params']['wte'])



# %%
# jnp.save('ELO_2000_UCI_Token.npy', JtokenizedGames)
def getLossEstimate(size=100):
    b = getBatch(JtokenizedGames, size)
    d,t = makeTargets(b)
    logits= chessModel.apply(params, d)

    loss = cross_entropy_loss(logits, t) 
    return loss

# %%
nBatches = 1000
losses = []
# for i in range(nBatches):
#     b = getBatch(JtokenizedGames, 10)
#     d,t = makeTargets(b)
#     logits, loss = chessModel.apply(params, d, t)
#     ans = logits[:, -1]
#     ansTokens = jnp.argmax(ans, axis=-1)
#     print(i, " | Loss", loss)
# import jax.numpy as jnp

# # ...

# Create the Adam optimizer
optimizer = optax.adam(learning_rate=1e-3)

# Define the update function\
def buildUpdateJit(optimizer):
    
    @jax.jit
    def update(params, opt_state, batch):
        d,t = makeTargets(batch)
        logits = chessModel.apply(params, d)
    
        loss, grads = jax.value_and_grad(cross_entropy_loss)(logits, t)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    return update
# ...
update = buildUpdateJit(optimizer)
# Initialize the optimizer state
opt_state = optimizer.init(params)


# ...

# Training loop
nBatches = 1000
losses = []
for i in tqdm(range(nBatches)):
    b = getBatch(JtokenizedGames, 100)
    # d, t = makeTargets(b)
    # logits= chessModel.apply(params, d)
    # loss, grads = jax.value_and_grad(cross_entropy_loss)(logits, t)
    print(opt_state)
    params, opt_state, loss = update(params, opt_state, b)
    # ans = logits[:, -1]
    # ansTokens = jnp.argmax(ans, axis=-1)
    print(i, " | Loss", loss)
    # print('ans', ans.shape, ansTokens.shape)
    # print(tokenizer.decodeArray(ansTokens, vocabDecode))
# %%
# d
# jnp.equal()
# %%
