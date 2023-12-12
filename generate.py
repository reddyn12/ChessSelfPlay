
from model import Tranformer, GPTConfig
import os
import jax.numpy as jnp
import tokenizer
import sys
import jax
import optax
from tqdm import tqdm
import pickle
from utils import saveWeights, loadWeights
CONTEXT_LENGTH = tokenizer.MAX_MOVES*3+1
def getPred(model, params, s, vocabDecode):
    arr = jnp.array([tokenizer.tokenizeLine(s, vocab)], dtype=jnp.int32)
    print('TOKENIZED ARRAY', arr)
    logits = model.apply(params, arr)
    # print(logits.shape)
    ll = logits[0, 0, :]
    # print(logits.shape)
    # print(logits)
    # yemp = jax.nn.softmax(logits)
    # print(yemp.shape)
    # print(yemp)
    # print(jnp.argmin(yemp))
    # print(jnp.argmin(logits))
    # return vocabDecode[jnp.argmin(logits)]
    return vocabDecode[jnp.argmax(ll)],logits

    sys.exit()
    logits = logits[:, -1, :]
    ans = jnp.argmax(logits, axis=-1)
    ansMove = vocabDecode[ans[0]]
    # print(ans)
    print(ansMove)

    # s = s + ' ' + ansMove
    # print(s)
    # print(s.split(' '))
    return ansMove


# sys.exit()
# weightsFile = 'model/params.pkl'
weightsFile = 'model_weights_PARALLEL.pkl'
params = loadWeights(weightsFile)
params = {'params': params}
print(params.keys())
# sys.exit()
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()

config = GPTConfig()
config.vocab_size = len(vocabDecode)
config.n_layer = 12
config.n_head = 12//2
config.n_embd = 768
config.dropout = 0.0
config.block_size = CONTEXT_LENGTH
config.bias = True

chessModel = Tranformer(config)

test = jnp.array([[1,0,0],[1,2,3],[4,5,6]], dtype=jnp.int32)
logits = chessModel.apply(params, test)

logits = logits[:, -1, :]
print(logits.shape)

s = '1.'
# s=''
for i in range(0, 4):
    ans, logits = getPred(chessModel, params, s, vocabDecode)
    print(logits.shape, ans)
    temp = logits[0]
    print(temp.shape)
    print(temp)
    # for row in logits:
        # print(jnp.argmax(row), row[jnp.argmax(row)])
    
    s = s + ' ' + ans
    # print(s)
# arr = jnp.array([tokenizer.tokenizeLine(s, vocab)], dtype=jnp.int32)
# logits = chessModel.apply(params, arr)
# logits = logits[:, -1, :]
# ans = jnp.argmax(logits, axis=-1)
# ansMove = vocabDecode[ans[0]]
# print(ans)
# print(vocabDecode[ans[0]])

# s = s + ' ' + ansMove
# print(s)
# print(s.split(' '))

