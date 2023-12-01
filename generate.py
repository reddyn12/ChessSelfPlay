
from model import Tranformer, GPTConfig, ChessGPT, cross_entropy_loss
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
    logits = model.apply(params, arr)
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
weightsFile = 'model1_CPU.pkl'
params = loadWeights(weightsFile)
vocab, vocabDecode = tokenizer.makeVocabUCI_SMALL()

config = GPTConfig()
config.vocab_size = len(vocabDecode)
config.n_layer = 12
config.n_head = 12
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
for i in range(0, 15):
    ans = getPred(chessModel, params, s, vocabDecode)
    s = s + ' ' + ans
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
