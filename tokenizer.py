import os
import sys
import jax
import jax.numpy as jnp

from train import CONTEXT_LENGTH

MAX_MOVES = 300
CONTEXT_LENGTH = (MAX_MOVES*3)+1
# def makeVocabUCI():
#     vocab = {}
#     v = ['<PAD>', '', '0-1', '1-0', '1/2-1/2', '*', 'O-O']     #'k', 'q', 'n', 'b', 'r'
#     c = []
#     for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
#         for j in ['1', '2', '3', '4', '5', '6', '7', '8']:
#             c.append(i+j)
#             # if j=='8' or j=='1':
#             #     v.append(i+j+'q')
#             #     v.append(i+j+'n')
#             #     v.append(i+j+'b')
#             #     v.append(i+j+'r')
#     for i in c:
#         for j in c:
#             if i!=j:
#                 v.append(i+j)
#                 if j[-1]=='8' or j[-1]=='1':
#                     # print(i+j)
#                     v.append(i+j+'q')
#                     v.append(i+j+'n')
#                     v.append(i+j+'b')
#                     v.append(i+j+'r')
    
#     # for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
#     #     for j in ['1', '2', '3', '4', '5', '6', '7', '8']:
#     #         v.append(i+j)
#     #         v.append(i+j+'+')
#     #         v.append(i+j+'#')
#     #         v.append('x'+i+j)
#     #         if j=='8' or j=='1':
#     #             v.append(i+j+'=')

#     for i in range(0, MAX_MOVES):
#         v.append(str(i+1)+'.')
#     # print(v)
#     # print(len(v))
#     for i, j in enumerate(v):
#         vocab[j] = i
#     return vocab
def makeVocabUCI_SMALL():
    vocab = {}
    v = ['<PAD>', '<EOL>','', '0-1', '1-0', '1/2-1/2', '*', 'O-O', '0000']     #'k', 'q', 'n', 'b', 'r'
    c = []
    for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        for j in ['1', '2', '3', '4', '5', '6', '7', '8']:
            c.append(i+j)
    numMin = ord('1')
    numMax = ord('8')
    charMin = ord('a')
    charMax = ord('h')
    for i in c:
        currChar = ord(i[0])
        currNum = ord(i[1])

        # Consider pawnPromotions
        tC = currChar
        tN = currNum
        while tN>numMin:
            tN-=1
            v.append(i+chr(tC)+chr(tN))
            if tN==numMin:
                v.append(i+chr(tC)+chr(tN)+'q')
                v.append(i+chr(tC)+chr(tN)+'n')
                v.append(i+chr(tC)+chr(tN)+'b')
                v.append(i+chr(tC)+chr(tN)+'r')
        tC = currChar
        tN = currNum
        while tN<numMax:
            tN+=1
            v.append(i+chr(tC)+chr(tN))
            if tN==numMax:
                v.append(i+chr(tC)+chr(tN)+'q')
                v.append(i+chr(tC)+chr(tN)+'n')
                v.append(i+chr(tC)+chr(tN)+'b')
                v.append(i+chr(tC)+chr(tN)+'r')

        
        tC = currChar
        tN = currNum
        while tC>charMin:
            tC-=1
            v.append(i+chr(tC)+chr(tN))
        tC = currChar
        tN = currNum
        while tC<charMax:
            tC+=1
            v.append(i+chr(tC)+chr(tN))
       
        tC = currChar
        tN = currNum
        while tC>charMin and tN>numMin:
            tC-=1
            tN-=1
            v.append(i+chr(tC)+chr(tN))
            if i[1]=='2':
                v.append(i+chr(tC)+chr(tN)+'q')
                v.append(i+chr(tC)+chr(tN)+'n')
                v.append(i+chr(tC)+chr(tN)+'b')
                v.append(i+chr(tC)+chr(tN)+'r')
        tC = currChar
        tN = currNum
        while tC<charMax and tN<numMax:
            tC+=1
            tN+=1
            v.append(i+chr(tC)+chr(tN))
            if i[1]=='7':
                v.append(i+chr(tC)+chr(tN)+'q')
                v.append(i+chr(tC)+chr(tN)+'n')
                v.append(i+chr(tC)+chr(tN)+'b')
                v.append(i+chr(tC)+chr(tN)+'r')
        tC = currChar
        tN = currNum
        while tC>charMin and tN<numMax:
            tC-=1
            tN+=1
            v.append(i+chr(tC)+chr(tN))
            if i[1]=='7':
                v.append(i+chr(tC)+chr(tN)+'q')
                v.append(i+chr(tC)+chr(tN)+'n')
                v.append(i+chr(tC)+chr(tN)+'b')
                v.append(i+chr(tC)+chr(tN)+'r')
        tC = currChar
        tN = currNum
        while tC<charMax and tN>numMin:
            tC+=1
            tN-=1
            v.append(i+chr(tC)+chr(tN))
            if i[1]=='2':
                v.append(i+chr(tC)+chr(tN)+'q')
                v.append(i+chr(tC)+chr(tN)+'n')
                v.append(i+chr(tC)+chr(tN)+'b')
                v.append(i+chr(tC)+chr(tN)+'r')
        knightMoves= [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for m in knightMoves:
            tC = currChar
            tN = currNum
            tC+=m[0]
            tN+=m[1]
            if tC>=charMin and tC<=charMax and tN>=numMin and tN<=numMax:
                v.append(i+chr(tC)+chr(tN))

    

    for i in range(0, MAX_MOVES):
        v.append(str(i+1)+'.')
    # print(v)
    # print(len(v))
    for i, j in enumerate(v):
        vocab[j] = i
    return vocab, v


# def tokenize(text: str):
#     elements = text.split(' ')
#     return [vocab[element] for element in elements]
def tokenizeLine(text, vocab, length=(MAX_MOVES*3)+1, truncate = True, pad = False):
    arr = []
    cnt = 0
    # elements = text.split(' ')
    # return [vocab[element] for element in elements]
    for e in text.split(' '):
            if cnt>length-2 and truncate:
                arr.append(vocab['<PAD>'])
                return arr
            arr.append(vocab[e])
            cnt+=1
    if pad:
        arr = fillPad(arr, vocab, length)
    return arr
def fillPad(arr, vocab, length=(MAX_MOVES*3)+1):
    while len(arr)<length:
        arr.append(vocab['<PAD>'])
    return arr
def pad_sequences(sequences, padding_value=0):
    max_length = max(len(seq) for seq in sequences)
    ans = jnp.array([], dtype=jnp.int16)
    # for seq in sequences:

    #     ans = jnp.vstack((ans,jnp.pad(seq, (0, max_length - len(seq)), constant_values=padding_value)))
    # return ans
    temp = [jnp.pad(seq, (0, max_length - len(seq)), constant_values=padding_value) for seq in sequences]
    return jnp.vstack(temp)

def decodeArray(arr, vocabDecode):
    return [vocabDecode[e] for e in arr]
def loadVocab():
    # write jax code to load dict
    pass
# print(tokenize('d j 4 r'))

# makeVocabUCI()