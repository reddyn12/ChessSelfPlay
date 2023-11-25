import os
import sys
import jax

vocab = {}
def makeVocab():

    v = ['0-1', '1-0', '1/2-1/2', 'O-O', 'K', 'Q', 'N', 'B', 'R']
    for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        for j in ['1', '2', '3', '4', '5', '6', '7', '8']:
            v.append(i+j)
            v.append(i+j+'+')
            v.append(i+j+'#')
            v.append('x'+i+j)
            if j=='8' or j=='1':
                v.append(i+j+'=')

    for i in range(0, 170):
        v.append(str(i+1)+'.')
def loadVocab():
    # write jax code to load dict
    pass


def tokenize(text: str):
    elements = text.split(' ')
    return [vocab[element] for element in elements]


print(tokenize('d j 4 r'))