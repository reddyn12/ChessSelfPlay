import os
import sys
import jax


def makeVocabUCI():
    vocab = {}
    v = ['', '0-1', '1-0', '1/2-1/2', '*', 'O-O']     #'k', 'q', 'n', 'b', 'r'
    c = []
    for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        for j in ['1', '2', '3', '4', '5', '6', '7', '8']:
            c.append(i+j)
            # if j=='8' or j=='1':
            #     v.append(i+j+'q')
            #     v.append(i+j+'n')
            #     v.append(i+j+'b')
            #     v.append(i+j+'r')
    for i in c:
        for j in c:
            if i!=j:
                v.append(i+j)
                if j[-1]=='8' or j[-1]=='1':
                    # print(i+j)
                    v.append(i+j+'q')
                    v.append(i+j+'n')
                    v.append(i+j+'b')
                    v.append(i+j+'r')
    
    # for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
    #     for j in ['1', '2', '3', '4', '5', '6', '7', '8']:
    #         v.append(i+j)
    #         v.append(i+j+'+')
    #         v.append(i+j+'#')
    #         v.append('x'+i+j)
    #         if j=='8' or j=='1':
    #             v.append(i+j+'=')

    for i in range(0, 300):
        v.append(str(i+1)+'.')
    # print(v)
    # print(len(v))
    for i, j in enumerate(v):
        vocab[j] = i
    return vocab



# def tokenize(text: str):
#     elements = text.split(' ')
#     return [vocab[element] for element in elements]
def tokenizeLine(text, vocab):

    elements = text.split(' ')
    return [vocab[element] for element in elements]

def loadVocab():
    # write jax code to load dict
    pass
# print(tokenize('d j 4 r'))

# makeVocabUCI()