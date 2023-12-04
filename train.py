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




time.sleep(1000)