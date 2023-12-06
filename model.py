import jax
import jax.numpy as jnp
import flax
import optax
import flax.linen as nn
from flax.training import train_state
from dataclasses import dataclass
import random
from tokenizer import makeVocabUCI_SMALL, CONTEXT_LENGTH
DETERMINISTIC = False
INT_DTYPE = jnp.int16
FLOAT_DTYPE = jnp.float32
vocab, vocabDecode = makeVocabUCI_SMALL()
VOCAB_SIZE = len(vocabDecode)
@dataclass
class GPTConfig:
    block_size: int = 901
    vocab_size: int = 2500
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
@dataclass
class HyperConfig:
    BATCH_SIZE: int = 128
    BATCH_ACC: int = 16
    BLOCK_SIZE: int = 512
    CONTEXT_LENGTH: int = CONTEXT_LENGTH
    RAND_SEED: int = 123
    VOCAB_SIZE: int = VOCAB_SIZE
    deviceCnt: int = 1
    nBatches: int = 10000
    BATCH_SIZE_CUM: int = None
    INT_DTYPE: int = INT_DTYPE
    FLOAT_DTYPE: int = FLOAT_DTYPE
class LayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450)."""
    # def __init__(self, config: GPTConfig):
    #     self.config = config
    config: GPTConfig
        
    def setup(self):
        self.layer = nn.LayerNorm(use_bias=self.config.bias)

    @nn.compact
    def __call__(self, x):
        return self.layer(x)

class CausalSelfAttention(nn.Module):
    """Causal self-attention (https://arxiv.org/abs/1803.02155)."""
    # def __init__(self, config: GPTConfig):
    #     self.config = config
    config: GPTConfig
        # self.qkv_features = config.n_embd
        # self.out_features = config.n_embd
    
    def setup(self):
        # nn.dot_product_attention()
        
        self.attend = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_head,
            # qkv_features=self.qkv_features,
            # out_features=self.out_features,
            dropout_rate=self.config.dropout,
            deterministic=DETERMINISTIC,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        # self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x):
        x = self.attend(x, x)
        # x = self.dropout(x)
        return x
class MLP(nn.Module):
    """MLP (https://arxiv.org/abs/1607.06450)."""
    # def __init__(self, config: GPTConfig):
    #     self.config = config
    config: GPTConfig
        
        # self.dropout_rate = config.dropout
    def setup(self):
        self.mlp_features = self.config.n_embd * 4
        self.out_features = self.config.n_embd
        self.layer1 = nn.Dense(features=self.mlp_features)
        self.gelu1 = nn.gelu
        self.layer2 = nn.Dense(features=self.out_features)
        self.dropout = nn.Dropout(rate=self.config.dropout, deterministic=DETERMINISTIC)

    @nn.compact
    def __call__(self, x):
        x = self.layer1(x)
        x = self.gelu1(x) # GELU??
        x = self.layer2(x)
        x = self.dropout(x)
        return x
class Block(nn.Module):
    """Transformer block (https://arxiv.org/abs/1706.03762)."""
    # def __init__(self, config: GPTConfig):
    #     self.config = config
    config: GPTConfig
        # self.num_heads = config.n_head
        # self.qkv_features = config.n_embd
        # self.out_features = config.n_embd
        # self.mlp_features = config.n_embd * 4
        # self.dropout_rate = config.dropout

    def setup(self):
        self.attend = CausalSelfAttention(
            self.config,
        )
        self.norm1 = LayerNorm(self.config)
        self.mlp = MLP(
            self.config,
        )
        self.norm2 = LayerNorm(self.config)
        # self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x):
        x = x + self.attend(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
        y = self.attend(x)
        x = x + self.dropout(y)
        x = self.norm1(x)
        y = self.mlp(x)
        x = x + self.dropout(y)
        x = self.norm2(x)
        return x
class Tranformer(nn.Module):
    # def __init__(self, config: GPTConfig):
    #     self.config = config
    config: GPTConfig

    def setup(self):
    # def __post_init__(self) -> None:
    #     super().__post_init__()
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=self.config.bias)
        self.blocks = [Block(self.config) for _ in range(self.config.n_layer)]
        self.layerNorm = LayerNorm(self.config)
        self.dropout = nn.Dropout(rate=self.config.dropout, deterministic=DETERMINISTIC)
        
    @nn.compact
    def __call__(self, x):
        # print(x.shape)
        B, T = x.shape
        # T = len(x)
        pos = jnp.arange(0, T)
        # torch.aran
        tokEmbed = self.wte(x)
        posEmbed = self.wpe(pos)
        x = self.dropout(tokEmbed + posEmbed)

        for block in self.blocks:
            x = block(x)
        x = self.layerNorm(x)
        # x = self.lm_head(x)

        x = self.lm_head(x)
        return x
        if targets is not None:
            x = self.lm_head(x)
            # nn.los
            loss = cross_entropy_loss(x, targets)
            # return jnp.mean(loss)
        else:
            x = self.lm_head(x[:, [-1], :])
            loss = None
        # x = self.lm_head(x)
        return x, loss
    

@jax.jit
def apply_model(state, d,t,idxs):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        logits = state.apply_fn({'params':params}, d)
        logits = logits[:, idxs-1, :]
        tt = t[:, idxs]
        tt = jax.nn.one_hot(tt, VOCAB_SIZE)
        loss = optax.softmax_cross_entropy(logits, tt)
        loss = jnp.mean(loss)
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == t[:, idxs])
    return grads, loss, accuracy
    
@jax.jit
def update_model(state:train_state.TrainState, grads):
  return state.apply_gradients(grads=grads)

def create_train_state(rng, config, hyperconfig):
    """Creates initial `TrainState`."""
    model = Tranformer(config)
#   cnn = CNN()
    d = jnp.ones((hyperconfig.BATCH_SIZE, hyperconfig.BLOCK_SIZE), dtype=hyperconfig.INT_DTYPE)
    # d_size_gb = d.size * d.itemsize / 1024**3
    # print('JNP Batch GB size',d_size_gb)
    params = model.init(rng, d)['params']
    tx = optax.adam(learning_rate=1e-3)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
def average_train_state(train_states):
    """Averages the parameters of multiple TrainState objects."""
    print('Averaging Train States')
    print('Train States:', len(train_states))
    keys = train_states[0].params.keys()
    averaged_params = {k: jax.lax.pmean(jnp.stack([state.params[k] for state in train_states]), axis=0) for k in keys}
    averaged_state = train_states[0].replace(params=averaged_params)
    return averaged_state
# def cross_entropy_loss(logits, labels):
#     log_probs = jax.nn.log_softmax(logits)
#     return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))




# class ChessGPT(nn.Module):
#     def __init__(self, config: GPTConfig):
#         self.config = config
    
#     def setup(self):
#         nn.Embed()
#         pass
        
#     def __call__(self, x):
#         pass

# class ChessTransformerRANDOM(nn.Module):
#     def setup(self):
#         print("SETUP")
#         self.embed = nn.Embed(
#             num_embeddings=100,
#             features=512,
#         )
#         self.transformer = nn.Transformer(
#             num_encoder_layers=6,
#             num_decoder_layers=6,
#             d_model=512,
#             nhead=8,
#             dim_feedforward=2048,
#             dropout=0.1,
#             activation=nn.gelu,
#         )
#         self.head = nn.Dense(features=100)

#     def __call__(self, x):
#         x = self.embed(x)
#         x = self.transformer(x, x)
#         x = self.head(x)
#         return x

    
# testModel = ChessTransformerRANDOM()
# nn.LayerNorm()
# nn.relu()
# nn.Embed()
# nn.Dense()
# nn.DenseGeneral()
# nn.softmax()
# nn.Dropout()
# jax.nn.softmax()