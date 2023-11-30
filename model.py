import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from dataclasses import dataclass
import random
DETERMINISTIC = False
@dataclass
class GPTConfig:
    block_size: int = 901
    vocab_size: int = 2500
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

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
def cross_entropy_loss(logits, labels):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))




class ChessGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        self.config = config
    
    def setup(self):
        nn.Embed()
        pass
        
    def __call__(self, x):
        pass

class ChessTransformerRANDOM(nn.Module):
    def setup(self):
        print("SETUP")
        self.embed = nn.Embed(
            num_embeddings=100,
            features=512,
        )
        self.transformer = nn.Transformer(
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation=nn.gelu,
        )
        self.head = nn.Dense(features=100)

    def __call__(self, x):
        x = self.embed(x)
        x = self.transformer(x, x)
        x = self.head(x)
        return x

    
# testModel = ChessTransformerRANDOM()
# nn.LayerNorm()
# nn.relu()
# nn.Embed()
# nn.Dense()
# nn.DenseGeneral()
# nn.softmax()
# nn.Dropout()
# jax.nn.softmax()