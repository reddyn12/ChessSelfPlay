import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

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
class ChessTransformer(nn.Module):
    def __init__(self, ctxLen, nEmbed, nHead, nLayers, nFF):
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
    
testModel = ChessTransformerRANDOM()
# nn.LayerNorm()
# nn.relu()
# nn.Embed()
# nn.Dense()
# nn.DenseGeneral()
# nn.softmax()
# nn.Dropout()
