from vit_flax.layers import Transformer, FeedForward

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

import flax
from flax import nn


class ViT(nn.Module):
  """ """
  def apply(self, x, patch_size, dim, depth, num_heads, dense_dims, img_size, num_classes):
    b, h, w, c = x.shape
    patch = x.reshape(b, c, h, w)
    num_patches = (img_size // patch_size) ** 2
    patch = patch.reshape(b, int((h*w)/(patch_size*patch_size)), int(patch_size*patch_size*c))
    embedding_dim = c * patch_size ** 2
    fc_embedding = nn.Dense(patch, dim)

    class_tokens = self.param('class_tokens', (b, 1, dim), nn.initializers.normal(stddev=1.0))
    pos_embedding = self.param('pos_embedding', (1, num_patches+1, dim), nn.initializers.normal(stddev=(1.0)))
    x = lax.concatenate([class_tokens, fc_embedding], dimension=1)
    x += pos_embedding
    x = Transformer(x, depth, num_heads, dense_dims[0])

    x = x[:, 0]
    x = nn.Dense(x, dense_dims[0])
    x = nn.gelu(x)
    x = nn.Dense(x, num_classes)
    return x
