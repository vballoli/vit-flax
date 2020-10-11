from vit_flax.layers import Transformer

from flax import nn
from jax import lax


class ViT(nn.Module):
  """Vision transformer

  :param x: Input tensor image
  :param patch_size: Patch dimension from image
  :param dim: Latent dim
  :param depth: Number of layers of Residual-normalized attention layers.
  :param num_heads: Number of attention heads
  :param dense_dims: Tuple(int, int) - (Transformer FC dim, Classifier FC dim) 
  :param img_size: Dimension of input image
  :param num_classes: Number of classification classes
  :param initializer: Flax initializer

  :return:Classification output
  """
  def apply(self, x,
            patch_size, dim, depth, num_heads, dense_dims,
            img_size, num_classes,
            initializer=nn.initializers.normal(stddev=1.0)):
    b, h, w, c = x.shape
    patch = x.reshape(b, c, h, w)
    num_patches = (img_size // patch_size) ** 2
    patch = patch.reshape(b, (h*w)//(patch_size*patch_size), c*patch_size**2)

    fc_embedding = nn.Dense(patch, dim)

    class_tokens = self.param('class_tokens', (b, 1, dim), initializer)
    pos_embedding = self.param(
      'pos_embedding', (1, num_patches+1, dim), initializer
    )
    x = lax.concatenate([class_tokens, fc_embedding], dimension=1)
    x += pos_embedding
    x = Transformer(x, depth, num_heads, dense_dims[0])

    x = x[:, 0]
    x = nn.Dense(x, dense_dims[1])
    x = nn.gelu(x)
    x = nn.Dense(x, num_classes)
    return x
