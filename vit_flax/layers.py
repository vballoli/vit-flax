from flax import nn


class FeedForward(nn.Module):
  """Simple FeedForward module: x -> Dense(x, latent_dim) -> GeLU -> Dense(x, original_dim)"""

  def apply(self, x, latent_dim):
    """Applies two linear transformations with Gelu to input

    :param x: Input tensor.
    :param latent_dim: FC latent dim.
    """
    dim = x.shape[-1]
    x = nn.Dense(x, latent_dim)
    x = nn.gelu(x)
    x = nn.Dense(x, dim)
    return x


class Residual(nn.Module):
  """Simple residual function: x -> fn(x) + x"""

  def apply(self, x, residual_fn):
    """Applies residual(skip) connection to a residual_fn block

    :param x: Input tensor.
    :param residual_fn: Callable function that takes in tensor as input.
    """
    return residual_fn(x) + x


class PreNorm(nn.Module):
  """A function applied to normalized input"""
  def apply(self, x, norm, fn):
    """Applies a function to normalized input

    :param x: Input tensor.
    :param norm: Normalization module
    :param fn: Callable function that takes in tensor as input

    :return: Output of function after normalizing input
    """
    return fn(norm(x))


class Transformer(nn.Module):
  """A simple implementation of a Transformer"""
  def apply(self, x, depth, num_heads, feed_forward_dim_1):
    """Applies a residual normalized attention(Transformer) to input

    :param x: Input tensor.
    :param depth: Number of layers of Residual-normalized attention layers.
    :param num_heads: Number of attention heads
    :param feed_forward_dim: FC dimension

    :return: Transformer output embedding
    """
    attention = nn.SelfAttention.partial(num_heads=num_heads)
    norm = nn.LayerNorm
    norm_attention = PreNorm.partial(norm=norm, fn=attention)
    residual_attention = Residual.partial(residual_fn=norm_attention)

    forward = FeedForward.partial(latent_dim=feed_forward_dim_1)
    norm_forward = PreNorm.partial(norm=norm, fn=forward)
    residual_forward = Residual.partial(residual_fn=norm_forward)

    for _ in range(depth):
      x = residual_forward(residual_attention(x))

    return x
