from flax import nn


class FeedForward(nn.Module):
  """Simple FeedForward module

  x -> Dense(x, latent_dim) -> GeLU -> Dense(x, original_dim)
  """
  def apply(self, x, latent_dim):
    dim = x.shape[-1]
    x = nn.Dense(x, latent_dim)
    x = nn.gelu(x)
    x = nn.Dense(x, dim)
    return x


class Residual(nn.Module):
  """Simple residual function

  x -> fn(x) + x"""
  def apply(self, x, residual_fn):
    return residual_fn(x) + x


class PreNorm(nn.Module):
  """Apply function to norm
  """
  def apply(self, x, norm, fn):
    return fn(norm(x))


class Transformer(nn.Module):
  """Simple implementation of a Transformer
  """
  def apply(self, x, depth, num_heads, feed_forward_dim_1):
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
