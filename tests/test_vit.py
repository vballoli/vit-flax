from vit_flax import ViT

import jax
from jax import numpy as jnp

def test_vit():
  rng = jax.random.PRNGKey(0)
  output, _ = ViT.init_by_shape(
    rng, [((5, 256, 256, 3), jnp.float32)], 32, 1024, 6, 8, (2048, 2048), 256, 10
  )
  assert output.shape == (5,10), "Wrong output shape"