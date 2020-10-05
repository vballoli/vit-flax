from vit_flax.layers import Transformer

import jax
from jax import numpy as jnp

def test_transformer():
  rng = jax.random.PRNGKey(0)
  output, _ = Transformer.init_by_shape(
    rng, [((3, 3, 3), jnp.float32)], 3, 3, 3
  )
  assert output.shape == (3,3,3), "Wrong Transformer shape"
