# Vision Transformer in JAX/Flax

This repository implements Vision Trasnformer(ViT) in Flax, introduced in an [ICLR paper 2021 submission](https://openreview.net/pdf?id=YicbFdNTTy), with further explanation by [Yannic Kilcher](https://www.youtube.com/watch?v=TrdevFK_am4). This repository is heavily inspired from [lucidrain's](https://github.com/lucidrains/vit-pytorch) implementation.

# Install
`pip install vit-flax`

# Usage
```python
import jax
from jax import numpy as jnp
from flax import nn
from vit_flax import ViT

rng = jax.random.PRNGKey(0)
module = ViT.partial(patch_size=32, dim=1024, depth=6, num_heads=8, dense_dims=(2048, 2048), img_size=256, num_classes=10)
_, initial_params = module.init_by_shape(
  rng, [((1, 256, 256, 3), jnp.float32)]
)
model = nn.Model(module, initial_params)

img = jax.random.uniform(rng, (1,256,256,3))
output = model(img)
```

`examples` directory contains code to train ViT on CIFAR datasets.
# Docs and references

Documentation for all the modules can be viewed [here](https://vit-flax.readthedocs.io/en/main/).

# Note
This repository is still in initial stages. Feel free to [Contact me](https://vballoli.github.io) or raise issues/PR for suggestions, improvements or bugs.

# Help needed

A recent commit introduces code for training CIFAR models in the `examples` directory. If you're using this code and have the resources to run, I'd be happy to include those reports here and give appropriate credits for the same.
