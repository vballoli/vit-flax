.. Vision Transformers in JAX/Flax(ViT-Flax) documentation master file, created by
   sphinx-quickstart on Mon Oct 12 01:49:11 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vision Transformers in JAX/Flax(ViT-Flax)'s documentation!
=====================================================================

Vision Transformers(ViT) in JAX/Flax is a re-implementation of the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

.. toctree::
   :maxdepth: 2
   :caption: Contents:


*********
Install
*********

.. code-block:: console

   pip install vit-flax

*********
Sample usage
*********

.. code-block:: python

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

.. toctree::
   :maxdepth: 2
   :caption: API reference

   vit_flax

