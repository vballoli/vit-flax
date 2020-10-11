import os

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import jax
from jax import numpy as jnp
from flax import nn
from vit_flax import ViT

from tensorflow.io import gfile

from train.dataset import Cifar10, Cifar100
from train.utils import create_optimizer, save_checkpoint, train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', os.path.join(os.path.dirname(__file__), 'configs/default.py'),
    'File path to the Training hyperparameter configuration.')

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Directory to store model data.'))

def main(_):
  config = FLAGS.config
  output_dir_suffix = os.path.join(
      'lr_' + str(config.learning_rate),
      'seed_' + str(config.seed))

  output_dir = os.path.join(config.output_dir, output_dir_suffix)

  if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)

  num_devices = jax.local_device_count()
  assert config.batch_size % num_devices == 0, "Batch size must be a multiple of number of GPU devices available"
  local_batch_size = config.batch_size // num_devices
  info = 'Total batch size: {} ({} x {} replicas)'.format(
      config.batch_size, local_batch_size, num_devices)
  logging.info(info)

  if config.dataset.lower() == 'cifar10':
    dataset_source = Cifar10(config.batch_size, 
                              config.image_level_augmentations,
                              config.batch_level_augmentations)
  elif config.dataset.lower() == 'cifar100':
    dataset_source = Cifar100(
        config.batch_size, config.image_level_augmentations,
        config.batch_level_augmentations)
  else:
    raise ValueError('Available datasets: cifar10 and cifar100')

  image_size = 32
  num_channels = 3

  num_classes = 100 if config.dataset.lower() == 'cifar100' else 10
  rng = jax.random.PRNGKey(config.seed)

  module = ViT.partial(
    patch_size=config.patch_size, dim=config.latent_dim, depth=config.depth, 
    num_heads=config.num_heads, 
    dense_dims=(config.transformer_dense_dim, config.fc_dense_dim), 
    img_size=image_size, num_classes=num_classes
  )
  _, initial_params = module.init_by_shape(
    rng, [((1, image_size, image_size, num_channels), jnp.float32)]
  )
  model = nn.Model(module, initial_params)
  # Learning rate will be overwritten by the lr schedule, we set it to zero.
  optimizer = create_optimizer(model, 0.0)

  train(optimizer, initial_params, dataset_source, output_dir,
                      config.num_epochs)


if __name__ == '__main__':
  flags.mark_flags_as_required(['model_dir'])
  app.run(main)
