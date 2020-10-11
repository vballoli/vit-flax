import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.1
  config.batch_size = 128
  config.num_epochs = 10
  config.seed = 42
  config.image_level_augmentations = 'basic'
  config.batch_level_augmentations = 'cutout'
  config.output_dir = './output/'

  config.patch_size = 8
  config.latent_dim = 64
  config.depth = 3
  config.num_heads = 4
  config.transformer_dense_dim = 2048
  config.fc_dense_dim = 2048
  config.dataset = 'cifar10' #or 'cifar100'

  return config