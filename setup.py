from setuptools import setup, find_packages

setup(
  name = 'vit-flax',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Vision Transformer, ViT - Flax',
  author = 'Vaibhav Balloli',
  author_email = 'balloli.vb@gmail.com',
  url = 'https://github.com/vballoli/vit-flax',
  keywords = [
    'computer vision',
    'transformers',
    'flax',
    'jax',
    'transformers in computer vision'
  ],
  install_requires=[
    'jax',
    'jaxlib',
    'flax',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
