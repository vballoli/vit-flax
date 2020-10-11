# Examples to train Vision Transformers using ViT-Flax

Vision transformer can be trained on the CIFAR dataset using `cifar_train.py` file. The `train` dir contains utils to train Vision transformer model based on CIFAR training scripts based on [google-research's code](https://github.com/google-research/google-research/blob/master/flax_models/cifar/).

# Install
This code uses [ml_collections](https://github.com/google/ml_collections) for hyperparameters and `vit-flax` for the . To run the code, make sure `JAX`, `Flax` and `Tensorflow` are pre-installed(these aren't included in the requirements to make sure there aren't any conflicts with the CUDA versions on your machine):
```bash
pip install -r requirements.txt
```

# Run
To train ViT on CIFAR10/100 (setup MODEL_DIR environment variable):
```bash
python cifar_train.py ----model_dir=$MODEL_DIR
```

For custom hyperparameters:
```bash
python cifar_train.py ----model_dir=$MODEL_DIR --config=configs/default.py --config.dataset=cifar100 --config.learning_rate=0.001
```
