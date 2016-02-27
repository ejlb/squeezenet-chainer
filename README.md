# SqueezeNet

An implementation of [SqueezeNet](http://arxiv.org/abs/1602.07360) in [chainer](https://github.com/pfnet/chainer). 

Below are some benchmarks on a ImageNet-like dataset (2 million 255x255 images with 192 batch size)

|-------------|-----------|----------------------|---------------------|---------------------|-------------------|
| Model       | Size (mb) | Parameters (million) | CPU prediction (ms) | GPU prediction (ms) | Accuracy @ e = 10 |
| AlexNet     | 667       | 58.49                | 439                 | 4                   |                   |
| SqueezeNet  | 4.4       | 1.225                | 1652                | 16                  |                   |
