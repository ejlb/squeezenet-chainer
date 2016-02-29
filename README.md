# SqueezeNet

An implementation of [SqueezeNet](http://arxiv.org/abs/1602.07360) in [chainer](https://github.com/pfnet/chainer). 

Below are some benchmarks on a ImageNet-like dataset (1 million 255x255 images with 128 batch size)

| Model       | Size (mb) | Parameters (million) | CPU prediction (ms) | GPU prediction (ms) | Accuracy  |
| ----------- | --------- | -------------------- | ------------------- | ------------------- | --------- |
| ZFNet       | 117       | 16.42                |                     |                     |           |
| SqueezeNet  | 4.6       | 1.277                |                     |                     |           |


## Details

* [ZFNet](http://arxiv.org/abs/1311.2901) instead of AlexNet (I already had a trained ZFNet model). Initial filters are 7x7 instead of 11x11
* I had to add a dense layer to the end of SqueezeNet to get correct shape for my labels.
* This is the basic squeezenet without deep compression
* Paper says images are 224x224 but code and parameters suggests they use 227x227
* Seeing an 18x size reduction over ZFNet (SqueezeNet size is the same as the paper) and an 13x
  reduction in params.
