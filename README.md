# SqueezeNet

An implementation of [SqueezeNet](http://arxiv.org/abs/1602.07360) in [chainer](https://github.com/pfnet/chainer). 

Below are some benchmarks on a ImageNet-like dataset (1 million 255x255 images with 128 batch size)

| Model       | Size (mb) | Parameters (million) |  Accuracy  |
| ----------- | --------- | -------------------- | ---------- |
| ZFNet       | 117       | 16.42                | 0.5835     |
| SqueezeNet  | 4.6       | 1.288                | 0.5207     |


## Details

* [ZFNet](http://arxiv.org/abs/1311.2901) instead of AlexNet (I already had a trained ZFNet model).
* This is the basic squeezenet without deep compression. I had to add a dense layer to the end of 
  SqueezeNet to get correct shape for my labels. I also had to add batch normalisation to the fire
  modules so it would fit.
* The paper says images are 224x224 but the code and parameters suggests they use 227x227. I also added some
  padding, relu and initialisation that was used in the squeezenet code but not mentioned in the
  paper.
* Accuracy of squeezenet is lower in this test but there is a 25x size reduction over ZFNet (SqueezeNet size is the same as in the paper) and a 12x
  reduction in params.

