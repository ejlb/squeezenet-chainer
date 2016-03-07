import chainer
import chainer.functions as F
import chainer.links as L


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super(Fire, self).__init__(
            conv1=L.Convolution2D(in_size, s1, 1),
            conv2=L.Convolution2D(s1, e1, 1),
            conv3=L.Convolution2D(s1, e3, 3, pad=1),
            bn4=L.BatchNormalization(e1 + e3)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat([h_1, h_3], axis=1)

        return F.relu(self.bn4(h_out))


class SqueezeNet(chainer.Chain):
    def __init__(self, ):
        super(SqueezeNet, self).__init__(
            conv1=L.Convolution2D(3, 96, 7, stride=2),
            fire2=Fire(96, 16, 64, 64),
            fire3=Fire(128, 16, 64, 64),
            fire4=Fire(128, 16, 128, 128),
            fire5=Fire(256, 32, 128, 128),
            fire6=Fire(256, 48, 192, 192),
            fire7=Fire(384, 48, 192, 192),
            fire8=Fire(384, 64, 256, 256),
            fire9=Fire(512, 64, 256, 256),
            conv10=L.Convolution2D(512, 1000, 1,
                pad=1, initialW=normal(0, 0.01 (1000, 512, 1, 1)))
        )

    def __call__(self, x, train=False):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire2(h)
        h = self.fire3(h)
        h = self.fire4(h)

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire5(h)
        h = self.fire6(h)
        h = self.fire7(h)
        h = self.fire8(h)

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fire9(h)
        h = F.dropout(h, ratio=0.5, train=train)

        h = F.relu(self.conv10(h))
        h = F.average_pooling_2d(h, 13)

        return F.reshape(h, (-1, 1000))
