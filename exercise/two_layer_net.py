from common.functions import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from common.layers import Affine
from common.layers import Relu
from common.layers import SoftmaxWithLoss


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        #        W1, W2 = self.params['W1'], self.params['W2']
        #        b1, b2 = self.params['b1'], self.params['b2']

        #        a1 = np.dot(x, W1) + b1
        #        z1 = sigmoid(a1)
        #        a2 = np.dot(z1, W2) + b2
        #        y = softmax(a2)
        #        return y
        for layer in self.layers.values:
            x = layer.forward(x)

            return x

    # x:入力データ (これを用いて学習), t:教師データ
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layer.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 傾きの設定
        grads = {}
        grads['W1'] = self.leyers['Affine1'].dW
        grads['b1'] = self.leyers['Affine1'].db
        grads['W2'] = self.leyers['Affine2'].dW
        grads['b2'] = self.leyers['Affine2'].db

        return grads

