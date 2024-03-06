from input.mnist import load_mnist


class getData:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            load_mnist(flatten=True, normalize=False)

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_train_data(self):
        return self.x_train, self.y_train
