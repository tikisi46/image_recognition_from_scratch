import numpy as np
from model import model


model = model()

network = model.init_network()
x = np.array([1.0, 0.5])
y = model.forward(network, x)
print(y)

