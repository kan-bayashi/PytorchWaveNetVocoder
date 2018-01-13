import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable

from wavenet import UpSampling, initialize

aux = np.random.randn(1, 28, 1000)
conv = UpSampling(80)
conv.apply(initialize)
batch = Variable(torch.from_numpy(aux).float())
out = conv(batch)
out = out.data.numpy()
plt.subplot(1, 2, 1)
plt.imshow(aux.reshape(28, -1)[3:], aspect="auto")
plt.subplot(1, 2, 2)
plt.imshow(out.reshape(28, -1)[3:], aspect="auto")
plt.show()
