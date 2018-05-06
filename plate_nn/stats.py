import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
from imageio import imread

files = ['dataset/positive/' + s for s in os.listdir('dataset/positive') if s.endswith('.png')]


def f(fn):
    img = imread(fn, pilmode='RGB')
    return img.shape


pool = ThreadPool(cpu_count())

result = pool.map(f, files)

a = np.array(result)

print(np.min(a, axis=0), np.max(a, axis=0), np.average(a, axis=0), np.std(a, axis=0))
print(np.average(a[:, 0]/a[:, 1]), np.std(a[:, 0]/a[:, 1]))
