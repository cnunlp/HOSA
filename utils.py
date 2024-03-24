import numpy

from models.hosa import HOAS
from sympy.physics.quantum import TensorProduct

def get_model(config):
    model = HOAS(config)
    return model


def dot_sim(x, y):
    return numpy.dot(x, y.T)


def cosine_sim(x, y):
    x = x / numpy.expand_dims(numpy.linalg.norm(x, axis=1), 1)
    y = y / numpy.expand_dims(numpy.linalg.norm(y, axis=1), 1)
    return numpy.dot(x, y.T)


def t_pro(x,y):
    return TensorProduct(x, y)
