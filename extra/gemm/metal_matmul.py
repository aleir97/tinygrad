import os
import numpy as np
from tinygrad import  dtypes
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.helpers import Timing, Profiling, getenv, DEBUG, colored, Context
from examples.llama import QK4_0Linear


weight = Tensor.rand(131072, 16, dtype=dtypes.half)
scale  = Tensor.rand(131072, 1,  dtype=dtypes.half)
x,y = 2048, 2048

quants = QK4_0Linear(2048, 2048)
quants.weight.assign(weight).realize()
quants.scale.assign(scale).realize()

quants(Tensor.rand(131072, 16))