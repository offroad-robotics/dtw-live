#!/usr/bin/env python3

import os
import ctypes
from ctypes import c_int

import numpy as np
from numpy.ctypeslib import ndpointer


# find & load library
ROOT = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.cdll.LoadLibrary(os.path.join(ROOT, 'libdtw.so'))

# ctypes typedefs
c_double_p1 = ndpointer(dtype=np.float64, ndim=1, flags='C')
c_double_p2 = ndpointer(dtype=np.float64, ndim=2, flags='C')
c_uint16_p = ndpointer(dtype=np.uint16, ndim=1, flags='C')

# initialize update_dist function
update_cost_c = lib.update_cost
update_cost_c.argtypes = [c_double_p1, c_double_p2, c_int, c_int,
                          c_double_p1]
update_cost_c.restype = None

# initialize update_dist_width function
update_cost_width_c = lib.update_cost_width
update_cost_width_c.argtypes = [c_double_p1, c_double_p2, c_int, c_int,
                                c_double_p1, c_uint16_p]
update_cost_width_c.restype = None

# initialize cost_matrix function
cost_matrix_c = lib.cost_matrix
cost_matrix_c.argtypes = [c_double_p2, c_double_p2, c_int, c_int,
                          c_int, c_int, c_int, c_double_p2]
cost_matrix_c.restype = None
