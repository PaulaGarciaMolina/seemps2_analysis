import numpy as np
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from seemps.state import MPS


def tensors2vector(tensors):
    mps = MPS(tensors)
    vector = mps.to_vector()
    print(vector)
    return vector


def vector2tensors(vector):
    n = int(np.log2(len(vector)))
    mps = MPS.from_vector(vector, [2] * n)
    for i in range(n):
        print(f"mps[{i}]: \n", round_arr(mps[i]))
    return mps


def trapezoidal_vector(sites):
    vector = np.ones(2**sites)
    vector[0] = vector[-1] = 0.5
    return vector


def simpson_vector(sites):
    if sites % 2 != 0:
        raise ValueError("The sites must be divisible by 2.")
    vector = np.ones(2**sites)
    vector[1:-1:3] = 3
    vector[2:-1:3] = 3
    vector[3:-1:3] = 2
    vector[0] = vector[-1] = 1
    return vector


def fifth_vector(sites):
    if sites % 4 != 0:
        raise ValueError("The sites must be divisible by 4.")
    vector = np.ones(2**sites)
    vector[1:-1:5] = 75
    vector[2:-1:5] = 50
    vector[3:-1:5] = 50
    vector[4:-1:5] = 75
    vector[5:-1:5] = 38
    vector[0] = vector[-1] = 19
    return vector


def boundary_vector(sites):
    vector = np.zeros(2**sites)
    vector[0] = vector[-1] = 1
    return vector


# I have to codify exactly a vector of this form to be able to build quadratures
def periodic_vector(sites, period, start=0):
    vector = np.zeros(2**sites)
    vector[start::period] = 1
    return vector


def round_list(arr_list):
    for idx, arr in enumerate(arr_list):
        arr_list[idx] = round_arr(arr)
    return arr_list


def round_arr(arr, tolerance=1e-10):
    return np.where(np.abs(arr) < tolerance, 0, arr)


print(trapezoidal_vector(3))
print(simpson_vector(4))
# print(fifth_vector(4))
# print(boundary_vector(4))
# print(periodic_vector(4, period=5, start=0))

# n = 8
# vector = periodic_vector(4 * n, period=5, start=0)
# vector = fifth_vector(n)
# tensors = vector2tensors(vector)
# pass


#############################################################################


# MPS RECIPE FOR A VECTOR [1, 0, ..., 0, 1]
# mps[0] =     [[[1. 0.]
#                [0. 1.]]]
# mps[1:n-3] = [[[1. 0.]
#                [0. 0.]]
#               [[0. 0.]
#                [0. 1.]]]
# mps[n-2] =   [[[1. 0.]
#                [0. 0.]]
#               [[0. 0.]
#                [0.-1.]]]
# mps[n-1] =   [[[np.sqrt(2)/2]
#                [0.]]
#               [[0.]
#                [-np.sqrt(2)/2]]]


# PURPOSE: TRY TO ENCODE THE VECTOR
# [1,    0,    0,    1,    0,    0,    1,    0,    0,    1,    0,    0,    1,    0,    0,    1]
# [0000, 0001, 0010, 0011, 0100, 0101, 0110, 0111, 1000, 1001, 1010, 1011, 1100, 1101, 1110, 1111]
tensor_1 = np.zeros((1, 2, 2))  # 4 elementos
tensor_2 = np.zeros((2, 2, 3))  # 12 elementos
tensor_3 = np.zeros((3, 2, 2))  # 12 elementos
tensor_4 = np.zeros((2, 2, 1))  # 4 elementos

tensor_1[0, 0, 0] = 1
tensor_1[0, 0, 1] = 0
tensor_1[0, 1, 0] = 1
tensor_1[0, 1, 1] = 1

tensor_2[0, 0, 0] = 1
tensor_2[0, 0, 1] = 0
tensor_2[0, 0, 2] = 0
tensor_2[0, 1, 0] = 1
tensor_2[0, 1, 1] = 0
tensor_2[0, 1, 2] = 0
tensor_2[1, 0, 0] = 0
tensor_2[1, 0, 1] = 0
tensor_2[1, 0, 2] = 0
tensor_2[1, 1, 0] = 0
tensor_2[1, 1, 1] = -1
tensor_2[1, 1, 2] = 0

tensor_3[0, 0, 0] = 1
tensor_3[0, 0, 1] = 0
tensor_3[0, 1, 0] = 1
tensor_3[0, 1, 1] = 0
tensor_3[1, 0, 0] = 0
tensor_3[1, 0, 1] = 0
tensor_3[1, 1, 0] = 0
tensor_3[1, 1, 1] = 1
tensor_3[2, 0, 0] = 0
tensor_3[2, 0, 1] = 0
tensor_3[2, 1, 0] = 0
tensor_3[2, 1, 1] = 0

tensor_4[0, 0, 0] = 1
tensor_4[0, 1, 0] = 1
tensor_4[1, 0, 0] = 0
tensor_4[1, 1, 0] = -1

tensors = [tensor_1, tensor_2, tensor_3, tensor_4]
tensors2vector(tensors)
