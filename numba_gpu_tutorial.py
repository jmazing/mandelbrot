#!/usr/bin/env python3

from numba import cuda
import numpy as np

@cuda.jit
def print_indices():
    i = cuda.grid(1)
    if i < 10:
        print(i)


@cuda.jit
def fill_array(arr):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = i * 2

@cuda.jit
def fill_coords(out):
    x, y = cuda.grid(2)   # get the 2D global thread index

    if x < out.shape[1] and y < out.shape[0]:
        out[y, x] = x + y * 100


print(cuda.gpus)
print("============================================")
print_indices[1, 10]()  # 1 block, 10 threads


n = 16
arr = np.zeros(n, np.int32)
d_arr = cuda.to_device(arr)

threads = 8
blocks = (n + threads - 1) // threads

fill_array[blocks, threads](d_arr)
result = d_arr.copy_to_host()

print(result)


H, W = 4, 6
out = np.zeros((H, W), dtype=np.int32)

