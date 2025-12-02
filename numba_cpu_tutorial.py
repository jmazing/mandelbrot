#!/usr/bin/env python3

from numba import njit
import time
import numpy as np

def slow_sum(n):
    s = 0
    for _ in range(n):
        s += 1
    return s

@njit
def fast_sum(n):
    s = 0
    for _ in range(n):
        s += 1
    return s

def slow_double_array():
    array_size = 100000000
    arr = np.ones(array_size, dtype=float)
    for i in range(arr.size):
        arr[i] *= 2

@njit
def fast_double_array():
    array_size = 100000000
    arr = np.ones(array_size, dtype=float)
    for i in range(arr.size):
        arr[i] *= 2


start_time = time.perf_counter()
result = slow_sum(1000000000)
end_time = time.perf_counter()



elapsed_time = end_time - start_time
print(f"Function slow_sum executed in {elapsed_time:.4f} seconds")

start_time = time.perf_counter()
result = fast_sum(1000000000)
end_time = time.perf_counter()



elapsed_time = end_time - start_time
print(f"Function fast_sum executed in {elapsed_time:.4f} seconds")



start_time = time.perf_counter()
result = slow_double_array()
end_time = time.perf_counter()



elapsed_time = end_time - start_time
print(f"Function double_arr executed in {elapsed_time:.4f} seconds")

start_time = time.perf_counter()
result = fast_double_array()
end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Function fast_sum executed in {elapsed_time:.4f} seconds")


