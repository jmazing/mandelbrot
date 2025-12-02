#!/usr/bin/env python3

from numba import njit
import math

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

import os

# View Settings
WIDTH_PX = 1600
HEIGHT_PX = 800
DPI = 100

WIDTH_INCHES = WIDTH_PX / DPI
HEIGHT_INCHES = HEIGHT_PX / DPI

# Mandelbrot settings
BOUNDARY_THRESHOLD = 0.9
ITERATIONS = 100
ZOOM_FACTOR = 1.02
ZOOM_ITERATIONS = 10 # Around .25 GB

X_CENTER = -0.743643887037158704752191506114774
Y_CENTER =  0.131825904205311970493132056385139

@njit
def in_mandelbrot(x_pt, y_pt, max_iters) -> float:
    zx = 0.0
    zy = 0.0
    i = 0
    
    while i < max_iters:
        # computer z^2
        zx2 = zx*zx - zy*zy
        zy2 = 2*zx*zy

        # add constant C
        zx = zx2 + x_pt
        zy = zy2 + y_pt

        r2 = zx*zx + zy*zy
        if r2 > 4.0:
            r = math.sqrt(r2)
            nu = i + 1.0 - math.log(math.log(r)) / math.log(2.0)
            return nu
        
        i += 1
    
    return float(max_iters)


"""
def find_boundary_pixels(img, threshold) -> tuple:
    max_iter_count = -1
    j_pixel = -1
    i_pixel = -1
    for i in range(0, WIDTH_PX):
        for j in range(0, HEIGHT_PX):
            iter_count = img[j][i]
            if threshold < iter_count < ITERATIONS:
                if iter_count > max_iter_count:
                    max_iter_count = iter_count
                    j_pixel = j
                    i_pixel = i
    
    return j_pixel, i_pixel
"""


def compute_row(j, x_min, x_range, y_min, y_range, height_px, width_px, iterations):
    # allocate an array for this row
    row_data = []

    dx = x_range / width_px
    dy = y_range / height_px
    for i in range(width_px):
        x_pt = x_min + i * dx
        y_pt = y_min + j * dy
        iter_count = in_mandelbrot(x_pt, y_pt, iterations)
        row_data.append(iter_count)

    return (j, row_data)


if __name__ == '__main__':
    directory_path = "/home/julius/work/mandelbrot_set/images"
    os.makedirs(directory_path, exist_ok=True)

    # initial view settings
    x_min = -2.5
    x_max = 1
    y_min = -1
    y_max = 1
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=DPI)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.axis("off")
    img = np.zeros((HEIGHT_PX, WIDTH_PX))

    with Pool(processes=cpu_count()) as pool:
        for z in range(0, ZOOM_ITERATIONS):
            tasks = []
            for j in range(HEIGHT_PX):
                tasks.append((j, x_min, x_range, y_min, y_range, HEIGHT_PX, WIDTH_PX, ITERATIONS))

            results = pool.starmap(compute_row, tasks)
            
            for (j, row_data) in results:
                img[j, :] = row_data

            plt.imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', cmap='turbo')
            
            img_name = f"{directory_path}/mandebrot_{z}.png"
            plt.savefig(img_name)

            x_range_new = x_range / ZOOM_FACTOR
            y_range_new = y_range / ZOOM_FACTOR
            x_half = x_range_new / 2
            y_half = y_range_new / 2

            x_min = X_CENTER - x_half
            x_max = X_CENTER + x_half
            y_min = Y_CENTER - y_half
            y_max = Y_CENTER + y_half

            x_range = abs(x_max - x_min)
            y_range = abs(y_max - y_min)

            ITERATIONS *= 1.02
