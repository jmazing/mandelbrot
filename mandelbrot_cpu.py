#!/usr/bin/env python3

from numba import njit, prange
import math

import matplotlib.pyplot as plt
import numpy as np

import os

# View Settings
WIDTH_PX = 1920
HEIGHT_PX = 1080
DPI = 100

WIDTH_INCHES = WIDTH_PX / DPI
HEIGHT_INCHES = HEIGHT_PX / DPI

# Mandelbrot settings
BOUNDARY_THRESHOLD = 0.9
NUM_FRAMES = 500

BASE_ITERATIONS = 300
K = 0.6      # detail growth factor
ZOOM_FACTOR = 1.02

X_CENTER = -0.743643887037158704752191506114774
Y_CENTER =  0.131825904205311970493132056385139


@njit
def in_mandelbrot(x_pt, y_pt, max_iters) -> float:
    zx = 0.0
    zy = 0.0

    for i in range(max_iters):
        # compute z^2
        zx2 = zx*zx - zy*zy
        zy2 = 2*zx*zy

        # add constant C
        zx = zx2 + x_pt
        zy = zy2 + y_pt

        r2 = zx*zx + zy*zy
        if r2 > 4.0:
            log_r = 0.5 * math.log(r2)
            nu = i + 1.0 - math.log(log_r) / math.log(2.0)
            return nu
    
    return float(max_iters)


@njit(parallel=True)
def render_frame(x_min, x_max, y_min, y_max, height_px, width_px, iterations) -> np.ndarray:
    img = np.empty((height_px, width_px), dtype=np.float32)
    dx = (x_max - x_min) / width_px
    dy = (y_max - y_min) / height_px

    for j in prange(height_px):        # parallel over rows
        y_pt = y_min + j * dy
        for i in range(width_px):      # inner loop over columns
            x_pt = x_min + i * dx
            img[j, i] = in_mandelbrot(x_pt, y_pt, iterations)

    return img


if __name__ == '__main__':
    directory_path = f"{os.getcwd()}/images"
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
    img = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)

    magnification = 1.0  
    for n in range(0, NUM_FRAMES):
        iterations = int(BASE_ITERATIONS * (magnification ** K))
        
        img = render_frame(x_min, x_max, y_min, y_max,
                           HEIGHT_PX, WIDTH_PX, iterations)

        plt.imshow(img, 
                    extent=[x_min, x_max, y_min, y_max], 
                    origin='lower', 
                    aspect='auto', 
                    cmap='turbo', 
                    vmin=np.min(img), 
                    vmax=np.percentile(img, 90)
                )  # ignore the top 10% "deep" values)
        
        img_name = f"{directory_path}/mandebrot_{n}.png"
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

        magnification *= ZOOM_FACTOR
