#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# View Settings
WIDTH_PX = 1600
HEIGHT_PX = 800
DPI = 100

WIDTH_INCHES = WIDTH_PX / DPI
HEIGHT_INCHES = HEIGHT_PX / DPI

# Mandelbrot settings
BOUNDARY_THRESHOLD = 0.9
ITERATIONS = 80
ZOOM_FACTOR = 1.02
ZOOM_ITERATIONS = 5


def in_mandelbrot_set(x_pt, y_pt, ITERATIONS) -> int:
    i = 0
    z_prev = 0
    while i < ITERATIONS:
        z_n = z_prev ** 2 + complex(x_pt, y_pt)
        
        if abs(z_n) > 2:
            return i
        
        z_prev = z_n
        i += 1
    
    return ITERATIONS


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


if __name__ == '__main__':
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

    for z in range(0, ZOOM_ITERATIONS - 1):
        
        # fill in image with iterations
        # new function here
        for i in range(0, WIDTH_PX):
            x_pt = x_min +  i *  (x_range / WIDTH_PX)
            for j in range(0, HEIGHT_PX):
                y_pt = y_min + j * (y_range / HEIGHT_PX)

                iter_count = in_mandelbrot_set(x_pt, y_pt, ITERATIONS)
                img[j][i] = iter_count


        plt.imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', cmap='inferno')
        plt.draw()
        plt.pause(0.5)

        # Find first boundary to zoom in on
        threshold = BOUNDARY_THRESHOLD * ITERATIONS
        j_zoom, i_zoom = find_boundary_pixels(img, threshold)
        
        # new function here
        x_zoom = x_min + i_zoom * (x_range / WIDTH_PX)
        y_zoom = y_min + j_zoom * (y_range / HEIGHT_PX)
        x_range_new = x_range / ZOOM_FACTOR
        y_range_new = y_range / ZOOM_FACTOR
        x_half = x_range_new / 2
        y_half = y_range_new / 2

        x_min = x_zoom - x_half
        x_max = x_zoom + x_half
        y_min = y_zoom - y_half
        y_max = y_zoom + y_half

        x_range = abs(x_max - x_min)
        y_range = abs(y_max - y_min)

        ITERATIONS *= 1.02


    # fill in image with iterations
    for i in range(0, WIDTH_PX):
        x_pt = x_min +  i *  (x_range / WIDTH_PX)
        for j in range(0, HEIGHT_PX):
            y_pt = y_min + j * (y_range/HEIGHT_PX)

            iter_count = in_mandelbrot_set(x_pt, y_pt, ITERATIONS)
            img[j][i] = iter_count

    plt.imshow(img, extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', cmap='inferno')
    plt.show()
