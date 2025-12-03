#!/usr/bin/env python3

from numba import cuda
import math

import matplotlib.pyplot as plt
import numpy as np

import os

# ----------------------------
# View / render settings
# ----------------------------
WIDTH_PX = 1920
HEIGHT_PX = 1080
DPI = 100

WIDTH_INCHES = WIDTH_PX / DPI
HEIGHT_INCHES = HEIGHT_PX / DPI

# ----------------------------
# Mandelbrot / zoom settings
# ----------------------------
NUM_FRAMES = 700

BASE_ITERATIONS = 300        # starting detail level
K = 0.6                      # how fast iterations grow with magnification
ZOOM_FACTOR = 1.02           # per-frame zoom factor

# Zoom center: Misiurewicz point M_{23,2} in Seahorse Valley (on boundary)
X_CENTER = -0.77568377
Y_CENTER =  0.13646737


# ----------------------------
# Device-side Mandelbrot eval
# ----------------------------
@cuda.jit(device=True)
def in_mandelbrot(x_pt, y_pt, max_iters):
    """
    Smooth escape-time Mandelbrot iteration.
    Returns a continuous escape value (nu) for coloring.
    """
    zx = 0.0
    zy = 0.0

    for i in range(max_iters):
        # z^2
        zx2 = zx * zx - zy * zy
        zy2 = 2.0 * zx * zy

        # add constant c
        zx = zx2 + x_pt
        zy = zy2 + y_pt

        r2 = zx * zx + zy * zy
        if r2 > 4.0:
            # Smooth escape time (continuous coloring)
            log_r2 = math.log(r2)
            log_abs = 0.5 * log_r2  # log|z|

            if log_abs <= 0.0:
                return float(i)

            log2 = math.log(2.0)
            nu = i + 1.0 - math.log(log_abs / log2) / log2
            return nu

    # Interior points: treat as max_iters (large, "deep" value)
    return float(max_iters)


@cuda.jit
def render_frame_kernel(x_min, x_max, y_min, y_max, iterations, img):
    """
    Kernel: computes smooth escape-time values over the viewport.
    img[y, x] holds the nu value for each pixel.
    """
    height_px, width_px = img.shape

    x_idx, y_idx = cuda.grid(2)  # (x, y) ordering

    if x_idx < width_px and y_idx < height_px:
        dx = (x_max - x_min) / width_px
        dy = (y_max - y_min) / height_px

        x_pt = x_min + x_idx * dx
        y_pt = y_min + y_idx * dy

        img[y_idx, x_idx] = in_mandelbrot(x_pt, y_pt, iterations)


# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    # Output directory
    directory_path = os.path.join(os.getcwd(), "images")
    os.makedirs(directory_path, exist_ok=True)

    # Initial view (classic full set)
    x_min = -2.5
    x_max = 1.0
    y_min = -1.0
    y_max = 1.0
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    # Matplotlib setup
    plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=DPI)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.axis("off")

    # CUDA grid config
    threads_per_block = (16, 16)
    blocks_x = math.ceil(WIDTH_PX / threads_per_block[0])
    blocks_y = math.ceil(HEIGHT_PX / threads_per_block[1])
    blocks_per_grid = (blocks_x, blocks_y)

    # Device image buffer (float64 for more precision)
    d_img = cuda.device_array((HEIGHT_PX, WIDTH_PX), dtype=np.float64)

    magnification = 1.0

    for n in range(NUM_FRAMES):
        # Iterations grow with magnification
        iterations = int(BASE_ITERATIONS * (magnification ** K))

        # ----------------------------
        # Render frame on GPU
        # ----------------------------
        render_frame_kernel[blocks_per_grid, threads_per_block](
            x_min, x_max, y_min, y_max, iterations, d_img
        )

        # Bring back to host (img = nu values)
        img = d_img.copy_to_host()

        # Basic raw stats (nu)
        nu_min = float(img.min())
        nu_max = float(img.max())
        nu_mean = float(img.mean())
        nu_std = float(img.std())
        nu_p05, nu_p50, nu_p95, nu_p99 = np.percentile(img, [5, 50, 95, 99])

        # ---- boundary-aware cyclic coloring ----
        interior_mask = img >= (iterations - 1)
        exterior = img[~interior_mask]
        interior_fraction = float(exterior.size < img.size) and float(
            np.mean(img >= (iterations - 1))
        )  # or simply np.mean(interior_mask)

        # cyclic phase from smooth iteration count
        color_freq = 0.05  # tweak for more/less bands
        phase = (img * color_freq) % 1.0
        phase[interior_mask] = 0.0  # solid interior

        # stretch contrast based on *exterior* only
        if exterior.size > 0:
            ext_phase = phase[~interior_mask]
            p_lo = np.percentile(ext_phase, 5.0)
            p_hi = np.percentile(ext_phase, 95.0)
        else:
            p_lo, p_hi = 0.0, 1.0

        if p_hi <= p_lo:
            p_hi = p_lo + 1e-6

        norm = (phase - p_lo) / (p_hi - p_lo)
        norm = np.clip(norm, 0.0, 1.0)

        gamma = 0.8
        norm_gamma = norm ** gamma

        # ---- plot ----
        plt.imshow(
            norm_gamma,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            aspect='auto',
            cmap='turbo',  # 'twilight', 'twilight_shifted', or 'hsv' also fun
            vmin=0.0,
            vmax=1.0,
        )

        img_name = os.path.join(directory_path, f"mandelbrot_{n:04d}.png")
        plt.savefig(img_name, dpi=DPI)
        plt.clf()

        # ----------------------------
        # Telemetry logs (per-frame)
        # ----------------------------
        print(
            f"[Frame {n:4d}] "
            f"iters={iterations:7d}, "
            f"x_range={x_range:.3e}, y_range={y_range:.3e}"
        )
        print(
            f"    nu_min={nu_min:.3e}, nu_max={nu_max:.3e}, "
            f"nu_mean={nu_mean:.3e}, nu_std={nu_std:.3e}"
        )
        print(
            f"    nu_p05={nu_p05:.3e}, nu_p50={nu_p50:.3e}, "
            f"nu_p95={nu_p95:.3e}, nu_p99={nu_p99:.3e}, "
            f"interior_frac={np.mean(interior_mask):.3f}"
        )

        # ----------------------------
        # Update zoom window around center
        # ----------------------------
        x_range_new = x_range / ZOOM_FACTOR
        y_range_new = y_range / ZOOM_FACTOR
        x_half = x_range_new / 2.0
        y_half = y_range_new / 2.0

        x_min = X_CENTER - x_half
        x_max = X_CENTER + x_half
        y_min = Y_CENTER - y_half
        y_max = Y_CENTER + y_half

        x_range = x_range_new
        y_range = y_range_new

        magnification *= ZOOM_FACTOR
