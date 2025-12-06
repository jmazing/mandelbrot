#!/usr/bin/env python3

from numba import cuda
import math

import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp

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
NUM_FRAMES = 1500

BASE_ITERATIONS = 300        # starting detail level
K = 0.6                      # how fast iterations grow with magnification
ZOOM_FACTOR = 1.02           # per-frame zoom factor

# Zoom center: Misiurewicz point M_{23,2} in Seahorse Valley (on boundary)
X_CENTER = -0.77568377
Y_CENTER =  0.13646737

# When the view is wider than this in x, we use direct iteration.
# When narrower, we switch to perturbation.
USE_PERTURB_THRESHOLD = 0.10  # tweak as needed

# Optional: cap iterations while experimenting so mpmath doesn't kill you
MAX_ITERS_CAP = 5000          # set None to disable


# ----------------------------
# High-precision reference orbit (CPU, mpmath)
# ----------------------------
def compute_reference_orbit(x_center, y_center, max_iters, mp_dps=80):
    """
    Compute high-precision reference orbit z_n(c0) for c0 = x_center + i*y_center
    using mpmath, then downcast to float64 arrays for the GPU.
    """
    mp.mp.dps = mp_dps  # decimal digits of precision

    c0 = mp.mpc(x_center, y_center)
    z = mp.mpc(0, 0)

    ref_re = np.empty(max_iters, dtype=np.float64)
    ref_im = np.empty(max_iters, dtype=np.float64)

    for n in range(max_iters):
        # store z_n BEFORE computing z_{n+1}
        ref_re[n] = float(mp.re(z))
        ref_im[n] = float(mp.im(z))
        z = z * z + c0

    return ref_re, ref_im


# ----------------------------
# Direct Mandelbrot device function (your original style)
# ----------------------------
@cuda.jit(device=True)
def direct_in_mandelbrot(x_pt, y_pt, max_iters):
    """
    Smooth escape-time Mandelbrot iteration (direct, no perturbation).
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


# ----------------------------
# Direct Mandelbrot kernel (wide views)
# ----------------------------
@cuda.jit
def render_frame_direct_kernel(x_min, x_max, y_min, y_max, iterations, img):
    """
    Kernel: computes smooth escape-time values over the viewport directly.
    img[y, x] holds the nu value for each pixel.
    """
    height_px, width_px = img.shape

    x_idx, y_idx = cuda.grid(2)

    if x_idx < width_px and y_idx < height_px:
        dx = (x_max - x_min) / width_px
        dy = (y_max - y_min) / height_px

        x_pt = x_min + x_idx * dx
        y_pt = y_min + y_idx * dy

        img[y_idx, x_idx] = direct_in_mandelbrot(x_pt, y_pt, iterations)


# ----------------------------
# Perturbation device function (deep zoom)
# ----------------------------
@cuda.jit(device=True)
def perturb_escape(x_pt, y_pt,
                   x_center, y_center,
                   iterations,
                   ref_re, ref_im):
    """
    Per-pixel escape computation using perturbation:

        z_{n+1} = z_n^2 + c
        c = c0 + delta
        epsilon_{n+1} = 2 z_n(c0) * epsilon_n + epsilon_n^2 + delta
        z_n(c) = z_n(c0) + epsilon_n

    All arithmetic here is float64.
    """
    # delta = c - c0
    delta_re = x_pt - x_center
    delta_im = y_pt - y_center

    eps_re = 0.0
    eps_im = 0.0

    for i in range(iterations):
        zref_re = ref_re[i]
        zref_im = ref_im[i]

        # eps^2
        eps2_re = eps_re * eps_re - eps_im * eps_im
        eps2_im = 2.0 * eps_re * eps_im

        # 2 * z_ref * eps
        tmp_re = 2.0 * (zref_re * eps_re - zref_im * eps_im)
        tmp_im = 2.0 * (zref_re * eps_im + zref_im * eps_re)

        # epsilon_{n+1}
        eps_re = tmp_re + eps2_re + delta_re
        eps_im = tmp_im + eps2_im + delta_im

        # reconstruct z_n(c) = z_ref + eps
        z_re = zref_re + eps_re
        z_im = zref_im + eps_im

        r2 = z_re * z_re + z_im * z_im
        if r2 > 4.0:
            # Smooth escape time, same formula as direct
            log_r2 = math.log(r2)
            log_abs = 0.5 * log_r2  # log|z|

            if log_abs <= 0.0:
                return float(i)

            log2 = math.log(2.0)
            nu = i + 1.0 - math.log(log_abs / log2) / log2
            return nu

    # treated as interior / max depth
    return float(iterations)


# ----------------------------
# Perturbation kernel
# ----------------------------
@cuda.jit
def render_frame_perturb_kernel(x_min, x_max,
                                y_min, y_max,
                                x_center, y_center,
                                iterations,
                                ref_re, ref_im,
                                img):
    """
    Kernel: per pixel, use perturbation relative to reference orbit ref_re/ref_im.
    """
    height_px, width_px = img.shape

    x_idx, y_idx = cuda.grid(2)

    if x_idx < width_px and y_idx < height_px:
        dx = (x_max - x_min) / width_px
        dy = (y_max - y_min) / height_px

        x_pt = x_min + x_idx * dx
        y_pt = y_min + y_idx * dy

        img[y_idx, x_idx] = perturb_escape(
            x_pt, y_pt,
            x_center, y_center,
            iterations,
            ref_re, ref_im
        )


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

    # Device image buffer (float64)
    d_img = cuda.device_array((HEIGHT_PX, WIDTH_PX), dtype=np.float64)

    magnification = 1.0

    for n in range(NUM_FRAMES):
        iterations = int(BASE_ITERATIONS * (magnification ** K))
        if MAX_ITERS_CAP is not None:
            iterations = min(iterations, MAX_ITERS_CAP)

        use_perturb = (n >= 1500)

        if use_perturb:
            # 1) reference orbit on CPU (high precision)
            ref_re_host, ref_im_host = compute_reference_orbit(
                X_CENTER, Y_CENTER,
                iterations,
                mp_dps=80  # tweak as needed
            )

            # Send reference orbit to GPU
            d_ref_re = cuda.to_device(ref_re_host)
            d_ref_im = cuda.to_device(ref_im_host)

            # 2) render frame on GPU using perturbation
            render_frame_perturb_kernel[blocks_per_grid, threads_per_block](
                x_min, x_max,
                y_min, y_max,
                X_CENTER, Y_CENTER,
                iterations,
                d_ref_re, d_ref_im,
                d_img
            )
        else:
            # Wide view: direct iteration (no perturbation)
            render_frame_direct_kernel[blocks_per_grid, threads_per_block](
                x_min, x_max,
                y_min, y_max,
                iterations,
                d_img
            )

        # Copy results to host
        img = d_img.copy_to_host()

        # ----------------------------
        # Stats + coloring (same as your original)
        # ----------------------------
        nu_min = float(img.min())
        nu_max = float(img.max())
        nu_mean = float(img.mean())
        nu_std = float(img.std())
        nu_p05, nu_p50, nu_p95, nu_p99 = np.percentile(img, [5, 50, 95, 99])

        # boundary-aware cyclic coloring
        interior_mask = img >= (iterations - 1)
        exterior = img[~interior_mask]

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
            cmap='turbo',
            vmin=0.0,
            vmax=1.0,
        )

        img_name = os.path.join(directory_path, f"mandelbrot_{n:04d}.png")
        plt.savefig(img_name, dpi=DPI)
        plt.clf()

        # ----------------------------
        # Telemetry logs (per-frame)
        # ----------------------------
        mode = "PERT" if use_perturb else "DIR "
        print(
            f"[Frame {n:4d}] mode={mode} "
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
