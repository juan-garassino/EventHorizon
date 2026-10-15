#!/usr/bin/env python3
"""
Small deterministic preview render + automated image checks (CPU, ~10-30 s).

Uses the production physics path (eq.13 b-table -> periodic interpolation ->
redshift_vec -> flux_observed_vec) on a dense regular grid of disk points, splats
the observed flux into a <= 300x200 raster and reports:

  * NaN / inf pixels                (must be 0)
  * dark background                 (image corners must be black)
  * primary + secondary image       (both must cover pixels)
  * shadow radius                   (innermost ghost-image b ~ b_crit)
  * holes / banding                 (unlit pixels enclosed by lit ones)

Usage:
    python eval/preview_render.py [--out results/preview.png] [--max-steps 10]
        [--p-points 1000] [--json metrics.json]

--max-steps / --p-points control the eq.13 root solver (bisection steps and
periastron scan resolution); halving the effective step is used for the
convergence study in eval/convergence.py.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from eventHorizon.physics import geodesics as geo  # noqa: E402
from eventHorizon.physics.fast_geodesics import (  # noqa: E402
    flux_observed_vec,
    impact_table_interpolator,
    redshift_vec,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

M = 1.0
INCL = np.radians(80.0)
R_IN, R_OUT = 6.0, 30.0
WIDTH, HEIGHT = 300, 200
X_HALF = 36.0                       # image spans x in [-36, 36]
PIX = 2 * X_HALF / WIDTH            # world units per pixel
B_CRIT = 3.0 * np.sqrt(3.0) * M


def solve_b_table(n, alpha, radii, max_steps, p_points):
    """b(alpha, r) for image order n via the production eq.13 solver."""
    objective = geo.lambda_objective()
    p_scan = np.linspace(2.1, 50, p_points)
    table = np.empty((alpha.size, radii.size))
    for j, r in enumerate(radii):
        alpha_solve = (alpha + np.pi) % (2 * np.pi) if n % 2 == 1 else alpha
        p = geo.fast_root(objective, p_scan, alpha_solve, (INCL, r, n, M),
                          max_steps=max_steps)
        b_of_p = np.sqrt(p**3 / (p - 2 * M))
        ellipse = r / np.sqrt(1 + np.tan(INCL) ** 2 * np.cos(alpha_solve) ** 2)
        table[:, j] = np.where(np.isnan(p), ellipse, b_of_p)
    return table


def render(max_steps=10, p_points=1000, n_alpha=200, n_r=49,
           n_alpha_fine=2880, n_r_fine=480):
    """Solve b on a (n_alpha x n_r) table, interpolate onto a fine grid.

    Mirrors the production path (build_impact_table + periodic linear
    interpolation in generate_particles_fast), but on a deterministic grid.
    """
    alpha = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False)
    radii = np.linspace(R_IN, R_OUT, n_r)
    alpha_f = np.linspace(0, 2 * np.pi, n_alpha_fine, endpoint=False)
    radii_f = np.linspace(R_IN, R_OUT, n_r_fine)
    aa, rr = np.meshgrid(alpha_f, radii_f, indexing="ij")
    pts = np.column_stack([aa.ravel(), rr.ravel()])

    layers = {}
    for n in (0, 1):
        table = solve_b_table(n, alpha, radii, max_steps, p_points)
        b = impact_table_interpolator(alpha, radii, table)(pts).reshape(aa.shape)
        opz = redshift_vec(rr, aa, INCL, M, b)
        flux = flux_observed_vec(rr, M, opz)
        x = b * np.cos(aa - np.pi / 2)
        y = b * np.sin(aa - np.pi / 2)
        layers[n] = dict(b=b, flux=flux, x=x, y=y, table=table)

    img = np.zeros((HEIGHT, WIDTH))
    order_map = np.full((HEIGHT, WIDTH), -1)
    # Ghost first, then the direct image on top (it is in front).
    for n in (1, 0):
        L = layers[n]
        col = np.floor((L["x"] + X_HALF) / PIX).astype(int)
        row = np.floor((HEIGHT / 2 * PIX - L["y"]) / PIX).astype(int)
        ok = (col >= 0) & (col < WIDTH) & (row >= 0) & (row < HEIGHT) & (L["flux"] > 0)
        np.maximum.at(img, (row[ok], col[ok]), L["flux"][ok])
        order_map[row[ok], col[ok]] = n
    return img, order_map, layers


def metrics(img, order_map, layers):
    lit = img > 0
    m = {}
    m["nonfinite_pixels"] = int((~np.isfinite(img)).sum())
    c = 5
    corners = np.concatenate([img[:c, :c].ravel(), img[:c, -c:].ravel(),
                              img[-c:, :c].ravel(), img[-c:, -c:].ravel()])
    m["corner_max_rel"] = float(corners.max() / img.max())
    m["primary_pixels"] = int((order_map == 0).sum())
    m["secondary_pixels"] = int((order_map == 1).sum())
    m["secondary_below_center_frac"] = float(
        (layers[1]["y"] < 0).mean())
    # Shadow edge: the inner rim of the ghost image hugs the photon ring.
    m["shadow_radius"] = float(np.nanmin(layers[1]["b"]))
    m["shadow_radius_over_bcrit"] = m["shadow_radius"] / B_CRIT
    # Holes/banding: unlit pixels with >= 7 of 8 lit neighbours.
    p = np.pad(lit, 1)
    neigh = sum(np.roll(np.roll(p, dy, 0), dx, 1)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0))[1:-1, 1:-1]
    m["hole_pixels"] = int(((~lit) & (neigh >= 7)).sum())
    m["hole_frac_of_lit"] = m["hole_pixels"] / max(int(lit.sum()), 1)
    # Doppler: approaching (left, x<0) vs receding (right) half of the direct image.
    d = layers[0]
    left = d["flux"][d["x"] < 0].mean()
    right = d["flux"][d["x"] > 0].mean()
    m["left_right_flux_ratio"] = float(left / right)

    m["checks"] = {
        "no_nan_inf": m["nonfinite_pixels"] == 0,
        "dark_background": m["corner_max_rel"] < 1e-3,
        "primary_image": m["primary_pixels"] > 2000,
        "secondary_image": m["secondary_pixels"] > 100,
        "shadow_radius_near_bcrit": 1.0 <= m["shadow_radius_over_bcrit"] < 1.35,
        "no_banding": m["hole_frac_of_lit"] < 0.01,
        "approaching_side_brighter": m["left_right_flux_ratio"] > 1.5,
    }
    m["checks"] = {k: bool(v) for k, v in m["checks"].items()}
    return m


def save_png(img, path):
    from PIL import Image

    norm = img / img.max()
    # Luminet-style display: compress dynamic range so the ghost is visible.
    v = np.clip(norm, 0, 1) ** 0.4
    rgb = np.stack([v, v ** 1.6, v ** 4], axis=-1)  # warm black-body-ish ramp
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--p-points", type=int, default=1000)
    ap.add_argument("--n-alpha", type=int, default=200,
                    help="alpha resolution of the b table (production: 200)")
    args = ap.parse_args()

    img, order_map, layers = render(args.max_steps, args.p_points, n_alpha=args.n_alpha)
    m = metrics(img, order_map, layers)
    if args.out:
        save_png(img, args.out)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(m, fh, indent=2)
    for k, v in m.items():
        if k != "checks":
            print(f"{k:32s} {v}")
    print()
    for k, v in m["checks"].items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    sys.exit(0 if all(m["checks"].values()) else 1)


if __name__ == "__main__":
    main()
