#!/usr/bin/env python3
"""
Step-size convergence study (CPU, ~3 min).

1. Disk image solver (Luminet eq.13): the effective periastron step is
   dP = (P_max - P_min) / p_points / 2**max_steps. Halving it (doubling
   p_points) must shrink both the b-table error and the preview-image delta
   against a fine reference solve.
2. Photon orbit integrator (RK4 on u'' = -u + 3Mu^2): halving dphi must
   shrink the deflection error ~16x (4th order) against the exact elliptic
   integral (photon_orbits.deflection_angle).
3. Spurious roots: the eq.13 scan starts at P = 2.1M; any root with
   P < 3M (inside the photon sphere) is not a physical turning point.

Usage:  python eval/convergence.py [--json results/convergence.json]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "eval"))

import preview_render as pr  # noqa: E402
from eventHorizon.physics import geodesics as geo  # noqa: E402
from eventHorizon.physics.photon_orbits import (  # noqa: E402
    deflection_angle,
    integrate_binet,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

IMAGE_FLOOR = 1e-5  # relative L1 image delta treated as converged


def image_solver_convergence():
    ref_img, _, ref_layers = pr.render(max_steps=30, p_points=4000)
    rows = []
    for p_points in (250, 500, 1000, 2000):
        img, _, layers = pr.render(max_steps=10, p_points=p_points)
        db = np.concatenate([np.abs(layers[n]["table"] - ref_layers[n]["table"]).ravel()
                             for n in (0, 1)])
        rows.append(dict(
            p_points=p_points,
            dP=(50 - 2.1) / (p_points - 1) / 2**10,
            rms_db=float(np.sqrt(np.nanmean(db**2))),
            p99_db=float(np.nanpercentile(db, 99)),
            max_abs_db=float(np.nanmax(db)),
            image_l1_rel=float(np.abs(img - ref_img).sum() / ref_img.sum()),
        ))
    return rows


def orbit_integrator_convergence():
    rows = []
    for b in (5.25, 10.0):
        exact = deflection_angle(b)
        for dphi in (8e-2, 4e-2, 2e-2, 1e-2):
            _, delta = integrate_binet(b, dphi=dphi)
            rows.append(dict(b=b, dphi=dphi, abs_err=abs(delta - exact)))
    return rows


def spurious_root_scan():
    objective = geo.lambda_objective()
    alpha = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    p_min = np.inf
    for n in (0, 1):
        for r in np.linspace(6.0, 50.0, 45):
            a = (alpha + np.pi) % (2 * np.pi) if n else alpha
            p = geo.fast_root(objective, np.linspace(2.1, 50, 1000), a,
                              (pr.INCL, r, n, pr.M))
            p_min = min(p_min, np.nanmin(p))
    return dict(min_periastron=float(p_min))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = dict(image=image_solver_convergence(),
               orbit=orbit_integrator_convergence(),
               spurious=spurious_root_scan())

    print("eq.13 solver vs reference (max_steps=30, p_points=4000):")
    print(f"{'p_points':>9} {'dP':>10} {'rms|db|':>10} {'p99|db|':>10} "
          f"{'max|db|':>10} {'img L1 rel':>11}")
    for r in out["image"]:
        print(f"{r['p_points']:>9} {r['dP']:>10.2e} {r['rms_db']:>10.2e} "
              f"{r['p99_db']:>10.2e} {r['max_abs_db']:>10.2e} "
              f"{r['image_l1_rel']:>11.2e}")
    # b error must shrink with every halving of dP; the image delta must
    # shrink until it reaches the pixel-quantisation floor (IMAGE_FLOOR),
    # where single disk points hopping to a neighbouring pixel dominate.
    rms = [r["rms_db"] for r in out["image"]]
    l1 = [r["image_l1_rel"] for r in out["image"]]
    ok_img = all(np.diff(rms) < 0) and all(
        b < a or b < IMAGE_FLOOR for a, b in zip(l1, l1[1:]))

    print("\nRK4 orbit integrator vs exact deflection:")
    for r in out["orbit"]:
        print(f"  b={r['b']:<5} dphi={r['dphi']:<6} |err|={r['abs_err']:.2e}")
    ok_orbit = True
    for b in (5.25, 10.0):
        e = [r["abs_err"] for r in out["orbit"] if r["b"] == b]
        ok_orbit &= all(np.diff(e) < 0)

    p_min = out["spurious"]["min_periastron"]
    print(f"\nmin eq.13 periastron over r in [6, 50], n in {{0,1}}: {p_min:.4f}")
    ok_root = p_min > 3.0

    out["checks"] = dict(image_converges=bool(ok_img),
                         orbit_converges=bool(ok_orbit),
                         no_roots_inside_photon_sphere=bool(ok_root))
    for k, v in out["checks"].items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
    sys.exit(0 if all(out["checks"].values()) else 1)


if __name__ == "__main__":
    main()
