#!/usr/bin/env python3
"""
Visual validation: render a low-res preview and check key visual features.
"""
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from eventHorizon.math.geodesics import (
    simulate_flux, lambdify, expr_b, expr_one_plus_z, expr_ellipse
)

# ============================================================================
# Parameters (Luminet 1979)
# ============================================================================
M = 1.0
INCL = np.radians(80.0)
RADII_DIRECT = [6.5, 7, 8, 9, 10, 12, 15, 20, 25, 30]
RADII_GHOST = [7, 10, 15, 20, 30]
N_ANGLES = 200

output_dir = os.path.join(project_root, "eval")

def render_isoradials():
    """Render isoradial curves with flux coloring."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')

    for ax in axes:
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_xlim(-35, 35)
        ax.set_ylim(-35, 35)
        ax.axis('off')

    axes[0].set_title("Direct Image (n=0)", color='white', fontsize=14)
    axes[1].set_title("Ghost Image (n=1)", color='white', fontsize=14)

    alpha = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)

    # Direct images
    all_x, all_y, all_flux = [], [], []
    for r in RADII_DIRECT:
        try:
            alpha_out, b_out, opz, flux = simulate_flux(alpha, r, INCL, 0, M)
            # Convert to Cartesian with -pi/2 rotation (Luminet convention)
            x = b_out * np.cos(alpha_out - np.pi/2)
            y = b_out * np.sin(alpha_out - np.pi/2)

            # Plot the isoradial as a colored line
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux)
            if np.any(valid):
                scatter = axes[0].scatter(x[valid], y[valid], c=flux[valid],
                                         cmap='hot', s=3, vmin=0, vmax=np.nanmax(flux[valid])*1.2)
                all_x.extend(x[valid])
                all_y.extend(y[valid])
                all_flux.extend(flux[valid])
        except Exception as e:
            print(f"  Direct r={r}: {e}")

    # Ghost images
    for r in RADII_GHOST:
        try:
            alpha_out, b_out, opz, flux = simulate_flux(alpha, r, INCL, 1, M)
            x = b_out * np.cos(alpha_out - np.pi/2)
            y = b_out * np.sin(alpha_out - np.pi/2)

            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux)
            if np.any(valid):
                axes[1].scatter(x[valid], y[valid], c=flux[valid],
                               cmap='hot', s=3, vmin=0)
        except Exception as e:
            print(f"  Ghost r={r}: {e}")

    # Add event horizon circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    for ax in axes:
        ax.plot(2*M*np.cos(theta_circle), 2*M*np.sin(theta_circle), 'r-', linewidth=1, alpha=0.5)
        # Photon sphere
        ax.plot(3*np.sqrt(3)*M*np.cos(theta_circle), 3*np.sqrt(3)*M*np.sin(theta_circle),
                'b--', linewidth=0.5, alpha=0.3)

    fig.suptitle("evenHorizon - Isoradial Preview (Luminet 1979)", color='white', fontsize=16)

    save_path = os.path.join(output_dir, "visual_preview_isoradials.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"Saved isoradial preview to {save_path}")
    return save_path


def render_combined_image():
    """Render combined direct + ghost image (Luminet's actual figure)."""
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.axis('off')

    alpha = np.linspace(0, 2*np.pi, N_ANGLES, endpoint=False)

    max_flux_global = 0

    # Collect all data first to find global max flux
    all_data = []

    # Direct images
    for r in RADII_DIRECT:
        try:
            alpha_out, b_out, opz, flux = simulate_flux(alpha, r, INCL, 0, M)
            x = b_out * np.cos(alpha_out - np.pi/2)
            y = b_out * np.sin(alpha_out - np.pi/2)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux) & (flux > 0)
            if np.any(valid):
                all_data.append((x[valid], y[valid], flux[valid], 'direct'))
                max_flux_global = max(max_flux_global, np.nanmax(flux[valid]))
        except Exception:
            pass

    # Ghost images
    for r in RADII_GHOST:
        try:
            alpha_out, b_out, opz, flux = simulate_flux(alpha, r, INCL, 1, M)
            x = b_out * np.cos(alpha_out - np.pi/2)
            y = b_out * np.sin(alpha_out - np.pi/2)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(flux) & (flux > 0)
            if np.any(valid):
                all_data.append((x[valid], y[valid], flux[valid], 'ghost'))
                max_flux_global = max(max_flux_global, np.nanmax(flux[valid]))
        except Exception:
            pass

    # Plot everything with consistent color scale
    for x, y, flux, img_type in all_data:
        s = 4 if img_type == 'direct' else 2
        ax.scatter(x, y, c=flux, cmap='hot', s=s, vmin=0, vmax=max_flux_global*0.3)

    # Event horizon
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.fill(2*M*np.cos(theta_circle), 2*M*np.sin(theta_circle), color='black', zorder=10)
    ax.plot(2*M*np.cos(theta_circle), 2*M*np.sin(theta_circle), 'dimgray', linewidth=0.5, zorder=11)

    ax.set_title("evenHorizon - Combined Image (Luminet 1979 style)\n"
                 "Observer inclination = 80 deg", color='white', fontsize=14)

    save_path = os.path.join(output_dir, "visual_preview_combined.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"Saved combined preview to {save_path}")
    return save_path


def check_visual_features():
    """Automated visual feature checks."""
    print("\n" + "=" * 60)
    print("Visual Feature Checks")
    print("=" * 60)

    alpha = np.linspace(0, 2*np.pi, 500, endpoint=False)

    # Check A: Brightness asymmetry at r=10M
    try:
        alpha_out, b_out, opz, flux = simulate_flux(alpha, 10.0, INCL, 0, M)

        # Left side (approaching): x < 0 in Luminet's convention
        x = b_out * np.cos(alpha_out - np.pi/2)

        left_mask = (x < 0) & np.isfinite(flux) & (flux > 0)
        right_mask = (x > 0) & np.isfinite(flux) & (flux > 0)

        if np.any(left_mask) and np.any(right_mask):
            left_mean = np.nanmean(flux[left_mask])
            right_mean = np.nanmean(flux[right_mask])
            ratio = max(left_mean, right_mean) / min(left_mean, right_mean)
            brighter_side = "LEFT" if left_mean > right_mean else "RIGHT"

            print(f"[{'PASS' if ratio > 2.0 else 'WARN'}] Brightness asymmetry at r=10M:")
            print(f"       Left mean flux  = {left_mean:.6e}")
            print(f"       Right mean flux = {right_mean:.6e}")
            print(f"       Ratio = {ratio:.2f}x ({brighter_side} side brighter)")
            print()
    except Exception as e:
        print(f"[ERROR] Brightness asymmetry check: {e}\n")

    # Check B: Ghost image exists and is dimmer than direct
    try:
        _, _, _, flux_direct = simulate_flux(alpha, 15.0, INCL, 0, M)
        _, _, _, flux_ghost = simulate_flux(alpha, 15.0, INCL, 1, M)

        direct_valid = np.isfinite(flux_direct) & (flux_direct > 0)
        ghost_valid = np.isfinite(flux_ghost) & (flux_ghost > 0)

        has_ghost = np.any(ghost_valid)
        if has_ghost:
            ghost_mean = np.nanmean(flux_ghost[ghost_valid])
            direct_mean = np.nanmean(flux_direct[direct_valid])
            ghost_dimmer = ghost_mean < direct_mean

            print(f"[{'PASS' if ghost_dimmer else 'WARN'}] Ghost image present and dimmer:")
            print(f"       Direct mean flux = {direct_mean:.6e}")
            print(f"       Ghost mean flux  = {ghost_mean:.6e}")
            print(f"       Ghost/Direct = {ghost_mean/direct_mean:.4f}")
        else:
            print("[FAIL] No ghost image detected at r=15M")
        print()
    except Exception as e:
        print(f"[ERROR] Ghost image check: {e}\n")

    # Check C: Shadow edge. The ghost (n=1) image hugs the photon ring from
    # outside, so its innermost b sits just above b_crit. (The direct image
    # of the near side legitimately crosses in front of the shadow with
    # b < b_crit, Luminet 1979 Fig. 4 trajectory 3, so it is not used here.)
    try:
        b_crit = 3 * np.sqrt(3) * M
        _, b_out, _, _ = simulate_flux(alpha, 6.5, INCL, 1, M)
        b_min = np.nanmin(b_out)
        ok = b_crit <= b_min < 1.1 * b_crit

        print(f"[{'PASS' if ok else 'WARN'}] Shadow edge check (ghost isoradial just outside b_crit):")
        print(f"       b_crit = {b_crit:.4f}")
        print(f"       min(b) of ghost at r=6.5M = {b_min:.4f}")
        print()
    except Exception as e:
        print(f"[ERROR] Shadow edge check: {e}\n")

if __name__ == "__main__":
    print("Rendering visual preview...")
    render_isoradials()
    render_combined_image()
    check_visual_features()
