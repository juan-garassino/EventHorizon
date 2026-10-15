# Physics Overview

This document explains the physics behind evenHorizon at an accessible level,
then gives the exact equations implemented in the code.

---

## Plain language

A black hole warps space so severely that light doesn't travel in straight
lines. Each ray of light curves around the black hole following a path called
a **geodesic**. The accretion disk -- a flat ring of hot glowing gas orbiting
the black hole -- appears distorted:

- The **back half** of the disk is bent up and over the top, creating a
  bright arc above the black hole (the "hat" shape).
- **One side is much brighter** than the other because gas on the approaching
  side is Doppler-boosted (like a siren getting louder as it approaches).
- A faint **ghost image** appears below the black hole from light that
  traveled around the back.

This is what Luminet computed by hand in 1979 and what evenHorizon
reproduces numerically.

---

## Spacetime

We use the **Schwarzschild metric** (non-rotating black hole) in geometric
units where G = c = 1:

```
ds^2 = -(1 - 2M/r) dt^2 + (1 - 2M/r)^{-1} dr^2 + r^2 dOmega^2
```

Key radii with M = 1:

| Radius | Name | Meaning |
|--------|------|---------|
| r = 0 | Singularity | Point of infinite curvature |
| r = 2M | Event horizon | Point of no return for anything |
| r = 3M | Photon sphere | Unstable circular photon orbit |
| r = 6M | ISCO | Innermost stable circular orbit; inner edge of the disk |

The accretion disk is modelled as infinitely thin, lying in the equatorial
plane from r = 6M to r = 50M.

---

## Step 1: Impact parameter (gravitational lensing)

For each point on the disk at radius r and azimuthal angle alpha, we need
to find the **impact parameter** b -- the perpendicular distance between
the light ray and the black hole in flat space. This determines where the
point appears on the observer's image plane.

### Luminet's equation 13

The emission angle gamma relates the disk point to the observer:

```
cos(gamma) = cos(alpha) / sqrt(cos^2(alpha) + 1/tan^2(i))
```

where i is the observer inclination.

The impact parameter is found by solving an implicit equation involving
**elliptic integrals**. For a photon with closest approach (periastron) P:

```
b = sqrt(P^3 / (P - 2M))
```

The deflection angle as a function of P involves the complete and incomplete
elliptic integrals of the first kind:

```
gamma = 2 * sqrt(P/Q) * [F(zeta_inf, k^2) - F(zeta_r, k^2)]  (direct image)
gamma = 2 * sqrt(P/Q) * [2K(k^2) - F(zeta_inf, k^2) + F(zeta_r, k^2)]  (ghost image)
```

where:
- Q = sqrt((P - 2M)(P + 6M))
- k^2 = (Q - P + 6M) / (2Q)
- zeta_inf = arcsin(sqrt((Q - P + 2M) / (Q - P + 6M)))

We solve this numerically: scan over periastron values, find sign changes
in the residual of eq. 13, then refine by bisection to find the exact P
that matches the required deflection angle. This is done in
`_calc_enhanced_impact_parameter_exact()`.

### Coordinate transformation

The impact parameter b and angle alpha are converted to Cartesian observer
coordinates with a 90-degree rotation:

```
X = b * cos(alpha - pi/2)
Y = b * sin(alpha - pi/2)
```

---

## Step 2: Redshift (Luminet's equation 19)

The combined gravitational + Doppler redshift factor (1+z) determines how
much the light is dimmed or brightened:

```
(1 + z) = [1 + sqrt(M/r^3) * b * sin(i) * sin(alpha)] * [1 - 3M/r]^{-1/2}
```

Breaking this down:

- **Doppler term**: `1 + sqrt(M/r^3) * b * sin(i) * sin(alpha)`
  - sqrt(M/r^3) is the Keplerian orbital angular velocity
  - The product with b, sin(i), sin(alpha) gives the line-of-sight velocity
  - Positive for receding gas (redshift), negative for approaching (blueshift)

- **Gravitational term**: `[1 - 3M/r]^{-1/2}`
  - This is u^t of a circular Keplerian orbit,
    `u^t = (1 - 2M/r - r^2 Omega^2)^{-1/2}` with `Omega = sqrt(M/r^3)`:
    gravitational redshift *plus* the transverse Doppler shift of the
    orbiting gas. It is deliberately not the static-emitter factor
    `[1 - 2M/r]^{-1/2}`. At the ISCO, sqrt(1 - 3M/r) = sqrt(1/2) ~ 0.707.
  - Always > 1, tends to 1 far from the hole, diverges at r = 3M
  - The same expression is implemented in `redshift_vec`,
    `disk_physics.redshift_factor` and `geodesics.expr_one_plus_z`
    (checked against u^t (1 + Omega L/E) in
    `tests/test_physics_ground_truth.py`)

When (1+z) < 1, the light is blueshifted (brighter). When (1+z) > 1, it is
redshifted (dimmer). This is why the left side of the disk appears much
brighter than the right.

---

## Step 3: Intrinsic flux (Novikov-Thorne)

The intrinsic flux emitted at radius r uses the **Page-Thorne (1974)**
formula with the full logarithmic correction:

```
F_intrinsic = (3M * mdot) / (8 * pi) * 1/((r/M - 3) * r^{5/2}) *
              [sqrt(r/M) - sqrt(6) + (sqrt(3)/2) * ln(L)]
```

where:
```
L = [(sqrt(r/M) + sqrt(3)) * (sqrt(6) - sqrt(3))] /
    [(sqrt(r/M) - sqrt(3)) * (sqrt(6) + sqrt(3))]
```

This is zero at the ISCO (r = 6M) and peaks at r = 9.55M. The log-term
factor is sqrt(3)/2 (Page & Thorne 1974); the sqrt(3)/3 printed in
Luminet (1979) eq. 15 is a typo. The implementation agrees with a direct
numerical evaluation of the Page-Thorne conservation-law integral to 1e-6
(see `tests/test_physics_ground_truth.py`).

Because the zero-torque inner boundary suppresses the flux near the ISCO,
the local temperature T ~ F^{1/4} is *lower* at 7M than at 15M
(F(7M)/F(15M) ~ 0.73). Only the Newtonian Shakura-Sunyaev profile
T^4 ~ r^-3 (1 - sqrt(6M/r)) has T(7M) > T(15M).

### Observed flux

The observed flux includes the relativistic beaming law:

```
F_observed = F_intrinsic / (1 + z)^4
```

The fourth power comes from: one factor each for energy shift, time
dilation, solid angle compression, and frequency-to-frequency mapping.
This dramatically amplifies the approaching (blueshifted) side and
suppresses the receding side.

---

## Step 4: Ghost image

Light that passes behind the black hole and wraps around by more than
pi radians creates a secondary image. In the code this is the n=1 solution
to eq. 13. A ghost photon arriving at plate position (b, alpha) sweeps
pi + gamma(alpha) around the hole before reaching the disk, half a turn
more than the direct photon at the same alpha. Eq. 13's n=1 branch encodes
a sweep of 2*pi - gamma, so it is evaluated at alpha + pi (since
gamma(alpha + pi) = pi - gamma(alpha)); the resulting b is then plotted at
plate angle alpha with no further flip. The wide part of the ghost image
is therefore below the shadow (the underside of the far disk) and the thin
part hugs the photon ring above it. This geometry is checked by direct
orbit integration in `tests/test_physics_ground_truth.py`. Ghost particles
are plotted with reduced opacity.

Ghost particles are filtered by the apparent inner edge: those inside
the lensed ISCO boundary form the inner ghost ring, those outside form
the outer ghost region. The two groups are rendered separately with
different opacity levels.

---

## Step 5: Black fill (apparent inner edge)

The black region in the centre is not a simple circle -- it is the
**gravitationally lensed apparent inner edge** of the disk. The code
traces this curve by computing the impact parameter for r = 6M at
200 angles around the disk, producing a teardrop-like shape that is
filled black.

---

## Rendering

### Contour mode (`luminet`)

Particles are assembled into a Delaunay triangulation and rendered with
`matplotlib.tricontourf` using the Greys_r colormap. Flux values are
power-scaled: `display = (flux/max_flux)^{power_scale}`.

### Scatter mode (`scatter`)

Each particle is drawn as a dot with the `hot` colormap. Dot size is
proportional to flux. This reveals the individual sampling structure
and is suitable for pen plotter export.

### Isoradial mode (`isoradials`)

Constant-radius curves are traced by computing the impact parameter at
many angles for each radius. These polylines are drawn in white and can
be exported as SVG paths or G-code toolpaths.

---

## References

1. Luminet, J.-P. (1979). "Image of a spherical black hole with thin
   accretion disk." *Astronomy and Astrophysics*, 75, 228-235.

2. Page, D. N. & Thorne, K. S. (1974). "Disk-accretion onto a black hole.
   Time-averaged structure of accretion disk." *The Astrophysical Journal*,
   191, 499-506.

3. Cunningham, C. T. & Bardeen, J. M. (1973). "The optical appearance of
   a star orbiting an extreme Kerr black hole." *The Astrophysical Journal*,
   183, 237-264.
