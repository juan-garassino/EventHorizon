# GUARDIAN

Automated security & bug-hygiene ledger for `juan-garassino/event-horizon`.
Work happens only on branch `claude/guardian`; changes reach `main` only via the
draft PR "Guardian: security & bug fixes", after owner review.

## Checks

| Area | Tool / rule | Scope |
|---|---|---|
| Dependency vulns | `uv export --frozen --no-hashes` -> `pip-audit -r <pins> --no-deps --disable-pip` | 21 pinned packages exported from `uv.lock` (no requirements*.txt / package.json in repo). `--disable-pip` is required when the audit venv's Python is older than a pin's `Requires-Python` (contourpy 1.4.0 needs 3.12) |
| Risky code | `bandit -r -q` + grep for `eval`/`exec`, pickle/`torch.load`, `yaml.load`, `shell=True`, SQL formatting, path traversal, SSRF, `verify=False`, CORS, `0.0.0.0` servers | first-party code |
| Secrets | `gitleaks detect --log-opts=--all --redact` (history) + `gitleaks dir . --redact` (tree) | whole repo, all branches/commits |
| CI | GitHub Actions pinning / permissions / `${{ github.event.* }}` injection | none present (no `.github/`) |
| Containers | Dockerfile root user / unpinned base | none present |
| Tests | `pytest` (venv, Python 3.12, deps pinned from `uv.lock`) | `tests/` |
| Lint | `ruff check --select E9,F63,F7,F82` (real errors only; repo has no ruff config) | first-party code |
| Physics smoke | `python eval/physics_checks.py` | 8 known-answer physics checks |
| Solver cross-check (run 2) | tests: vectorized eq.13 solver vs scalar bisection solver, incl. r up to 300M | `tests/test_guardian_regressions.py` |
| Renderer consistency (run 2) | tests: ghost isoradial polyline vs particle-path ghost convention; plus a smoke render of `main.py --mode {luminet,scatter,isoradials} -q draft --no-display` | handlers + `main.py` |

**Exclusions:** `references/` (vendored third-party reference implementations and
validation CSVs, ~21 MB), `build/` (compiled object files), notebook outputs
(`eventHorizon_demo.ipynb`), `results/` (generated run artifacts). These were still
covered by the gitleaks scans.

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-1 | medium | code-injection (`eval` of ini values, B307) | `eventHorizon/core/isoradial_model.py` (`BaseModel.read_parameters`) | fixed | 2026-10-17 |
| G-2 | medium | bug: `fast_root` early exit used `nanmax(f)` not `nanmax(abs(f))` -> returns after 1 bisection when all residuals are negative | `eventHorizon/math/geodesics.py`, `eventHorizon/physics/geodesics.py` | fixed | 2026-10-17 |
| G-3 | medium | test suite: all 3 test modules fail collection (import removed modules); root `pytest` also collects `scripts/dev/*` which call `sys.exit` | `tests/`, `pyproject.toml` | fixed (legacy modules ignored, `testpaths` set) | 2026-10-17 |
| G-4 | low | B318 `minidom.parseString` (parses only self-generated SVG -> not exploitable) | `eventHorizon/{rendering/export,visualization}/plotter_export.py` | fixed (run 1, pass 2: `ElementTree.indent`) | 2026-10-17 |
| G-5 | low | bug: `eval/physics_checks.py` checks 7 & 8 import removed modules (`math.flux_calculations`, `core.physics_engine`) -> always FAIL | `eval/physics_checks.py` | fixed (run 1, pass 2) | 2026-10-17 |
| G-6 | low | B603/B404 `subprocess.run` of ffmpeg with list args (no shell) | `eventHorizon/rendering/animation.py`, `eventHorizon/visualization/animation_engine.py` | wontfix (safe usage) | 2026-10-17 |
| G-7 | low | B110/B112 silent `try/except: pass/continue` (x~30) | handlers/, visualization/ | partly fixed (pass 3: 4 fast-path fallbacks now log at DEBUG; per-point `continue` loops left as is) | 2026-10-17 |
| G-9 | medium | bug: impact-parameter interpolator built on `linspace(0, 2pi, endpoint=False)` with NaN fill -> particles with alpha in (alpha_max, 2pi) dropped (missing disk wedge; 2.5% at n_alpha=40) | `{math,physics}/fast_geodesics.py`, `handlers/{contour,isoradial}.py`, `visualization/mode_router.py` | fixed (pass 3: periodic wrap `_periodic_interp`) | 2026-10-17 |
| G-10 | medium | bug: `SpacetimeGeometry.calculate_disk_flux` zeroed only r <= 3M, so 3M < r < 6M emitted and diverged as r -> 3M | `math/spacetime_geometry.py`, `physics/spacetime.py` | fixed (pass 3) | 2026-10-17 |
| G-11 | low | bare `except:` (x6) in `LuminetPointsHandler` swallowed KeyboardInterrupt/SystemExit | `visualization/mode_router.py` | fixed (pass 3: arithmetic helpers -> `(ArithmeticError, ValueError, TypeError)`, solver -> `Exception`; same narrowing in `physics/disk_physics.py`) | 2026-10-17 |
| G-12 | low | DoS: G-1 parser allowed `9 ** 9 ** 9` (hangs) | `core/isoradial_model.py` | fixed (pass 3: exponent bound 64) | 2026-10-17 |
| G-13 | medium | bug: periastron scan grid hard-capped at P=50 -> for disk radii beyond ~120M eq.13 was never bracketed and `impact_parameter` silently used the Newtonian ellipse (13% error at r=150; `calculate_periastron` returned None) | `{math,physics}/geodesics.py` | fixed (run 2: grid to `max(50, 1.05 r)`) | 2026-10-17 |
| G-14 | medium | bug: ghost isoradials drawn upside down: the n=1 solver flips alpha by pi and the isoradial handlers negated Y again (double flip), so the ghost ring bulged above the shadow and disagreed with the particle renderers | `handlers/isoradial.py`, `visualization/mode_router.py` | fixed (run 2) | 2026-10-17 |
| G-15 | low | `fast_root` bisected all-NaN brackets and emitted `RuntimeWarning: All-NaN slice` when nothing was bracketed | `{math,physics}/geodesics.py` | fixed (run 2) | 2026-10-17 |
| G-16 | low | dead code: `build_impact_table` computed `p_scan` and ignored `p_grid_size` | `{math,physics}/fast_geodesics.py` | fixed (run 2: removed, arg documented as unused) | 2026-10-17 |
| G-8 | info | duplicated modules: `eventHorizon/math/*` and `eventHorizon/physics/*` are near-identical copies (fixes must be applied twice) | `eventHorizon/math`, `eventHorizon/physics` | open (backlog) | 2026-10-17 |

No dependency vulnerabilities and no secrets were found (runs 1 and 2).

## Metrics

| metric | before | after |
|---|---|---|
| dependency vulns (crit/high/med/low) | 0/0/0/0 | 0/0/0/0 |
| bandit, first-party code excl. `tests/` (high/med/low) | 0/3/26 | 0/0/20 |
| secrets (gitleaks history + tree) | 0 | 0 |
| tests collected / passing | 0 / 0 (3 collection errors) -> run 2 start 52 / 52 | 67 / 67 |
| failing tests | 3 modules error | 0 |
| ruff real errors (E9,F63,F7,F82) | 0 | 0 |
| bare `except:` in `eventHorizon/` (E722) | 6 | 0 |
| `eval/physics_checks.py` | 6/8 | 8/8 |
| pytest warnings | 2 (All-NaN slice) | 0 |
| open medium bugs found in run 2 | 2 (G-13, G-14) | 0 |

## Backlog

1. Port or delete legacy tests (`tests/test_black_hole*.py`, `tests/test_isoradials.py`, `tests/benchmark.py`) — they target the old flat layout / `src.math`.
2. G-7 remainder: per-point `except Exception: continue` loops in `handlers/isoradial.py`, `handlers/edge_fill.py`, `visualization/mode_router.py` (keep broad, consider counting/logging skipped points).
3. G-8: de-duplicate `eventHorizon/math` vs `eventHorizon/physics` (owner decision; large refactor). Run 2 again had to patch every fix twice.
4. Add a CI workflow (pytest + ruff) with SHA-pinned actions and `permissions: contents: read`. **Owner action**: deliberately not added by the guardian (outward-facing change).
5. `edge_fill.add_inner_disk_edge_fill(y_flip=True)` and the `mode_router` copy: no caller passes `y_flip=True`; after G-14 the "ghost region is negative Y" docstring is the only remaining trace of the double-flip convention. Delete or document.
6. Ruff F841 (11 unused locals, e.g. `quality_config` in `api.py`/`luminet.py`, `body` in `main.py`) — cosmetic, batch them in a quality pass.
7. Physics-owner question (from run 1 backlog 6, analysed in run 2): the vendored `references/luminet` does NOT flip alpha for n=1 and instead negates Y at plot time; commit 012d51b flips alpha at solve time. For the particle path the two are equivalent (screen image identical up to the left/right mirror, Doppler preserved). Resolved as consistent; kept here only so a future reference update re-checks it.

## Ideas (run 2)

- Golden-image check: render `--mode isoradials -q draft` and assert a few polyline extrema (cheap, catches convention flips like G-14 across all renderers).
- Property test: every renderer's ghost polyline/particles should lie at |b| >= 3*sqrt(3) M.
- Run `pip-audit` inside a Python 3.12 venv so the default (pip-resolving) mode works and `--disable-pip` is not needed.

## Log

- **Run 1 — 2026-10-17**: first scan. pip-audit clean, gitleaks clean (28 commits), no CI/Docker. Fixed G-1 (safe ini literal parser replaces `eval`), G-2 (`fast_root` convergence check), G-3 (runnable test suite). Added `tests/test_guardian_regressions.py` (4 tests, 3 of which failed before the fixes).
- **Run 1, pass 2 — 2026-10-17**: fixed G-5 (`eval/physics_checks.py` now 8/8, pointed at `physics.fast_geodesics.flux_intrinsic_vec` / `physics.disk_physics.redshift_factor`), fixed G-4 (dropped `minidom` re-parse in both `plotter_export.py` copies). Added 4 known-answer/regression tests (vector vs scalar redshift & flux, Doppler asymmetry, ISCO gravitational redshift = sqrt 2, SVG well-formedness). Suite 8/8.
- **Run 1, pass 3 — 2026-10-17**: 19 parser-rejection tests (`__import__`, attribute/dunder access, calls, lambdas, comprehensions, dicts, bad syntax) and 17 parser-acceptance cases. Fixed G-12 (pow DoS), G-9 (alpha wrap-around NaN wedge, all 5 interpolator sites), G-10 (disk flux inside ISCO), G-11 (bare excepts), part of G-7 (fallbacks now log). Each fix has a test that failed first. Suite 52/52, physics checks 8/8. Leaving the math/physics duplication and CI to the owner.
- **Run 2 — 2026-10-17**: continuation from the run-1 backlog. Scans: pip-audit clean (needed `--disable-pip`, see Checks), gitleaks clean (31 commits + tree), bandit 0 high / 0 medium, no CI / Docker added upstream. No open PR review comments.
  - Pass 1: backlog 2 known-answer tests (vectorized vs scalar eq.13 solver, mirror symmetry, ghost b >= b_crit). While extending them to large radii found **G-13** (periastron grid capped at P=50) and fixed both copies plus `UnifiedGeodesics.calculate_periastron`; removed dead `p_scan` (G-16, backlog 7).
  - Pass 2: analysed backlog 6 (ghost alpha convention) and found **G-14**: the isoradial handlers double-flipped the ghost image. Confirmed visually (before: dashed ghost ring above the shadow; after: below, as in Luminet 1979) and with a regression test against the particle convention.
  - Pass 3: G-15 (`fast_root` early return, no more All-NaN warnings). Suite 52 -> 67, physics checks 8/8.
  - **META-REVIEW (4th Log entry).** Added two checks: a solver cross-check (two independent eq.13 implementations must agree, including far outside the default disk) and a renderer-consistency check (the same image order must land in the same place in every renderer). Why: both medium bugs this run were invisible to unit tests of single functions; they only showed up when comparing two code paths or extending the input range, and run 1's checks were all single-path. Tightened the pip-audit invocation (documented `--disable-pip` failure mode). Dropped backlog 6 as a standing open item (resolved by analysis, kept as a note). No false positives to cut: bandit's remaining 20 lows are the known G-6/G-7 patterns.
