# GUARDIAN

Automated security & bug-hygiene ledger for `juan-garassino/event-horizon`.
Work happens only on branch `claude/guardian`; changes reach `main` only via the
draft PR "Guardian: security & bug fixes", after owner review.

## Checks

| Area | Tool / rule | Scope |
|---|---|---|
| Dependency vulns | `pip-audit -r <pins> --no-deps` | 21 pinned packages exported from `uv.lock` (no requirements*.txt / package.json in repo) |
| Risky code | `bandit -r -q` + grep for `eval`/`exec`, pickle/`torch.load`, `yaml.load`, `shell=True`, SQL formatting, path traversal, SSRF, `verify=False`, CORS, `0.0.0.0` servers | first-party code |
| Secrets | `gitleaks detect --log-opts=--all --redact` (history) + `gitleaks dir . --redact` (tree) | whole repo, all branches/commits |
| CI | GitHub Actions pinning / permissions / `${{ github.event.* }}` injection | none present (no `.github/`) |
| Containers | Dockerfile root user / unpinned base | none present |
| Tests | `pytest` (venv, Python 3.12, deps pinned from `uv.lock`) | `tests/` |
| Lint | `ruff check --select E9,F63,F7,F82` (real errors only; repo has no ruff config) | first-party code |
| Physics smoke | `python eval/physics_checks.py` | 8 known-answer physics checks |

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
| G-7 | low | B110/B112 silent `try/except: pass/continue` (x~30) | handlers/, visualization/ | open (quality) | 2026-10-17 |
| G-8 | info | duplicated modules: `eventHorizon/math/*` and `eventHorizon/physics/*` are near-identical copies (fixes must be applied twice) | `eventHorizon/math`, `eventHorizon/physics` | open (backlog) | 2026-10-17 |

No dependency vulnerabilities and no secrets were found.

## Metrics

| metric | before | after |
|---|---|---|
| dependency vulns (crit/high/med/low) | 0/0/0/0 | 0/0/0/0 |
| bandit, first-party code excl. `tests/` (high/med/low) | 0/3/26 | 0/0/24 |
| secrets (gitleaks history + tree) | 0 | 0 |
| tests collected / passing | 0 / 0 (3 collection errors) | 8 / 8 |
| failing tests | 3 modules error | 0 |
| ruff real errors (E9,F63,F7,F82) | 0 | 0 |
| `eval/physics_checks.py` | 6/8 | 8/8 |

## Backlog

1. Port or delete legacy tests (`tests/test_black_hole*.py`, `tests/test_isoradials.py`, `tests/benchmark.py`) — they target the old flat layout / `src.math`.
2. Known-answer test for `impact_parameter` vs `calc_impact_parameter_exact` (agree to ~1e-4 at r=10, i=80°; slow-ish, needs a tolerance decision).
3. G-7: replace silent `except: pass/continue` with logged warnings.
4. G-8: de-duplicate `eventHorizon/math` vs `eventHorizon/physics` (owner decision; large refactor).
5. Add a CI workflow (pytest + ruff) with SHA-pinned actions and `permissions: contents: read`.

## Log

- **Run 1 — 2026-10-17**: first scan. pip-audit clean, gitleaks clean (28 commits), no CI/Docker. Fixed G-1 (safe ini literal parser replaces `eval`), G-2 (`fast_root` convergence check), G-3 (runnable test suite). Added `tests/test_guardian_regressions.py` (4 tests, 3 of which failed before the fixes).
- **Run 1, pass 2 — 2026-10-17**: fixed G-5 (`eval/physics_checks.py` now 8/8, pointed at `physics.fast_geodesics.flux_intrinsic_vec` / `physics.disk_physics.redshift_factor`), fixed G-4 (dropped `minidom` re-parse in both `plotter_export.py` copies). Added 4 known-answer/regression tests (vector vs scalar redshift & flux, Doppler asymmetry, ISCO gravitational redshift = sqrt 2, SVG well-formedness). Suite 8/8.
