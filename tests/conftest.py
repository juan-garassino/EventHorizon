# Legacy test modules that cannot be collected on main: they import modules
# that no longer exist in this repo (`black_hole` from the old flat layout,
# `src.math.*`, `bh`, `data`). Skipped so `pytest` runs the live suite; see
# GUARDIAN.md backlog (port or delete them).
collect_ignore = [
    "test_black_hole.py",
    "test_black_hole_math.py",
    "test_isoradials.py",
    "benchmark.py",
]
