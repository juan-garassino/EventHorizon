"""
SVG and G-code export for pen-plotter output.

Converts isoradial curves (polylines) and scatter particles (dots)
into SVG paths and G-code toolpaths suitable for pen plotters.
"""
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent, tostring

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Polyline = List[Tuple[float, float]]  # list of (x, y) points


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

class PlotterBed:
    """Maps world coordinates (centered on black hole) to plotter bed coordinates."""

    def __init__(
        self,
        bed_width_mm: float = 297.0,   # A4 landscape
        bed_height_mm: float = 210.0,
        margin_mm: float = 15.0,
        world_range: Tuple[float, float] = (-40, 40),
    ):
        self.bed_width = bed_width_mm
        self.bed_height = bed_height_mm
        self.margin = margin_mm
        self.world_min, self.world_max = world_range
        self.world_span = self.world_max - self.world_min

        # Drawable area
        draw_w = self.bed_width - 2 * self.margin
        draw_h = self.bed_height - 2 * self.margin

        # Uniform scale (preserve aspect ratio)
        self.scale = min(draw_w, draw_h) / self.world_span

        # Offset so (0,0) world maps to centre of bed
        self.offset_x = self.bed_width / 2
        self.offset_y = self.bed_height / 2

    def to_plotter(self, wx: float, wy: float) -> Tuple[float, float]:
        """World coords → plotter mm.  Y is flipped (SVG/G-code Y-down)."""
        px = self.offset_x + wx * self.scale
        py = self.offset_y - wy * self.scale  # flip Y
        return round(px, 3), round(py, 3)

    def in_bounds(self, px: float, py: float) -> bool:
        return (self.margin <= px <= self.bed_width - self.margin and
                self.margin <= py <= self.bed_height - self.margin)


# ---------------------------------------------------------------------------
# SVG export
# ---------------------------------------------------------------------------

def polylines_to_svg(
    polylines: List[Polyline],
    output_path: str,
    bed: Optional[PlotterBed] = None,
    stroke_width: float = 0.5,
    stroke_color: str = "white",
    background: str = "black",
) -> str:
    """
    Export a list of polylines as an SVG file.

    Parameters
    ----------
    polylines : list of list of (x, y)
        Each polyline is a list of world-coordinate points.
    output_path : str
        Where to save the SVG.
    bed : PlotterBed, optional
        Coordinate mapper.  If None, uses defaults.
    stroke_width : float
        SVG stroke width in mm.
    stroke_color : str
        Stroke color.
    background : str
        Background rectangle color (use "none" to omit).

    Returns
    -------
    str
        Path to the saved file.
    """
    bed = bed or PlotterBed()
    w, h = bed.bed_width, bed.bed_height

    svg = Element("svg", xmlns="http://www.w3.org/2000/svg",
                  width=f"{w}mm", height=f"{h}mm",
                  viewBox=f"0 0 {w} {h}")

    # Background
    if background and background != "none":
        SubElement(svg, "rect", x="0", y="0",
                   width=str(w), height=str(h), fill=background)

    # Group for all paths
    g = SubElement(svg, "g", fill="none", stroke=stroke_color,
                   **{"stroke-width": str(stroke_width),
                      "stroke-linecap": "round",
                      "stroke-linejoin": "round"})

    for poly in polylines:
        if len(poly) < 2:
            continue
        pts = [bed.to_plotter(x, y) for x, y in poly]
        d_parts = [f"M{pts[0][0]},{pts[0][1]}"]
        for px, py in pts[1:]:
            d_parts.append(f"L{px},{py}")
        SubElement(g, "path", d=" ".join(d_parts))

    _write_svg(svg, output_path)
    return output_path


def scatter_to_svg(
    points: List[Tuple[float, float]],
    output_path: str,
    bed: Optional[PlotterBed] = None,
    dot_radius_mm: float = 0.3,
    intensities: Optional[List[float]] = None,
    stroke_color: str = "white",
    background: str = "black",
) -> str:
    """
    Export scatter points as small circles in SVG.

    Parameters
    ----------
    points : list of (x, y)
        World-coordinate particle positions.
    output_path : str
        Where to save.
    bed : PlotterBed, optional
        Coordinate mapper.
    dot_radius_mm : float
        Circle radius in mm.
    intensities : list of float, optional
        Per-point intensity [0,1] controlling opacity.
    stroke_color : str
        Stroke color for circles.
    background : str
        Background color.

    Returns
    -------
    str
        Path to saved file.
    """
    bed = bed or PlotterBed()
    w, h = bed.bed_width, bed.bed_height

    svg = Element("svg", xmlns="http://www.w3.org/2000/svg",
                  width=f"{w}mm", height=f"{h}mm",
                  viewBox=f"0 0 {w} {h}")

    if background and background != "none":
        SubElement(svg, "rect", x="0", y="0",
                   width=str(w), height=str(h), fill=background)

    g = SubElement(svg, "g", fill="none", stroke=stroke_color,
                   **{"stroke-width": str(min(dot_radius_mm, 0.3))})

    for i, (wx, wy) in enumerate(points):
        px, py = bed.to_plotter(wx, wy)
        if not bed.in_bounds(px, py):
            continue
        attrs = {"cx": str(px), "cy": str(py), "r": str(dot_radius_mm)}
        if intensities is not None and i < len(intensities):
            opacity = max(0.05, min(1.0, intensities[i]))
            attrs["opacity"] = f"{opacity:.2f}"
        SubElement(g, "circle", **attrs)

    _write_svg(svg, output_path)
    return output_path


def _write_svg(root: Element, path: str) -> None:
    # Pretty-print in place with ElementTree (no re-parse through minidom).
    indent(root, space="  ")
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             tostring(root, encoding="unicode"), ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# G-code export
# ---------------------------------------------------------------------------

def polylines_to_gcode(
    polylines: List[Polyline],
    output_path: str,
    bed: Optional[PlotterBed] = None,
    feed_rate: float = 3000.0,
    travel_rate: float = 5000.0,
    pen_down_cmd: str = "M3 S1000",
    pen_up_cmd: str = "M5",
    z_pen_down: Optional[float] = None,
    z_pen_up: Optional[float] = None,
) -> str:
    """
    Export polylines as G-code for a pen plotter.

    Supports both servo-based (M3/M5) and Z-axis pen control.

    Parameters
    ----------
    polylines : list of polyline
        World-coordinate polylines.
    output_path : str
        Where to save.
    bed : PlotterBed, optional
        Coordinate mapper.
    feed_rate : float
        Drawing feed rate (mm/min).
    travel_rate : float
        Travel (pen-up) feed rate.
    pen_down_cmd / pen_up_cmd : str
        G-code commands for pen servo.  Ignored if z_pen_down is set.
    z_pen_down / z_pen_up : float, optional
        If set, use Z-axis movement instead of servo commands.

    Returns
    -------
    str
        Path to saved file.
    """
    bed = bed or PlotterBed()
    lines = _gcode_header(feed_rate, travel_rate)

    use_z = z_pen_down is not None and z_pen_up is not None

    def pen_up():
        if use_z:
            return f"G0 Z{z_pen_up}"
        return pen_up_cmd

    def pen_down():
        if use_z:
            return f"G1 Z{z_pen_down} F{feed_rate}"
        return pen_down_cmd

    lines.append(pen_up())

    for poly in polylines:
        if len(poly) < 2:
            continue
        pts = [bed.to_plotter(x, y) for x, y in poly]

        # Travel to start
        lines.append(f"G0 X{pts[0][0]} Y{pts[0][1]} F{travel_rate}")
        lines.append(pen_down())

        # Draw
        for px, py in pts[1:]:
            lines.append(f"G1 X{px} Y{py} F{feed_rate}")

        lines.append(pen_up())

    lines.extend(_gcode_footer())
    _write_gcode(lines, output_path)
    return output_path


def scatter_to_gcode(
    points: List[Tuple[float, float]],
    output_path: str,
    bed: Optional[PlotterBed] = None,
    feed_rate: float = 3000.0,
    travel_rate: float = 5000.0,
    pen_down_cmd: str = "M3 S1000",
    pen_up_cmd: str = "M5",
    z_pen_down: Optional[float] = None,
    z_pen_up: Optional[float] = None,
    dot_dwell_ms: int = 50,
    sort_nearest: bool = True,
) -> str:
    """
    Export scatter points as G-code dot marks.

    Each point is a pen-down then immediate pen-up (with optional dwell).
    Points are optionally sorted by nearest-neighbor to minimise travel.

    Parameters
    ----------
    points : list of (x, y)
        World coordinates.
    output_path : str
        Where to save.
    dot_dwell_ms : int
        Milliseconds to dwell with pen down at each dot.
    sort_nearest : bool
        If True, reorder points by nearest-neighbor to reduce travel.

    Returns
    -------
    str
        Path to saved file.
    """
    bed = bed or PlotterBed()

    use_z = z_pen_down is not None and z_pen_up is not None

    def pen_up():
        return f"G0 Z{z_pen_up}" if use_z else pen_up_cmd

    def pen_down():
        return f"G1 Z{z_pen_down} F{feed_rate}" if use_z else pen_down_cmd

    # Map to plotter coords and filter
    mapped = []
    for wx, wy in points:
        px, py = bed.to_plotter(wx, wy)
        if bed.in_bounds(px, py):
            mapped.append((px, py))

    if sort_nearest and mapped:
        mapped = _nearest_neighbor_sort(mapped)

    lines = _gcode_header(feed_rate, travel_rate)
    lines.append(pen_up())

    for px, py in mapped:
        lines.append(f"G0 X{px} Y{py} F{travel_rate}")
        lines.append(pen_down())
        if dot_dwell_ms > 0:
            lines.append(f"G4 P{dot_dwell_ms}")
        lines.append(pen_up())

    lines.extend(_gcode_footer())
    _write_gcode(lines, output_path)
    return output_path


# ---------------------------------------------------------------------------
# G-code helpers
# ---------------------------------------------------------------------------

def _gcode_header(feed: float, travel: float) -> List[str]:
    return [
        "; evenHorizon plotter export",
        "G21 ; mm",
        "G90 ; absolute positioning",
        f"; draw feed: {feed} mm/min",
        f"; travel feed: {travel} mm/min",
        "G28 ; home",
    ]


def _gcode_footer() -> List[str]:
    return [
        "G0 X0 Y0 ; return home",
        "M5 ; pen up",
        "M2 ; end",
    ]


def _write_gcode(lines: List[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _nearest_neighbor_sort(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Greedy nearest-neighbor sort to reduce travel distance."""
    remaining = list(points)
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]
        dists = [(math.hypot(p[0] - last[0], p[1] - last[1]), i) for i, p in enumerate(remaining)]
        _, idx = min(dists)
        ordered.append(remaining.pop(idx))
    return ordered


# ---------------------------------------------------------------------------
# High-level helpers used by mode_router / CLI
# ---------------------------------------------------------------------------

def export_render_data(
    render_type: str,
    data: Dict[str, Any],
    output_dir: str,
    formats: List[str],
    bed_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Export render data to requested formats.

    Parameters
    ----------
    render_type : str
        'isoradials' or 'scatter'.
    data : dict
        For isoradials: {'polylines': [[(x,y),...], ...]}
        For scatter: {'points': [(x,y),...], 'intensities': [...]}
    output_dir : str
        Directory for output files.
    formats : list of str
        Subset of ['svg', 'gcode'].
    bed_kwargs : dict, optional
        Overrides for PlotterBed constructor.

    Returns
    -------
    list of str
        Paths to exported files.
    """
    bed = PlotterBed(**(bed_kwargs or {}))
    exported = []
    base = Path(output_dir)

    if render_type == "isoradials":
        polylines = data.get("polylines", [])
        if "svg" in formats:
            p = str(base / "isoradials.svg")
            polylines_to_svg(polylines, p, bed=bed)
            exported.append(p)
        if "gcode" in formats:
            p = str(base / "isoradials.gcode")
            polylines_to_gcode(polylines, p, bed=bed)
            exported.append(p)

    elif render_type == "scatter":
        points = data.get("points", [])
        intensities = data.get("intensities")
        if "svg" in formats:
            p = str(base / "scatter.svg")
            scatter_to_svg(points, p, bed=bed, intensities=intensities)
            exported.append(p)
        if "gcode" in formats:
            p = str(base / "scatter.gcode")
            scatter_to_gcode(points, p, bed=bed)
            exported.append(p)

    return exported
