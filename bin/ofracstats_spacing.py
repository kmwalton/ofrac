#!/usr/bin/env python
"""Compute P10 and produce raw fracture spacing data.

Parse one or more DFN files using ofracs, merge all resulting OFracGrid
objects into a single grid, run random scan-line sampling per zone, and
write a JSON summary.  If no zones are specified the whole merged domain
is used as the single zone.
"""

import argparse
import copy
import datetime
import getpass
import json
import os
import socket
import sys
from decimal import Decimal

import numpy as np
from ofrac import ofracs
from ofrac.ofrac.aabbox import AABBox


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _decimal_default(obj):
    """JSON serialiser that converts Decimal → float (and catches the rest)."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def load_and_merge(dfn_files: list[str]) -> ofracs.OFracGrid:
    """
    Parse every DFN_FILE with ofracs.parse(), merge all resulting OFracGrid
    objects into one and return it.
    """
    grids = []

    for path in dfn_files:
        # --- parse ---
        try:
            grid = ofracs.parse(path)
        except ofracs.NotValidOFracGridError as exc:
            print(f"error: {exc.message}", file=sys.stderr)
            sys.exit(1)

        grids.append(grid)

    if not grids:
        print("error: no DFN files were successfully parsed", file=sys.stderr)
        sys.exit(1)

    # merge: the first grid's .merge() accepts the rest as *args
    merged = grids[0].merge(*grids[1:])
    return merged


def zone_grid(merged: ofracs.OFracGrid, box: AABBox) -> ofracs.OFracGrid:
    """Return a copy of *merged* with its domain clipped to *box*.

    Uses setDomainSize(origin, size) which also excludes any gridlines and
    fractures that fall outside the new domain bounds.
    """
    grid = copy.deepcopy(merged)
    origin = [box.x0, box.y0, box.z0]
    size   = [box.x1 - box.x0, box.y1 - box.y0, box.z1 - box.z0]
    grid.setDomainSize(origin, size)
    return grid


def make_fx_array(grid: ofracs.OFracGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinate and orientation arrays for all fractures.

    Returns
    -------
    fx : ndarray, shape (N, 6), float64
        Columns: xfrom, xto, yfrom, yto, zfrom, zto  (OFrac.d order).
    orient : ndarray, shape (N,), int8
        Per-fracture orientation index as returned by
        ``OFrac.determineFracOrientation``:
          0 → fracture plane is yz  (perpendicular to x-axis)
          1 → fracture plane is xz  (perpendicular to y-axis)
          2 → fracture plane is xy  (perpendicular to z-axis)
    """
    rows = []
    orients = []
    for f in grid.iterFracs():
        rows.append(list(map(float, f.d)))
        orients.append(ofracs.OFrac.determineFracOrientation(f))

    fx     = np.array(rows,    dtype=np.float64)
    orient = np.array(orients, dtype=np.int8)
    return fx, orient


# ---------------------------------------------------------------------------
# scan-line sampling
# ---------------------------------------------------------------------------

# For each perpendicular-axis orientation o ∈ {0,1,2}:
#   scan_axis   = o          (the axis the scan line travels along)
#   plane_axes  = the other two axes, in (first, second) order
#   col indices in fx:
#     scan_axis  → columns 2*o, 2*o+1   (the constant coordinate of the plane)
#     plane_axes → columns for the remaining two axes
#
# Axis→column mapping in fx:  x→(0,1), y→(2,3), z→(4,5)

def _axis_columns(axis: int) -> tuple[int, int]:
    """Return the (from, to) column indices in fx for a given axis (0,1,2)."""
    return 2 * axis, 2 * axis + 1


class _OrientIndex:
    """Pre-computed binary-search structures for one fracture orientation.

    For a scan point q on plane-axis a, a fracture spans [from_a, to_a] and
    is hit when  from_a <= q <= to_a.

    Equivalently:
      - q >= from_a  →  fracture is in the *prefix* of the from-sorted order
                        up to (but not including) the first row where from > q
      - q <= to_a    →  fracture is in the *suffix* of the to-sorted order
                        starting from the first row where to >= q

    Intersecting those two sets of *original* row indices gives the hits.
    Because the two searchsorted calls each touch O(log M) entries
    (M = number of fractures of this orientation) instead of O(M), and the
    set intersection is over the (typically small) hit sets, this is
    substantially faster than a full mask scan when M is large.
    """

    __slots__ = (
        "fx_o",          # (M, 6) sub-array for this orientation
        "orig_indices",  # (M,) original row indices in the full fx array
        "perp_col",      # column index of the constant perp-plane coordinate
        # for each of the two plane axes: sorted values and their orig indices
        "from_sorted",   # list of 2 arrays: sorted from-values for a1, a2
        "from_order",    # list of 2 arrays: orig-row indices in from sort order
        "to_sorted",     # list of 2 arrays: sorted to-values for a1, a2
        "to_order",      # list of 2 arrays: orig-row indices in to sort order
    )

    def __init__(self, fx: np.ndarray, orient: np.ndarray, o: int):
        mask            = orient == o
        self.orig_indices = np.where(mask)[0]
        self.fx_o       = fx[mask]
        self.perp_col   = _axis_columns(o)[0]   # from==to for perp axis

        plane_axes = [a for a in range(3) if a != o]
        self.from_sorted = []
        self.from_order  = []
        self.to_sorted   = []
        self.to_order    = []

        for a in plane_axes:
            fc, tc = _axis_columns(a)

            f_ord = np.argsort(self.fx_o[:, fc], kind="stable")
            self.from_order.append(f_ord)
            self.from_sorted.append(self.fx_o[f_ord, fc])

            t_ord = np.argsort(self.fx_o[:, tc], kind="stable")
            self.to_order.append(t_ord)
            self.to_sorted.append(self.fx_o[t_ord, tc])

    def query(self, q1: float, q2: float) -> np.ndarray:
        """Return the perp-plane coordinates of fractures hit by point (q1,q2).

        For each plane axis i ∈ {0,1}:
          candidates_from_i = from_order[i][:searchsorted(from_sorted[i], q, side='right')]
          candidates_to_i   = to_order[i][searchsorted(to_sorted[i], q, side='left'):]
          hits_i            = intersect(candidates_from_i, candidates_to_i)

        Final hits = hits_0 ∩ hits_1  (row indices into fx_o).
        """
        hits = None
        for i, q in enumerate((q1, q2)):
            # all fractures whose from-value <= q  (q is within or past start)
            n_from = int(np.searchsorted(self.from_sorted[i], q, side="right"))
            cands_from = set(self.from_order[i][:n_from].tolist())

            # all fractures whose to-value >= q  (q has not passed the end)
            n_to = int(np.searchsorted(self.to_sorted[i], q, side="left"))
            cands_to = set(self.to_order[i][n_to:].tolist())

            axis_hits = cands_from & cands_to
            hits = axis_hits if hits is None else hits & axis_hits

        return sorted(float(self.fx_o[r, self.perp_col]) for r in hits)


def build_orient_indices(
    fx: np.ndarray,
    orient: np.ndarray,
) -> dict[int, _OrientIndex]:
    """Build an _OrientIndex for every orientation present in the network."""
    return {
        o: _OrientIndex(fx, orient, o)
        for o in range(3)
        if np.any(orient == o)
    }


def run_scan_lines(
    orient_idx: dict[int, _OrientIndex],
    grid: ofracs.OFracGrid,
    n: int,
    rng: np.random.Generator,
) -> dict:
    """Generate random scan lines and find fracture-plane intersections.

    For each orientation axis o present in ``orient_idx``:
      - Draw ``n`` random 2-D points uniformly from the domain face
        spanned by the two *other* axes.
      - For each point use binary search + set intersection to collect the
        perp-plane coordinates of every fracture whose in-plane extent
        contains the point.

    Parameters
    ----------
    orient_idx : mapping returned by ``build_orient_indices``.
    grid       : merged OFracGrid (used only for domain bounds).
    n          : number of scan lines per orientation axis.
    rng        : numpy random Generator.

    Returns
    -------
    dict keyed by axis name ('x', 'y', 'z'), each value a list of dicts::

        {
          "<pt_axis_1>": float,   # e.g. "y" and "z" for an x-direction line
          "<pt_axis_2>": float,
          "domain_length": float, # extent of the scan axis across the domain
          "intersections": [float, ...]  # sorted plane coords hit by the line
        }
    """
    origin = [float(v) for v in grid.getDomainStart()]
    end    = [float(v) for v in grid.getDomainEnd()]
    axis_name = ('x', 'y', 'z')
    results: dict[str, list[dict]] = {}

    for o, idx in orient_idx.items():
        plane_axes = [a for a in range(3) if a != o]
        a1, a2 = plane_axes

        domain_length = end[o] - origin[o]

        p1 = rng.uniform(origin[a1], end[a1], size=n)
        p2 = rng.uniform(origin[a2], end[a2], size=n)

        scan_results = []
        for i in range(n):
            scan_results.append({
                axis_name[a1]: float(p1[i]),
                axis_name[a2]: float(p2[i]),
                "domain_length": domain_length,
                "intersections": idx.query(p1[i], p2[i]),
            })

        results[axis_name[o]] = scan_results

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_run_metadata(args: argparse.Namespace) -> dict:
    """Return a dict of provenance information about this invocation."""
    return {
        "datetime":        datetime.datetime.now().astimezone().isoformat(),
        "script":          os.path.abspath(sys.argv[0]),
        "invocation_dir":  os.getcwd(),
        "arguments":       sys.argv[1:],
        "user":            getpass.getuser(),
        "machine":         socket.gethostname(),
    }


def summarise_scan_results(
    scan_results: dict[str, list[dict]],
) -> dict[str, dict]:
    """Compute per-axis P10 statistics from the raw scan-line results.

    For each axis key in scan_results the returned dict contains:
      n_scan_lines       – number of scan lines run
      total_intersections – total fracture-plane crossings across all lines
      total_length       – sum of domain_length over all lines (= n × L)
      p10                – total_intersections / total_length
      mean_spacing       – 1 / p10  (or None when p10 == 0)
    """
    summary = {}
    for axis, lines in scan_results.items():
        n          = len(lines)
        total_hits = sum(len(line["intersections"]) for line in lines)
        total_len  = sum(line["domain_length"]      for line in lines)
        p10        = total_hits / total_len if total_len > 0 else 0.0
        summary[axis] = {
            "n_scan_lines":        n,
            "total_intersections": total_hits,
            "total_length":        total_len,
            "p10":                 p10,
            "mean_spacing":        (1.0 / p10) if p10 > 0 else None,
        }
    return summary


def _print_initial_message(
    n_files: int,
    merged: ofracs.OFracGrid,
) -> None:
    """Print the brief opening banner to stderr."""
    origin = [float(v) for v in merged.getDomainStart()]
    end    = [float(v) for v in merged.getDomainEnd()]
    size   = [e - o for o, e in zip(origin, end)]
    print(
        f"Merged {n_files} DFN file(s) → "
        f"{merged.getFxCount()} fractures, "
        f"domain {size[0]:.3g} × {size[1]:.3g} × {size[2]:.3g} "
        f"(origin {origin[0]:.3g},{origin[1]:.3g},{origin[2]:.3g})",
        file=sys.stderr,
    )


def _print_scan_summary(summary: dict[str, dict]) -> None:
    """Print the scan-line summary table to stderr."""
    axis_label = {"x": "x (yz-fracs)", "y": "y (xz-fracs)", "z": "z (xy-fracs)"}
    header = f"{'Axis':<14} {'N lines':>8} {'Crossings':>10} {'Tot. length':>12} {'P10':>10} {'Spacing':>10}"
    print("\nScan-line summary:", file=sys.stderr)
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for axis, s in summary.items():
        spacing_str = f"{s['mean_spacing']:.4g}" if s["mean_spacing"] is not None else "—"
        print(
            f"{axis_label.get(axis, axis):<14} "
            f"{s['n_scan_lines']:>8d} "
            f"{s['total_intersections']:>10d} "
            f"{s['total_length']:>12.4g} "
            f"{s['p10']:>10.4g} "
            f"{spacing_str:>10}",
            file=sys.stderr,
        )


def _print_zone_summary(
    zone_label: str,
    grid: ofracs.OFracGrid,
    summary: dict[str, dict],
) -> None:
    """Print the per-zone banner and scan-line summary table to stderr."""
    origin = [float(v) for v in grid.getDomainStart()]
    end    = [float(v) for v in grid.getDomainEnd()]
    size   = [e - o for o, e in zip(origin, end)]
    print(
        f"\n{zone_label}: {grid.getFxCount()} fractures, "
        f"domain {size[0]:.3g} × {size[1]:.3g} × {size[2]:.3g} "
        f"(origin {origin[0]:.3g},{origin[1]:.3g},{origin[2]:.3g})",
        file=sys.stderr,
    )
    _print_scan_summary(summary)


def open_json_output(json_file: str) -> tuple:
    """
    Return a (file, indent) pair for JSON output.

      '-'   → (stdout, 2)    pretty-printed; progress output suppressed
      FILE  → (opened file, None)  compact, no indentation
    """
    if json_file == "-":
        return sys.stdout, 2
    try:
        return open(json_file, "w", encoding="utf-8"), None
    except OSError as exc:
        print(f"error: cannot open JSON output file: {exc}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one or more DFN files using ofracs, merge them into a "
            "single OFracGrid, run random scan-line sampling, and write "
            "JSON output.  Use '-' as JSON_FILE to write pretty-printed JSON "
            "to stdout (all other output is suppressed in that case); a file "
            "path writes compact JSON directly to disk."
        ),
    )

    parser.add_argument(
        "-n", "--n-scan-lines",
        metavar="INT",
        type=int,
        default=10,
        help=(
            "Number of random scan lines to generate per orientation axis "
            "(default: 10; minimum: 1)."
        ),
    )

    parser.add_argument(
        "-z", "--zone",
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        nargs=6,
        type=float,
        action="append",
        dest="zones",
        default=None,
        help=(
            "Define an analysis zone as six floats in blockspec order: "
            "x0 x1 y0 y1 z0 z1.  May be repeated for multiple zones.  "
            "If omitted, the full merged domain is used as a single zone."
        ),
    )

    parser.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=None,
        help="Random seed for reproducible scan-line point generation.",
    )

    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help=(
            "Destination for JSON output.  Use '-' to write pretty-printed "
            "JSON to stdout (suppresses all other output)."
        ),
    )

    parser.add_argument(
        "dfn_files",
        metavar="DFN_FILE",
        nargs="+",
        help="One or more DFN files to parse and merge.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.n_scan_lines < 1:
        parser.error("--n-scan-lines must be >= 1")

    to_stdout = (args.json_file == "-")

    metadata = collect_run_metadata(args)

    merged = load_and_merge(args.dfn_files)
    if not to_stdout:
        _print_initial_message(len(args.dfn_files), merged)

    # --- build zone list ---
    # Each zone is an (label, AABBox) pair.  When no --zone flags are given,
    # fall back to the full merged domain as "zone_0".
    if args.zones:
        zones = [
            (f"zone_{i}", AABBox(v[0], v[2], v[4], v[1], v[3], v[5]))
            for i, v in enumerate(args.zones)
        ]
    else:
        s = merged.getDomainStart()
        e = merged.getDomainEnd()
        zones = [("zone_0", AABBox(float(s[0]), float(s[1]), float(s[2]),
                                   float(e[0]), float(e[1]), float(e[2])))]

    # --- run analysis per zone ---
    rng = np.random.default_rng(args.seed)
    zone_data: dict[str, dict] = {}

    for label, box in zones:
        grid = zone_grid(merged, box)

        fx, orient = make_fx_array(grid)
        orient_idx = build_orient_indices(fx, orient)

        scan_results = run_scan_lines(orient_idx, grid, args.n_scan_lines, rng)
        scan_summary = summarise_scan_results(scan_results)

        if not to_stdout:
            _print_zone_summary(label, grid, scan_summary)

        zone_data[label] = {
            "zone_bbox":    list(box),       # [x0, y0, z0, x1, y1, z1]
            "scan_lines":   scan_results,
            "scan_summary": scan_summary,
        }

    # --- JSON output ---
    json_out, indent = open_json_output(args.json_file)
    data = {
        "metadata": metadata,
        "zones":    zone_data,
    }
    json.dump(data, json_out, indent=indent, default=_decimal_default)
    print(file=json_out)
    if json_out is not sys.stdout:
        json_out.close()


if __name__ == "__main__":
    main()
