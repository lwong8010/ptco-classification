#!/usr/bin/env python3
"""
Build processed_pt_xanes_exafs.pkl from the reorganized Pt-only tree.

Reads xmu.dat (XANES) and chi.dat (EXAFS) directly from each site directory,
processes each curve with the Pt reference ranges, reads the integer
multiplicity from each site's multiplicity.json, and stores per-cluster
entries with multiplicity-weighted averages.

Per-cluster entry:
    {
        "total_atoms": int,
        "isomer_num": int,
        "pt_xanes":  [(site_str, mult_int, (e_new, mu_new)), ...],
        "pt_shifts": [(site_str, mult_int, shift_float), ...],
        "pt_avg_xanes": np.ndarray or None,   # multiplicity-weighted
        "pt_exafs":  [(site_str, mult_int, (k_new, chi_new)), ...],
        "pt_avg_exafs": np.ndarray or None,   # multiplicity-weighted
    }

A site with failed XANES (no xmu.dat) is excluded from pt_xanes / pt_shifts /
pt_avg_xanes but still contributes its EXAFS to pt_exafs / pt_avg_exafs, and
vice versa.
"""

import re
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d

PT_ROOT = Path("/media/nvme1/scientific_data_np_data/pt")
OUT = "processed_pt_xanes_exafs.pkl"

PT_L3_EDGE = 11564.0
XANES_PT = dict(target_min=11553.590, target_max=11629.425,
                target_norm_factor=3.000250, e_ref=PT_L3_EDGE)
EXAFS_PT = dict(target_norm_factor=0.262502)

SITE_RE = re.compile(r"^site_(\d+)$")


# ---------------------------------------------------------------------------
# curve processing (Pt-specialized; lifted from the reference implementation)
# ---------------------------------------------------------------------------
def _left_transform(E, E0, eps):
    """
    Map energies to a reciprocal coordinate that diverges at E0.

    Used to stretch the pre-edge/near-edge region so a PCHIP fit can
    represent the steep near-threshold behavior with better resolution.

    Args:
        E: Energy values to transform.
        E0: Reference energy at which the transform diverges.
        eps: Small offset added to the denominator to avoid division by zero.

    Returns:
        Transformed values, same shape as ``E``.
    """
    return 1.0 / (E0 - E + eps)


def process_xanes_curve(energy, mu, target_min, target_max,
                        target_norm_factor, e_ref, n_interp=256, n_fit=1024):
    """
    Resample, edge-align, and normalize a XANES (mu vs. energy) curve.

    The curve is sorted by energy, then reconstructed on a fine internal
    grid using a reciprocal-transformed PCHIP fit near the low-energy end
    and a cubic spline near the high-energy end (falling back to plain
    linear interpolation for very short inputs). The edge position is
    located from the maximum of the second derivative near ``e_ref`` and
    the curve is shifted so that edge aligns with ``e_ref``, then rescaled
    by ``target_norm_factor``.

    Args:
        energy: 1-D array of energy values (need not be sorted).
        mu: 1-D array of absorption values matching ``energy``.
        target_min: Lower bound of the output energy grid.
        target_max: Upper bound of the output energy grid.
        target_norm_factor: Divisor applied to the aligned curve for
            normalization.
        e_ref: Reference edge energy that the curve is aligned to.
        n_interp: Number of points in the returned energy grid.
        n_fit: Max number of points used at each end for the boundary fits.

    Returns:
        Tuple ``(energy_new, mu_aligned, shift)`` where ``energy_new`` is
        the output energy grid (length ``n_interp``), ``mu_aligned`` is the
        edge-aligned and normalized absorption on that grid, and ``shift``
        is the energy shift applied to align the edge to ``e_ref`` (``0.0``
        when the input is too short to fit and edge-align).
    """
    energy = np.asarray(energy, float)
    mu = np.asarray(mu, float)
    order = np.argsort(energy)
    energy, mu = energy[order], mu[order]

    if energy.size < 4:
        e_new = np.linspace(target_min, target_max, n_interp)
        mu_new = np.interp(e_new, energy, mu, left=0.0, right=0.0)
        return e_new, np.maximum(mu_new, 0.0), 0.0

    left_count = min(n_fit, max(3, len(energy) // 3))
    right_count = min(n_fit, max(3, len(energy) // 3))
    e_left, mu_left = energy[:left_count], mu[:left_count]
    e_right, mu_right = energy[-right_count:], mu[-right_count:]

    E0 = float(e_left[-1])
    left_span = max(1e-6, E0 - float(e_left[0]))
    eps = max(1e-6, left_span * 0.01)

    t_left = _left_transform(e_left, E0, eps)
    if t_left[0] > t_left[-1]:
        t_left = t_left[::-1]
        mu_left_fit = mu_left[::-1]
    else:
        mu_left_fit = mu_left.copy()

    t_with_zero = np.concatenate(([0.0], t_left))
    mu_with_zero = np.concatenate(([0.0], mu_left_fit))
    try:
        left_pchip = PchipInterpolator(t_with_zero, mu_with_zero, extrapolate=True)
    except Exception:
        left_pchip = interp1d(t_with_zero, mu_with_zero, kind="linear",
                              bounds_error=False,
                              fill_value=(0.0, mu_with_zero[-1]))
    spline_right = CubicSpline(e_right, mu_right, extrapolate=True)

    n_full = max(len(energy) + 2 * n_fit, 4 * n_interp)
    e_full = np.linspace(target_min, target_max, n_full)
    mu_full = np.empty_like(e_full)

    left_mask = e_full <= energy[0]
    right_mask = e_full >= energy[-1]
    mid_mask = (~left_mask) & (~right_mask)

    if np.any(left_mask):
        e_le = np.minimum(e_full[left_mask], E0)
        mu_full[left_mask] = left_pchip(_left_transform(e_le, E0, eps))
    if np.any(right_mask):
        mu_full[right_mask] = spline_right(e_full[right_mask])
    if np.any(mid_mask):
        mu_full[mid_mask] = interp1d(energy, mu, bounds_error=False,
                                     fill_value="extrapolate")(e_full[mid_mask])
    mu_full = np.maximum(mu_full, 0.0)

    energy_new = np.linspace(target_min, target_max, n_interp)
    mu_new = interp1d(e_full, mu_full, kind="linear", bounds_error=False,
                      fill_value="extrapolate")(energy_new)
    mu_new = np.maximum(mu_new, 0.0)

    # second-derivative edge alignment to e_ref
    mu_smooth = savgol_filter(mu_new, 11, 3)
    d2 = np.gradient(np.gradient(mu_smooth, energy_new), energy_new)
    window = 2.0
    mask = (energy_new >= e_ref - window) & (energy_new <= e_ref + window)
    E_edge = energy_new[mask][np.argmax(d2[mask])]
    shift = E_edge - e_ref

    mu_aligned = interp1d(e_full, mu_full, kind="linear", bounds_error=False,
                          fill_value="extrapolate")(energy_new - shift)
    mu_aligned = np.maximum(mu_aligned, 0.0) / target_norm_factor
    return energy_new, mu_aligned, shift


def process_exafs_curve(k, chi, target_min=0.0, target_max=20.0,
                        target_norm_factor=1.0, k_weight=None,
                        n_interp=256, window=None):
    """
    Resample and normalize an EXAFS (chi vs. k) curve onto a fixed k-grid.

    The curve is sorted by k, pinned to (0, 0) if it doesn't already start
    at k=0, and interpolated with a PCHIP spline (or plain linear
    interpolation for very short inputs). Optionally applies a k-weighting
    and a windowing function before normalization.

    Args:
        k: 1-D array of k values (need not be sorted).
        chi: 1-D array of chi values matching ``k``.
        target_min: Lower bound of the output k grid.
        target_max: Upper bound of the output k grid.
        target_norm_factor: Divisor applied to chi for normalization.
        k_weight: If given, multiply chi by ``k_new ** k_weight``.
        n_interp: Number of points in the returned k grid.
        window: Optional apodization window to apply, one of ``"hann"`` or
            ``"hamming"`` (case-insensitive); ``None`` for no window.

    Returns:
        Tuple ``(k_new, chi_new)``: the output k grid (length ``n_interp``)
        and the corresponding processed chi values.
    """
    k = np.asarray(k, float)
    chi = np.asarray(chi, float)
    order = np.argsort(k)
    k, chi = k[order], chi[order]

    if k.size < 4:
        k_new = np.linspace(target_min, target_max, n_interp)
        return k_new, np.interp(k_new, k, chi, left=0.0, right=0.0)

    if k[0] > 0:
        k = np.concatenate(([0.0], k))
        chi = np.concatenate(([0.0], chi))

    interp_func = PchipInterpolator(k, chi, extrapolate=True)
    k_new = np.linspace(target_min, target_max, n_interp)
    chi_new = interp_func(k_new)
    if k_weight is not None:
        chi_new = chi_new * k_new ** k_weight
    chi_new /= target_norm_factor
    if window is not None:
        if window.lower() == "hann":
            chi_new *= np.hanning(len(k_new))
        elif window.lower() == "hamming":
            chi_new *= np.hamming(len(k_new))
    return k_new, chi_new


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def read_xmu(path):
    """
    Read absolute energy and mu from a FEFF xmu.dat file.

    Args:
        path: Path to the xmu.dat file. Expected columns are
            omega(0) e(1) k(2) mu(3) ...

    Returns:
        Tuple ``(energy, mu)`` of 1-D arrays: column 1 (absolute energy,
        omega) and column 4 (mu).
    """
    data = np.loadtxt(path, comments="#")
    energy = data[:, 0]   # column 1: absolute energy (omega)
    mu = data[:, 3]       # column 4: mu
    return energy, mu


def read_chi(path):
    """
    Read k and chi from a FEFF chi.dat file.

    Args:
        path: Path to the chi.dat file. Expected columns are k(0) chi(1) ...

    Returns:
        Tuple ``(k, chi)`` of 1-D arrays taken from columns 0 and 1.
    """
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


def read_multiplicity(site_dir):
    """
    Read the integer multiplicity from a site directory's multiplicity.json.

    Args:
        site_dir: Path to the site directory containing
            ``multiplicity.json``.

    Returns:
        The site's multiplicity as an int.
    """
    mj = site_dir / "multiplicity.json"
    with open(mj) as f:
        return int(json.load(f)["multiplicity"])


def site_index(name):
    """
    Extract the integer index from a ``site_<N>`` directory name.

    Args:
        name: Directory name matching ``SITE_RE`` (e.g. ``"site_12"``).

    Returns:
        The site index as an int (e.g. ``12``).
    """
    return int(SITE_RE.match(name).group(1))


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------
def find_clusters(root):
    """
    Find cluster directories under ``root``.

    A cluster dir is identified by containing a ``site_multiplicity.json``
    file.

    Args:
        root: Root directory to search recursively.

    Yields:
        Paths to cluster directories.
    """
    for smj in root.rglob("site_multiplicity.json"):
        yield smj.parent


def weighted_average(items):
    """
    Compute a multiplicity-weighted average of curves.

    Args:
        items: List of ``(site, mult, (x, y))`` tuples. All ``y`` arrays
            must be the same shape; ``x`` is unused (curves are assumed to
            already share a common grid).

    Returns:
        The multiplicity-weighted average ``sum(mult * y) / sum(mult)`` as
        an ``np.ndarray``, or ``None`` if ``items`` is empty.
    """
    if not items:
        return None
    mults = np.array([m for (_s, m, _d) in items], float)
    stack = np.stack([d[1] for (_s, _m, d) in items], axis=0)  # (n_sites, n_points)
    return (mults[:, None] * stack).sum(axis=0) / mults.sum()


def build():
    """
    Build the processed Pt XANES/EXAFS dataset and pickle it to ``OUT``.

    Walks every cluster directory under ``PT_ROOT``, and for each site
    within a cluster reads its multiplicity plus any available XANES
    (xmu.dat) and EXAFS (chi.dat) data, processes each curve with the Pt
    reference ranges, and aggregates per-cluster entries (including
    multiplicity-weighted averages) into a dict keyed by the cluster's
    relative path. The result is written to ``OUT`` as a pickle.

    Raises:
        SystemExit: If no cluster directories are found under ``PT_ROOT``.
    """
    processed = {}
    clusters = sorted(find_clusters(PT_ROOT))
    if not clusters:
        raise SystemExit(f"No clusters found under {PT_ROOT}")

    for cluster in tqdm(clusters, desc="Clusters"):
        rel = cluster.relative_to(PT_ROOT)          # e.g. 40/1
        parts = rel.parts
        total_atoms = int(parts[0])
        isomer_num = int(parts[1])

        pt_xanes, pt_shifts, pt_exafs = [], [], []

        xdir = cluster / "XANES"
        edir = cluster / "EXAFS"

        # union of site names across XANES and EXAFS
        site_names = set()
        if xdir.is_dir():
            site_names |= {d.name for d in xdir.iterdir()
                           if d.is_dir() and SITE_RE.match(d.name)}
        if edir.is_dir():
            site_names |= {d.name for d in edir.iterdir()
                           if d.is_dir() and SITE_RE.match(d.name)}

        for site in sorted(site_names, key=site_index):
            # multiplicity: prefer XANES site dir, fall back to EXAFS
            mult = None
            for base in (xdir, edir):
                sd = base / site
                if (sd / "multiplicity.json").is_file():
                    mult = read_multiplicity(sd)
                    break
            if mult is None:
                print(f"[WARN] no multiplicity.json for {rel}/{site}; skipping")
                continue

            # XANES (skip this side if xmu.dat missing/failed)
            xmu = xdir / site / "xmu.dat"
            if xmu.is_file():
                try:
                    e, mu = read_xmu(xmu)
                    e_new, mu_new, shift = process_xanes_curve(e, mu, **XANES_PT)
                    pt_xanes.append((site, mult, (e_new, mu_new)))
                    pt_shifts.append((site, mult, shift))
                except Exception as ex:
                    print(f"[WARN] XANES {rel}/{site}: {ex}")

            # EXAFS (skip this side if chi.dat missing)
            chi = edir / site / "chi.dat"
            if chi.is_file():
                try:
                    k, c = read_chi(chi)
                    k_new, chi_new = process_exafs_curve(k, c, **EXAFS_PT)
                    pt_exafs.append((site, mult, (k_new, chi_new)))
                except Exception as ex:
                    print(f"[WARN] EXAFS {rel}/{site}: {ex}")

        processed[str(rel)] = {
            "total_atoms": total_atoms,
            "isomer_num": isomer_num,
            "pt_xanes": pt_xanes,
            "pt_shifts": pt_shifts,
            "pt_avg_xanes": weighted_average(pt_xanes),
            "pt_exafs": pt_exafs,
            "pt_avg_exafs": weighted_average(pt_exafs),
        }

    with open(OUT, "wb") as f:
        pickle.dump(processed, f)
    print(f"\nSaved {len(processed)} clusters -> {OUT}")


if __name__ == "__main__":
    build()
