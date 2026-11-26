#!/usr/bin/env python3
"""
Build processed_ptco_xanes_exafs.pkl from the reorganized PtCo tree.

Per site: read xmu.dat (XANES) and chi.dat (EXAFS) directly, read species and
multiplicity from multiplicity.json, route to Pt (L3) or Co (K) bucket with the
matching processing ranges. Per-cluster averages are multiplicity-weighted.

Per-cluster entry:
    {
        "total_atoms": int, "isomer_num": int,
        "motif": "random"|"core_shell"|"ordered",
        "csr": 15|50 (core_shell only) else None,
        "ratio": "ratio_1to1"|"ratio_1to3"|"ratio_3to1"|"ratio_none_iv",
        "pt_xanes":  [(site, mult, (e_new, mu_new)), ...],
        "co_xanes":  [...],
        "pt_shifts": [(site, mult, shift), ...],
        "co_shifts": [...],
        "pt_avg_xanes": ndarray|None,  # multiplicity-weighted
        "co_avg_xanes": ndarray|None,
        "pt_exafs":  [(site, mult, (k_new, chi_new)), ...],
        "co_exafs":  [...],
        "pt_avg_exafs": ndarray|None,
        "co_avg_exafs": ndarray|None,
    }

A site whose XANES failed (no xmu.dat) is excluded from *_xanes/*_shifts only;
its EXAFS still contributes, and vice versa.
"""

import re
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d

PTCO_ROOT = Path("/media/nvme1/nanoparticle_data/ptco")
OUT = "../ptco_nanoparticles.pkl"

PT_L3_EDGE = 11564.0
CO_K_EDGE = 7709.0

XANES_RANGES = {
    "Co": dict(target_min=7704.233, target_max=7781.897,
               target_norm_factor=1.612260, e_ref=CO_K_EDGE),
    "Pt": dict(target_min=11553.590, target_max=11629.425,
               target_norm_factor=3.000250, e_ref=PT_L3_EDGE),
}
EXAFS_RANGES = {
    "Co": dict(target_norm_factor=0.813328),
    "Pt": dict(target_norm_factor=0.262502),
}

SITE_RE = re.compile(r"^site_(\d+)$")


# ---------------------------------------------------------------------------
# curve processing (identical numerics to the reference implementations)
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
                              bounds_error=False, fill_value=(0.0, mu_with_zero[-1]))
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
                        n_interp=401, window=None):
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
# IO
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
    return data[:, 0], data[:, 3]   # energy (omega), mu


def read_chi(path):
    """
    Read k and chi from a FEFF chi.dat file.

    Args:
        path: Path to the chi.dat file. Expected columns are k(0) chi(1) ...

    Returns:
        Tuple ``(k, chi)`` of 1-D arrays taken from columns 0 and 1.
    """
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]   # k, chi


def read_mult_species(site_dir):
    """
    Read the multiplicity and species from a site directory's multiplicity.json.

    Args:
        site_dir: Path to the site directory possibly containing
            ``multiplicity.json``.

    Returns:
        Tuple ``(multiplicity, species)`` where ``multiplicity`` is an int
        and ``species`` is the stripped species string (e.g. ``"Pt"`` or
        ``"Co"``), or ``None`` if the site has no ``multiplicity.json``.
    """
    mj = site_dir / "multiplicity.json"
    if not mj.is_file():
        return None
    with open(mj) as f:
        d = json.load(f)
    return int(d["multiplicity"]), d["species"].strip()


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
# path metadata: motif / csr / ratio
# ---------------------------------------------------------------------------
def parse_meta(rel_parts):
    """
    Parse cluster metadata (motif/csr/ratio) from a path relative to PTCO_ROOT.

    Args:
        rel_parts: Path parts relative to ``PTCO_ROOT``, e.g.
            ``('40', '1', 'core_shell', '15', 'ratio_1to1')`` -> core_shell,
            csr=15; ``('40', '1', 'random', 'ratio_1to1')`` -> random,
            csr=None; ``('40', '1', 'ordered', 'ratio_none_iv')`` ->
            ordered, csr=None.

    Returns:
        Tuple ``(total_atoms, isomer_num, motif, csr, ratio)``, where
        ``csr`` is ``None`` unless ``motif == "core_shell"``.
    """
    total_atoms = int(rel_parts[0])
    isomer_num = int(rel_parts[1])
    motif = rel_parts[2]
    if motif == "core_shell":
        csr = int(rel_parts[3])
        ratio = rel_parts[4]
    else:
        csr = None
        ratio = rel_parts[3]
    return total_atoms, isomer_num, motif, csr, ratio


# ---------------------------------------------------------------------------
# build
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
    stack = np.stack([d[1] for (_s, _m, d) in items], axis=0)
    return (mults[:, None] * stack).sum(axis=0) / mults.sum()


def build():
    """
    Build the processed PtCo XANES/EXAFS dataset and pickle it to ``OUT``.

    Walks every cluster directory under ``PTCO_ROOT``, and for each site
    within a cluster reads its multiplicity and species plus any available
    XANES (xmu.dat) and EXAFS (chi.dat) data, processes each curve with the
    species-appropriate reference ranges, routes it into the Pt or Co
    bucket, and aggregates per-cluster entries (including
    multiplicity-weighted averages and motif/csr/ratio metadata) into a
    dict keyed by the cluster's relative path. The result is written to
    ``OUT`` as a pickle.

    Raises:
        SystemExit: If no cluster directories are found under
            ``PTCO_ROOT``.
    """
    processed = {}
    clusters = sorted(find_clusters(PTCO_ROOT))
    if not clusters:
        raise SystemExit(f"No clusters found under {PTCO_ROOT}")

    for cluster in tqdm(clusters, desc="Clusters"):
        rel = cluster.relative_to(PTCO_ROOT)
        total_atoms, isomer_num, motif, csr, ratio = parse_meta(rel.parts)

        buckets = {
            "Pt": {"xanes": [], "shifts": [], "exafs": []},
            "Co": {"xanes": [], "shifts": [], "exafs": []},
        }

        xdir = cluster / "XANES"
        edir = cluster / "EXAFS"

        site_names = set()
        for base in (xdir, edir):
            if base.is_dir():
                site_names |= {d.name for d in base.iterdir()
                               if d.is_dir() and SITE_RE.match(d.name)}

        for site in sorted(site_names, key=site_index):
            # multiplicity + species: prefer XANES dir, fall back to EXAFS
            ms = None
            for base in (xdir, edir):
                ms = read_mult_species(base / site)
                if ms is not None:
                    break
            if ms is None:
                print(f"[WARN] no multiplicity.json for {rel}/{site}; skipping")
                continue
            mult, species = ms
            if species not in XANES_RANGES:
                print(f"[WARN] unknown species {species} in {rel}/{site}; skipping")
                continue

            xr = XANES_RANGES[species]
            er = EXAFS_RANGES[species]

            xmu = xdir / site / "xmu.dat"
            if xmu.is_file():
                try:
                    e, mu = read_xmu(xmu)
                    e_new, mu_new, shift = process_xanes_curve(e, mu, **xr)
                    buckets[species]["xanes"].append((site, mult, (e_new, mu_new)))
                    buckets[species]["shifts"].append((site, mult, shift))
                except Exception as ex:
                    print(f"[WARN] XANES {rel}/{site}: {ex}")

            chi = edir / site / "chi.dat"
            if chi.is_file():
                try:
                    k, c = read_chi(chi)
                    k_new, chi_new = process_exafs_curve(k, c, **er)
                    buckets[species]["exafs"].append((site, mult, (k_new, chi_new)))
                except Exception as ex:
                    print(f"[WARN] EXAFS {rel}/{site}: {ex}")

        processed[str(rel)] = {
            "total_atoms": total_atoms,
            "isomer_num": isomer_num,
            "motif": motif,
            "csr": csr,
            "ratio": ratio,
            "pt_xanes": buckets["Pt"]["xanes"],
            "co_xanes": buckets["Co"]["xanes"],
            "pt_shifts": buckets["Pt"]["shifts"],
            "co_shifts": buckets["Co"]["shifts"],
            "pt_avg_xanes": weighted_average(buckets["Pt"]["xanes"]),
            "co_avg_xanes": weighted_average(buckets["Co"]["xanes"]),
            "pt_exafs": buckets["Pt"]["exafs"],
            "co_exafs": buckets["Co"]["exafs"],
            "pt_avg_exafs": weighted_average(buckets["Pt"]["exafs"]),
            "co_avg_exafs": weighted_average(buckets["Co"]["exafs"]),
        }

    with open(OUT, "wb") as f:
        pickle.dump(processed, f)
    print(f"\nSaved {len(processed)} clusters -> {OUT}")


if __name__ == "__main__":
    build()