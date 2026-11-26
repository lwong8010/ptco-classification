import os
import random
import numpy as np
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree 

SOURCE_BASE = "/media/hdd1/pt_np_nomad_dataset/JVkvSYsCQEmSx18lbb8Ehw/Pt"  # Source base path for the downloaded original dataset
DEST_BASE = "/media/nvme1/nanoparticle_data/code/test"  # Destination base path for the PtCo structures

def generate_random_ptco_structures(source_base, dest_base):
    """
    Generate PtCo structures by randomly substituting Pt atoms with Co.

    Walks the source dataset's numbered folders (40-55), reads each
    sub-structure's CONTCAR (or vasprun.xml), and for each of the 1:1, 1:3,
    and 3:1 Pt:Co ratios randomly replaces a fraction of the Pt atoms with
    Co. Each resulting structure is written as an XYZ file under a mirrored
    directory layout in dest_base.

    Args:
      source_base: Base directory of the original Pt-only dataset.
      dest_base: Base directory to write the generated PtCo structures to.

    Returns:
      None
    """
    ratios = {"1to1": 0.5, "1to3": 0.75, "3to1": 0.25}  # fraction of Co atoms
    folder_range = range(40, 56)

    # === Processing ===
    for i in folder_range:
        folder = Path(source_base) / str(i)
        if not folder.exists():
            continue

        for sub in sorted(folder.iterdir()):
            if not sub.is_dir():
                continue

            # Try to read structure from CONTCAR or vasprun.xml
            struct_path = sub / "CONTCAR"
            if not struct_path.exists():
                struct_path = sub / "vasprun.xml"
                if not struct_path.exists():
                    print(f"Skipping {sub} (no CONTCAR or vasprun.xml)")
                    continue

            try:
                structure = Structure.from_file(struct_path)
            except Exception as e:
                print(f"Error reading {struct_path}: {e}")
                continue

            pt_indices = [i for i, s in enumerate(structure.species) if s.symbol == "Pt"]
            num_pt = len(pt_indices)
            if num_pt == 0:
                print(f"No Pt atoms found in {sub}, skipping.")
                continue

            # Generate random Co substitutions for each ratio
            for ratio_name, frac_co in ratios.items():
                n_replace = max(1, int(round(num_pt * frac_co)))
                replace_indices = random.sample(pt_indices, n_replace)

                new_struct = structure.copy()
                for idx in replace_indices:
                    new_struct[idx] = "Co"

                # Prepare output path (new structure directory layout)
                out_dir = Path(dest_base) / str(i) / sub.name / "random" / f"ratio_{ratio_name}"
                out_dir.mkdir(parents=True, exist_ok=True)

                out_file = out_dir / "structure_initial.xyz"

                # Convert to ASE and write
                atoms = AseAtomsAdaptor.get_atoms(new_struct)
                write(out_file, atoms)

    print("Done generating random PtCo structures.")
    
def cluster_radius_metrics(contcar_path):
    """
    Compute radius of gyration and max radial distance for a Pt nanoparticle.

    Args:
      contcar_path: Path to a CONTCAR (or other pymatgen-readable) structure
        file for the nanoparticle.

    Returns:
      A dict with keys:
        num_atoms: Number of atoms in the structure.
        radius_of_gyration: Root-mean-square distance of atoms from the
          center of geometry.
        max_radial_distance: Maximum distance of any atom from the center
          of geometry.
        center_of_geometry: The center of geometry as a list of 3 floats.
    """
    structure = Structure.from_file(contcar_path)

    # Cartesian coordinates (ignore lattice)
    coords = np.array([site.coords for site in structure])

    # Center of geometry
    cog = coords.mean(axis=0)

    # Displacements from center
    disp = coords - cog
    dists = np.linalg.norm(disp, axis=1)

    # Metrics
    r_max = np.max(dists)
    r_g = np.sqrt(np.mean(dists**2))

    return {
        "num_atoms": len(structure),
        "radius_of_gyration": r_g,
        "max_radial_distance": r_max,
        "center_of_geometry": cog.tolist(),
    }
    
def generate_core_shell_structures(source_base, dest_base, shell_percent=50):
    """
    Generate core-shell Co-core / Pt-Co shell structures.

    - Input structures are pure Pt
    - Entire structure is first converted to Co
    - Pt is substituted back ONLY in the shell
    - Ratios define Pt fraction in the shell

    Walks the source dataset's numbered folders (40-55), reads each
    sub-structure's CONTCAR, determines the core/shell boundary from
    cluster_radius_metrics using shell_percent, and for each of the 1:1,
    1:3, and 3:1 Pt:Co shell ratios writes a structure (all Co in the core,
    a Pt/Co mix in the shell) as an XYZ file under a mirrored directory
    layout in dest_base.

    Args:
      source_base: Base directory of the original Pt-only dataset.
      dest_base: Base directory to write the generated core-shell
        structures to.
      shell_percent: Percentage (0-100) of the particle radius, measured
        inward from the surface, that is treated as shell.

    Returns:
      None
    """

    # Pt fraction in the SHELL
    ratios = {
        "1to1": 0.5,     # Pt fraction = 1 / (1+1)
        "1to3": 0.25,    # Pt fraction = 1 / (1+3)
        "3to1": 0.75,    # Pt fraction = 3 / (3+1)
    }

    source_base = Path(source_base)
    dest_base = Path(dest_base)

    for i in range(40, 56):
        parent_dir = source_base / str(i)
        if not parent_dir.is_dir():
            continue

        for sub in parent_dir.iterdir():
            contcar_path = sub / "CONTCAR"
            if not contcar_path.is_file():
                continue

            metrics = cluster_radius_metrics(contcar_path)
            structure = Structure.from_file(contcar_path)

            coords = np.array([site.coords for site in structure])
            cog = np.array(metrics["center_of_geometry"])
            dists = np.linalg.norm(coords - cog, axis=1)

            # Core / shell boundary
            r_max = metrics["max_radial_distance"]
            shell_cutoff = (1 - shell_percent / 100.0) * r_max

            shell_indices = np.where(dists >= shell_cutoff)[0]
            core_indices = np.where(dists < shell_cutoff)[0]

            # Ensure outermost atom is shell
            max_atom = np.argmax(dists)
            if max_atom not in shell_indices:
                shell_indices = np.append(shell_indices, max_atom)
                core_indices = core_indices[core_indices != max_atom]

            n_atoms = len(structure)

            for ratio_name, pt_frac in ratios.items():
                # Step 1: start from ALL Co
                species = ["Co"] * n_atoms

                # Step 2: substitute Pt into shell only
                n_shell = len(shell_indices)
                n_pt_shell = int(np.round(pt_frac * n_shell))

                pt_indices = random.sample(list(shell_indices), n_pt_shell)

                for idx in pt_indices:
                    species[idx] = "Pt"

                new_structure = Structure(
                    lattice=structure.lattice,
                    species=species,
                    coords=coords,
                    coords_are_cartesian=True,
                )

                out_dir = (
                    dest_base
                    / str(i)
                    / sub.name
                    / "core_shell"
                    / str(shell_percent)
                    / f"ratio_{ratio_name}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                xyz_path = out_dir / "structure_initial.xyz"
                atoms = AseAtomsAdaptor.get_atoms(new_structure)
                write(xyz_path, atoms)

    print(f"Done generating PtCo core-shell structures ({shell_percent}% shell).")
    
def format_lattice(mat):
    """Format a 3x3 lattice matrix as a flat, space-separated string.

    Args:
      mat: 3x3 lattice matrix (numpy array or list-of-lists).

    Returns:
      String "a b c d e f g h i" of the 9 matrix entries (row-major) at
      full double precision.
    """
    flat = [float(x) for row in mat for x in row]
    return " ".join(f"{val:.14g}" for val in flat)

def write_structure_xyz_from_structure(structure: Structure, out_path: str):
    """Write a structure to an extended-XYZ file with a VASP-style header.

    Output format:
      Line 1: atom count
      Line 2: Lattice="ax ay az bx by bz cx cy cz" Properties=species:S:1:pos:R:3 pbc="T T T"
      Remaining lines: "Element  x  y  z" (cartesian coordinates in Angstroms)

    Args:
      structure: pymatgen Structure to write out.
      out_path: File path to write the XYZ file to.

    Returns:
      None
    """
    # number of sites
    n = len(structure.sites)

    # lattice matrix (3x3). Use structure.lattice.matrix
    lattice_mat = np.array(structure.lattice.matrix)  # shape (3,3)

    lattice_str = format_lattice(lattice_mat)

    # header line with lattice and properties
    header_line = f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3 pbc="T T T"'

    # gather element and cart coords
    # use site.specie.symbol for element label, and site.coords for cart coords
    lines = []
    for site in structure.sites:
        elem = site.specie.symbol if hasattr(site, "specie") else site.species_string
        x, y, z = site.coords  # cart coords in Å
        # match spacing/precision similar to example: use 8-11 significant digits
        lines.append(f"{elem:2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")

    # write file
    with open(out_path, "w") as fh:
        fh.write(f"{n}\n")
        fh.write(f"{header_line}\n")
        for l in lines:
            fh.write(l + "\n")

N_LAYERS = 5
NEIGHBOR_SMOOTHING = True
SMOOTH_ITERATIONS = 2
NEIGHBOR_CUTOFF = 2.3
SURFACE_ELEMENT = "Co"
TARGET_PT_FRACTION = 0.25

def make_ordered_structure(
    contcar_path,
    out_path,
    N_layers,
    neighbor_smoothing,
    smooth_iterations,
    neighbor_cutoff,
    surface_element,
    target_pt_fraction,
    reinject_per_co_layer,
    ):
    """
    Generate a layered Pt/Co structure from a single CONTCAR.

    Projects atom coordinates onto the first principal component (PCA) of
    the particle, bins the projection into N_layers slabs, and assigns
    alternating Pt/Co species by layer parity so that surface_element ends
    up on the outermost layer. Optionally smooths the layer assignment by
    neighbor majority vote, then adjusts Co-layer compositions (via
    reinject_per_co_layer and/or target_pt_fraction) by converting the
    Co atoms with the most Pt neighbors to Pt (or vice versa), so Pt
    reintroduction favors interface sites. The result is written as an XYZ
    file named "structure_initial.xyz" in out_path.

    Args:
      contcar_path: Path to the source CONTCAR structure file.
      out_path: Directory to write "structure_initial.xyz" into.
      N_layers: Number of layers to bin atoms into along the principal axis.
      neighbor_smoothing: Whether to smooth layer assignment using
        neighbor-majority voting.
      smooth_iterations: Number of smoothing passes to run when
        neighbor_smoothing is True.
      neighbor_cutoff: Distance cutoff (in Angstroms) defining neighbors for
        smoothing and for ranking reinjection/removal candidates.
      surface_element: Element ("Pt" or "Co") that should occupy the
        outermost layer.
      target_pt_fraction: Desired Pt fraction within each Co layer, or None
        to skip this global top-up/trim step.
      reinject_per_co_layer: Optional dict mapping layer index to a Pt
        fraction to convert within that Co layer, applied before the
        target_pt_fraction top-up.

    Returns:
      None. Returns early (without writing a file) if contcar_path cannot
      be read or the structure is empty.
    """

    try:
        s = Structure.from_file(contcar_path)
    except Exception as e:
        print(f"Failed to read {contcar_path}: {e}")
        return

    coords_cart = s.cart_coords.copy()
    N = len(coords_cart)
    if N == 0:
        print(f"Skipped empty structure: {contcar_path}")
        return

    # --- PCA + binning ---
    com = coords_cart.mean(axis=0)
    coords_centered = coords_cart - com
    pca = PCA(n_components=3)
    pca.fit(coords_centered)
    pc1 = pca.components_[0]
    proj = coords_centered.dot(pc1)
    pmin, pmax = proj.min(), proj.max()
    edges = np.linspace(pmin, pmax, N_layers + 1)
    layer_idx = np.digitize(proj, edges) - 1
    layer_idx = np.clip(layer_idx, 0, N_layers - 1)

    # --- Layer assignment (alternating) ---
    outer_layer = N_layers - 1
    elem0, elem1 = "Pt", "Co"
    if surface_element == elem1:
        parity_map = {0: elem1, 1: elem0} if (outer_layer % 2) == 0 else {0: elem0, 1: elem1}
    else:
        parity_map = {0: elem0, 1: elem1} if (outer_layer % 2) == 0 else {0: elem1, 1: elem0}
    species = np.array([parity_map[int(li % 2)] for li in layer_idx])

    # --- Optional initial neighbor smoothing ---
    if neighbor_smoothing:
        tree = cKDTree(coords_cart)
        for _ in range(smooth_iterations):
            old_layer_idx = layer_idx.copy()
            new_species = species.copy()
            new_layer_idx = layer_idx.copy()
            for i in range(N):
                neighbors = tree.query_ball_point(coords_cart[i], neighbor_cutoff)
                if not neighbors:
                    continue
                neigh_layers = old_layer_idx[neighbors]
                vals, counts = np.unique(neigh_layers, return_counts=True)
                maj_layer = vals[np.argmax(counts)]
                new_species[i] = parity_map[int(maj_layer % 2)]
                new_layer_idx[i] = maj_layer
            species = new_species
            layer_idx = new_layer_idx
            
    # --- Ratio control: reinject Pt into Co layers ---
    # Helper: indices grouped by layer and species
    indices_by_layer = {li: np.where(layer_idx == li)[0] for li in range(N_layers)}
    co_layers = [li for li in range(N_layers) if parity_map[int(li % 2)] == "Co"]
    pt_layers = [li for li in range(N_layers) if parity_map[int(li % 2)] == "Pt"]

    # Build neighbor tree for selecting reinjection candidates by Pt-neighbor count
    tree = cKDTree(coords_cart)
    
    def count_Pt_neighbors(idx):
        neigh = tree.query_ball_point(coords_cart[idx], neighbor_cutoff)
        # exclude self if it is Pt we still count neighbors
        return sum(1 for j in neigh if species[j] == "Pt")

    # Apply per-layer reinjection first (if provided)
    if reinject_per_co_layer:
        for li, frac in reinject_per_co_layer.items():
            if li not in co_layers:
                # skip if this layer is not a Co layer
                continue
            layer_inds = indices_by_layer.get(li, np.array([], dtype=int))
            # among these, choose only atoms currently Co
            co_inds_in_layer = [i for i in layer_inds if species[i] == "Co"]
            if len(co_inds_in_layer) == 0:
                continue
            n_to_convert = int(round(len(co_inds_in_layer) * float(frac)))
            if n_to_convert <= 0:
                continue
            # rank by Pt-neighbor count so conversions occur near interfaces
            ranked = sorted(co_inds_in_layer, key=lambda ii: count_Pt_neighbors(ii), reverse=True)
            pick = ranked[:n_to_convert]
            for idx in pick:
                species[idx] = "Pt"
                
    # If global target specified, top-up conversions per individual Co layer
    if target_pt_fraction is not None:
        for li in co_layers:
            layer_inds = indices_by_layer.get(li, np.array([], dtype=int))
            n_layer = len(layer_inds)
            if n_layer == 0:
                continue

            desired_Pt_in_layer = int(round(n_layer * float(target_pt_fraction)))
            current_Pt_in_layer = sum(1 for i in layer_inds if species[i] == "Pt")
            need_add = desired_Pt_in_layer - current_Pt_in_layer

            if need_add > 0:
                candidate_inds = [i for i in layer_inds if species[i] == "Co"]
                candidate_ranking = sorted(candidate_inds,
                                           key=lambda ii: count_Pt_neighbors(ii),
                                           reverse=True)
                to_convert = candidate_ranking[:need_add]
                for idx in to_convert:
                    species[idx] = "Pt"
            elif need_add < 0:
                # too much Pt already in this layer — flip some back
                candidates = [i for i in layer_inds if species[i] == "Pt"]
                candidates_sorted = sorted(candidates,
                                           key=lambda ii: count_Pt_neighbors(ii))  # isolated Pt first
                for idx in candidates_sorted[:-need_add]:
                    species[idx] = "Co"

    # --- Write new POSCAR ---
    new_structure = Structure(s.lattice, [], [])
    for i, site in enumerate(s.sites):
        new_structure.append(species[i], site.coords, coords_are_cartesian=True)

    out_file = os.path.join(out_path, "structure_initial.xyz")
    write_structure_xyz_from_structure(new_structure, out_file)

def generate_ordered_ptco_structures(
    source_base, 
    dest_base, 
    ratio,
    N_layers=None,
    neighbor_smoothing=None,
    smooth_iterations=None,
    neighbor_cutoff=None,
    surface_element=None,
    target_pt_fraction=None,
    reinject_per_co_layer=None
    ):
    """
    Generate ordered layered Pt/Co structures from the original Pt dataset.

    Walks the source dataset's numbered folders (40-55) and, for each
    sub-structure's CONTCAR, calls make_ordered_structure to produce a
    layered Pt/Co structure, writing it under a mirrored directory layout
    in dest_base. For the built-in ratios ("1to1", "1to3", "3to1", "none"),
    the N_layers/neighbor_smoothing/smooth_iterations/neighbor_cutoff/
    surface_element/target_pt_fraction parameters are overridden with
    preset values; for "custom" they must be supplied by the caller.

    Args:
      source_base: Base directory of the original Pt-only dataset.
      dest_base: Base directory to write the generated ordered structures
        to.
      ratio: One of "1to1", "1to3", "3to1", "none", or "custom", selecting
        a preset (or custom) target Pt:Co ratio for the ordered structures.
      N_layers: Number of layers to bin atoms into (ignored except for
        ratio="custom", where it is passed through to
        make_ordered_structure).
      neighbor_smoothing: Whether to smooth layer assignment (ignored
        except for ratio="custom").
      smooth_iterations: Number of smoothing passes (ignored except for
        ratio="custom").
      neighbor_cutoff: Neighbor distance cutoff in Angstroms (ignored
        except for ratio="custom").
      surface_element: Element to place on the surface layer (ignored
        except for ratio="custom", where it is required).
      target_pt_fraction: Target Pt fraction within Co layers (ignored
        except for ratio="custom").
      reinject_per_co_layer: Optional dict mapping layer index to a Pt
        fraction to reinject into that Co layer, passed through to
        make_ordered_structure.

    Returns:
      None

    Raises:
      ValueError: If ratio is "custom" and surface_element is not
        specified.
    """
    source_base = Path(source_base)
    dest_base = Path(dest_base)
    
    if ratio == "1to1":
        N_layers = N_LAYERS
        neighbor_smoothing = NEIGHBOR_SMOOTHING
        smooth_iterations = SMOOTH_ITERATIONS
        neighbor_cutoff = NEIGHBOR_CUTOFF
        surface_element = "Co"
        target_pt_fraction = 0.5
    elif ratio == "1to3":
        N_layers = N_LAYERS
        neighbor_smoothing = NEIGHBOR_SMOOTHING
        smooth_iterations = SMOOTH_ITERATIONS
        neighbor_cutoff = NEIGHBOR_CUTOFF
        surface_element = "Co"
        target_pt_fraction = 0.25
    elif ratio == "3to1":
        N_layers = N_LAYERS
        neighbor_smoothing = NEIGHBOR_SMOOTHING
        smooth_iterations = SMOOTH_ITERATIONS
        neighbor_cutoff = NEIGHBOR_CUTOFF
        surface_element = "Co"
        target_pt_fraction = 0.75
    elif ratio == "none":
        N_layers = N_LAYERS
        neighbor_smoothing = NEIGHBOR_SMOOTHING
        smooth_iterations = SMOOTH_ITERATIONS
        neighbor_cutoff = NEIGHBOR_CUTOFF
        surface_element = "Pt"
        target_pt_fraction = None
    elif ratio == "custom":
        if surface_element is None:
            raise ValueError("For 'custom' ratio, all arguments must be specified.")

    for i in range(40, 56):
        parent_dir = source_base / str(i)
        if not parent_dir.is_dir():
            continue

        for sub in parent_dir.iterdir():
            contcar_path = sub / "CONTCAR"
            if not contcar_path.is_file():
                continue

            out_dir = dest_base / str(i) / sub.name / "ordered" / f"ratio_{ratio}"
            out_dir.mkdir(parents=True, exist_ok=True)

            make_ordered_structure(contcar_path, out_dir, N_layers, neighbor_smoothing, smooth_iterations, neighbor_cutoff, surface_element, target_pt_fraction, reinject_per_co_layer)
            
    print(f"Done generating PtCo ratio_{ratio} ordered structures.")
    
def display_xyz_number_info(dest_path):
    """
    Display the count of generated XYZ files, grouped by top-level folder.

    Recursively finds all "*.xyz" files under dest_path, groups them by
    their immediate child folder of dest_path (the numbered dataset
    folders), and prints the total count and per-folder counts.

    Args:
      dest_path: Base directory containing the generated PtCo structures.

    Returns:
      None
    """
    from collections import Counter
    dest_path = Path(dest_path)
    xyz_files = list(dest_path.rglob("*.xyz"))

    print(f"Total .xyz files found: {len(xyz_files)}\n")

    counts = Counter(p.parts[len(dest_path.parts)] for p in xyz_files)
    for folder, count in sorted(counts.items(), key=lambda x: int(x[0])):
        print(f"{folder}: {count} files")
    
def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Script for creating the PtCo dataset from the original QCD dataset using algorithms for random, core-shell, and ordered structures."
    )
    parser.add_argument(
        "--source", type=str, default=SOURCE_BASE,
        help="Base directory of the source dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--dest", type=str, default=DEST_BASE,
        help="Base directory for the PtCo dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--ratio", type=str, choices=["1to1", "1to3", "3to1", "none", "custom"],
        help="Pt:Co ratio to generate (choices: 1to1, 1to3, 3to1, none, custom)"
    )
    parser.add_argument(
        "--N_layers", type=int, default=N_LAYERS,
        help="Number of layers for ordered structures (default: %(default)s)"
    )
    parser.add_argument(
        "--neighbor_smoothing", action='store_true',
        help="Enable neighbor smoothing for ordered structures"
    )
    parser.add_argument(
        "--smooth_iterations", type=int, default=SMOOTH_ITERATIONS,
        help="Number of smoothing iterations (default: %(default)s)"
    )
    parser.add_argument(
        "--neighbor_cutoff", type=float, default=NEIGHBOR_CUTOFF,
        help="Neighbor cutoff distance in Å (default: %(default)s)"
    )
    parser.add_argument(
        "--surface_element", type=str, choices=["Pt", "Co"], default=None,
        help="Element to be on the surface layer (Pt or Co)"
    )
    parser.add_argument(
        "--target_pt_fraction", type=float,
        help="Target Pt fraction in Co layers (0-1) for ordered structures"
    )
    parser.add_argument(
        "--reinject_per_co_layer", type=dict, default=None,
        help="Comma-separated dict of layer_index:pt_fraction for reinjection into Co layers (e.g., {1: 0.2, 3: 0.1})"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    generate_random_ptco_structures(
        source_base=args.source,
        dest_base=args.dest
    )
    generate_core_shell_structures(
        source_base=args.source,
        dest_base=args.dest,
        shell_percent=15
    )
    generate_core_shell_structures(
        source_base=args.source,
        dest_base=args.dest,
        shell_percent=50
    )
    generate_ordered_ptco_structures(
        source_base=args.source,
        dest_base=args.dest,
        ratio="1to1"
    )
    generate_ordered_ptco_structures(
        source_base=args.source,
        dest_base=args.dest,
        ratio="1to3"
    )
    generate_ordered_ptco_structures(
        source_base=args.source,
        dest_base=args.dest,
        ratio="3to1"
    )
    generate_ordered_ptco_structures(
        source_base=args.source,
        dest_base=args.dest,
        ratio="none"
    )
    display_xyz_number_info(args.dest)
    
if __name__ == "__main__":
    main()