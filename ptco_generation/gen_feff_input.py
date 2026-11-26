import os
import warnings
import tempfile
from ase import io
from pymatgen.core.structure import Structure, Lattice
from pymatgen.io.cif import CifParser
from tqdm import tqdm
from larixite.cif_cluster import cif2feffinp

# Suppress larixite/pymatgen primitive warning
warnings.filterwarnings(
    "ignore",
    message="The default value of primitive was changed from True to False",
    category=UserWarning,
    module="larixite.cif_cluster"
)

warnings.filterwarnings(
    "ignore",
    message=".*Issues encountered while parsing CIF: .* fractional coordinates rounded.*",
    category=UserWarning
)

def process_feff(content: str, spectrum: str) -> str:
    """
    Post-process a FEFF input string for an EXAFS or XANES calculation.

    The function rewrites the FEFF header so that conflicting directives are
    removed or commented out, then inserts the spectrum-specific control block
    immediately before the POTENTIALS section.

    Args:
      content: Raw FEFF input content as a single multiline string.
      spectrum: Desired spectrum mode. Supported values are "EXAFS" and
        "XANES".

    Returns:
      The updated FEFF input content as a newline-terminated string.
    """
    lines = content.splitlines()
    # find POTENTIALS index
    pot_idx = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("POTENTIALS"):
            pot_idx = i
            break
    if pot_idx is None:
        raise ValueError("No POTENTIALS block found in feff.inp")

    header = lines[:pot_idx]
    potentials = lines[pot_idx:]

    def is_keyword(line, kw):
        # ignore leading comment '*' and whitespace, case-insensitive
        s = line.strip()
        s2 = s.lstrip('*').strip().upper()
        return s2.startswith(kw.upper())

    new_header = []
    for line in header:
        # always delete NLEG lines
        if is_keyword(line, "NLEG"):
            continue

        if spectrum == "EXAFS":
            # drop any EXAFS or RPATH directives found in the header (we'll add the correct ones)
            if is_keyword(line, "EXAFS") or is_keyword(line, "RPATH"):
                continue
            # comment existing SCF so it won't conflict with EXAFS settings
            if is_keyword(line, "SCF"):
                if line.lstrip().startswith('*'):
                    new_header.append(line)
                else:
                    new_header.append('*' + line)
                continue
            new_header.append(line)

        elif spectrum == "XANES":
            # remove EXAFS and RPATH lines entirely for XANES
            if is_keyword(line, "EXAFS") or is_keyword(line, "RPATH"):
                continue
            # comment any existing SCF so we can add the proper SCF below
            if is_keyword(line, "SCF"):
                if line.lstrip().startswith('*'):
                    new_header.append(line)
                else:
                    new_header.append('*' + line)
                continue
            new_header.append(line)

        else:
            # unknown spectrum: keep header unchanged
            new_header.append(line)

    # Prepare insertion block and place it right before POTENTIALS, with a blank line after it
    insert_lines = []
    if spectrum == "EXAFS":
        insert_lines = [
            "EXAFS 20",
            "RPATH 12.0",
        ]
    elif spectrum == "XANES":
        insert_lines = [
            "SCF 7.0",
            "XANES 4.0 0.07 0.5",
            "FMS 6.0",
            "CORRECTIONS 1.5 0.3",
        ]

    # Combine: header + insert_lines + blank line + potentials
    combined = new_header + insert_lines + [""] + potentials
    return "\n".join(combined) + "\n"

SOURCE_PATH = "/media/nvme1/nanoparticle_data/"
EDGE = "L3"

EXAFS_ROOT = os.path.join(SOURCE_PATH, "EXAFS")
XANES_ROOT = os.path.join(SOURCE_PATH, "XANES")

def generate_feff_inputs(source_path):
    """
    Generate FEFF input files from relaxed XYZ structures.

    Recursively scans the source directory for files ending in
    ``relaxed.xyz``, reads each structure with ASE's extxyz parser,
    converts it into a pymatgen Structure, and then writes temporary CIF
    data before generating one FEFF input file per absorber site for both
    EXAFS and XANES output trees.

    Args:
      source_path: Root directory containing the relaxed structure dataset.

    Returns:
      None
    """
    # Find all relaxed xyz files in the source path
    xyz_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith("relaxed.xyz"):
                xyz_files.append(os.path.join(root, file))

    print(f"Found {len(xyz_files)} .xyz structures to process")

    for xyz_path in tqdm(xyz_files, desc="Generating FEFF inputs"):
        try:
            # read with extxyz so lattice/pbc are preserved
            atoms = io.read(xyz_path, format="extxyz")

            # ensure a valid cell (should be present in your extxyz)
            cell = atoms.get_cell()
            if not cell.any():
                # fallback cell if somehow missing
                cell = [[30.0, 0, 0], [0, 30.0, 0], [0, 0, 30.0]]

            # build pymatgen Structure
            structure = Structure(
                lattice=Lattice(cell),
                species=[atom.symbol for atom in atoms],
                coords=atoms.get_positions(),
                coords_are_cartesian=True,
            )

            rel_path = os.path.relpath(os.path.dirname(xyz_path), source_path)

            # create a unique temporary CIF file in the same directory (avoid basename collisions)
            fd, tmp_cif_path = tempfile.mkstemp(prefix="larix_cif_", suffix=".cif", dir=os.path.dirname(xyz_path))
            os.close(fd)
            structure.to(fmt="cif", filename=tmp_cif_path, symprec=None)

            # Re-parse the CIF with CifParser to ensure larixite and we see the same parsed structure
            parsed_structs = CifParser(tmp_cif_path).parse_structures(primitive=False)
            if len(parsed_structs) == 0:
                raise RuntimeError(f"No structures parsed from {tmp_cif_path}")
            parsed_struct = parsed_structs[0]
            n_sites = len(parsed_struct)

            # Use a different variable name than `root` to avoid shadowing outer variables
            for spectrum, out_root in [("EXAFS", EXAFS_ROOT), ("XANES", XANES_ROOT)]:
                spectrum_outdir = os.path.join(out_root, rel_path)
                os.makedirs(spectrum_outdir, exist_ok=True)

                # iterate parsed sites (use parsed_struct for absorber species and indexing)
                for idx in range(n_sites):
                    atom_dir = os.path.join(spectrum_outdir, f"atom_{idx}")
                    os.makedirs(atom_dir, exist_ok=True)

                    if parsed_struct[idx].specie.symbol == "Co":
                        EDGE = "K"
                    else:
                        EDGE = "L3"
                    
                    try:
                        # absorber_site is 1-based indexing for FEFF
                        feff_inp = cif2feffinp(
                            tmp_cif_path,
                            absorber=parsed_struct[idx].species_string,
                            absorber_site=idx + 1,
                            edge=EDGE,
                            cluster_size=20
                        )
                    except Exception as sub_e:
                        print(f"  ↳ Skipped site {idx+1} in {xyz_path}: {sub_e}")
                        continue

                    feff_inp = process_feff(feff_inp, spectrum)

                    with open(os.path.join(atom_dir, "feff.inp"), "w") as f:
                        f.write(feff_inp)

            # clean up temp cif file
            try:
                os.remove(tmp_cif_path)
            except Exception:
                pass

        except Exception as e:
            print(f"Error processing {xyz_path}: {e}")

def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate FEFF input files from relaxed structures."
    )
    parser.add_argument(
        "--source", type=str, default=SOURCE_PATH,
        help="Base directory of the source dataset (default: %(default)s)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    generate_feff_inputs(args.source)
    
if __name__ == "__main__":
    main()