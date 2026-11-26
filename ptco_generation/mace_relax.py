import argparse
from ase import io
from ase.optimize import BFGS
from mace.calculators import MACECalculator

MACE_MODEL = ["/home/lwong/mace/mace-osaka24-large.model"] # Path to the MACE model file
DEVICE = "cuda"
FMAX = 0.02
MAX_STEPS = 10000

def parse_args():
    parser = argparse.ArgumentParser(
        description="Relax a structure with a MACE machine-learned potential using BFGS."
    )
    parser.add_argument(
        "structure_file",
        help='Path to the structure file to relax (expected to end in "_initial.xyz")',
    )
    parser.add_argument(
        "--model", type=str, default=MACE_MODEL[0],
        help="Path to the MACE model file (default: %(default)s)"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE,
        help="Device to run the MACE calculator on (default: %(default)s)"
    )
    parser.add_argument(
        "--fmax", type=float, default=FMAX,
        help="Force convergence threshold for BFGS, in eV/Angstrom (default: %(default)s)"
    )
    parser.add_argument(
        "--steps", type=int, default=MAX_STEPS,
        help="Maximum number of BFGS steps to run (default: %(default)s)"
    )
    return parser.parse_args()

def main():
    """
    Relax a structure with MACE and write the relaxed geometry.

    Reads the structure from args.structure_file, attaches a MACE
    calculator (using args.model on args.device), and runs a BFGS
    geometry optimization until either args.fmax is reached or args.steps
    is exceeded. Optimizer logging is written to a "_relax.log" file
    derived from structure_file, and the final structure is written to a
    "_relaxed.xyz" file derived from structure_file (by replacing an
    "_initial.xyz" suffix).

    Returns:
      None
    """
    args = parse_args()

    calc = MACECalculator(model_paths=[args.model], device=args.device)

    atoms = io.read(args.structure_file)
    atoms.calc = calc

    dyn = BFGS(atoms, logfile=args.structure_file.replace(".xyz", "_relax.log"))
    dyn.run(fmax=args.fmax, steps=args.steps)

    atoms.write(args.structure_file.replace("_initial.xyz", "_relaxed.xyz"))

if __name__ == "__main__":
    main()
