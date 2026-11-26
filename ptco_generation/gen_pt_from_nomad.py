import os
from pathlib import Path
from ase.io import read, write

# Source and destination base paths
SOURCE_BASE = "/media/hdd1/pt_np_nomad_dataset/JVkvSYsCQEmSx18lbb8Ehw/Pt" # Source base path for the downloaded original dataset
DEST_BASE = "/media/nvme1/pt_only" # Destination base path for the Pt-only dataset

def create_pt_only_dataset(source_base, dest_base):
    """
    Create a new dataset containing only Pt-containing entries from the QCD dataset.

    Args:
        source_base (str): The base directory of the source dataset.
        dest_base (str): The base directory where the Pt-only dataset will be created.

    Returns:
        None
    """
    
    # Ensure destination base exists
    os.makedirs(dest_base, exist_ok=True)
    
    # Iterate through directories (40-55)
    for temp in range(40, 56):
        temp_dir = f"{temp}"
        
        # Find all numbered subdirectories (1-##)
        source_temp_path = os.path.join(source_base, temp_dir)
        
        if not os.path.exists(source_temp_path):
            continue
        
        # Get all subdirectories matching pattern
        for subdir in os.listdir(source_temp_path):
            subdir_path = os.path.join(source_temp_path, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
            
            source_file = os.path.join(subdir_path, "vasprun.xml")
            
            if not os.path.exists(source_file):
                continue
            
            # Convert to xyz format using ASE
            atoms = read(str(source_file), index=-1, format="vasp-xml")
            
            # Create destination directory
            dest_file = os.path.join(dest_base, temp_dir, subdir, "structure.xyz")
            dest_dir = os.path.dirname(dest_file)
            
            os.makedirs(dest_dir, exist_ok=True)
            
            write(dest_file, atoms, format="xyz")

    print("Write operation completed!")
    
def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Script for creating a Pt-only dataset from the original QCD dataset."
    )
    parser.add_argument(
        "--source", type=str, default=SOURCE_BASE,
        help="Base directory of the source dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--dest", type=str, default=DEST_BASE,
        help="Base directory for the Pt-only dataset (default: %(default)s)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_pt_only_dataset(args.source, args.dest)