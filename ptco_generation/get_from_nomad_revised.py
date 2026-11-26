import json
import os
import time
import requests
from tqdm import tqdm
import argparse

BASE_URL = "https://nomad-lab.eu/prod/v1/api/v1"
DATASET_ID = "10.17172/NOMAD/2023.02.01-1"
ELEMENT = "Pt"
PAGE_SIZE = 100

def query_entries(dataset_id, element):
    """
    Query the NOMAD API for all entries in a dataset containing a given element.

    Pages through the /entries endpoint using cursor-based pagination
    (page_after_value) until no further pages are returned.

    Args:
      dataset_id: NOMAD dataset ID to restrict the search to.
      element: Element symbol used to filter entries (e.g. "Pt").

    Returns:
      List of entry_id strings for all matching entries.

    """
    print(f"Querying NOMAD for {element} entries in dataset {dataset_id}...")
    all_entry_ids = []
    page_after = None

    while True:
        params = {
            "dataset_id": dataset_id,
            "elements": element,
            "per_page": PAGE_SIZE,
        }
        if page_after:
            params["page_after_value"] = page_after

        resp = requests.get(f"{BASE_URL}/entries", params=params)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("data", [])
        if not results:
            break

        entry_ids = [entry["entry_id"] for entry in results]
        all_entry_ids.extend(entry_ids)
        print(f"  Retrieved {len(entry_ids)}, total so far: {len(all_entry_ids)}")

        page_after = data.get("pagination", {}).get("next_page_after_value")
        if not page_after:
            break

        time.sleep(0.2)  # Add stagger for rate limiting on NOMAD API

    print(f"Total {element}-containing entries in dataset: {len(all_entry_ids)}")
    return all_entry_ids


def download_archives(entry_ids, output_dir):
    """
    Download all entries as a single archive zip.

    Saves the list of entry IDs to "pt_entry_ids.json" and the downloaded
    archive to "pt_entries.zip", both inside output_dir.

    Args:
      entry_ids: List of NOMAD entry_id strings to download.
      output_dir: Directory to save the entry-ID list and archive zip to;
        created if it does not already exist.

    Returns:
      None

    """
    print("Downloading entries as archive zip...")
    os.makedirs(output_dir, exist_ok=True)

    ids_path = os.path.join(output_dir, "pt_entry_ids.json")
    with open(ids_path, "w") as f:
        json.dump(entry_ids, f, indent=2)
    print(f"Entry-ID list saved to {ids_path}")

    zip_path = os.path.join(output_dir, "pt_entries.zip")
    resp = requests.post(
        f"{BASE_URL}/entries/archive/download",
        json={"entry_ids": entry_ids},
    )
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(resp.content)
    print(f"Archive saved to {zip_path}")
    
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""
Extract all Pt-containing entries from the Quantum Cluster Database (QCD)
deposited in the NOMAD repository, and download them as a single archive.

QCD Data Descriptor: Manna, S. et al. Sci. Data 10, 308 (2023).
                    https://doi.org/10.1038/s41597-023-02200-4
NOMAD dataset:       https://doi.org/10.17172/NOMAD/2023.02.01-1
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.getcwd(),
        help="Directory to save the archive and entry-ID list (default: current directory)"
    )
    parser.add_argument(
        "-e", "--element",
        default=ELEMENT,
        help=f"Element to filter entries by (default: {ELEMENT})"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    entry_ids = query_entries(DATASET_ID, args.element)
    if entry_ids:
        download_archives(entry_ids, args.output_dir)
    else:
        print("No matching entries found.")

if __name__ == "__main__":
    main()
