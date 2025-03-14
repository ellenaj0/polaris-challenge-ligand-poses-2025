from pathlib import Path
from tqdm import tqdm
import polaris as po
import fastpdb


def write_pdb_files():
    """ Write protein structures of the train dataset from Polaris to PDB files."""

    competition_cache_dir = Path("./data/antiviral-ligand-poses-2025")
    out_dir = Path("../data/polaris_train")

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load competition
    competition = po.load_competition("asap-discovery/antiviral-ligand-poses-2025")
    competition.cache(competition_cache_dir)
    train_data, _ = competition.get_train_test_split()

    # Write PDB files to specified directory
    for idx in tqdm(range(len(train_data)), desc=f"Writing proteins from polaris train set"):
        pdb_file = fastpdb.PDBFile()

        protein_structure = competition[idx]["Protein Structure"]
        pdb_file.set_structure(protein_structure)

        pdb_file.write(Path(out_dir) / f"protein_{idx}.pdb")


if __name__ == '__main__':
    write_pdb_files()