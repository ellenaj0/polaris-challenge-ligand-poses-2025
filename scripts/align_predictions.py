from typing import List
import os
import argparse
from rdkit import Chem
import pandas as pd
import pymol


def preprocess_pdb_before_combine(protein_pdb_lines: List[str]) -> List[str]:
    # Remove the END line from the protein pdb if exists since we want to append the ligand
    if protein_pdb_lines[-1].startswith("END"):
        protein_pdb_lines = protein_pdb_lines[:-1]

    # Add a TER line if it doesn't exist to separate the protein and ligand
    if not protein_pdb_lines[-1].startswith("TER"):
        protein_pdb_lines.append("TER\n")

    return protein_pdb_lines


def renumber_ligand(ligand_pdb_lines: List[str], atom_num_start: int) -> List[str]:
    ATOM_NUM_START_POSITION = 6
    ATOM_NUM_END_POSITION = 11

    current_atom_num = atom_num_start
    old_to_new_atom_num = {}

    ligand_pdb_lines_renumbered = []

    for i, line in enumerate(ligand_pdb_lines):
        if line.startswith("HETATM"):
            line_renumbered = line[:ATOM_NUM_START_POSITION] + str(current_atom_num).rjust(5) + line[ATOM_NUM_END_POSITION:]
            old_to_new_atom_num[i + 1] = current_atom_num

            ligand_pdb_lines_renumbered.append(line_renumbered)
            current_atom_num += 1

        elif line.startswith("CONECT"):
            conect_line_parts = line.split()
            new_conect_line = "CONECT"
            for atom_num in conect_line_parts[1:]:  # Skip the first element, which is "CONECT"
                new_conect_line += f"{old_to_new_atom_num[int(atom_num)]:5}"
            new_conect_line += "\n"
            ligand_pdb_lines_renumbered.append(new_conect_line)

        else:  # Else it's an END line
            ligand_pdb_lines_renumbered.append(line)

    return ligand_pdb_lines_renumbered


def combine_proteins_and_ligand_preds_in_pdb(protein_dir: str, preds_dir: str, complex_dir: str) -> List[str]:
    """
    Combine the protein and the ligand prediction into a single pdb file, so we can align the whole complex later.
    """

    # Get all protein pdb files
    protein_files = [f"{protein_dir}/protein_{i}.pdb" for i in range(len(os.listdir(protein_dir)))]
    # Get all DiffDock rank 1 predictions from sdf files
    ligand_files = [f"{preds_dir}/complex_{i}/rank1.sdf" for i in range(len(os.listdir(preds_dir)))]

    # Complex paths to save the protein-ligand complexes
    os.makedirs(complex_dir, exist_ok=True)
    complex_paths = [f"{complex_dir}/complex_{i}.pdb" for i in range(len(os.listdir(protein_dir)))]

    for protein_file, ligand_file, complex_path in zip(protein_files, ligand_files, complex_paths):
        complex_pdb_lines = []

        if not os.path.exists(ligand_file):
            print(f"Skipping {ligand_file} because it does not exist")
            continue

        # Get the ligand from .sdf and convert it to a pdb block
        ligand_mol = Chem.MolFromMolFile(ligand_file, removeHs=True, sanitize=False)
        ligand_pdb_block = Chem.MolToPDBBlock(ligand_mol)

        # Append the ligand pdb block to the protein pdb and save the complex to complex_path
        ligand_pdb_lines = ligand_pdb_block.splitlines(keepends=True)

        with open(protein_file, 'r') as f:
            protein_pdb_lines = f.readlines()

        # Preprocess the protein pdb lines before combining it with the ligand
        protein_pdb_lines = preprocess_pdb_before_combine(protein_pdb_lines)
        complex_pdb_lines.extend(protein_pdb_lines)

        # Renumber the ligand atoms and append them to the complex
        ligand_pdb_lines = renumber_ligand(ligand_pdb_lines, atom_num_start=len(protein_pdb_lines) + 1)
        complex_pdb_lines.extend(ligand_pdb_lines)

        with open(complex_path, 'w') as f:
            f.writelines(complex_pdb_lines)

    return complex_paths


def align_prediction(protein_path, protein_label, aligned_pred_path) -> float:
    # Find out which protein label to align to
    if protein_label == "MERS-CoV Mpro":
        ref_path = "../data/ALIGNMENT_REFERENCES/MERS-CoV-Mpro/reference_structure/protein.pdb"
    elif protein_label == "SARS-CoV-2 Mpro":
        ref_path = "../data/ALIGNMENT_REFERENCES/SARS-CoV-2-Mpro/reference_structure/protein.pdb"
    else:
        raise ValueError(f"Unknown protein label: {protein_label}")

    # Align the structure to the reference structure
    pymol.cmd.delete("all")
    pymol.cmd.load(protein_path, "mobile")
    pymol.cmd.load(ref_path, "reference")

    alignment_info = pymol.cmd.align(
        "polymer and name CA and mobile",
        "polymer and name CA and reference",
    )

    # Save only the aligned ligand to given path
    pymol.cmd.save(aligned_pred_path, "mobile and hetatm")

    # Return RMSD of alignment as info
    return alignment_info[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_dir", type=str, default="../data/polaris_test/protein_structures")
    parser.add_argument("--preds_dir", type=str, default="../predictions/polaris_test_40")
    parser.add_argument("--complex_dir", type=str, default="../data/polaris_test/proteins_with_pred_ligand")
    parser.add_argument("--aligned_preds_dir", type=str, default="../data/polaris_test/predictions_aligned")
    args = parser.parse_args()

    # Combine the protein and ligand predictions into a single pdb file, so we can align the whole complex
    complex_paths = combine_proteins_and_ligand_preds_in_pdb(
        protein_dir=args.protein_dir,
        preds_dir=args.preds_dir,
        complex_dir=args.complex_dir
    )

    # Paths to save the aligned predictions to
    os.makedirs(args.aligned_preds_dir, exist_ok=True)
    aligned_pred_paths = [f"{args.aligned_preds_dir}/ligand_pose_{i}.sdf" for i in range(len(complex_paths))]

    # Get the protein labels from csv
    polaris_test_df = pd.read_csv("../data/polaris_test/polaris_test_data.csv")
    protein_labels = polaris_test_df["Protein Label"]

    # Align predictions to reference structures
    for i, (complex_path, protein_label, aligned_pred_path) in (
            enumerate(zip(complex_paths, protein_labels, aligned_pred_paths))):
        print(protein_label)
        alignment_rmsd = align_prediction(complex_path, protein_label, aligned_pred_path)
        print(f"Aligned complex {i} with RMSD: {alignment_rmsd:.2f}")



if __name__ == "__main__":
    main()
