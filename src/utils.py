from pathlib import Path
import yaml
from typing import Tuple
import re

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from spyrmsd import rmsd
from spyrmsd.molecule import Molecule

from datasets.pdbbind import NoiseTransform
from datasets.dataloader import DataLoader, DataListLoader

from src.datasets import DiffDockDataset


def scaffold_split(dataset, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train/test sets based on their scaffolds groups.
    """

    data = dataset.data.copy()   # Copy data to avoid modifying the original dataset

    # Compute scaffolds for each molecule
    data["scaffold"] = data["cxsmiles"].apply(compute_scaffold)

    # Group by scaffold
    scaffold_groups = data.groupby("scaffold").apply(lambda x: x.index.tolist())

    # Sort scaffolds by frequency (larger groups first for balanced splitting)
    scaffold_groups = sorted(scaffold_groups, key=len, reverse=True)

    # Prepare train, validation, and test sets
    train_scaffolds_indices, test_scaffolds_indices = train_test_split(scaffold_groups, test_size=test_size, random_state=42)
    train_scaffolds_indices = sum(train_scaffolds_indices, start=[])  # Flatten list
    test_scaffolds_indices = sum(test_scaffolds_indices, start=[])  # Flatten list

    # Create final splits
    train_data_split = data.loc[train_scaffolds_indices].drop(columns="scaffold").reset_index(drop=True)
    test_data_split = data.loc[test_scaffolds_indices].drop(columns="scaffold").reset_index(drop=True)
    return train_data_split, test_data_split


def compute_scaffold(smiles: str):
    """
    Convert SMILES to scaffold using Murcko Scaffolds.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to construct molecule from SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold


def compute_symmetry_rmsd(true_pose: Chem.rdchem.Mol, pred_pose: Chem.rdchem.Mol) -> float:
    """
    Compute the symmetry-corrected heavy-atom RMSD between two poses.
    """

    true_pose_mol = Molecule.from_rdkit(true_pose)
    pred_pose_mol = Molecule.from_rdkit(pred_pose)

    symm_rmsd = rmsd.rmsdwrapper(true_pose_mol, pred_pose_mol, symmetry=True, strip=True)

    return symm_rmsd[0].item()


def get_mol_from_sdf(sdf_path: str) -> Chem.rdchem.Mol:
    """
    Load a molecule from an SDF file that contains a single molecule.
    """

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = suppl[0]
    if mol is None:
        raise ValueError(f"Failed to get molecule from SDF: {sdf_path}")
    return mol


def extract_confidence_from_filename(filename: str) -> float:
    """
    Extract the confidence value from a filename where the corresponding file contains a ligand pose prediction (SDF file) by DiffDock.
    """

    match = re.search(r"confidence(-?\d+\.\d+)", filename)
    if match:
        confidence_value = float(match.group(1))
    else:
        raise ValueError(f"Failed to extract confidence value from filename: {filename}")
    return confidence_value


def load_yaml_to_dict(config_filename: str) -> dict:
    path = Path(".") / "config" / config_filename
    with open(path, "r") as file:
        config: dict = yaml.safe_load(file)

    return config


def print_rmsd_statistics(rmsds: np.array) -> None:
    print(f"Min RMSD: {rmsds.min():.2f}")
    print(f"Mean RMSD: {rmsds.mean():.2f}")
    print(f"Median RMSD: {np.median(rmsds):.2f}")
    print(f"Max RMSD: {rmsds.max():.2f}")
    print(f"RMSD<2: {(100 * (rmsds < 2).sum() / len(rmsds)):.4f}%")
    print(f"RMSD<3: {(100 * (rmsds < 3).sum() / len(rmsds)):.4f}%")
    print(f"RMSD<5: {(100 * (rmsds < 5).sum() / len(rmsds)):.4f}%")


def construct_polaris_dataloaders(dataset: pd.DataFrame, args, t_to_sigma):
    """ 
    Construct dataloaders for the Polaris dataset for training of DiffDock. 
    """

    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms, alpha=args.sampling_alpha, beta=args.sampling_beta,
                               include_miscellaneous_atoms=False if not hasattr(args,'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                               crop_beyond_cutoff=args.crop_beyond)

    # TODO: For now, just randomly split the dataset into train and val 80/20
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_data = dataset.loc[train_indices].reset_index(drop=True)
    val_data = dataset.loc[val_indices].reset_index(drop=True)

    complex_name_list = [f"complex_{i}" for i in range(len(dataset))]

    # Construct train dataset
    train_complex_names = [complex_name_list[i] for i in train_indices]
    train_protein_files = train_data['protein_path'].tolist()
    train_ligand_descriptions = train_data['cxsmiles'].tolist()

    train_dataset = DiffDockDataset(root=args.polaris_dir, transform=transform,
                                    complex_names=train_complex_names,
                                    protein_files=train_protein_files,
                                    ligand_descriptions=train_ligand_descriptions,
                                    protein_sequences=None,
                                    lm_embeddings=True,
                                    receptor_radius=args.receptor_radius,
                                    remove_hs=args.remove_hs,
                                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                                    all_atoms=args.all_atoms, atom_radius=args.atom_radius,
                                    atom_max_neighbors=args.atom_max_neighbors,
                                    knn_only_graph=False)

    # Construct val dataset
    val_complex_names = [complex_name_list[i] for i in val_indices]
    val_protein_files = val_data['protein_path'].tolist()
    val_ligand_descriptions = val_data['cxsmiles'].tolist()

    val_dataset = DiffDockDataset(root=args.polaris_dir, transform=transform,
                                    complex_names=val_complex_names,
                                    protein_files=val_protein_files,
                                    ligand_descriptions=val_ligand_descriptions,
                                    protein_sequences=None,
                                    lm_embeddings=True,
                                    receptor_radius=args.receptor_radius,
                                    remove_hs=args.remove_hs,
                                    c_alpha_max_neighbors=args.c_alpha_max_neighbors,
                                    all_atoms=args.all_atoms, atom_radius=args.atom_radius,
                                    atom_max_neighbors=args.atom_max_neighbors,
                                    knn_only_graph=False)

    # Construct dataloaders
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size,
                                num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory,
                                drop_last=args.dataloader_drop_last)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers,
                              shuffle=False, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)

    print("Constructed dataloaders for Polaris dataset.")
    return train_loader, val_loader
