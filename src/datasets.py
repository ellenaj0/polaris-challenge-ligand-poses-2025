import os
import copy
from tqdm import tqdm
from esm import pretrained
import pandas as pd
from rdkit.Chem import AddHs, MolFromSmiles
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import Dataset as GeometricDataset
from torch.utils.data import Dataset

from datasets.process_mols import generate_conformer, get_lig_graph_with_matching, moad_extract_receptor_structure
from utils.inference_utils import get_sequences, compute_ESM_embeddings


class AntiviralLigandPosesDataset(Dataset):
    def __init__(self, polaris_dataset, test: bool = False):
        self.data: pd.DataFrame = self.get_dataset_from_polaris(polaris_dataset, test=test)
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_path = self.data["protein_path"].iloc[idx]
        cxsmiles = self.data["cxsmiles"].iloc[idx]

        # If not test, then also return label
        if not self.test:
            label = self.data["label"].iloc[idx]
            return (protein_path, cxsmiles), label

        return protein_path, cxsmiles
    
    def get_dataset_from_polaris(self, dataset, test: bool) -> pd.DataFrame:
        protein_pdb_dir = "data/polaris_train/protein_structures" if not test \
            else "data/polaris_test/protein_structures"

        # Extract CXSMILES strings from dataset
        cxsmiles = [dataset[i][0]["CXSMILES"] for i in range(len(dataset))] if not test \
            else [dataset[i]["CXSMILES"] for i in range(len(dataset))]

        data = {
            "protein_path": [f"{protein_pdb_dir}/protein_{i}.pdb" for i in range(len(dataset))],
            "cxsmiles": cxsmiles
        }

        # Add labels only for train set
        if not test:
            data["label"] = [dataset[i][1] for i in range(len(dataset))]

        return pd.DataFrame(data)


class DiffDockDataset(GeometricDataset):
    def __init__(self, root, transform, complex_names, protein_files, ligand_descriptions, protein_sequences, lm_embeddings=True,
                 receptor_radius=30, c_alpha_max_neighbors=None, precomputed_lm_embeddings=None,
                 remove_hs=False, all_atoms=False, atom_radius=5, atom_max_neighbors=None, knn_only_graph=False):

        super(DiffDockDataset, self).__init__(root, transform)
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.knn_only_graph = knn_only_graph

        self.polaris_dir = root
        self.complex_names = complex_names
        self.protein_files = protein_files
        self.ligand_descriptions = ligand_descriptions
        self.protein_sequences = protein_sequences

        # generate LM embeddings
        if lm_embeddings and (precomputed_lm_embeddings is None or precomputed_lm_embeddings[0] is None):
            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            protein_sequences = get_sequences(protein_files, protein_sequences)
            labels, sequences = [], []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')
                sequences.extend(s)
                labels.extend([complex_names[i] + '_chain_' + str(j) for j in range(len(s))])

            lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

            self.lm_embeddings = []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')
                self.lm_embeddings.append([lm_embeddings[f'{complex_names[i]}_chain_{j}'] for j in range(len(s))])

        elif not lm_embeddings:
            self.lm_embeddings = [None] * len(self.complex_names)

        else:
            self.lm_embeddings = precomputed_lm_embeddings

        # Preprocess all complexes and get complex graphs
        self.complex_graphs = self.get_all_complex_graphs()

    def len(self):
        return len(self.complex_names)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graphs[idx])
        return complex_graph

    def get_all_complex_graphs(self) -> list:
        print("Preprocessing all complexes...")
        all_complex_graphs = []

        for idx in tqdm(range(len(self.complex_names)), desc="Constructing graphs"):
            name, protein_file, ligand_description, lm_embedding = \
                self.complex_names[idx], self.protein_files[idx], self.ligand_descriptions[idx], self.lm_embeddings[idx]

            # build the pytorch geometric heterogeneous graph
            complex_graph = HeteroData()
            complex_graph['name'] = name

            # parse the ligand from smiles
            try:
                mol = MolFromSmiles(ligand_description)
                if mol is not None:
                    mol = AddHs(mol)
                    generate_conformer(mol)

            # TODO: Catching this exception should not be necessary, maybe remove later
            except Exception as e:
                print('Failed to read molecule ', ligand_description,
                      ' We are skipping it. The reason is the exception: ', e)
                complex_graph['success'] = False
                all_complex_graphs.append(complex_graph)

            try:
                # parse the receptor from the pdb file
                get_lig_graph_with_matching(mol, complex_graph, popsize=None, maxiter=None, matching=False,
                                            keep_original=False,
                                            num_conformers=1, remove_hs=self.remove_hs)

                moad_extract_receptor_structure(
                    path=os.path.join(protein_file),
                    complex_graph=complex_graph,
                    neighbor_cutoff=self.receptor_radius,
                    max_neighbors=self.c_alpha_max_neighbors,
                    lm_embeddings=lm_embedding,
                    knn_only_graph=self.knn_only_graph,
                    all_atoms=self.all_atoms,
                    atom_cutoff=self.atom_radius,
                    atom_max_neighbors=self.atom_max_neighbors)

            except Exception as e:
                print(f'Skipping {name} because of the error:')
                print(e)
                complex_graph['success'] = False
                all_complex_graphs.append(complex_graph)

            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            complex_graph['receptor'].pos -= protein_center
            if self.all_atoms:
                complex_graph['atom'].pos -= protein_center

            ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
            complex_graph['ligand'].pos -= ligand_center

            complex_graph.original_center = protein_center

            complex_graph.mol = mol

            complex_graph['success'] = True

            all_complex_graphs.append(complex_graph)

        print("Preprocessing all complexes done.")
        return all_complex_graphs
