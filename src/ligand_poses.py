from pathlib import Path
import glob
import os
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import numpy as np
import polaris as po
import torch

from src.datasets import AntiviralLigandPosesDataset
from src.inference import inference_diffdock
from src.training import train_diffdock
from src.utils import scaffold_split, compute_symmetry_rmsd, get_mol_from_sdf, extract_confidence_from_filename, \
    print_rmsd_statistics


class LigandPoses:
    def __init__(self, params: dict = None): 
        self.params: dict = params
        self.device: str = "cpu"
        self.competition = None
        self.train_dataset_polaris = None
        self.test_dataset_polaris = None
        self.train_dataset = None
        self.test_dataset = None

        self._init()

    def _init(self):
        self._init_device()
        self._init_competition()
        self._init_dataset()

    def _init_device(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device.type}")

    def _init_competition(self):
        self.competition = po.load_competition("asap-discovery/antiviral-ligand-poses-2025")
        competition_data_dir: Path = Path("./data/antiviral-ligand-poses-2025")
        self.competition.cache(competition_data_dir)

    def _init_dataset(self):
        train_dataset_polaris, test_dataset_polaris = self.competition.get_train_test_split()

        self.train_dataset_polaris = AntiviralLigandPosesDataset(train_dataset_polaris, test=False)
        self.test_dataset_polaris = AntiviralLigandPosesDataset(test_dataset_polaris, test=True)

        self.train_dataset, self.test_dataset = scaffold_split(dataset=self.train_dataset_polaris, test_size=0.2) # TODO: Param test size

    def train(self, args):
        print("Start training...")
        train_diffdock(self.train_dataset, self.device, args)

    def inference(self, args):
        print("Running inference...")
        inference_diffdock(self.test_dataset_polaris.data, args)

    def evaluate_min_rmsd(self, args):
        """
        Evaluate the minimum RMSD and corresponding confidence of the predicted poses by DiffDock for each complex.
        """

        min_rmsd_list = []
        min_confidence_list = []
        for i, true_pose in tqdm(enumerate(self.train_dataset_polaris.data["label"]), desc="Evaluating poses"):
            pred_pose_rmsds = []
            pred_pose_confidences = []
            current_complex_dir = f"{args.preds_dir}/complex_{i}"

            if len(os.listdir(current_complex_dir)) == 0:
                print(f"{current_complex_dir} is empty, skipping this complex...")
                continue

            # Compute RMSD for every predicted sample (for each rank)
            for rank in range(1, args.samples_per_complex + 1):
                pred_pose_path = glob.glob(f"{current_complex_dir}/rank{rank}_*.sdf")[0]

                pred_pose = get_mol_from_sdf(pred_pose_path)
                if pred_pose is not None:   # Todo: In case pred_pose_path does not contain a valid molecule
                    confidence = extract_confidence_from_filename(pred_pose_path)
                    pred_pose_confidences.append(confidence)

                    rmsd = compute_symmetry_rmsd(true_pose, pred_pose)
                    pred_pose_rmsds.append(rmsd)

            # Get minimum RMSD from all predicted samples and its corresponding confidence
            pred_pose_rmsds = np.array(pred_pose_rmsds)
            min_rmsd_idx = np.argmin(pred_pose_rmsds)

            min_rmsd = pred_pose_rmsds[min_rmsd_idx]
            min_rmsd_list.append(min_rmsd)

            min_confidence = pred_pose_confidences[min_rmsd_idx]
            min_confidence_list.append(min_confidence)

        # Save results to csv
        os.makedirs(args.results_dir, exist_ok=True)

        results_df = pd.DataFrame({
            "RMSD": min_rmsd_list,
            "Confidence": min_confidence_list
        })
        results_df.to_csv(f"{args.results_dir}/min_rmsds.csv", index=False) 

        # Print statistics
        rmsds = np.array(min_rmsd_list)
        print_rmsd_statistics(rmsds)


    def evaluate_rank1(self, args):
        """
        Evaluate the rank1 RMSD and corresponding confidence of the predicted poses by DiffDock for each complex.
        """

        rmsd_list = []
        confidence_list = []
        for i, true_pose in tqdm(enumerate(self.train_dataset_polaris.data["label"]), desc="Evaluating poses"):
            rank1_pose_path = glob.glob(f"{args.preds_dir}/complex_{i}/rank1_*.sdf")[0]

            if len(rank1_pose_path) == 0:
                print(f"Could not find rank1 prediction for complex_{i}, skipping it...")
                continue

            rank1_pose = get_mol_from_sdf(rank1_pose_path)

            rank1_confidence = extract_confidence_from_filename(rank1_pose_path)
            confidence_list.append(rank1_confidence)

            rmsd = compute_symmetry_rmsd(true_pose, rank1_pose)
            rmsd_list.append(rmsd)

            # Save results to csv
            os.makedirs(args.results_dir, exist_ok=True)

            results_df = pd.DataFrame({
                "RMSD": rmsd_list,
                "Confidence": confidence_list
            })
            results_df.to_csv(f"{args.results_dir}/rank1_rmsds.csv", index=False) 

        rmsds = np.array(rmsd_list)
        print_rmsd_statistics(rmsds)

    def evaluate_rank1_aligned(self, args):
        """
        Evaluate the rank1 RMSD of the predicted poses by DiffDock for each aligned complex to the reference structure.
        """

        rmsd_list = []
        for i, true_pose in tqdm(enumerate(self.train_dataset_polaris.data["label"]), desc="Evaluating poses"):
            pred_path = f"{args.preds_dir}/ligand_{i}.sdf"

            if not os.path.exists(pred_path):
                print(f"Could not find prediction for ligand {i} in {args.preds_dir}, skipping it...")
                continue

            rank1_pose = Chem.MolFromMolFile(pred_path, removeHs=True, sanitize=False)

            try:
                rmsd = compute_symmetry_rmsd(true_pose, rank1_pose)
            except Exception as e:
                print(f"An error occured while computing the rmsd of ligand {i}: {e}")

            rmsd_list.append(rmsd)

            # Save results to csv
            os.makedirs(args.results_dir, exist_ok=True)

            results_df = pd.DataFrame({
                "RMSD": rmsd_list,
            })
            results_df.to_csv(f"{args.results_dir}/rank1_rmsds_aligned.csv", index=False)  

        rmsds = np.array(rmsd_list)
        print_rmsd_statistics(rmsds)
