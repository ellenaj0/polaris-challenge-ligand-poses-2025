from typing import List
import base64
import os
import argparse

from rdkit import Chem
import polaris as po


def serialize_rdkit_mol(mol: Chem.Mol):
    props = Chem.PropertyPickleOptions.AllProps
    mol_bytes = mol.ToBinary(props)
    return base64.b64encode(mol_bytes).decode('ascii')


def sort_pred_files(pred_dir: str) -> List[str]:
    """
    Sort the prediction files in the given directory according to the order of the polaris test dataset.
    """

    pred_files_sorted = sorted([f for f in os.listdir(pred_dir) if f.startswith("ligand_pose")],
                               key=lambda x: int(x.split("_")[2].split(".")[0]))
    return pred_files_sorted


def submit_predictions_to_polaris(pred_dir: str):
    # Sort the paths of the prediction files to match with the order of the polaris test dataset
    pred_files = sort_pred_files(pred_dir)
    # Get the full path of the prediction files
    pred_paths = [os.path.join(pred_dir, pred_file) for pred_file in pred_files]

    # Convert ligand pose prediction in sdf file to mol object
    preds_mol = [Chem.MolFromMolFile(pred, removeHs=True, sanitize=False) for pred in pred_paths]

    # Serialize preds for submission
    preds_serialized = [serialize_rdkit_mol(mol) for mol in preds_mol]

    # Submit predictions to competition
    competition = po.load_competition("asap-discovery/antiviral-ligand-poses-2025")

    competition.submit_predictions(
        predictions=preds_serialized,
        prediction_name="final-predictions",
        prediction_owner="ellenaj0",
        report_url="https://drive.google.com/file/d/1leKiDHVeGDe5GL0_ll8fxZrDh05d4X7g/view?usp=sharing",   
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="../data/polaris_test/final_submission/predictions_aligned")
    args = parser.parse_args()

    print(f"Submitting predictions from {args.pred_dir} to Polaris...")
    submit_predictions_to_polaris(pred_dir=args.pred_dir)
