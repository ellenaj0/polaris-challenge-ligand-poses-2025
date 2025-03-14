import argparse
import requests
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import polaris as po


def generate_esm_protein_structures(out_dir: Path):
    """
    Generate the 3D structures of the given proteins using the NVIDIA ESMFold model through the provided API.
    """

    # Load the competition
    competition = po.load_competition("asap-discovery/antiviral-ligand-poses-2025")
    competition.cache()

    # Put the test data into a dataframe
    _, test = competition.get_train_test_split()
    df_test = pd.DataFrame(test.X)

    # Concat the protein chain sequences
    df_test["Concat Chain Sequence"] = df_test["Chain A Sequence"] + df_test["Chain B Sequence"]
    # Add a column with the length of the Chain A sequence to know where to separate the chains later
    df_test['Chain A Sequence Length'] = df_test['Chain A Sequence'].apply(len)

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Prepare the API request
    invoke_url = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"

    headers = {
        "Authorization": "your_api_key_here",  # Replace with your actual API key
        "Accept": "application/json",
    }

    # Predict protein structure for every concatenated sequence using the ESMFold model
    for i, protein_seq in tqdm(enumerate(df_test["Concat Chain Sequence"]), total=len(df_test),
                               desc="Generating protein structures"):

        # Send API request
        payload = {
            "sequence": protein_seq
        }

        session = requests.Session()

        response = session.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()

        # Get the response body as JSON
        response_body = response.json()

        # Write the generated protein structure to a .pdb file
        # The protein structure is saved in the "pdbs" key of the response body
        pdb_file_content = response_body["pdbs"][0]

        # Refactor the chain B in the PDB file
        pdb_file_content_processed = refactor_chain_B(pdb_file_content=pdb_file_content,
                                                      chain_a_len=df_test['Chain A Sequence Length'][i])

        protein_file = os.path.join(out_dir, f"protein_{i}.pdb")
        with open(protein_file, "w") as f:
            f.write(pdb_file_content_processed)

def refactor_chain_B(pdb_file_content: str, chain_a_len: int):
    """
    Rename the chain ID and residue sequence number of the B chain in the PDB file, since during the prediction
    of the 3D structure the concatenated protein sequences were used.
    """
    # Constants for the positions of the fields in the PDB file
    # See https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html
    ATOM_NAME_START_POSITION = 12
    ATOM_NAME_END_POSITION = 16
    RESIDUE_NAME_START_POSITION = 17
    RESIDUE_NAME_END_POSITION = 20
    CHAIN_ID_POSITION = 21
    RESIDUE_SEQ_NUM_START_POSITION = 23
    RESIDUE_SEQ_NUM_END_POSITION = 26

    # Split the PDB file content into lines and save in a list
    pdb_lines = pdb_file_content.splitlines()

    # Initialize the chain B sequence number and the previous residue name
    chain_B_seq_num = 0
    prev_res_name = None

    for i, line in enumerate(pdb_lines):
        # Only process ATOM and TER lines
        if line.startswith(("ATOM", "TER")):
            # Extract the residue sequence number and check if we have reached the B chain
            res_seq_num = int(line[RESIDUE_SEQ_NUM_START_POSITION: RESIDUE_SEQ_NUM_END_POSITION].strip())
            # If the residue sequence number is greater than the length of chain A, we are in chain B
            if res_seq_num > chain_a_len:
                # Get the current atom name and residue name
                atom_name = line[ATOM_NAME_START_POSITION: ATOM_NAME_END_POSITION].strip()
                cur_res_name = line[RESIDUE_NAME_START_POSITION: RESIDUE_NAME_END_POSITION].strip()
                # Update the chain B sequence number if we are in a new residue
                # This is the case either when the residue name changes or in case of another consecutive residue
                # with the same name but the atom name is "N" (the first atom in the residue)
                if cur_res_name != prev_res_name or (cur_res_name == prev_res_name and atom_name == "N"):
                    chain_B_seq_num += 1
                    prev_res_name = cur_res_name
                # Update the chain ID and residue sequence number in the current line
                line = line[:CHAIN_ID_POSITION] + "B" + f"{chain_B_seq_num:>4}" + line[RESIDUE_SEQ_NUM_END_POSITION:]

            pdb_lines[i] = line

    # Join the lines back to a single string
    pdb_file_content_processed = "\n".join(pdb_lines)

    return pdb_file_content_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data/polaris_test/protein_structures_processed",
                        help="Output directory for the generated protein structures")
    args = parser.parse_args()

    generate_esm_protein_structures(out_dir=Path(args.out_dir))
