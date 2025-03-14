import polaris as po
import pandas as pd
from pathlib import Path


def generate_test_data_csv(output_file: Path = "../data/polaris_test/polaris_test_data.csv"):
    """
    Generate a csv file to be used for DiffDock inference from the Polaris test data.
    Concatenates the protein chains A and B into a single sequence.
    """

    # Load the competition
    competition = po.load_competition("asap-discovery/antiviral-ligand-poses-2025")
    competition.cache()

    # Put the test data into a dataframe
    train, test = competition.get_train_test_split()
    df_test = pd.DataFrame(test.X)

    # Concat the protein chain sequences
    df_test["Concat Chain Sequence"] = df_test["Chain A Sequence"] + df_test["Chain B Sequence"]

    # Create a new dataframe for DiffDock inference
    df_test_inference = pd.DataFrame(columns=["complex_name", "protein_path", "ligand_description", "protein_sequence"])
    df_test_inference["ligand_description"] = df_test["CXSMILES"]
    df_test_inference["protein_sequence"] = df_test["Concat Chain Sequence"]
    df_test_inference["complex_name"] = df_test_inference["complex_name"].fillna("")
    df_test_inference["protein_path"] = df_test_inference["protein_path"].fillna("")

    # Write df to csv file
    df_test_inference.to_csv(output_file, index=False)


if __name__ == "__main__":
    generate_test_data_csv()
