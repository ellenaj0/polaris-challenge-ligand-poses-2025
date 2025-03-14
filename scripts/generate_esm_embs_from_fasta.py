import pickle
import torch
from esm import pretrained

from utils.inference_utils import compute_ESM_embeddings
from tqdm import tqdm


def generate_esm_embeddings_from_fasta():
    """
    Generate ESM embeddings from the fasta file and save to a .pt file.
    """
    
    with open("data/prepared_for_esm.fasta", "rb") as f:
        file = pickle.load(f)
        names = list(file.keys())

        print("Generating ESM language model embeddings")
        model_location = "esm2_t33_650M_UR50D"
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        protein_sequences = list(file.values())
        labels, sequences = [], []
        for i in tqdm(range(len(protein_sequences))):
            print(protein_sequences[i])
            s = protein_sequences[i].split(':')
            sequences.extend(s)
            labels.extend([f"{names[i][:8]}" + '_chain_' + str(j) for j in range(len(s))])

        lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)

        torch.save(lm_embeddings, "data/lm_embeddings.pt")

if __name__ == "__main__":
    generate_esm_embeddings_from_fasta()
