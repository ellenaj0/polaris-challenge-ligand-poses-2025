import logging
from argparse import Namespace
import copy
import os
from functools import partial
import warnings
import yaml

# Ignore pandas deprecation warning around pyarrow
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="(?s).*Pyarrow will become a required dependency of pandas.*")

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs

# TODO imports are a little odd, utils seems to shadow things
from utils.logging_utils import configure_logger, get_logger
import utils.utils
from datasets.process_mols import write_mol_with_coords
from utils.download import download_and_extract
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm

import sys
from src.utils import get_logger

if os.name != 'nt':  # The line does not work on Windows
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings("ignore", category=UserWarning,
                        message="The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`")

# Prody logging is very verbose by default
prody_logger = logging.getLogger(".prody")
prody_logger.setLevel(logging.ERROR)

REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")
REMOTE_URLS = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
               f"{REPOSITORY_URL}/releases/download/v1.1/diffdock_models.zip"]


def inference_diffdock(dataset_df: pd.DataFrame, args):

    configure_logger(loglevel=args.loglevel, logfile=args.logfile)
    logger = get_logger()

    # Download models if they don't exist locally
    if not os.path.exists(args.model_dir):
        logger.info(f"Models not found. Downloading")
        remote_urls = REMOTE_URLS
        downloaded_successfully = False
        for remote_url in remote_urls:
            try:
                logger.info(f"Attempting download from {remote_url}")
                files_downloaded = download_and_extract(remote_url, os.path.dirname(args.model_dir))
                if not files_downloaded:
                    logger.info(f"Download from {remote_url} failed.")
                    continue
                logger.info(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
                downloaded_successfully = True
                # Once we have downloaded the models, we can break the loop
                break
            except Exception as e:
                pass

        if not downloaded_successfully:
            raise Exception(f"Models not found locally and failed to download them from {remote_urls}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"DiffDock will run on {device}")

    # Use the dataset passed in to get protein paths and ligand descriptions
    protein_path_list = dataset_df['protein_path'].tolist()
    ligand_description_list = dataset_df['cxsmiles'].tolist()

    complex_name_list = [f"complex_{i}" for i in range(len(dataset_df))]
    for name in complex_name_list:
        write_dir = f'{args.out_dir}/{name}'
        os.makedirs(write_dir, exist_ok=True)

    # preprocessing of complexes into geometric graphs
    test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                    ligand_descriptions=ligand_description_list, protein_sequences=None,   # TODO: Add protein sequences to dataset passed in?
                                    lm_embeddings=True,
                                    receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                    c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                    all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                    atom_max_neighbors=score_model_args.atom_max_neighbors,
                                    knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
        logger.info('Confidence model uses different type of graphs than the score model. '
                    'Loading (or creating if not existing) the data for the confidence model now.')
        confidence_test_dataset = \
            InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                             ligand_descriptions=ligand_description_list, protein_sequences=None,   # TODO: Add protein sequences to dataset passed in?
                             lm_embeddings=True,
                             receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                             c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                             all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                             atom_max_neighbors=confidence_args.atom_max_neighbors,
                             precomputed_lm_embeddings=test_dataset.lm_embeddings,
                             knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
    else:
        confidence_test_dataset = None

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.confidence_model_dir is not None:
        confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                     confidence_mode=True, old=args.old_confidence_model)
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None

    tr_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')

    failures, skipped = 0, 0
    N = args.samples_per_complex
    test_ds_size = len(test_dataset)
    logger.info(f'Size of test dataset: {test_ds_size}')
    for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
        if not orig_complex_graph.success[0]:
            skipped += 1
            logger.warning(f"The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
            continue
        try:
            if confidence_test_dataset is not None:
                confidence_complex_graph = confidence_test_dataset[idx]
                if not confidence_complex_graph.success:
                    skipped += 1
                    logger.warning(f"The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                    continue
                confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
            else:
                confidence_data_list = None
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max,
                               initial_noise_std_proportion=args.initial_noise_std_proportion,
                               choose_residue=args.choose_residue)

            lig = orig_complex_graph.mol[0]

            # initialize visualisation
            pdb = None
            if args.save_visualisation:
                visualization_list = []
                for graph in data_list:
                    pdb = PDBFile(lig)
                    pdb.add(lig, 0, 0)
                    pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                    pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                    visualization_list.append(pdb)
            else:
                visualization_list = None

            # run reverse diffusion
            print("Running reverse diffusion...")
            data_list, confidence = sampling(data_list=data_list, model=model,
                                             inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                             tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                             device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                             visualization_list=visualization_list, confidence_model=confidence_model,
                                             confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                             batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise,
                                             temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot,
                                                            args.temp_sampling_tor],
                                             temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                             temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot,
                                                              args.temp_sigma_data_tor])

            # This array contains the predicted ligand positions for all ranks
            ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

            # reorder predictions based on confidence output
            if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                confidence = confidence[:, 0]
            if confidence is not None:
                confidence = confidence.cpu().numpy()
                re_order = np.argsort(confidence)[::-1]
                confidence = confidence[re_order]
                ligand_pos = ligand_pos[re_order]

            # save predictions
            write_dir = f'{args.out_dir}/{complex_name_list[idx]}'
            # Copy the original ligand and modify it according to the predicted positions, then save it
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                if score_model_args.remove_hs: mol_pred = RemoveAllHs(mol_pred)
                if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

            # save visualisation frames
            if args.save_visualisation:
                if confidence is not None:
                    for rank, batch_idx in enumerate(re_order):
                        visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
                else:
                    for rank, batch_idx in enumerate(ligand_pos):
                        visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

        except Exception as e:
            logger.warning("Failed on", orig_complex_graph["name"], e)
            failures += 1

    result_msg = f"""
    Failed for {failures} / {test_ds_size} complexes.
    Skipped {skipped} / {test_ds_size} complexes.
"""
    if failures or skipped:
        logger.warning(result_msg)
    else:
        logger.info(result_msg)
    logger.info(f"Results saved in {args.out_dir}")
