import argparse
from argparse import Namespace

from src.utils import load_yaml_to_dict
from src.ligand_poses import LigandPoses


def main(args: dict) -> None:
    inference_config_filename = args["inference_config_filename"]
    inference_config_dict = load_yaml_to_dict(inference_config_filename)
    inference_args = Namespace(**inference_config_dict)
    print(inference_args)

    eval_config_filename = args["eval_config_filename"]
    eval_config_dict = load_yaml_to_dict(eval_config_filename)
    eval_args = Namespace(**eval_config_dict)
    print(eval_args)

    ligand_poses = LigandPoses()

    ligand_poses.inference(inference_args)

    ligand_poses.evaluate_rank1(eval_args)
    ligand_poses.evaluate_min_rmsd(eval_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass config files.")
    parser.add_argument("--inference_config_filename", default= "inference_args.yaml",
                        help="Name of the config file in the config directory.")
    parser.add_argument("--eval_config_filename", default="eval_args.yaml",
                        help="Name of the config file in the config directory.")

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)
    main(input_args_dict)
