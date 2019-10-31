"""
A script to test models for all the tissues
Example usage:
python test_all_tissues_full.py -exp ../output/output_all_tissues_nn -testep 50 -arch fully_connected --cuda True
python test_all_tissues_full.py -exp ../output/output_all_tissues_conv -testep 50 -arch conv --cuda True
"""

import os
from tqdm import tqdm
from parser_util import get_parser


def main():
    options = get_parser().parse_args()
    opt = {}
    for arg in vars(options):
        opt[arg] = getattr(options, arg)

    tissues = os.listdir(options.experiment_root)
    split_dir = os.path.join(options.dataset_root, "tabula_muris_split")

    for tissue in tqdm(tissues):
        full_dir = os.path.join(options.experiment_root, tissue)
        opt["experiment_root"] = full_dir
        opt["split_file"] = os.path.join(split_dir, tissue, "split.json")
        command = "python eval_full.py"
        for arg in opt:
            command += " --{} {}".format(arg, opt[arg])
        print(command)
        try:
            os.system(command)
        except Exception as e:
            print("Skipping {} due to {}".format(tissue, e))


if __name__ == '__main__':
    main()