"""
A script to train and test models for all the tissues
Example usage:
python train_all_tissues.py -exp ../output/output_all_tissues_nn -nep 50 -testep 50 -arch fully_connected --cuda True
python train_all_tissues.py -exp ../output/output_all_tissues_conv -nep 50 -testep 50 -arch conv --cuda True
"""

import os
from tqdm import tqdm
from parser_util import get_parser
import utils


def main():
    options = get_parser().parse_args()
    opt = {}
    for arg in vars(options):
        opt[arg] = getattr(options, arg)

    split_dir = os.path.join(options.dataset_root, "tabula_muris_split")
    tissues = os.listdir(split_dir)

    for tissue in tqdm(tissues):
        full_dir = os.path.join(options.experiment_root, tissue)
        utils.mkdir_p(full_dir)
        opt["experiment_root"] = full_dir
        opt["split_file"] = os.path.join(split_dir, tissue, "split.json")
        command = "python train.py"
        for arg in opt:
            command += " --{} {}".format(arg, opt[arg])
        print(command)
        os.system(command)


if __name__ == '__main__':
    main()

