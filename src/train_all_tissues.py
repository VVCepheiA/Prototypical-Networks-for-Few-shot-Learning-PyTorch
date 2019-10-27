"""
A script to train and test models for all the tissues
Example usage:
python train_all_tissues.py -exp ../output/output_all_tissues_nn -nep 50 -testep 50 -arch fully_connected --cuda True
python train_all_tissues.py -exp ../output/output_all_tissues_conv -nep 50 -testep 50 -arch conv --cuda True
"""

import os
import pathlib
from tqdm import tqdm
from parser_util import get_parser


def mkdir_p(full_dir):
    """Simulate mkdir -p"""
    if not os.path.exists(full_dir):
        pathlib.Path(full_dir).mkdir(parents=True, exist_ok=True)


def main():
    options = get_parser().parse_args()
    opt = {}
    for arg in vars(options):
        opt[arg] = getattr(options, arg)
    tissues = ['Bladder',
               'Large_Intestine',
               'Thymus',
               'Skin',
               'Lung',
               'Liver',
               'Spleen',
               'Kidney',
               'Tongue',
               'Heart',
               'Pancreas',
               'Brain_Myeloid',
               'Marrow',
               'Mammary_Gland',
               'Limb_Muscle',
               'Brain_Non-Myeloid',
               'Trachea',
               'Fat']
    for tissue in tqdm(tissues):
        full_dir = os.path.join(options.experiment_root, tissue)
        mkdir_p(full_dir)
        opt["experiment_root"] = full_dir
        opt["split_file"] = "../../data/tabula_muris_split/{}/split.json".format(tissue)
        command = "python train.py"
        for arg in opt:
            command += " --{} {}".format(arg, opt[arg])
        print(command)
        os.system(command)


if __name__ == '__main__':
    main()

