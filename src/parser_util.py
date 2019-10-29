# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='../../data/tabula_muris')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-testep', '--test_epochs',
                        type=int,
                        help='number of epochs to test for',
                        default=20)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=20',
                        default=20)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=15)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5. Setting this to a very '
                             'high number will use all available classes in validation.',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)

    parser.add_argument('-cTe', '--classes_per_it_test',
                        type=int,
                        help='number of random classes per episode for test, default=5. Setting this to a very '
                             'high number will use all available classes in validation.',
                        default=5)

    parser.add_argument('-nsTe', '--num_support_test',
                        type=int,
                        help='number of samples per class to use as support for test, default=5',
                        default=5)

    parser.add_argument('-nqTe', '--num_query_test',
                        type=int,
                        help='number of samples per class to use as query for test, default=15',
                        default=15)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        type=bool,
                        help='enables cuda',
                        default=False)

    parser.add_argument('-gid', '--gpu_id',
                        type=int,
                        help='cuda dvice id',
                        default=0)

    parser.add_argument('-arch', '--nn_architecture',
                        type=str,
                        help='Support convolutional (conv) or fully connected network (fully_connected).',
                        default='fully_connected')

    parser.add_argument('-split', '--split_file',
                        type=str,
                        help='File that defines train, test, val split.',
                        default=None)

    parser.add_argument('-res', '--test_result_file',
                        type=str,
                        help='File that the test results are written to.',
                        default='test_metrics.txt')

    return parser
