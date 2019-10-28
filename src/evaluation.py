# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from protonet import ProtoNet
from parser_util import get_parser
from train import init_seed, init_dataloader, init_protonet
from sklearn.metrics import f1_score
from collections import defaultdict
from pprint import pprint
import json

from tqdm import tqdm
import numpy as np
import torch
import os


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:{}'.format(opt.gpu_id) if torch.cuda.is_available() and opt.cuda else 'cpu'
    scores = defaultdict(list)
    res = {}
    for epoch in tqdm(range(opt.test_epochs)):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            scores['accuracy'].append(acc.item())

            # DEBUG
            output_np = model_output.detach().numpy()
            y_np = y.detach().numpy()
            print("acc", acc, acc.item(), type(acc))
            print("Output", output_np, output_np.shape)
            print("y", y_np, y_np.shape)

            scores['macro_f1'].append(f1_score(model_output.detach().numpy(), y.detach().numpy(), average='macro'))
            scores['micro_f1'].append(f1_score(model_output.detach().numpy(), y.detach().numpy(), average='micro'))
            print("acc", acc, acc.item())

    for metric in scores:
        res[metric] = np.mean(scores[metric])
    pprint(res)

    with open(os.path.join(opt.experiment_root, 'test_metrics.txt'), 'w') as f:
        json.dump(res, f)

    return avg_acc


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    test_dataloader = init_dataloader(options, 'test')

    model = init_protonet(options, x_dim=test_dataloader.dataset.get_dim())
    model_path = os.path.join(options.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


if __name__ == '__main__':
    main()
