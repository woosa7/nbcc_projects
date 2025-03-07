import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from classification.data_loader import get_loaders
from classification.trainer import Trainer
from classification.model_loader import get_model

"""
# Train from scratch
python main.py --model_fn model_resnet.pt --model_name resnet --dataset_name catdog

# Train from pretrained weights
python main.py --model_fn model_resnet.pt --model_name resnet --dataset_name catdog --use_pretrained

# Train with freezed pretrained weights
python main.py --model_fn model_resnet.pt --model_name resnet --dataset_name catdog --use_pretrained --freeze
"""

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.6)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--test_ratio', type=float, default=.2)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model_name', type=str, default='resnet')   # resnet, vgg, alexnet, squeezenet, densenet
    p.add_argument('--dataset_name', type=str, default='catdog')  # or hymenoptera (ant/bee)
    p.add_argument('--n_classes', type=int, default=2)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--use_pretrained', action='store_true')

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    if config.verbose >= 2:
        print(config)

    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    model, input_size = get_model(config)
    model = model.to(device)

    train_loader, valid_loader, test_loader = get_loaders(config, input_size)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 3:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
