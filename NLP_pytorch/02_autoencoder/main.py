import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from trainer import Trainer
from utils import load_mnist
from model import Autoencoder

"""
AutoEncoder : MNIST
"""

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)  # model save filename

    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=70)
    p.add_argument('--verbose', type=int, default=1)

    p.add_argument('--latent_size', type=int, default=2)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    # prepare data
    train_x, train_y = load_mnist(flatten=True)
    test_x, test_y = load_mnist(is_train=False, flatten=True)

    train_cnt = int(train_x.size(0) * config.train_ratio)
    valid_cnt = train_x.size(0) - train_cnt

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(train_x.size(0))

    train_x, valid_x = torch.index_select(
        train_x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)

    train_y, valid_y = torch.index_select(
        train_y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)

    print("Train:", train_x.shape, train_y.shape)
    print("Valid:", valid_x.shape, valid_y.shape)
    print("Test:", test_x.shape, test_y.shape)

    # create model
    model = Autoencoder(config.latent_size).to(device)
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.MSELoss()

    # training
    trainer = Trainer(model, optimizer, loss_func)
    trainer.train((train_x, train_x), (valid_x, valid_x), config)

    # save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
