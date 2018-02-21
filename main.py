import os
import torch

from opts import get_opts
from trainer import train
from prepro import get_data


def main():
    opt = get_opts()
    opt.use_cuda = torch.cuda.is_available()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    if opt.mode == 'train':
        corpus = get_data(opt)
        train(opt, corpus)

    # if opt.mode == 'generate':
    #     model = load_model()
    #     generate(opt, model)


if __name__ == '__main__':
    main()
