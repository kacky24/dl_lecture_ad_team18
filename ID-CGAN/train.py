import argparse
import os
import torch
from torch.autograd import Variable
from data_loader import get_data_loader
from models import Generator, Discriminator, VggTransformar


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def main(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load data_loader
    input_dir = args.input_dir
    target_dir = args.target_dir
    batch_size = args.batch_size
    data_loader = get_data_loader(input_dir, target_dir, batch_size,
                                  shuffle=True, num_workers=6)

    # build model
    G = Generator()
    D = Discriminator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrained_models',
                        help='path for saving trained models')
    parser.add_argument('--input_dir', type=str,
                        default='../dataset/reverse_snow',
                        help='path for snow image directory')
    parser.add_argument('target_dir', type=str,
                        default='../dataset/original',
                        help='path for origial image directory')
    parser.add_argument('--batch_size', type=int, dafault=7)

    args = parser.parse_args()

    main(args)
