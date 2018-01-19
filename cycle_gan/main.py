import argparse
import os
from model import cyclegan

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint', help='models are saved here')
parser.add_argument('--data_dir', dest='dataset_dir', default='data', help='path of the dataset')
parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epoch')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--max_train_size', dest='max_train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=100, help='sampling fake images from test_dir')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--sample_dir', dest='sample_dir', default='sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test', help='test sample are saved here')
parser.add_argument('--weight_dir', dest='weight_dir', default='weight', help='weights are loaded from here')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')

args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        model.train() if args.phase == 'train' \
            else model.test()

if __name__ == '__main__':
    tf.app.run()
