import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loader import get_data_loader
from models import pix2pix_model
from utils import monitor_output_image
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def save_checkpoint(state, filename):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.ckpt')


def main(args):
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    monitor_path = args.monitor_path
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    # load data_loader
    input_dir = args.input_dir
    target_dir = args.target_dir
    img_list_path = args.img_list_path
    batch_size = args.batch_size
    data_loader = get_data_loader(input_dir, target_dir, img_list_path,
                                  batch_size, shuffle=True, num_workers=6)

    # build model
    G = pix2pix_model.Generator(3, 3)
    D = pix2pix_model.Discriminator70(6)
    if torch.cuda.is_available():
        G = G.cuda()
        D = D.cuda()

    criterion_l1 = nn.L1Loss()
    criterion_bce = nn.BCELoss()
    if torch.cuda.is_available():
        criterion_l1 = criterion_l1.cuda()
        criterion_bce = criterion_bce.cuda()
    lambda_a = args.lambda_a
    lambda_l1 = args.lambda_l1

    # optimizer
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002,
                                   betas=(0.5, 0.999), weight_decay=0.00001)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002,
                                   betas=(0.5, 0.999), weight_decay=0.00001)

    # train
    epoch_num = args.epoch_num
    total_step = len(data_loader)

    statement = "Epoch [%d/%d], Step [%d/%d], G_Loss: %.4f, " + \
                "g_loss_a: %.4f, g_loss_l1: %.4f, D_Loss: %.4f"

    for epoch in range(1, epoch_num + 1):
        for i, (input_imgs, target_imgs) in enumerate(data_loader):
            input_imgs = to_var(input_imgs)
            target_imgs = to_var(target_imgs)

            # generate images
            generated_imgs = G(input_imgs)

            # update discriminator
            optimizer_D.zero_grad()
            negative_examples = D(input_imgs, generated_imgs.detach())
            positive_examples = D(input_imgs, target_imgs)

            zero_labels = Variable(torch.zeros(negative_examples.size()))
            one_labels = Variable(torch.ones(positive_examples.size()))
            if torch.cuda.is_available():
                zero_labels = zero_labels.cuda()
                one_labels = one_labels.cuda()

            d_loss = 0.5 * (
                criterion_bce(negative_examples, zero_labels) +
                criterion_bce(positive_examples, one_labels)
                )
            d_loss.backward()
            optimizer_D.step()

            # update generator
            optimizer_G.zero_grad()
            negative_examples = D(input_imgs, generated_imgs)
            g_loss_a = criterion_bce(negative_examples, one_labels)
            g_loss_l1 = criterion_l1(generated_imgs, target_imgs)
            g_loss = lambda_a * g_loss_a + lambda_l1 * g_loss_l1
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(statement % (epoch, epoch_num, i + 1, total_step,
                      g_loss.data.mean(), g_loss_a.data.mean(),
                      g_loss_l1.data.mean(), d_loss.data.mean()))

        # save
        if epoch % 4 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict_g': G.state_dict(),
                'state_dict_d': D.state_dict(),
                'optimizer_g': optimizer_G.state_dict(),
                'optimizer_d': optimizer_D.state_dict()
                }, os.path.join(model_path, '%03d.ckpt' % (epoch)))

            # monitor
            g_imgs = generated_imgs.cpu().data.numpy()
            # t_imgs = target_imgs.cpu().data.numpy()
            i_imgs = input_imgs.cpu().data.numpy()
            monitor_img = monitor_output_image(g_imgs[0], i_imgs[0])
            cv2.imwrite(os.path.join(monitor_path, 'monitor_img%03d.jpg')
                        % (epoch,), monitor_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='pretrained_models/pix2pix/all',
                        help='path for saving trained models')
    parser.add_argument('--monitor_path', type=str,
                        default='monitor_images/pix2pix/all',
                        help='path for saving monitor images')
    parser.add_argument('--input_dir', type=str,
                        default='../dataset/train_all/snow',
                        help='path for snow image directory')
    parser.add_argument('--target_dir', type=str,
                        default='../dataset/train_all/original',
                        help='path for origial image directory')
    parser.add_argument('--img_list_path', type=str,
                        default='../dataset/valid_img_list3.json',
                        help='path for valid_img_list')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch_num', type=int, default=600)
    parser.add_argument('--lambda_a', type=float, default=0.01,
                        help='coefficient for adversarial loss')
    parser.add_argument('--lambda_l1', type=float, default=1,
                        help='coefficient for per-pixel loss')

    args = parser.parse_args()

    main(args)
