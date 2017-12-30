import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loader import get_data_loader
from models import idcgan_model
from utils import monitor_output_image
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

    # load data_loader
    input_dir = args.input_dir
    target_dir = args.target_dir
    img_list_path = args.img_list_path
    batch_size = args.batch_size
    data_loader = get_data_loader(input_dir, target_dir, img_list_path,
                                  batch_size, shuffle=True, num_workers=6)

    # build model
    G = idcgan_model.GeneratorInLuaCode(3, 3)
    D = idcgan_model.DiscriminatorInLuaCode(6)
    if torch.cuda.is_available():
        G = G.cuda()
        D = D.cuda()

    # loss
    vgg_model = idcgan_model.VggTransformar()
    criterion_mse = nn.L1Loss()
    criterion_bce = nn.BCELoss()
    if torch.cuda.is_available():
        vgg_model = vgg_model.cuda()
        criterion_mse = criterion_mse.cuda()
        criterion_bce = criterion_bce.cuda()
    lambda_a = args.lambda_a
    lambda_e = args.lambda_e
    lambda_p = args.lambda_p

    # optimizer
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.002,
                                   betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002,
                                   betas=(0.5, 0.999))
    # optimizer_D = torch.optim.SGD(D.parameters(), lr=0.0002,
    #                               momentum=0.9, weight_decay=0.0005)

    # train
    epoch_num = args.epoch_num
    total_step = len(data_loader)

    statement = "Epoch [%d/%d], Step [%d/%d], G_Loss: %.4f, g_loss_a: %.4f, g_loss_e: %.4f, g_loss_p: %.4f, D_Loss: %.4f"

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
            g_loss_e = criterion_mse(generated_imgs, target_imgs)
            g_loss_p = criterion_mse(vgg_model(generated_imgs),
                                     Variable(vgg_model(target_imgs).data,
                                              requires_grad=False)
                                     )
            g_loss = lambda_a * g_loss_a + lambda_e * g_loss_e + \
                lambda_p * g_loss_p
            g_loss.backward()
            optimizer_G.step()

            # print log
            # if i % 100 == 0:
            #     print("Epoch [%d/%d], Step [%d/%d], G_Loss: %.4f, D_Loss: %.4f"
            #           % (epoch, epoch_num, i + 1, total_step,
            #               g_loss.data[0], d_loss.data[0]))

            if i % 100 == 0:
                print(statement % (epoch, epoch_num, i + 1, total_step,
                      g_loss.data[0], g_loss_a.data[0], g_loss_e.data[0],
                      g_loss_p.data[0], d_loss.data[0]))

        # save
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
        cv2.imwrite('monitor_images/monitor_img%03d.jpg'
                    % (epoch,), monitor_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pretrained_models',
                        help='path for saving trained models')
    parser.add_argument('--input_dir', type=str,
                        default='../dataset/snow',
                        help='path for snow image directory')
    parser.add_argument('--target_dir', type=str,
                        default='../dataset/original',
                        help='path for origial image directory')
    parser.add_argument('--img_list_path', type=str,
                        default='../dataset/valid_img_list.json',
                        help='path for valid_img_list')
    parser.add_argument('--batch_size', type=int, default=7)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lambda_a', type=float, default=6.6e-3,
                        help='coefficient for adversarial loss')
    parser.add_argument('--lambda_e', type=float, default=1,
                        help='coefficient for per-pixel loss')
    parser.add_argument('--lambda_p', type=float, default=1,
                        help='coefficient for perceptual loss')

    args = parser.parse_args()

    main(args)
