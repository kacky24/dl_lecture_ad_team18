import numpy as np


def monitor_output_image(generated_img, target_img):
    generated_img = generated_img.transpose(1, 2, 0)
    target_img = target_img.transpose(1, 2, 0)
    cat_img = np.concatenate((target_img, generated_img), axis=1)
    return denormalize(cat_img)


def denormalize(img):
    return (img + 1) * 127.5


if __name__ == '__main__':
    import cv2
    img1 = cv2.imread('../dataset/original/9265.jpg')
    # img2 = cv2.imread('../dataset/original/8526.jpg')
    print(monitor_output_image(img1, img1).shape)
    print(monitor_output_image(img1, img1).max())
    print(monitor_output_image(img1, img1).min())
