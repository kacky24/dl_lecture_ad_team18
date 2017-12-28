import os
import json
import cv2


def main():
    input_dir = '../dataset/snow'
    target_dir = '../dataset/original'

    img_names = os.listdir(input_dir)
    valid_img_list_in = []
    for name in img_names:
        img = cv2.imread(os.path.join(input_dir, name))
        if img is not None and img.shape[2] == 3:
            valid_img_list_in.append(name)

    img_names = os.listdir(target_dir)
    valid_img_list_tar = []
    for name in img_names:
        img = cv2.imread(os.path.join(target_dir, name))
        if img is not None and img.shape[2] == 3:
            valid_img_list_tar.append(name)

    valid_img_list = set(valid_img_list_in) & set(valid_img_list_tar)
    valid_img_list = list(valid_img_list)
    print(len(valid_img_list))

    with open('../dataset/valid_img_list.json', 'w') as f:
        json.dump(valid_img_list, f)


if __name__ == '__main__':
    main()
