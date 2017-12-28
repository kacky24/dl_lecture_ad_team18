import os
import json
import cv2


def main():
    input_dir = '../dataset/snow'

    img_names = os.listdir(input_dir)
    valid_img_list = []
    for name in img_names:
        img = cv2.imread(os.path.join(input_dir, name))
        if img is not None and img.shape[2] == 3:
            valid_img_list.append(name)

    print(len(valid_img_list))

    with open('../dataset/valid_img_list.json', 'w') as f:
        json.dump(valid_img_list, f)


if __name__ == '__main__':
    main()
