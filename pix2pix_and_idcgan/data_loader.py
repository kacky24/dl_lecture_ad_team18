import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import transformer
import cv2


class SnowDataset(Dataset):
    '''snow dataset'''
    def __init__(self, input_dir, target_dir, img_list_path, transform=None):
        '''
        Args:
            input_dir: directory for snow images
            target_dir: directory for original images
        '''
        self.input_dir = input_dir
        self.target_dir = target_dir
        with open(img_list_path, 'r') as f:
            self.img_names = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, ix):
        img_name = self.img_names[ix]
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)

        input_img = cv2.imread(input_path).astype(np.float32)
        target_img = cv2.imread(target_path).astype(np.float32)
        if self.transform is not None:
            input_img, target_img = self.transform(input_img, target_img)

        return input_img, target_img


def get_data_loader(input_dir, target_dir, img_list_path, batch_size,
                    transform=None, shuffle=True, num_workers=0):
    '''return data loader'''
    if transform is None:
        transform = transformer.Compose([
            transformer.Rescale((286, 286)),
            transformer.RandomCrop((256, 256)),
            transformer.RandomHorizontalFlip(),
            transformer.Normalize(),
            transformer.ToTensor()
            ])

    snow_dataset = SnowDataset(input_dir, target_dir, img_list_path, transform)

    data_loader = DataLoader(dataset=snow_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    input_dir = '../dataset/snow'
    target_dir = '../dataset/original'
    img_list_path = '../dataset/valid_img_list.json'
    data_loader = get_data_loader(input_dir, target_dir, img_list_path, 7)

    for i, (input_img, target_img) in enumerate(data_loader):
        if target_img.numpy().shape != (7, 3, 256, 256):
            print(input_img.numpy().shape)
        if i == 100:
            break
