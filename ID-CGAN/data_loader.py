import os
from torch.utils.data import Dataset, DataLoader
import transformer
import cv2


class SnowDataset(Dataset):
    '''snow dataset'''
    def __init__(self, input_dir, target_dir, transform=None):
        '''
        Args:
            input_dir: directory for snow images
            target_dir: directory for original images
        '''
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.img_names = os.listdir(input_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, ix):
        img_name = self.img_names[ix]
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)

        input_img = cv2.imread(input_path)
        target_img = cv2.imread(target_path)
        if self.transform is not None:
            input_img, target_img = self.transform(input_img, target_img)

        return input_img, target_img


def get_data_loader(input_dir, target_dir, batch_size,
                    transform=None, shuffle=False, num_workers=0):
    '''return data loader'''
    if transform is None:
        transform = transformer.Compose([
            transformer.Rescale((256, 256)),
            transformer.ToTensor()
            ])

    snow_dataset = SnowDataset(input_dir, target_dir, transform)

    data_loader = DataLoader(dataset=snow_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader
