import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        transform = Compose([
            Rescale((256, 256)),
            ToTensor()
            ])

    snow_dataset = SnowDataset(input_dir, target_dir, transform)

    data_loader = DataLoader(dataset=snow_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader


class ToTensor:
    '''ndarray -> torch.Tonsor'''
    def __init__(self):
        pass

    def __call__(self, input_img, target_img):
        transform = transforms.ToTensor()
        input_img = transform(input_img)
        target_img = transform(target_img)
        return input_img, target_img


class Rescale:
    '''Rescale the image to a given size
    Args:
        output_size(int or tuple)
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, input_img, target_img):
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        input_img = cv2.resize(input_img, (new_h, new_w))
        target_img = cv2.resize(target_img, (new_h, new_w))

        return input_img, target_img


class Compose:
    """
    Composes several transforms together.
    Args:
        transforms: list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_img, target_img):
        for t in self.transforms:
            input_img, target_img = t(input_img, target_img)
        return input_img, target_img
