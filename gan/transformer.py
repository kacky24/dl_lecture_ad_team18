import torch
import cv2


class ToTensor:
    '''ndarray -> torch.Tonsor'''
    def __init__(self):
        pass

    def __call__(self, input_img, target_img):
        input_img = input_img.transpose(2, 0, 1)
        target_img = target_img.transpose(2, 0, 1)
        input_img = torch.from_numpy(input_img)
        target_img = torch.from_numpy(target_img)
        return input_img, target_img


class Normalize:
    '''0 - 255 -> -1 - 1'''
    def __init__(self):
        pass

    def __call__(self, input_img, target_img):
        input_img = input_img / 127.5 - 1
        target_img = target_img / 127.5 - 1
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
