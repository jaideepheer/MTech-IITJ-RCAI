import torch


def de_normalize_image(image: torch.Tensor):
    image = image - image.min()
    image = image * (image.max() - image.min())
    return image
