import torch


def de_normalize_image(image: torch.Tensor, min=-1.0, max=1.0):
    image = image - min
    image = image / (max - min)
    return image
