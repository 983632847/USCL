import torch
import numpy as np
import random

class Outpainting(object):
    """Randomly mask out one or more patches from an image, we only need mask regions.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=5):
        self.n_holes = n_holes

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes outpainting of it.
        """
        c, h, w = img.shape

        new_img = np.random.rand(c, h, w) * 1.0

        for n in range(self.n_holes):
            # length of edges
            block_noise_size_x = w - random.randint(3*w//7, 4*w//7)
            block_noise_size_y = h - random.randint(3*h//7, 4*h//7)
        
            # lower left corner
            noise_x = random.randint(3, w-block_noise_size_x-3)
            noise_y = random.randint(3, h-block_noise_size_y-3)

            # copy the original image
            new_img[:, noise_y:noise_y+block_noise_size_y,
                    noise_x:noise_x+block_noise_size_x] = img[:, noise_y:noise_y+block_noise_size_y,
                                                                noise_x:noise_x+block_noise_size_x]
            new_img = torch.tensor(new_img)
            new_img = new_img.type(torch.FloatTensor)

        return new_img
        # return torch.tensor(new_img)
