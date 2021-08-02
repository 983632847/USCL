import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter

class Sharpen(object):
    """ Sharpen an image before inputing it to networks
    Args:
        degree (int): The sharpen intensity, from -1 to 5. 
                    0 represents original image.
    """
    def __init__(self, degree=0):
        self.degree = degree

    def __call__(self, img):
        """
        Args:
            img (PIL image): input image
        Returns:
            img (PIL image): sharpened output image
        """
        if self.degree == -1:
            img = img.filter(ImageFilter.Kernel((3,3),(-1, -1/2, -1,
                                                       -1/2, 3, -1/2,
                                                       -1, -1/2, -1))) # 比原图还模糊
        elif self.degree == 0: # 原图
            pass
        elif self.degree == 1:
            img = img.filter(ImageFilter.Kernel((3,3),(1, -2, 1,
                                                       -2, 5, -2,
                                                       1, -2, 1))) # 很弱，几乎没有什么锐化
        elif self.degree == 2:
            img = img.filter(ImageFilter.Kernel((3,3),(0, -2/7, 0,
                                                       -2/7, 19/7, -2/7,
                                                       0, -2/7, 0))) # 能看出一丁点锐化
        elif self.degree == 3:
            img = img.filter(ImageFilter.Kernel((3,3),(0, -1, 0,
                                                       -1, 5, -1,
                                                       0, -1, 0))) # 锐化较明显
        elif self.degree == 4:
            img = img.filter(ImageFilter.Kernel((3,3),(-1, -1, -1,
                                                       -1, 9, -1,
                                                       -1, -1, -1))) # 锐化很明显
        elif self.degree == 5:
            img = img.filter(ImageFilter.Kernel((3,3),(-1, -4, -1,
                                                       -4, 21, -4,
                                                       -1, -4, -1))) # 最强
        else:
            raise ValueError('The degree must be integer between -1 and 5')

        return img
        
        
        
        
        
        
        
        
        
        
        
        