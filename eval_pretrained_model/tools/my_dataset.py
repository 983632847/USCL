import os
import random
import pickle
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)

class COVIDDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        """
        POCUS Dataset
            param data_dir: str
            param transform: torch.transform
        """
        self.label_name = {"covid19": 0, "pneumonia": 1, "regular": 2}
        with open(data_dir, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
        if train:
            self.X, self.y = X_train, y_train       # [N, C, H, W], [N]
        else:
            self.X, self.y = X_test, y_test         # [N, C, H, W], [N]
        self.transform = transform
    
    def __getitem__(self, index):
        img_arr = self.X[index].transpose(1,2,0)    # CHW => HWC
        img = Image.fromarray(img_arr.astype('uint8')).convert('RGB') # 0~255
        label = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.y)