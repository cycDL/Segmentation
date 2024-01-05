import glob
import os
import torch
from PIL import Image
from utils.utils import one_hot_it, get_label_info
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
# from utils import one_hot_it, get_label_info
# import matplotlib.pyplot as plt

class cloud_cloud_shadow(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, csv_path, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.img_list = glob.glob(os.path.join(img_path, '*.png'))
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))
        self.label_info = get_label_info(csv_path)  # {'same': [0, 0, 0], 'different': [255, 255, 255]}
        self.to_tensor = transforms.ToTensor()
        self.transforms = transform

    def __getitem__(self, index):

        img = Image.open(self.img_list[index]).convert('RGB')  # 这里.convert('RGB')可加可不加
        label = Image.open(self.label_list[index]).convert('RGB')
        if self.transforms:
            label = self.transforms(label)
            img = self.transforms(img)
        else:
            transform = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224), ])
            label = transform(label)
            img = transform(img)
        label = one_hot_it(label, self.label_info).astype(np.uint8)  # 将原来输出的布尔值转换为8位无符号整型
        label = np.transpose(label, [2, 0, 1]).astype(np.uint8)
        img = self.to_tensor(img).float()
        label = torch.from_numpy(label)  # 把 numpy变成tensor,tensor改 numpy也会改

        return img, label

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # from utils import one_hot_it, get_label_info

    # 改为224
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224)])
    data = cloud_cloud_shadow('./data/val/cloudshadow/',
                              './data/val/labels/',
                              './data/class_dict.csv',
                              transform=transform)

    dataloader_test = DataLoader(
        data,
        batch_size=1,  # 为啥像测试集载入数据
        shuffle=False,
        num_workers=0
    )

    for i, (img, label) in enumerate(dataloader_test):
        print(img.shape)
        if i >= 0:
            break
