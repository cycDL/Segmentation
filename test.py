import pandas as pd
import numpy as np
import torch
import cv2
from torchvision import transforms
import warnings
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
from tqdm import tqdm
import time
import os
from model.Unet import Unet
from model.segnet import SegNet

# model_path = 'G:/experiment/fanhua/SP_CSANet/miou_0.933543.pth'
model_path = './checkpoints/SegNet/miou_0.790502.pth'
csv_path = './data/class_dict.csv'
read_path = './data/val/cloudshadow/'
save_path = './demo/SegNet_test/'

tests_path = os.listdir(read_path)
print(tests_path)

# model = Unet(3).cuda()
model_name = 'SegNet'  # 要改
model = SegNet(3, 3).cuda()  # 要改
epoch = 1


def get_label_info(csv_path):
    data = pd.read_csv(csv_path)
    label = {}

    for _, row in data.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label


def one_hot_it(label, label_info = get_label_info(csv_path)):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    image = image.permute(1, 2, 0)  # [2, 512, 512] ==> [512, 512, 2]
    x = torch.argmax(image, dim=-1)  # [512, 512, 2] ==> [512, 512]
    return x


def colour_code_segmentation(image, label_values):
    label_values = [label_values[key] for key in label_values]  # [[128, 0, 0], [0, 128, 0], [0, 0, 0]]
    colour_codes = np.array(label_values)  # [[128   0   0][  0 128   0][  0   0   0]]
    x = colour_codes[image.astype(int)]    # 索引取值
    return x


def predict_on_image(model, epoch, csv_path, read_path, save_path):
    for test_path in tests_path:
        save_pre_path = save_path + test_path.split('.')[-2] + model_name + '_test.png'  # 批量保存
        image = cv2.imread(read_path + test_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 读取为BGR 这里要转RGB

        # image = Image.fromarray(image)
        image = transforms.ToTensor()(image).unsqueeze(0)

        # #read csv label path
        label_info = get_label_info(csv_path)

        # predict
        model.eval()
        predict = model(image.cuda()).squeeze()
        predict = reverse_one_hot(predict)  # (h,w)一张图上面每一个像素点对应分类好的数字
        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
        cv2.imwrite(save_pre_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # from torchsummary import summary
    # from tqdm import tqdm
    # import time

    warnings.filterwarnings('ignore')  # 忽略结果中警告语句

    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)

    # UNet = Unet(3).cuda()
    summary(model, input_size=(3, 224, 224))


    predict_on_image(model, epoch, csv_path, read_path, save_path)
###############################################################################
    # test 100张图
    pbar = tqdm(total=100)
    for j in range(10):
        for i in range(10):
            pbar.update(1)
            time.sleep(0.1)
        pbar.close()

##############################################################################

    # pbar = tqdm(total=100)
    # for i in range(100):
    #     pbar.update(1)
    #     time.sleep(0.1)
    # pbar.close()
    #
###############################################################################

    # try:
    #     print(a)
    # except:
    #     print("An exception was raised")
    # else:
    #     print("Thank God, no exceptions were raised.")


###############################################################################
    # a = 1132132
    # print('{}'.format(a))
    # print('%5d'%a)


# read_path = './demo/ceshi.png'
# image = cv2.imread(read_path, 1)
#
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.imshow(image)
# plt.show()
#
# cv2.imshow('fig', image)
# cv2.waitKey(0)
#
# image = Image.open(read_path)
# image.show()
# cv2.waitKey(0)



