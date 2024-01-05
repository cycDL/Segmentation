import pandas as pd
import numpy as np
import torch
from imgaug import augmenters as iaa
import cv2
from PIL import Image
from torchvision  import transforms

csv_path = './data/class_dict.csv'

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

def one_hot_it(label, label_info=get_label_info(csv_path)):
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
    x = torch.argmax(image, dim=-1) # [512, 512, 2] ==> [512, 512]
    return x


def colour_code_segmentation(image, label_values):
    label_values = [label_values[key] for key in label_values]  # [[128, 0, 0], [0, 128, 0], [0, 0, 0]]
    colour_codes = np.array(label_values)  # [[128   0   0][  0 128   0][  0   0   0]]
    x = colour_codes[image.astype(int)]
    return x


def predict_on_image(model, epoch, csv_path, read_path, save_path):
    image = cv2.imread(read_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #读取为BGR 这里要转RGB
    # image = Image.fromarray(image)
    image = transforms.ToTensor()(image).unsqueeze(0)
    # #read csv label path
    label_info = get_label_info(csv_path)
    # predict
    model.eval()
    predict = model(image.cuda()).squeeze()
    predict = reverse_one_hot(predict)  # (h,w)一张图上面每一个像素点对应分类好的数字
    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
    cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))