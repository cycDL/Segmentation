import cv2
from imgaug import augmenters as iaa
from PIL import Image
import torchvision.transforms as transforms
from utils.utils import get_label_info, reverse_one_hot, colour_code_segmentation
import numpy as np
import os
import torch
"""
马占明的test
"""
def predict_on_image(model, args, epoch, csv_path):
    # pre-processing on image
    test_list = os.listdir(args.test_path)

    for i in test_list:
        image = cv2.imread(args.test_path + i, -1)    # 读入一张图片数据，图片为BGR形式的数组
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # 将BGR图片数组 转换为RGB图片数组
        resize = iaa.Resize({'height': args.crop_height, 'width': args.crop_width})  # 剪裁尺寸
        resize_det = resize.to_deterministic()     # 保持坐标和图像同步改变，而不是随机
        image = resize_det.augment_image(image)       # 保存变换后的图片
        image = Image.fromarray(image).convert('RGB')   # 将数组转换为RGB图片
        image = transforms.ToTensor()(image).unsqueeze(0)   # 标准化后，在第0维增加一个维度

        # read csv label path
        label_info = get_label_info(csv_path)

        # predict
        model.eval()
        # 单loss输出
        predict = model(image.cuda())

        # 多loss输出 bisenetv2在后面加【0】  predict = model(image.cuda())
        # predict = model(image.cuda())[0]

        w = predict.size()[-1]
        # bisenetv2   w =predict.size()[-1]    加个[0]
        c = predict.size()[-3]
        predict = predict.resize(c, w, w)
        predict = reverse_one_hot(predict)     # 此处返回的是HWC 最后一维处最大值的序号，用来判断像素点的颜色

        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 对每一个像素点进行分类，得到分类后的图片数据
        predict = cv2.resize(np.uint8(predict), (args.crop_height, args.crop_width))  # 数据类型转换为unit8，， uint8为无符号整型数据, 范围是从0–255
        save_path = f'/{i[:-4]}_epoch_%d.png' % (epoch)
        cv2.imwrite('demo/'+args.save_model_path + save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        # cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


def predict1( model,test_path,size, load_state_dict,csv_path,save_path):

    test_list  = os.listdir(test_path)
    for i in test_list:
        image = cv2.imread(test_path + i, -1)    # 读入一张图片数据，图片为BGR形式的数组
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # 将BGR图片数组 转换为RGB图片数组

        image = Image.fromarray(image).convert('RGB')   # 将数组转换为RGB图片
        image = transforms.ToTensor()(image).unsqueeze(0)   # 标准化后，在第0维增加一个维度
        #read csv label path
        label_info = get_label_info(csv_path)
        # predict
        # model.eval()
        model.load_state_dict(torch.load(load_state_dict))
        model.eval()
        predict = model(image)

        w =predict.size()[-1]
        c =predict.size()[-3]
        predict = predict.resize(c,w,w)
        predict = reverse_one_hot(predict)     # 此处返回的是HWC 最后一维处最大值的序号，用来判断像素点的颜色

        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 对每一个像素点进行分类，得到分类后的图片数据
        predict = cv2.resize(np.uint8(predict), (size, size))   # 数据类型转换为unit8，， uint8为无符号整型数据, 范围是从0–255
        save_path1 = f'/{i[:-4]}.png'
        cv2.imwrite(save_path + save_path1, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # from models.ABC_74 import ABCNet
    from model.segnet import SegNet
    test_path = 'E:\seg\datasets\zzzz'
    load_state_dict = 'E:\seg\checkpoints\ 74 build  adam 7.8  0.001   adam\miou_0.8475428847616725_epoch_61.pth'
    model = SegNet(3, 3)  # test不用cuda
    # model = ABCNet(classes=2)
    size = 224
    csv_path = f'E:\seg\datasets\\build\class_dict.csv'
    save_path = './data/val/segNet_test'

    predict1(model=model, test_path = test_path, load_state_dict = load_state_dict, size=size, csv_path=csv_path, save_path=save_path)
