import os, torch, time
from torch.utils.data import DataLoader
from evaluation import *
# from model.SpCSAMNet.CSAMNet import CSAMNet
from loaddata import cloud_cloud_shadow
from utils import *
from torchvision import transforms
# from Segmentation.model.network.deeplabv3plus_resnet import resnet101
# from model.HRNet.HRNet import get_seg_model
# from model.unet_model import UNet
from model.Unet import Unet
import warnings
import yaml

from utils.utils import reverse_one_hot

'''
推理测试.py
'''
model_path = 'G:/experiment/fanhua/SegNet/epoch_215'
# with open('H:/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', encoding="utf-8") as f:
#     cfg = yaml.load(f, yaml.FullLoader)
# model = get_seg_model(cfg=cfg).cuda()

# model = SegNet(3,2).cuda()
model = Unet(num_class=3).cuda()

print('loading model......')

########################## 加载模型 ############################

try:
    pretrained_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(pretrained_dict)

    print('Successful ！')

except:

    print('----- defeat')


########################### 加载验证集 ##########################

val_path_img = 'H:/code/code_practice/Segmentation/fanhua/val/cloudshadow'
val_path_label = 'H:/code/code_practice/Segmentation/fanhua/val/labels'
csv_path = 'H:/code/code_practice/Segmentation/fanhua/class_dict.csv'


transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    ])

dataset_val = cloud_cloud_shadow(val_path_img,
                                 val_path_label,
                                 csv_path,
                                 mode='val',
                                 transform=transform)

dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,  # 必须为1
        shuffle=False,
        # num_workers=0
    )




def val(model, dataloader_val):
    print('Val...')
    start = time.time()
    #with torch.no_grad() 数据不需要计算梯度，也不会进行反向传播
    # 主要是用于停止autograd模块的工作，以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。
    with torch.no_grad():
        model.cuda()
        #model.eval()主要用于通知dropout层和batchnorm层在train和val模式间切换
        #在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新。
        #在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
        model.eval()
        PA_all = []
        MPA_all = []
        FWIoU_all = []
        miou_all= []
        metric = SegmentationMetric(3)   # 3类混淆矩阵
        for i, (img, label) in enumerate(dataloader_val):
            print('i=%d'%i)
            img, label = img.cuda(), label.cuda()
            predict = model(img)
            predict = predict.squeeze()     # [1, n_cl, h, w] ==> [n_cl, h, w]
            predict = reverse_one_hot(predict)

            label = label.squeeze()
            label = reverse_one_hot(label)

            metric.addBatch(predict, label)    # 加载混淆矩阵

            pa = Pixel_Accuracy(predict, label)
            MPA = metric.meanPixelAccuracy()
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
            miou = mean_IU(predict, label)

            PA_all.append(pa)
            MPA_all.append(MPA)
            FWIoU_all.append(FWIoU)
            miou_all.append(miou)

        # compute average value
        pa = np.mean(PA_all)
        mpa = np.mean(MPA_all)
        fwiou = np.mean(FWIoU_all)
        miou = np.mean(miou_all)

        s = ("%15s;" * 4) % ("PA", "MPA", "FWIoU", "Miou")
        with open('result.txt', 'a') as file:
            file.write(s + '\n')

        jilu = ('%15.5g;'*4)  % ( pa, mpa, fwiou, miou)
        # 将验证的结果记录在 result.txt文件中
        with open('result.txt', 'a') as file:
            file.write(jilu + '\n')
        # 打印结果在Run区---可视化
        print('PA:      {:}'.format(pa))
        print('MPA:     {:}'.format(mpa))
        print('FWIoU:   {:}'.format(fwiou))
        print('MIoU:    {:}'.format(miou))
        print('Time:    {:}'.format(time.time() - start))

    return miou




val(model, dataloader_val)
