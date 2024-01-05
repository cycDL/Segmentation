import os, argparse, torch
from loaddata import cloud_cloud_shadow
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from train import train
# from Segmentation.model.SpCSAMNet_new.CSAMNet import CSAMNet
# from model.pspnet_2.pspnet import PSPNet
# from Segmentation.model.Resnet_Unet.Resnet_Unet import Resnet_Unet
# from Segmentation.model.HRNet.HRNet import get_seg_model
# from Segmentation.model.network.deeplabv3plus_resnet import resnet101
# from model.unet_model import UNet

# from model.segnet import SegNet
# from model.Unet import Unet
# from model.PSPNet import PSPNet
from model.ppliteseg import PPLiteSeg

from torchvision import transforms
import warnings
# import yaml

warnings.filterwarnings('ignore')  # 忽略结果中警告语句


def main(params):
    parser = argparse.ArgumentParser('cloud and cloud shadow Segmentation')   # 创建解析器
    # 添加参数
    parser.add_argument('--num_epochs',             type=int,   default=100,        help='None')
    parser.add_argument('--num_epoch_decay',        type=int,   default=70,         help='change lr')
    parser.add_argument('--checkpoint_step',        type=int,   default=5,          help='save model for every X time')
    parser.add_argument('--validation_step',        type=int,   default=1,          help='check model for every X time')
    parser.add_argument('--batch_size',             type=int,   default=16,          help='None')  # 224*224应该能放16
    parser.add_argument('--num_workers',            type=int,   default=2,          help='None')
    parser.add_argument('--lr',                     type=float, default=0.001,     help='None')
    parser.add_argument('--lr_scheduler',           type=int,   default=3,          help='Update the learning rate every X times')
    parser.add_argument('--lr_scheduler_gamma',     type=float, default=0.99,       help='learning rate attenuation coefficient')
    parser.add_argument('--warmup',                 type=int,   default=1,          help='warm up')
    parser.add_argument('--warmup_num',             type=int,   default=1,          help='warm up the number')
    parser.add_argument('--cuda',                   type=str,   default='0',        help='GPU ids used for training')
    parser.add_argument('--beta1',                  type=float, default=0.5,        help='momentum1 in Adam')
    parser.add_argument('--beta2',                  type=float, default=0.999,      help='momentum2 in Adam')
    parser.add_argument('--miou_max',               type=float, default=0.85,       help='If Miou greater than it ,will be saved and update it')
    parser.add_argument('--crop_height',            type=int,   default=224,        help='None')  # 512
    parser.add_argument('--crop_width',             type=int,   default=224,        help='None')
    parser.add_argument('--pretrained_model_path',  type=str,   default=None,       help='None')
    parser.add_argument('--save_model_path',        type=str,   default="./checkpoints/",   help='path to save model')
    parser.add_argument('--data',                   type=str,   default='./data/',      help='path of training data')
    parser.add_argument('--log_path',               type=str,   default='./log/',           help='path to save the log')
    parser.add_argument('--test_path',              type=str,   default='./checkpoints/test_model', help='path to save model')
    # 解析参数
    args = parser.parse_args(params)

    tb = PrettyTable(['Index', 'Key', 'Value'])
    args_str = str(args)[10:-1].split(',')  # 从（）内的内容开始以‘，’分隔作为list

    for i, key_value in enumerate(args_str):
        key, value = key_value.split('=')[0], key_value.split('=')[1]
        tb.add_row([i + 1, key, value])
    print(tb)

    # 检测是否有参数列表中的 save_model_path参数下对应的路径："./checkpoints/"
    # 检测是否有参数列表中的 log_path参数下对应的路径："./log/"
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # 创建训练数据集和验证集的（image和label）的路径
    train_path_img = os.path.join(args.data, 'train/cloudshadow')
    train_path_label = os.path.join(args.data, 'train/labels')
    val_path_img = os.path.join(args.data, 'val/cloudshadow')
    val_path_label = os.path.join(args.data, 'val/labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')

    # 训练数据集的数据增强
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    ])

    # 训练数据加载
    dataset_train = cloud_cloud_shadow(train_path_img,
                                  train_path_label,
                                  csv_path,
                                  mode='train',
                                  transform=transform)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 验证数据加载
    dataset_val = cloud_cloud_shadow(val_path_img,
                                    val_path_label,
                                    csv_path,
                                    mode='val',
                                     transform=transform)

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,  # 必须为1
        shuffle=False,
        num_workers=args.num_workers
    )

    # 在 PyTorch 程序开头将其值设置为 True，就可以大大提升卷积神经网络的运行速度。
    torch.backends.cudnn.benchmark = True

    # 设置模型和参数
    # model = CSAMNet(num_class=3).cuda()
    # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=3, zoom_factor=8, use_ppm=True,
    #                    pretrained=False).cuda()
    # model = Resnet_Unet(3).cuda()
    # model = resnet101(n_class=3, output_stride=16, pretrained=False).cuda()

    model = PPLiteSeg(3).cuda()

    # model = SegNet(3, 3).cuda()
    # with open('H:/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml', encoding="utf-8") as f:
    #     cfg = yaml.load(f, yaml.FullLoader)
    # model = get_seg_model(cfg=cfg).cuda()

    # 优化器,学习率下降策略
    optimizer = torch.optim.Adam(model.parameters(), args.lr, [args.beta1, args.beta2])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler, gamma=args.lr_scheduler_gamma)

    # 如果存在预训练好的模型，加载预训练模型
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        # loading the part of network params
        pretrained_dict = torch.load(args.pretrained_model_path)  # 加载path文件
        model_dict = model.state_dict()  # 加载模型参数结构
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        print('Done!')

    # 开始训练
    train(args, model, optimizer, dataloader_train, dataloader_val, exp_lr_scheduler)


if __name__ == '__main__':
    params = [
        '--num_epochs', '150',
        '--batch_size', '16',  # 最重要
        '--lr', '0.001',
        '--warmup', '2',
        '--lr_scheduler_gamma', '0.95',
        '--lr_scheduler', '3',
        '--miou_max', '0.6',
        # '--crop_height', '224'
        # '--crop_width' , '224'
        # '--pretrained_model_path', 'G:/pretrained_model/resnet18-5c106cde.pth'
        # '--cuda', '7,2',  # model put in the cuda[0]
        # '--pretrained_model_path', './checkpoints/miou_0.857.pth'
    ]

    s = ("%15s;" * 6) % ("epoch", "loss", "PA", "MPA", "FWIoU", "Miou")
    with open('result.txt', 'a') as file:
        file.write(s + '\n')

    main(params)