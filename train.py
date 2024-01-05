import tqdm
import time
import numpy
import torch.nn
import torch
from tensorboardX import SummaryWriter
from utils import *
from evaluation import *
from utils.utils import reverse_one_hot, predict_on_image


def val(model, dataloader_val, epoch, loss_train_mean, writer, csv_path):
    print('Val...')
    start = time.time()
    # 梯度不优化，不影响dropout和BN层的行为。
    with torch.no_grad():
        model.cuda()
        # model.eval()主要用于通知dropout层和BN层在train和val模式间切换
        # 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); BN层会继续计算数据的mean和var等参数并更新。
        # 在val模式下，dropout层会让所有的激活单元都通过，而BN层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
        model.eval()
        PA_all = []
        MPA_all = []
        FWIoU_all = []
        miou_all = []
        metric = SegmentationMetric(3)  # 3类混淆矩阵
        for i, (img, label) in enumerate(dataloader_val):
            img, label = img.cuda(), label.cuda()
            predict = model(img)
            predict = predict.squeeze()  # [1, n_cl, h, w] ==> [n_cl, h, w]
            predict = reverse_one_hot(predict)

            label = label.squeeze()
            label = reverse_one_hot(label)

            metric.addBatch(predict, label)  # 加载混淆矩阵

            pa = Pixel_Accuracy(predict, label)
            MPA = metric.meanPixelAccuracy()
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
            miou = mean_IU(predict, label)

            PA_all.append(pa)
            MPA_all.append(MPA)
            FWIoU_all.append(FWIoU)
            miou_all.append(miou)

        #################输出预测图片#############################
        read_path = './demo/ceshi.png'
        save_path = './demo/epoch_%d.png' % (epoch + 1)

        if (epoch + 1) % 1 == 0:
            predict_on_image(model, epoch, csv_path, read_path, save_path)

        # compute average value
        pa = np.mean(PA_all)
        mpa = np.mean(MPA_all)
        fwiou = np.mean(FWIoU_all)
        miou = np.mean(miou_all)

        jilu = ('%15.5g;' * 6) % (epoch + 1, loss_train_mean, pa, mpa, fwiou, miou)
        # 将验证的结果记录在 result.txt文件中
        with open('result.txt', 'a') as file:
            file.write(jilu + '\n')
        # 打印结果在Run区---可视化
        print('PA:      {:}'.format(pa))
        print('MPA:     {:}'.format(mpa))
        print('FWIoU:   {:}'.format(fwiou))
        print('MIoU:    {:}'.format(miou))
        print('Time:    {:}'.format(time.time() - start))

        # 写进log
        writer.add_scalar('{}_PA'.format('val'), pa, epoch + 1)
        writer.add_scalar('{}_MPA'.format('val'), mpa, epoch + 1)
        writer.add_scalar('{}_FWIoU'.format('val'), fwiou, epoch + 1)
        writer.add_scalar('{}_MIoU'.format('val'), miou, epoch + 1)

        return miou


def train(args, model, optimizer, dataloader_train, dataloader_val, exp_lr_scheduler):
    print('Train...')
    miou_max = args.miou_max
    writer = SummaryWriter(logdir=args.log_path)

    for epoch in range(args.num_epochs):
        model.train()
        exp_lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch + 1, lr))
        loss_record = []

        for i, (img, label) in enumerate(dataloader_train):
            img, label = img.cuda(), label.cuda()
            # 在起步的时候减小学习率防止数据震荡（因为开始的时候参数是随机初始化的）
            # if args.warmup == 1 and epoch == 0:
            #     lr = args.lr / (len(dataloader_train) - i)
            #     tq.set_description('epoch %d, lr %f' % (epoch + 1, lr))
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr

            output = model(img)
            label = torch.argmax(label, dim=1)

            loss = torch.nn.CrossEntropyLoss()(output, label)

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # computer mean loss
            loss_record.append(loss.item())
        tq.close()

        loss_train_mean = np.mean(loss_record)
        print('Loss for train :{:.6f}'.format(loss_train_mean))

        # written to the log
        writer.add_scalar('{}_loss'.format('train'), loss_train_mean, epoch + 1)  # 可视化工具里添加标量
        # 保存模型的参数字典
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            save_path = args.save_model_path + 'epoch_{:}'.format(epoch)
            torch.save(model.state_dict(), save_path)

        if epoch % args.validation_step == 0:
            csv_path = './data/class_dict.csv'
            miou = val(model, dataloader_val, epoch, loss_train_mean, writer, csv_path)

            if miou > miou_max:
                save_path = args.save_model_path + 'miou_{:.6f}.pth'.format(miou)
                torch.save(model.state_dict(), save_path)
                miou_max = miou
                predict_on_image(model, epoch, csv_path,
                                 read_path='E:/work/WorkSpace/AI-CV/Segmentation_Qu/miou_max/whitecity/3459.png',
                                 save_path='E:/work/WorkSpace/AI-CV/Segmentation_Qu/miou_max/whitecity/miou_%f.png' % (
                                     miou_max))

    writer.close()
    save_path = args.save_model_path + 'last.path'
    torch.save(model.state_dict(), save_path)
