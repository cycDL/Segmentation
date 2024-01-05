import numpy as np
import cv2
import torch

# test中计算各项指标
class SegmentationMetric(object):
    def __init__(self, numClass):  # 云 云影 背景 三类
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def PixelAccuarcy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

if __name__ == '__main__':



    def reverse_one_hot(image):
        #     image = image.permute(1, 2, 0)  # [2, 512, 512] ==> [512, 512, 2]
        x = torch.argmax(image, dim=-1)  # [512, 512, 2] ==> [512, 512]
        return x


    def Pixel_Accuracy(SR, GT):
        SR = SR.flatten()  # flatten() 默认按照行将一个维度 [[1,0],[1,1]]  --->[1,0,1,1]
        GT = GT.flatten()
        corr = torch.sum(SR == GT)
        tensor_size = SR.size(0)
        PA = float(corr) / float(tensor_size)
        return PA

    # 单张图 预测指标
    predict_path = 'C:/Users/admin/Desktop/zhibiao/miou_0.890000.png'
    label_path = 'C:/Users/admin/Desktop/zhibiao/3459.png'

    imgPredict = cv2.imread(predict_path, 1)
    imgPredict = cv2.cvtColor(imgPredict, cv2.COLOR_BGR2RGB)
    imgPredict = torch.tensor(imgPredict)
    imgPredict = reverse_one_hot(imgPredict)  # 去通道数

    imgLabel = cv2.imread(label_path, 1)
    imgLabel = cv2.cvtColor(imgLabel, cv2.COLOR_BGR2RGB)
    imgLabel = torch.tensor(imgLabel)
    imgLabel = reverse_one_hot(imgLabel)

    # imgPredict = np.array([0, 0, 1, 1, 2, 2])
    # imgLabel = np.array([0, 0, 1, 1, 2, 2])
    # imgPredict = torch.tensor(imgPredict)
    # imgLabel = torch.tensor(imgLabel)
    print(imgLabel.shape)

    PA = Pixel_Accuracy(imgPredict, imgLabel)

    metric = SegmentationMetric(3)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.PixelAccuarcy()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    MPA = metric.meanPixelAccuracy()
    print(acc, mIoU, FWIoU, MPA, PA)


    #######################################################

    def union_classes(SR, GT):
        eval_cl, _ = extract_classes(SR)
        gt_cl, _ = extract_classes(GT)

        cl = torch.unique(torch.cat([eval_cl, gt_cl]).view(-1))
        n_cl = len(cl)
        return cl, n_cl


    def extract_classes(GT):
        cl = torch.unique(GT)
        n_cl = len(cl)

        return cl, n_cl


    def segm_size(segm):
        try:
            height = segm.shape[0]
            width = segm.shape[1]
        except IndexError:
            raise

        return height, width


    def extract_masks(segm, cl, n_cl):
        h, w = segm_size(segm)
        masks = torch.zeros((n_cl, h, w))
        for i, c in enumerate(cl):
            masks[i, :, :] = segm == c

        return masks


    def extract_both_masks(SR, GT, cl, n_cl):
        eval_mask = extract_masks(SR, cl, n_cl)
        gt_mask = extract_masks(GT, cl, n_cl)

        return eval_mask, gt_mask


    def mean_IU(SR, GT):

        cl, n_cl = union_classes(SR, GT)
        _, n_cl_gt = extract_classes(GT)
        eval_mask, gt_mask = extract_both_masks(SR, GT, cl, n_cl)

        IU = torch.FloatTensor(list([0]) * n_cl)

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]
            if (torch.sum(curr_eval_mask) == 0) or (torch.sum(curr_gt_mask) == 0):
                continue

            n_ii = torch.sum((curr_eval_mask == 1) & (curr_gt_mask == 1))
            t_i = torch.sum(curr_gt_mask)
            n_ij = torch.sum(curr_eval_mask)

            IU[i] = n_ii / (t_i + n_ij - n_ii)

        miou = torch.sum(IU) / n_cl_gt
        return miou


    miou = mean_IU(imgPredict, imgLabel)
    print(miou)