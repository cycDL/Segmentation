import cv2, torch
import numpy as np

label_info_green = {'green':[0,255,0]}

label_info_red = {'red':[255,0,0]}

# 定义PA
def Pixel_Accuracy(SR, GT):
    SR = SR.flatten()  #flatten() 默认按照行将一个维度 [[1,0],[1,1]]  --->[1,0,1,1]
    GT = GT.flatten()
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)
    PA = float(corr) / float(tensor_size)
    return PA

# 对比预测和label 定义指标
def ying(predict, label, label_info):
    color = label_info['green']
    equality_label = np.equal(label, color)
    equality_predict = np.equal(predict, color)

    label_class_map = np.all(equality_label, axis=-1).flatten().astype(int)
    predict_class_map = np.all(equality_predict, axis=-1).flatten().astype(int)

    index_False = (predict_class_map == 0)
    predict_class_map[index_False] = 2

    same = sum(label_class_map == predict_class_map)

    element_green = sum(label_class_map == True)

    return same, element_green

def yun(predict, label, label_info):
    color = label_info['red']
    equality_label = np.equal(label, color)
    equality_predict = np.equal(predict, color)

    label_class_map = np.all(equality_label, axis=-1).flatten().astype(int)
    predict_class_map = np.all(equality_predict, axis=-1).flatten().astype(int)

    index_False = (predict_class_map == 0)
    predict_class_map[index_False] = 2

    same = sum(label_class_map == predict_class_map)

    element_red = sum(label_class_map == True)

    return same, element_red


ying_label_path = 'C:/Users/admin/Desktop/edge/label/newying.png'
yun_label_path = 'C:/Users/admin/Desktop/edge/label/newyun.png'

ying_predict_path = 'C:/Users/admin/Desktop/edge/DeepLabV3+/ying.png'
yun_predict_path = 'C:/Users/admin/Desktop/edge/DeepLabV3+/yun.png'


ying_label = cv2.imread(ying_label_path)
ying_label = cv2.cvtColor(ying_label, cv2.COLOR_BGR2RGB)
yun_label = cv2.imread(yun_label_path)
yun_label = cv2.cvtColor(yun_label, cv2.COLOR_BGR2RGB)

ying_predict = cv2.imread(ying_predict_path)
ying_predict = cv2.cvtColor(ying_predict, cv2.COLOR_BGR2RGB)
yun_predict = cv2.imread(yun_predict_path)
yun_predict = cv2.cvtColor(yun_predict, cv2.COLOR_BGR2RGB)

same_ying, element_green = ying(ying_predict, ying_label, label_info_green)

same_yun, element_red = yun(yun_predict, yun_label, label_info_red)

print(same_ying,  element_green)
print('*'*100)
print(same_yun, element_red)
print('*'*100)
edge = (same_ying + same_yun) / (element_green + element_red)
print(edge)


################################################################################

# label_info_green = {'green':[0,255,0]}
# color = label_info_green['green']
#
# equality_label = np.zeros((224, 224))
# label_index = (label == color)
# equality_label = equality_label[label_index]
# print(equality_label)