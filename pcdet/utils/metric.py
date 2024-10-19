import torch
import numpy as np



# 对每一行计算混淆矩阵
def _fast_hist(row_label, row_image, n_class):
    mask = (row_label>= 0) & (row_label < n_class)
    # print(mask)
    # 关键步骤
    # print('row_label:{} row_image:{}'.format(row_label[mask], row_image[mask]))
    # tmp = n_class * row_label[mask].astype(int) + row_image[mask]  # 这句不理解 ？？
    # print('tmp:', tmp)
    hist = np.bincount(
        n_class * row_label[mask].astype(int) + row_image[mask], minlength=n_class ** 2).reshape(n_class, n_class)  # 关键句？？
    # print('single_hist: \n', hist)
    return hist
def get_hist(hist,pred,label,n_class):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    for single_image,single_label in zip(pred, label):  # 取出 batch 中的一张图片进行计算(这里 batchsize=1 方便计算)
        for row_image,row_label in zip(single_image,single_label):  # 取出 single_image 和 single_label 中的一行
            # 每一行得到的混淆矩阵累加
            hist += _fast_hist(row_label.flatten(), row_image.flatten(), n_class)
            # print('sum_hist:\n{} \n\n'.format(hist))

    return hist

def get_miou(hist):
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.mean(iu)
    return miou


if __name__=='__main__':
    out = torch.tensor([[[[0.4, -0.45, 0, 3.4],
                          [0.2, -0.5, 3.44, 3.4],
                          [0.43, -0.05, 0.76, -0.4],
                          [-0.11, -0.15, 0.67, 2.4]],

                         [[1.4, -0.45, 0, -2.44],
                          [-0.4, -0.45, 0, 0.14],
                          [3.4, 0.3, 0.9, 1.4],
                          [0.24, 0.46, 2.0, 0.44]],

                         [[0.44, -0.95, 0.88, 0.74],
                          [0.56, 0.58, -0.09, 0.84],
                          [-0.37, -0.445, 0.66, 1.4],
                          [0.73, 0.15, 0, -3.4]]]])
    label = torch.tensor([[[0, 0, 2, 1],
                           [1, 2, 0, 0],
                           [2, 1, 2, 1],
                           [2, 2, 0, 1]]])
    _, index = out.max(dim=1)  # 注意：index 并非混淆矩阵，只是每个像素点的类别，混淆矩阵应为（classes, classes）即（3, 3）
    # label = label.numpy()
    # index = index.numpy()
    #
    hist = np.zeros((3, 3))  # 混淆矩阵
    hist = get_hist(hist,index,label,3)
    # for single_image,single_label in zip(index, label):  # 取出 batch 中的一张图片进行计算(这里 batchsize=1 方便计算)
    #     for row_image,row_label in zip(single_image,single_label):  # 取出 single_image 和 single_label 中的一行
    #         # 每一行得到的混淆矩阵累加
    #         hist += _fast_hist(row_label.flatten(), row_image.flatten(), 3)
    #         # print('sum_hist:\n{} \n\n'.format(hist))

    print(hist)
    # 根据混淆矩阵计算iou
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    print(f'miou={np.mean(iu)}')

