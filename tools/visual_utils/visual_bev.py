import os.path

import numpy as np
import matplotlib.pyplot as plt

def vis_pred_label_bev(pred,label,save_path,frame_id):
    save_path = save_path + '/vis/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    B,_,_ = pred.shape
    pred = pred.cpu()
    label = label.cpu()
    for i in range(B):
        frame = frame_id[i]
        save_frame_path = save_path + str(frame) + '.png'
        pred_frame = pred[i]
        label_frame = label[i]
        pred_bev = np.zeros((320,320,3))
        label_bev = np.zeros((320,320,3))
        pred_bev = draw_bev_image(pred_frame,pred_bev)
        label_bev = draw_bev_image(label_frame, label_bev)
        plt.subplot(121)
        plt.imshow(pred_bev)
        plt.subplot(122)
        plt.imshow(label_bev)
        # plt.show()
        plt.savefig(save_frame_path)
        plt.clf()
def draw_bev_image(pred_frame,pred_bev):
    xx,yy = np.where(pred_frame!=0)
    for index in range(len(xx)):
        if pred_frame[xx[index],yy[index]]==1:
            pred_bev[xx[index],yy[index],0]=1
        elif pred_frame[xx[index],yy[index]]==2:
            pred_bev[xx[index],yy[index],1]=1
        elif pred_frame[xx[index],yy[index]]==3:
            pred_bev[xx[index],yy[index],2]=1
        # pass
    return pred_bev
    # pass