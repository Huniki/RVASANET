import cv2
import pickle
import numpy as np
from pcdet.utils.box_utils import in_hull, boxes_to_corners_3d
from pcdet.utils.calibration_kitti import Calibration


def draw_box(box_class,image_data,p1,p2,p3,p4,p5,p6,p7,p8):
    colors = {1:(255,255,0),2:(0,255,0),3:(255,0,255)}
    cv2.line(image_data,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),color=colors[box_class],thickness=2)
    cv2.line(image_data, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data,(int(p3[0]),int(p3[1])),(int(p4[0]),int(p4[1])),color=colors[box_class],thickness=2)
    cv2.line(image_data, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data, (int(p5[0]), int(p5[1])), (int(p6[0]), int(p6[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data, (int(p6[0]), int(p6[1])), (int(p7[0]), int(p7[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data, (int(p7[0]), int(p7[1])), (int(p8[0]), int(p8[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data,(int(p8[0]),int(p8[1])),(int(p5[0]),int(p5[1])),color=colors[box_class],thickness=2)
    cv2.line(image_data, (int(p1[0]), int(p1[1])), (int(p5[0]), int(p5[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data, (int(p2[0]), int(p2[1])), (int(p6[0]), int(p6[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data, (int(p3[0]), int(p3[1])), (int(p7[0]), int(p7[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data, (int(p8[0]), int(p8[1])), (int(p4[0]), int(p4[1])), color=colors[box_class], thickness=2)
    return image_data

val_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_val.pkl'
with open(val_pkl,'rb') as f:
    val_data = pickle.load(f)

for i in range(len(val_data)):
    # i = 0
    data_frame = val_data[i]
    pts_index = data_frame['point_cloud']['lidar_idx']
    image_index = data_frame['image']['image_idx']
    pts2cam = data_frame['calib']['Tr_velo_to_cam']
    cam_ins = data_frame['calib']['P2']
    _ = data_frame['calib']['R0_rect']

    # 旋转平移矩阵接口搭建
    pts2cam = pts2cam[:3,:]
    cam_ins = cam_ins[:3,:]
    _ = _[:3,:3]
    radar_calib = {'Tr_velo2cam':pts2cam,'P2':cam_ins,'R0':_}
    calib = Calibration(radar_calib)
    # 读取点云
    pts_root_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/training/velodyne/'
    pts_path = pts_root_path + pts_index + '.bin'
    # Radar
    pts_data = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 7)[:,:3]
    # Lidar
    # pts_data = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    # 读取图片
    image_root_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/training/image_2/'
    image_path = image_root_path + image_index + '.jpg'
    image_data = cv2.imread(image_path)
    # 读取box，并过滤
    boxes_names = data_frame['annos']['name']
    gt_boxes = data_frame['annos']['gt_boxes_lidar']
    box_mask = []
    boxes_class = []
    for i in range(len(boxes_names)):
        if boxes_names[i] =='Car' or boxes_names[i] =='Pedestrian' or  boxes_names[i] =='Cyclist':
            box_mask.append(True)
            if boxes_names[i] =='Car':
                boxes_class.append(1)
            elif boxes_names[i] =='Pedestrian':
                boxes_class.append(2)
            else:
                boxes_class.append(3)
        else:
            box_mask.append(False)
    gt_boxes = gt_boxes[box_mask]
    # 将box转换为顶点
    gt_boxes_corners = boxes_to_corners_3d(gt_boxes)
    # 过滤框外的点  # box框的坐标系转换
    file_pts_list = []
    boxes_img = []
    pts_class = []
    for i in range(len(gt_boxes_corners)):
        gt_box_corners = gt_boxes_corners[i]
        box_class = boxes_class[i]
        box_img, box_depth = calib.lidar_to_img(gt_box_corners)
        file_pts = pts_data[in_hull(pts_data,gt_box_corners)]
        file_pts_list.append(file_pts)
        for idx in range(len(file_pts)):
            pts_class.append(box_class)
        boxes_img.append(box_img)
    file_pts = np.concatenate(file_pts_list,axis=0)
    # 点云坐标系转换
    pts_img, pts_depth = calib.lidar_to_img(file_pts)
    # 将点画在图片上！
    colors = {1:(255,255,0),2:(0,255,0),3:(255,0,255)}
    for i in range(len(pts_img)):
        # Lidar
        # cv2.circle(image_data,(int(pts_img[i][0]),int(pts_img[i][1])),1,color=colors[pts_class[i]],thickness=2)

        # Radar
        cv2.circle(image_data, (int(pts_img[i][0]), int(pts_img[i][1])), 3, color=colors[pts_class[i]], thickness=6)
        # cv2.imshow('image_ret',image_data)
        # cv2.waitKey()
        # cv2.destroyWindow()
    # for box_img in boxes_img:
    for box_index in range(len(boxes_img)):
        box_img = boxes_img[box_index]
        p1,p2,p3,p4,p5,p6,p7,p8 = box_img[0],box_img[1],box_img[2],box_img[3],box_img[4],box_img[5],box_img[6],box_img[7]
        box_class = boxes_class[box_index]
        image_data = draw_box(box_class,image_data,p1,p2,p3,p4,p5,p6,p7,p8)

    # cv2.imshow('image_ret',image_data)
    # cv2.waitKey()
    # cv2.destroyWindow()
    save_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/lidar2cam/'+ pts_index
    cv2.imwrite(save_path+'.png',image_data)
    print('done')

