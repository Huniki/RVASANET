#可视化初始radar模型、最终radar模型、lidar模型
#初始radar模型：/home/rpf/Pillarseg/OpenPCDet/output/home/rpf/Pillarseg/OpenPCDet/tools/cfgs/kitti_models/vod_pointpillar/raw_radar_det/eval/epoch_no_number/val/default/result.pkl
#最终radar模型：/home/rpf/Pillarseg/OpenPCDet/output/home/rpf/Pillarseg/OpenPCDet/tools/cfgs/kitti_models/pillarseg/pillarseg_radar_4class_celoss_repts_bs4/eval/epoch_95/val/default/result.pkl
#lidar模型：/home/rpf/Pillarseg/OpenPCDet/output/home/rpf/Pillarseg/OpenPCDet/tools/cfgs/kitti_models/vod_pointpillar_lidar/test_lidar_bs4/eval/epoch_no_number/val/default/result.pkl
import cv2
import numpy as np
import pickle
from pcdet.utils.box_utils import in_hull, boxes_to_corners_3d
def pcl2bev(pcl_path, ratio, width, height):
    """
        pcl点云中的一般都是m的坐标，
        ratio: 1m代表几个像素的意思.
        比如width5m ,height,10m的一个范围
    """
    img_width = int(ratio * width)
    img_height = int(ratio * height)
    img = np.zeros((img_height, img_width))
    # 4D radar
    # radar_pts = np.fromfile(str(pcl_path), dtype=np.float32).reshape(-1, 7)
    # Lidar
    radar_pts = np.fromfile(str(pcl_path), dtype=np.float32).reshape(-1, 4)
    radar_pts = radar_pts[:,:3]
    # pcd, pcl_points = read_pcd(pcl_path)
    # colors = np.asarray(pcd.colors) * 255
    # colors = colors.astype("uint8")

    for i, pt in enumerate(radar_pts):
        x, y, z = pt
        # u = int(x*ratio) - (- img_width//2)
        # v = int(y*ratio) - (- img_height//2)
        u = (img_width) - int(x*ratio)
        v = int(y*ratio) - (- img_height//2)

        if (u>=0 and u<= img_width-1) and (v>=0 and v<= img_height-1):
            # cv2.circle(img, (v,u), 1, 255, 1)
            img[u,v] = 255
    # img = np.flip(img, 0)
    # cv2.imshow('bev_image',img)
    # cv2.waitKey()
    # cv2.destroyWindow()
    return img

def draw_box(image_data,corners_list,box_class):
    colors = {1:(255,255,0),2:(0,255,0),3:(255,0,255)}
    p1,p2,p3,p4 = corners_list[0],corners_list[1],corners_list[2],corners_list[3]
    cv2.line(image_data,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),color=colors[box_class],thickness=2)
    cv2.line(image_data, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), color=colors[box_class], thickness=2)
    cv2.line(image_data,(int(p3[0]),int(p3[1])),(int(p4[0]),int(p4[1])),color=colors[box_class],thickness=2)
    cv2.line(image_data, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), color=colors[box_class], thickness=2)
    # cv2.imshow('bev_image',image_data)
    # cv2.waitKey()
    # cv2.destroyWindow()
    return image_data

result_pkl = '/home/rpf/Pillarseg/OpenPCDet/output/home/rpf/Pillarseg/OpenPCDet/tools/cfgs/kitti_models/vod_pointpillar_lidar/test_lidar_bs4/eval/epoch_no_number/val/default/result.pkl'

#原始点云地址读取
radar_pts_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/training/velodyne/'
lidar_pts_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/training/velodyne/'
dir = '/home/rpf/Pillarseg/OpenPCDet/output/vis/lidar/'
ratio = 10
img_width = int(ratio * 51.2)
img_height = int(ratio * 51.2)
with open(result_pkl,'rb') as f:
    result_datas = pickle.load(f)

for index in range(len(result_datas)):
    result_data = result_datas[index]
    frame_id = result_data['frame_id']

    # radar_path = radar_pts_path + frame_id + '.bin'
    lidar_path = lidar_pts_path + frame_id + '.bin'
    # raw_img = pcl2bev(radar_path,ratio,51.2,51.2)
    raw_img = pcl2bev(lidar_path, ratio, 51.2, 51.2)

    raw_img = raw_img[:,:,np.newaxis]
    raw_img = np.concatenate([raw_img,raw_img,raw_img],axis=2)

    result_boxes = result_data['boxes_lidar']
    result_boxes =boxes_to_corners_3d(result_boxes)
    names = result_data['name']
    for i in range(len(names)):
        if names[i] =='Pedestrian':
            box_corners = result_boxes[i]
            box_corners = box_corners[:4]
            corners_list = []
            for i, pt in enumerate(box_corners):
                x, y, z = pt
                # u = int(x*ratio) - (- img_width//2)
                # v = int(y*ratio) - (- img_height//2)
                u = (img_width) - int(x * ratio)
                v = int(y * ratio) - (- img_height // 2)
                corners_list.append((v,u))
            raw_img = draw_box(raw_img,corners_list,2)
        elif names[i] == 'Car':
            box_corners = result_boxes[i]
            box_corners = box_corners[:4]
            corners_list = []
            for i, pt in enumerate(box_corners):
                x, y, z = pt
                # u = int(x*ratio) - (- img_width//2)
                # v = int(y*ratio) - (- img_height//2)
                u = (img_width) - int(x * ratio)
                v = int(y * ratio) - (- img_height // 2)
                corners_list.append((v,u))
            raw_img = draw_box(raw_img,corners_list,1)
        else:
            box_corners = result_boxes[i]
            box_corners = box_corners[:4]
            corners_list = []
            for i, pt in enumerate(box_corners):
                x, y, z = pt
                # u = int(x*ratio) - (- img_width//2)
                # v = int(y*ratio) - (- img_height//2)
                u = (img_width) - int(x * ratio)
                v = int(y * ratio) - (- img_height // 2)
                corners_list.append((v,u))
            raw_img = draw_box(raw_img,corners_list,3)
    save_dir = dir + frame_id + '.png'
    cv2.imwrite(save_dir, raw_img)
    # cv2.imshow('bev_image',raw_img)
    # cv2.waitKey()
    # cv2.destroyWindow()


print('done!')