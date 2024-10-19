import pickle
from pcdet.utils.box_utils import in_hull, boxes_to_corners_3d
train_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_train.pkl'
test_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_test.pkl'
trainval_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_trainval.pkl'
val_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_val.pkl'


with open(train_pkl,'rb') as f:
    train_data = pickle.load(f)

with open(test_pkl,'rb') as f:
    test_data = pickle.load(f)

with open(trainval_pkl,'rb') as f:
    trainval_data = pickle.load(f)

with open(val_pkl,'rb') as f:
    val_data = pickle.load(f)
print(' ')
Car_box_num = 0
Ped_box_num = 0
Cyc_box_num = 0
Car_box_height = 0
Ped_box_height = 0
Cyc_box_height = 0
height_list = []
height_class = []
for i in range(len(train_data)):
    # i = 1287
    data_frame = train_data[i]
    boxes_names = data_frame['annos']['name']
    gt_boxes = data_frame['annos']['gt_boxes_lidar']
    num_points_in_gt = data_frame['annos']['num_points_in_gt']
    box_mask = []
    for j in range(len(boxes_names)):
        if boxes_names[j] =='Car' or boxes_names[j] =='Pedestrian' or  boxes_names[j] =='Cyclist':
            box_mask.append(True)
        else:
            box_mask.append(False)
    if gt_boxes.shape[0]!=0:
        gt_boxes = gt_boxes[box_mask]
        # num_points_in_gt = num_points_in_gt[box_mask]
        boxes_names = boxes_names[box_mask]
    else:
        continue

    for index in range(len(boxes_names)):
        if boxes_names[index]=='Car':
            Car_box_num+=1
            Car_box_height += gt_boxes[index,2]
            height_list.append(gt_boxes[index,2])
            height_class.append(1)
        elif boxes_names[index]=='Pedestrian':
            Ped_box_num+=1
            Ped_box_height += gt_boxes[index, 2]
            height_list.append(gt_boxes[index, 2])
            height_class.append(2)
        else:
            Cyc_box_num+=1
            Cyc_box_height += gt_boxes[index, 2]
            height_list.append(gt_boxes[index, 2])
            height_class.append(3)
av_car_height = Car_box_height/Car_box_num
av_ped_height = Ped_box_height/Ped_box_num
av_cyc_height = Cyc_box_height/Cyc_box_num
print(f'average car height:{av_car_height}')
print(f'average ped height:{av_ped_height}')
print(f'average cyc height:{av_cyc_height}')
print('done!')

# coding=utf-8
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(height_class,height_list)
plt.show()