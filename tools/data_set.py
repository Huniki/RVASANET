import pickle
import numpy as np
from pcdet.utils.box_utils import in_hull, boxes_to_corners_3d
import matplotlib.pyplot as plt
import numpy as np
#
val_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/kitti_infos_val.pkl'
with open(val_pkl,'rb') as f:
    val_data = pickle.load(f)

train_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/kitti_infos_train.pkl'
with open(train_pkl,'rb') as f:
    train_data = pickle.load(f)

Car_box_num = 0
Ped_box_num = 0
Cyc_box_num = 0
Car_box_num_no_pts = 0
Ped_box_num_no_pts = 0
Cyc_box_num_no_pts = 0
Car_box_num_no_pts_our = 0
Ped_box_num_no_pts_our = 0
Cyc_box_num_no_pts_our = 0
max_car = 0
max_ped = 0
max_cyc = 0
# Lidar
car_box_num = [0] * 150
ped_box_num = [0] * 150
cyc_box_num = [0] * 150
# Radar
# car_box_num = [0] * 80
# ped_box_num = [0] * 54
# cyc_box_num = [0] * 73
for i in range(len(val_data)):
    # i = 1287
    data_frame = val_data[i]
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
        num_points_in_gt = num_points_in_gt[box_mask]
        boxes_names = boxes_names[box_mask]

        # 读取点云
        pts_index = data_frame['point_cloud']['lidar_idx']
        pts_root_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/training/velodyne/'
        pts_path = pts_root_path + pts_index + '.bin'
        # pts_data = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 7)[:,:3]
        pts_data = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        # 将box转换为顶点
        gt_boxes_corners = boxes_to_corners_3d(gt_boxes)
        # 过滤框外的点  # box框的坐标系转换
        file_pts_list = []
        for k in range(len(gt_boxes_corners)):
            gt_box_corners = gt_boxes_corners[k]
            file_pts = pts_data[in_hull(pts_data,gt_box_corners)]
            num_pts_in_box = file_pts.shape[0]

            if num_pts_in_box==0:
                if boxes_names[k]=='Car':
                    Car_box_num_no_pts_our+=1
                elif boxes_names[k]=='Pedestrian':
                    Ped_box_num_no_pts_our+=1
                else:
                    Cyc_box_num_no_pts_our+=1
            else:
                if boxes_names[k] == 'Car':
                    max_car = max(max_car, num_pts_in_box)
                    if num_pts_in_box <= 150:
                        car_box_num[num_pts_in_box - 1] += 1
                elif boxes_names[k] == 'Pedestrian':
                    max_ped = max(max_ped, num_pts_in_box)
                    if num_pts_in_box <= 150:
                        ped_box_num[num_pts_in_box - 1] += 1
                else:
                    max_cyc = max(max_cyc, num_pts_in_box)
                    if num_pts_in_box <= 150:
                        cyc_box_num[num_pts_in_box - 1] += 1
            if num_points_in_gt[k]==0:
                if boxes_names[k]=='Car':
                    Car_box_num_no_pts+=1
                elif boxes_names[k]=='Pedestrian':
                    Ped_box_num_no_pts+=1
                else:
                    Cyc_box_num_no_pts+=1

            if boxes_names[k] == 'Car':
                Car_box_num += 1
            elif boxes_names[k] == 'Pedestrian':
                Ped_box_num += 1
            else:
                Cyc_box_num += 1
print(f'Car_box_num:{Car_box_num}')
print(f'Ped_box_num:{Ped_box_num}')
print(f'Cyc_box_num:{Cyc_box_num}')
print(f'Car_box_num_no_pts:{Car_box_num_no_pts}')
print(f'Ped_box_num_no_pts:{Ped_box_num_no_pts}')
print(f'Cyc_box_num_no_pts:{Cyc_box_num_no_pts}')
print(f'Car_box_num_no_pts_our:{Car_box_num_no_pts_our}')
print(f'Ped_box_num_no_pts_our:{Ped_box_num_no_pts_our}')
print(f'Cyc_box_num_no_pts_our:{Cyc_box_num_no_pts_our}')
print(f'max_car:{max_car}')
print(f'max_ped:{max_ped}')
print(f'max_cyc:{max_cyc}')

car_box_num_fl = []
ped_box_num_fl = []
cyc_box_num_fl = []
for idx in range(len(car_box_num)):
#     car_box_num_fl.append(car_box_num[idx])
# for idx in range(len(ped_box_num)):
#     ped_box_num_fl.append(ped_box_num[idx])
# for idx in range(len(cyc_box_num)):
#     cyc_box_num_fl.append(cyc_box_num[idx])


    if car_box_num[idx] !=0:
        car_box_num_fl.append(car_box_num[idx])
    if ped_box_num[idx] !=0:
        ped_box_num_fl.append(ped_box_num[idx])
    if cyc_box_num[idx] !=0:
        cyc_box_num_fl.append(cyc_box_num[idx])


#设定画布。dpi越大图越清晰，绘图时间越久
fig=plt.figure(figsize=(4, 4), dpi=300)
# #导入数据
x1=list(np.arange(1, 151,2))
x2=list(np.arange(1, 151,2))
x3=list(np.arange(1, 151,2))
# x1=list(np.arange(1, 81))
# x2=list(np.arange(1, 55))
# x3=list(np.arange(1, 74))
y1=np.array(car_box_num)
y2=np.array(ped_box_num)
y3=np.array(cyc_box_num)
y1=np.array(car_box_num_fl)
y2=np.array(ped_box_num_fl)
y3=np.array(cyc_box_num_fl)

#绘图命令
plt.plot(x1, y1, lw=4, ls='-', c='b', label='Car')
plt.plot(x2, y2, lw=4, ls='-', c='r', label='Pedestrian')
plt.plot(x3, y3, lw=4, ls='-', c='k', label='Cyclist')
# plt.plot(x1, y1, lw=4, ls='-', c='b', alpha=0.5, label='Car')
# plt.plot(x2, y2, lw=4, ls='-', c='r', alpha=0.5, label='Pedestrian')
# plt.plot(x3, y3, lw=4, ls='-', c='k', alpha=0.5, label='Cyclist')
plt.xlabel("The number of points in the box", fontdict={'size': 10})
plt.ylabel("The number of boxes", fontdict={'size': 10})
plt.legend(loc='best')
# plt.title("", fontdict={'size': 20})
plt.plot()
#show出图形
plt.show()

print('done!')

# import matplotlib.pyplot as plt
#
#
# plt.figure(figsize=(20, 10), dpi=100)
# game = ['1-G1', '1-G2', '1-G3', '1-G4', '1-G5', '2-G1', '2-G2', '2-G3', '2-G4', '2-G5', '3-G1', '3-G2', '3-G3',
#         '3-G4', '3-G5', '总决赛-G1', '总决赛-G2', '总决赛-G3', '总决赛-G4', '总决赛-G5', '总决赛-G6']
# scores = [23, 10, 38, 30, 36, 20, 28, 36, 16, 29, 15, 26, 30, 26, 38, 34, 33, 25, 28, 40, 28]
# rebounds = [17, 6, 12, 6, 10, 8, 11, 7, 15, 11, 6, 11, 10, 9, 16, 13, 9, 10, 12, 13, 14]
# assists = [16, 7, 8, 10, 10, 7, 9, 5, 9, 7, 12, 4, 11, 8, 10, 9, 9, 8, 8, 7, 10]
# plt.plot(game, scores, c='red', label="得分")
# plt.plot(game, rebounds, c='green', linestyle='--', label="篮板")
# plt.plot(game, assists, c='blue', linestyle='-.', label="助攻")
# plt.scatter(game, scores, c='red')
# plt.scatter(game, rebounds, c='green')
# plt.scatter(game, assists, c='blue')
# plt.legend(loc='best')
# plt.yticks(range(0, 50, 5))
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("赛程", fontdict={'size': 16})
# plt.ylabel("数据", fontdict={'size': 16})
# plt.title("NBA2020季后赛詹姆斯数据", fontdict={'size': 20})
# plt.show()



#
#导入库
# import matplotlib.pyplot as plt
# import numpy as np
# #设定画布。dpi越大图越清晰，绘图时间越久
# fig=plt.figure(figsize=(4, 4), dpi=300)
# #导入数据
# x=list(np.arange(1, 21))
# y=np.random.randn(20)
# #绘图命令
# plt.plot(x, y, lw=4, ls='-', c='b', alpha=0.1)
# plt.plot()
# #show出图形
# plt.show()
# #保存图片
# fig.savefig("画布")