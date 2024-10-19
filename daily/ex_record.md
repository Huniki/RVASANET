# 实验记录
# Radar部分！

## 检测模型
config文件：vod_pointpillar
这部分实验是因为在做分割辅助检测任务时，最大的bs设置只能是4，所以考虑到bs对训练的影响“vod论文中，作者提到bs大训练效果会好”。这里对bs为4进行detection的单独训练。
epoch：100
bs：4
extra_tag:only_detection_bs4
实验结果：
可视化地址：
loss设置：

loss最终降低到
评价结果：
2024-02-11 23:07:29,226   INFO  recall_roi_0.3: 0.000000
2024-02-11 23:07:29,226   INFO  recall_rcnn_0.3: 0.443002
2024-02-11 23:07:29,226   INFO  recall_roi_0.5: 0.000000
2024-02-11 23:07:29,226   INFO  recall_rcnn_0.5: 0.300084
2024-02-11 23:07:29,226   INFO  recall_roi_0.7: 0.000000
2024-02-11 23:07:29,226   INFO  recall_rcnn_0.7: 0.103019
Entire annotated area: 
Car: 39.322073439466024 
Pedestrian: 30.244303900112783 
Cyclist: 63.81408586403591 
mAP: 44.4601544012049 
Driving corridor area: 
Car: 71.72858607505911 
Pedestrian: 41.05866065562582 
Cyclist: 84.85895006257851 
mAP: 65.88206559775448

## 分割辅助检测
config文件：pillarseg
Unet分割辅助检测，分割loss使用ce_loss，控制bs为4，分割类别为4类别。
epoch：100
bs：4
extra_tag:pillarseg_radar_4class_celoss
实验结果：
可视化地址：
loss设置：
2024-03-03 12:51:25,267   INFO  recall_roi_0.3: 0.000000
2024-03-03 12:51:25,267   INFO  recall_rcnn_0.3: 0.463796
2024-03-03 12:51:25,267   INFO  recall_roi_0.5: 0.000000
2024-03-03 12:51:25,267   INFO  recall_rcnn_0.5: 0.306101
2024-03-03 12:51:25,267   INFO  recall_roi_0.7: 0.000000
2024-03-03 12:51:25,267   INFO  recall_rcnn_0.7: 0.103863
Entire annotated area: 
Car: 40.40572789178932 
Pedestrian: 32.69896724260389 
Cyclist: 67.10654096588887 
mAP: 46.737078700094024 
Driving corridor area: 
Car: 71.7520278222058 
Pedestrian: 43.111430902498476 
Cyclist: 87.60962779004902 
mAP: 67.4910288382511

## 分割辅助检测
config文件：pillarseg
Unet分割辅助检测，分割loss使用ce_loss，这里使用bs为12，分割类别为4类别。
epoch：100
bs：12
extra_tag:pillarseg_radar_4class_celoss_bs12
实验结果：
可视化地址：
loss设置：
2024-03-08 19:02:30,332   INFO  recall_roi_0.3: 0.000000
2024-03-08 19:02:30,332   INFO  recall_rcnn_0.3: 0.452185
2024-03-08 19:02:30,332   INFO  recall_roi_0.5: 0.000000
2024-03-08 19:02:30,332   INFO  recall_rcnn_0.5: 0.305151
2024-03-08 19:02:30,333   INFO  recall_roi_0.7: 0.000000
2024-03-08 19:02:30,333   INFO  recall_rcnn_0.7: 0.105658
Entire annotated area: 
Car: 38.86159496290139 
Pedestrian: 32.88176307046161 
Cyclist: 65.20466172535026 
mAP: 45.649339919571084 
Driving corridor area: 
Car: 70.83764324746954 
Pedestrian: 42.87079386912905 
Cyclist: 87.34236818736902 
mAP: 67.01693510132253 

## 分割辅助检测_单类别
config文件：pillarseg
Unet分割辅助检测，分割loss使用ce_loss，控制bs为4，分割类别为1类别。
epoch：100
bs：4
extra_tag:pillarseg_radar_1class_celoss
实验结果：
可视化地址：
loss设置：
2024-03-06 12:58:51,828   INFO  recall_roi_0.3: 0.000000
2024-03-06 12:58:51,828   INFO  recall_rcnn_0.3: 0.454190
2024-03-06 12:58:51,828   INFO  recall_roi_0.5: 0.000000
2024-03-06 12:58:51,828   INFO  recall_rcnn_0.5: 0.308423
2024-03-06 12:58:51,828   INFO  recall_roi_0.7: 0.000000
2024-03-06 12:58:51,828   INFO  recall_rcnn_0.7: 0.107769
Entire annotated area: 
Car: 39.77978769115688 
Pedestrian: 31.655659856121304 
Cyclist: 65.31907594176839 
mAP: 45.58484116301552 
Driving corridor area: 
Car: 71.74375961360849 
Pedestrian: 42.22760382725264 
Cyclist: 86.72314410589628 
mAP: 66.89816918225246 
**Q** : 为什么训练的时候loss为负数啦！----结果：多类别分割和单类别分割更改时需要注意label的更改！
**结论** ： 相比与多类别分割的，单类别的效果反而差。这个可以通过lidar进行实验验证！

## 分割检测模型+attn
python tools/train.py --cfg_file tools/cfgs/kitti_models/pillarseg.yaml --batch_size 4 --epochs 100 --extra_tag pillarseg_radar_1class_celoss_atth
config文件：pillarseg
radar作为输入，分割辅助进行目标检测---单类别分割+attn
epoch：100
bs：4
extra_tag: pillarseg_radar_1class_celoss_attn
实验结果：
可视化地址：
loss设置：
2024-03-06 13:05:31,230   INFO  recall_roi_0.3: 0.000000
2024-03-06 13:05:31,230   INFO  recall_rcnn_0.3: 0.458940
2024-03-06 13:05:31,230   INFO  recall_roi_0.5: 0.000000
2024-03-06 13:05:31,230   INFO  recall_rcnn_0.5: 0.311906
2024-03-06 13:05:31,230   INFO  recall_roi_0.7: 0.000000
2024-03-06 13:05:31,230   INFO  recall_rcnn_0.7: 0.106396
Entire annotated area: 
Car: 39.56493879244713 
Pedestrian: 31.837798328498778 
Cyclist: 66.4519580128246 
mAP: 45.951565044590176 
Driving corridor area: 
Car: 71.6855899014935 
Pedestrian: 43.35313729081767 
Cyclist: 88.63821858923761 
mAP: 67.89231526051627 
**结论** ：这相比与单类别分割辅助的话，是有一些涨点的！


## 分割检测模型+attn
python tools/train.py --cfg_file tools/cfgs/kitti_models/pillarseg.yaml --batch_size 4 --epochs 120 --extra_tag pillarseg_radar_4class_celoss_atth
config文件：pillarseg
radar作为输入，分割辅助进行目标检测---4类别分割+attn，这里训练120个epochs
epoch：120
bs：4
extra_tag: pillarseg_radar_4class_celoss_attn
实验结果：
可视化地址：
loss设置：
2024-03-07 11:05:30,918   INFO  recall_roi_0.3: 0.000000
2024-03-07 11:05:30,918   INFO  recall_rcnn_0.3: 0.447118
2024-03-07 11:05:30,918   INFO  recall_roi_0.5: 0.000000
2024-03-07 11:05:30,918   INFO  recall_rcnn_0.5: 0.284568
2024-03-07 11:05:30,918   INFO  recall_roi_0.7: 0.000000
2024-03-07 11:05:30,918   INFO  recall_rcnn_0.7: 0.095313
Entire annotated area: 
Car: 37.822430439461684 
Pedestrian: 31.64571119116574 
Cyclist: 67.37546170445725 
mAP: 45.61453444502823 
Driving corridor area: 
Car: 71.45708040714598 
Pedestrian: 41.88384581187408 
Cyclist: 88.01793168636682 
mAP: 67.11961930179562 
**结论：**注意力看样子效果一般，是不是注意力的地方加错啦。在pillar_vfe部分，attn加载是有问题的！

## 分割检测模型+attn
python tools/train.py --cfg_file tools/cfgs/kitti_models/pillarseg.yaml --batch_size 4 --epochs 100 --extra_tag pillarseg_radar_4class_celoss_fusion_attn
config文件：pillarseg
radar作为输入，分割辅助进行目标检测---4类别分割+attn这里的attn是指在fusion模块做的通道注意力
epoch：100
bs：4
extra_tag: pillarseg_radar_4class_celoss_fusion_attn
实验结果：
可视化地址：
loss设置：
2024-03-07 20:46:41,465   INFO  recall_roi_0.3: 0.000000
2024-03-07 20:46:41,465   INFO  recall_rcnn_0.3: 0.461157
2024-03-07 20:46:41,465   INFO  recall_roi_0.5: 0.000000
2024-03-07 20:46:41,465   INFO  recall_rcnn_0.5: 0.303884
2024-03-07 20:46:41,465   INFO  recall_roi_0.7: 0.000000
2024-03-07 20:46:41,465   INFO  recall_rcnn_0.7: 0.096158
Entire annotated area: 
Car: 38.23744786549993 
Pedestrian: 30.91300148140157 
Cyclist: 67.78353178867037 
mAP: 45.64466037852395 
Driving corridor area: 
Car: 71.73726797545632 
Pedestrian: 41.729629555832325 
Cyclist: 87.85928716735344 
mAP: 67.1087282328807 

## 分割检测模型+attn
python tools/train.py --cfg_file tools/cfgs/kitti_models/pillarseg.yaml --batch_size 4 --epochs 100 --extra_tag pillarseg_radar_4class_celoss_fusion_attn_nobnrelu
config文件：pillarseg
radar作为输入，分割辅助进行目标检测---4类别分割+attn这里的attn是指在fusion模块做的通道注意力，在降维那部分，没有使用bn、relu
epoch：100
bs：4
extra_tag: pillarseg_radar_4class_celoss_fusion_attn_nobnrelu
实验结果：
可视化地址：
loss设置：
2024-03-09 11:37:58,234   INFO  recall_roi_0.3: 0.000000
2024-03-09 11:37:58,235   INFO  recall_rcnn_0.3: 0.461474
2024-03-09 11:37:58,235   INFO  recall_roi_0.5: 0.000000
2024-03-09 11:37:58,235   INFO  recall_rcnn_0.5: 0.311484
2024-03-09 11:37:58,235   INFO  recall_roi_0.7: 0.000000
2024-03-09 11:37:58,235   INFO  recall_rcnn_0.7: 0.105658
Entire annotated area: 
Car: 40.56460191228534 
Pedestrian: 32.5153657021698 
Cyclist: 66.61238428306248 
mAP: 46.564117299172544 
Driving corridor area: 
Car: 72.35919604459257 
Pedestrian: 43.491874497280655 
Cyclist: 87.89021753225987 
mAP: 67.9137626913777 
**结论：** 近距离有提升！

## 分割检测模型+attn
python tools/train.py --cfg_file tools/cfgs/kitti_models/pillarseg.yaml --batch_size 16 --epochs 100 --extra_tag pillarseg_radar_4class_celoss_fusion_attn_nobnrelu
config文件：pillarseg
radar作为输入，分割辅助进行目标检测---4类别分割+attn这里的attn是指在fusion模块做的通道注意力，在降维那部分，没有使用bn、relu，同时将bs增加到16
epoch：100
bs：16
extra_tag: pillarseg_radar_4class_celoss_fusion_attn_nobnrelu_bs16
实验结果：
可视化地址：
loss设置：
2024-03-09 22:11:45,540   INFO  recall_roi_0.3: 0.000000
2024-03-09 22:11:45,540   INFO  recall_rcnn_0.3: 0.440046
2024-03-09 22:11:45,540   INFO  recall_roi_0.5: 0.000000
2024-03-09 22:11:45,540   INFO  recall_rcnn_0.5: 0.297551
2024-03-09 22:11:45,541   INFO  recall_roi_0.7: 0.000000
2024-03-09 22:11:45,541   INFO  recall_rcnn_0.7: 0.102174
Entire annotated area: 
Car: 37.94005976250451 
Pedestrian: 30.202265814733174 
Cyclist: 65.19174963710145 
mAP: 44.44469173811304 
Driving corridor area: 
Car: 70.871780416708 
Pedestrian: 41.37392340098627 
Cyclist: 87.03769950222348 
mAP: 66.42780110663925 

## 分割检测模型+attn_bs12
python tools/train.py --cfg_file tools/cfgs/kitti_models/pillarseg.yaml --batch_size 12 --epochs 100 --extra_tag pillarseg_radar_4class_celoss_fusion_attn_bs12
config文件：pillarseg
radar作为输入，分割辅助进行目标检测---4类别分割+attn这里的attn是指在fusion模块做的通道注意力,同时把bs增加到12
epoch：100
bs：12
extra_tag: pillarseg_radar_4class_celoss_fusion_attn_bs12
实验结果：
可视化地址：
loss设置：
2024-03-08 10:29:46,792   INFO  recall_roi_0.3: 0.000000
2024-03-08 10:29:46,793   INFO  recall_rcnn_0.3: 0.468018
2024-03-08 10:29:46,793   INFO  recall_roi_0.5: 0.000000
2024-03-08 10:29:46,793   INFO  recall_rcnn_0.5: 0.322039
2024-03-08 10:29:46,793   INFO  recall_roi_0.7: 0.000000
2024-03-08 10:29:46,793   INFO  recall_rcnn_0.7: 0.105552
Entire annotated area: 
Car: 40.52085128534207 
Pedestrian: 31.939266513493912 
Cyclist: 66.10534489975997 
mAP: 46.18848756619865 
Driving corridor area: 
Car: 71.69492532232815 
Pedestrian: 43.34962366368682 
Cyclist: 86.72162721755872 
mAP: 67.2553920678579 


# Lidar部分
## 检测模型
config文件：vod_pointpillar_lidar
lidar作为输入，仅进行目标检测
epoch：100
bs：8  （可能存在一个问题，bs对模型的影响，可能需要把这个调整到4，统一实验）
only_detection_bs4_lidar
实验结果：
可视化地址：
loss设置：

loss最终降低到
评价结果：
2024-03-02 10:51:46,585   INFO  recall_roi_0.3: 0.000000
2024-03-02 10:51:46,585   INFO  recall_rcnn_0.3: 0.570825
2024-03-02 10:51:46,585   INFO  recall_roi_0.5: 0.000000
2024-03-02 10:51:46,585   INFO  recall_rcnn_0.5: 0.491767
2024-03-02 10:51:46,585   INFO  recall_roi_0.7: 0.000000
2024-03-02 10:51:46,586   INFO  recall_rcnn_0.7: 0.216593
Entire annotated area: 
Car: 59.69912272947302 
Pedestrian: 37.55790065254944 
Cyclist: 54.9565313902144 
mAP: 50.73785159074561 
Driving corridor area: 
Car: 89.94565807473755 
Pedestrian: 48.60044059023051 
Cyclist: 81.47621179234474 
mAP: 73.3407701524376


## 检测模型
config文件：vod_pointpillar_lidar
lidar作为输入，仅进行目标检测
epoch：100
bs：4
extra_tag: lidar_detection_bs4
实验结果：
可视化地址：
loss设置：
2024-03-02 22:32:18,853   INFO  recall_roi_0.3: 0.000000
2024-03-02 22:32:18,853   INFO  recall_rcnn_0.3: 0.558792
2024-03-02 22:32:18,853   INFO  recall_roi_0.5: 0.000000
2024-03-02 22:32:18,853   INFO  recall_rcnn_0.5: 0.498944
2024-03-02 22:32:18,853   INFO  recall_roi_0.7: 0.000000
2024-03-02 22:32:18,853   INFO  recall_rcnn_0.7: 0.225459
Entire annotated area: 
Car: 60.03112127378244 
Pedestrian: 32.193026649840014 
Cyclist: 62.92615817542795 
mAP: 51.71676869968346 
Driving corridor area: 
Car: 90.87748867586643 
Pedestrian: 41.72371765038436 
Cyclist: 83.05683039055208 
mAP: 71.8860122389343 

## 分割检测模型
config文件：pillarseg_lidar
lidar作为输入，分割辅助进行目标检测----4类别分割
epoch：100
bs：4
extra_tag: pillarseg_lidar_4class_celoss
实验结果：
可视化地址：
loss设置：
2024-03-04 10:20:54,858   INFO  recall_roi_0.3: 0.000000
2024-03-04 10:20:54,858   INFO  recall_rcnn_0.3: 0.632257
2024-03-04 10:20:54,859   INFO  recall_roi_0.5: 0.000000
2024-03-04 10:20:54,859   INFO  recall_rcnn_0.5: 0.559848
2024-03-04 10:20:54,859   INFO  recall_roi_0.7: 0.000000
2024-03-04 10:20:54,859   INFO  recall_rcnn_0.7: 0.275491
Entire annotated area: 
Car: 61.29235850304009 
Pedestrian: 47.92118729542595 
Cyclist: 66.74254991223563 
mAP: 58.65203190356723 
Driving corridor area: 
Car: 90.85614206581948 
Pedestrian: 51.9571553170166 
Cyclist: 87.5729101557916 
mAP: 76.7954025128759

## 分割检测模型
config文件：pillarseg_lidar
lidar作为输入，分割辅助进行目标检测----1类别分割
epoch：100
bs：4
extra_tag: pillarseg_lidar_1class_celoss
实验结果：
可视化地址：
loss设置：
2024-03-06 21:55:51,325   INFO  recall_roi_0.3: 0.000000
2024-03-06 21:55:51,325   INFO  recall_rcnn_0.3: 0.628457
2024-03-06 21:55:51,325   INFO  recall_roi_0.5: 0.000000
2024-03-06 21:55:51,325   INFO  recall_rcnn_0.5: 0.564915
2024-03-06 21:55:51,325   INFO  recall_roi_0.7: 0.000000
2024-03-06 21:55:51,325   INFO  recall_rcnn_0.7: 0.284146
Entire annotated area: 
Car: 62.19128711071442 
Pedestrian: 41.37350945133439 
Cyclist: 65.71477583114763 
mAP: 56.426524131065484 
Driving corridor area: 
Car: 90.81493880112983 
Pedestrian: 57.3481939369032 
Cyclist: 85.65762882865218 
mAP: 77.94025385556174 
**结论：** 标注区域结果下降，近距离结果提升！
