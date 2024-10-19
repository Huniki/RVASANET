import pickle

train_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/kitti_infos_train.pkl'
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

print('done!')
