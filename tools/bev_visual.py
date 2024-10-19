import cv2
import numpy as np
import pickle
def pcl2bev(pcl_path, save_dir, ratio, width, height):
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
            cv2.circle(img, (v,u), 1, 255, 2)
            # img[u,v] = 0
    # img = np.flip(img, 0)
    # cv2.imshow('bev_image',img)
    # cv2.waitKey()
    # cv2.destroyWindow()
    save_dir = save_dir + pcd_path.split('/')[-1][:-3] + 'png'
    cv2.imwrite(save_dir, img.astype("uint8"))


if __name__=="__main__":
    # 4D radar
    # val_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_val.pkl'
    # # train_pkl_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_train.pkl'
    # with open(val_pkl, 'rb') as f:
    #     train_data = pickle.load(f)
    # data_root = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar/training/velodyne/'
    # save_dir = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar/bev_pts/'

    # Lidar
    val_pkl = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/kitti_infos_val.pkl'
    # train_pkl_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar_5frames/kitti_infos_train.pkl'
    with open(val_pkl, 'rb') as f:
        train_data = pickle.load(f)
    data_root = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/training/velodyne/'
    save_dir = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/bev_pts/'
    for i in range(len(train_data)):

        pcd_path = train_data[i]['point_cloud']['lidar_idx']
        pcd_path = data_root + pcd_path + '.bin'
        pcl2bev(pcd_path,save_dir,6.25,51.2,51.2)
        print(i)

