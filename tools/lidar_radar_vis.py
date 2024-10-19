import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt




####  彩虹色
if __name__ == '__main__':
    # 加载点云数据
    radar_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/radar/training/velodyne/08000.bin'
    lidar_path = '/home/rpf/Pillarseg/OpenPCDet/data/view_of_delft/lidar/training/velodyne/08000.bin'
    radar_pts = o3d.io.read_point_cloud(radar_path)
    lidar_pts = o3d.io.read_point_cloud(lidar_path)


#     points = np.load('C:/Users/mi/Desktop/trash/bed_ret_mn40.npy')
#     points = points[0, :, :]  # 里面有多个点云  [5, 8192,3]
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
#
#     # 获取高度范围
#     min_height = np.min(points[:, 2])
#     max_height = np.max(points[:, 2])
#
#     # 根据高度值计算归一化的值
#     normalized_heights = (points[:, 2] - min_height) / (max_height - min_height)
#
#     # 创建彩虹色映射
#     cmap = plt.cm.get_cmap('rainbow')
#
#     # 将归一化的高度值映射到彩虹色映射上
#     gradient_colors = cmap(normalized_heights)
#
#     # 将颜色数组赋值给点云对象
#     pcd.colors = o3d.utility.Vector3dVector(gradient_colors[:, :3])
#
#     # 显示点云
#     o3d.visualization.draw_geometries([pcd], window_name="pc show", point_show_normal=False,
#                                       width=800, height=600)
#
# # !/usr/bin/python3
# # -*- coding: utf-8 -*-
#
# import os
# import open3d as o3d
# import numpy as np
# import time
#
#
# def save_view_point(pcd_numpy, filename):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     pcd = o3d.open3d.geometry.PointCloud()
#     pcd.points = o3d.open3d.utility.Vector3dVector(pcd_numpy)
#     vis.add_geometry(pcd)
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
#     vis.add_geometry(axis)
#     vis.run()  # user changes the view and press "q" to terminate
#     param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#     o3d.io.write_pinhole_camera_parameters(filename, param)
#     vis.destroy_window()
#
#
# def draw_color(color_h, color_l, pcd):
#     color_h = np.array(color_h, np.float32).reshape(1, 3)
#     color_l = np.array(color_l, np.float32).reshape(1, 3)
#
#     raw_points = np.array(pcd.points).copy()
#     hight = raw_points[:, 2:]
#     hight = np.clip(hight, -3, 1)
#     colors = color_l + (hight - (-3)) * (color_h - color_l) / 4.0
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     return pcd
#
#
# def vis_cons(files_dir, vis_detect_result=False):
#     files = os.listdir(files_dir)
#     pcds = []
#     for f in files:
#         pcd_path = os.path.join(files_dir, f)
#         pcd = o3d.open3d.geometry.PointCloud()  # 创建点云对象
#         raw_point = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)[:, :3]
#         pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)  # 将点云数据转换为Open3d可以直接使用的数据类型
#         pcd = draw_color([1.0, 0.36, 0.2], [1.0, 0.96, 0.2], pcd)
#         pcds.append(pcd)
#     # if vis_detect_result:
#     #    batch_results = np.load('batch_results.npy',allow_pickle=True)
#
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     opt = vis.get_render_option()
#     opt.background_color = np.asarray([0, 0, 0])
#     opt.point_size = 1
#     opt.show_coordinate_frame = False
#     if os.path.exists("viewpoint.json"):
#         ctr = vis.get_view_control()
#         param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
#         ctr.convert_from_pinhole_camera_parameters(param)
#
#     for i in range(len(pcds)):
#         vis.clear_geometries()
#         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
#         vis.add_geometry(axis)
#         vis.add_geometry(pcds[i])
#         ctr.convert_from_pinhole_camera_parameters(param)
#         time.sleep(0.1)
#         vis.run()
#     vis.destroy_window()
#
#
# if __name__ == '__main__':
#     exp_pcd_file = r"F:\Datasets\PointCloud\KITTI_track\KITTI_tracking\training\velodyne\0001"
#     view_pcd = np.fromfile(os.path.join(exp_pcd_file, '000000.bin'), dtype=np.float32).reshape(-1, 4)[:, :3]
#     save_view_point(view_pcd, "viewpoint.json")
#     vis_cons(exp_pcd_file)

